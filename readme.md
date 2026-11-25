# multilevel ternary hash tables


## about

This is a hashtable which:
- uses a novel design
- has good performance when keys or values are >8 byte strings
- uses a relatively small amount of code
- can recover an item's key + value from a single pointer
- can be iterated fast in insertion order

I thought of the concept for this in 2019, and implemented a proof of concept in 2020. I didn't have the low-level programming knowledge or patience to get good performance from it, but now, LLMs do. Also, for various reasons, I need something public and technical that was made with AI. So I had Deepseek V4.1 Flash make this, but I really had to push it.

There are some benchmarks and tests, but you probably should not use something like a hashtable in production if 3rd-party ones haven't been done.


## concept

### multilevel linear probing

Linear probing hash tables (LPHTs) have good performance at 0.5 load factor.

Suppose we have a LPHT "T1" with a high load factor, but after a small number of collisions we move to a second LPHT "T2" which has a low load factor. Insertions would then have the good performance of T2 with some constant overhead from checking T1 first.

For good read performance, we want a good chance of finding results in the first table checked. Let's say T2 is 1/4 the size of T1, and we check 4 slots in T1 before going to T2.

That gives a better load factor, but every read from T2 will obviously require at least 5 probes, and a lot of the data is in T2. So, the average number of probes is relatively high.

Obviously reads would require less probes at lower load factors, but can we do better on a conceptual level?

### ternary hash tables

Suppose T2.size = T1.size/8 and we aim for no more than 3 probes in T1 before going to T2. To get a good load factor, insertions must check more than 3 slots for empty spaces in T1 before going to T2. This is possible by shifting existing elements to either the right or left to make space for insertions, while keeping offsets in the range (-1 to +1).

Add a 1-byte header to each table element, containing its offset from its original hash position. This header value means:
- 0: empty element
- 1: offset -1
- 2: offset 0
- 3: offset 1

When inserting an element in an occupied slot, push other elements out of the way to make room for it.

Initially, every insertion has an offset of 0. When a table is mostly full, about half of elements will have been pushed an even number of times, so about half the elements will have an offset of 0.

Note that if the initial probe offset is -1 or +1, we only need to check in one direction from it.


### implementation notes

Let's supppose keys and data are often long, so we need a pointer to them instead of packing them in the hash table directly. And let's assume a 64-bit system.

User space pointers are 48 bits, and headers are 2 bits. That leaves 14 bits for other uses.

We can use 6 bits to hold the highest offset of elements pushed from that position to T2, and search from that position backwards in T2. That makes searches in T2 fast.

The other 8 bits can be a "fingerprint" of the key, a part of the key hash that we check before loading and comparing against the full key.

T2 is 1/8 the size of T1. If T2 fills to 3/4 the table size doubles. This generally happens at 3/4 loading of T1.



## usage

Header-only. Copy `include/mtht.hpp` or add `-Iinclude`.

```cpp
#include "mtht.hpp"

// make a table
std::size_t slot_count = 1u << 16;  // Initial slots; can grow.
// Initial key-value storage; can grow. Default is 4kb.
std::size_t kv_store_bytes = 64u << 20;
mtht::Table table(slot_count, kv_store_bytes);

// utility
std::size_t occupied = table.size();

// strings
std::string key = "user-1042";
std::string val = "ada@example.com";
table.insert(key, val);
std::string email = table.find_str(key);

// a value stored as a pointer
int foo = 42;
int* p = &foo;
table.insert("cfg", &p, sizeof p);
int* got = table.find_as<int*>("cfg");  // nullptr if absent

// a value copied out as any fixed-size type
table.insert("answer", &foo, sizeof foo);
int answer = table.find_as<int>("answer");

// raw bytes
const char key[4] = {'a', 'b', 'c', 'd'};
std::size_t key_len = sizeof key;
std::uint8_t val[64] = {};
void* entry = table.insert(key, key_len, val, sizeof val);  // the stored item
void* found = table.find(key, key_len);
std::string_view k = mtht::read_key(found);   // the stored key
std::string_view v = mtht::read_str(found);   // its value
int answer = mtht::read_as<int>(table.find("answer"));  // 0 if absent
table.erase(key, key_len);

// get a pointer to a stored item (outlives an erase)
void* held = table.find(key, key_len);
if (mtht::ptr_is_live(held)) { /* value still valid */ }

// iterate every live entry, in order of latest insertion
auto cur = table.iter_start();
while (void* entry = table.iter(cur)) {
    // iter gives the same item pointers as find()
    std::string_view k = mtht::read_key(entry);
    std::string_view v = mtht::read_str(entry);
}

// iterate in the order each key was first stored
// overwritten elements will require lookups and make this slower
auto first = table.iter_start();
while (void* entry = table.iter_first(first)) {
    std::string_view k = mtht::read_key(entry);
    std::string_view v = mtht::read_str(entry);
}

// iterate a function over the live table entries
table.map([](std::string_view k, std::string_view v) {
    // ...
});

// return an array of function results over the live table entries
std::unique_ptr<std::size_t[]> lens = table.map(
    [](std::string_view k, std::string_view v) { return k.size() + v.size(); }
);

// walk the same table twice at once, one state per walk
auto a = table.iter_start();
auto b = table.iter_start();
void* from_a = table.iter(a);
void* from_b = table.iter(b);  // independent of a

// using custom hash functions
struct Fnv {
    // hash a span, len != 8
    std::uint64_t operator()(const void* p, std::size_t len) const { ... }
    // hash a 8-byte key
    std::uint64_t operator()(const void* p) const { ... }
};
std::size_t slot_count = 1u << 16;
mtht::BasicTable<Fnv> custom_table(slot_count);
```

On a miss: `find` returns nullptr, `find_str` returns `""`.


## benchmarks


### with various strings

#### code

200k keys by default, same hash function for every table. `-n <keys>` changes the count.
```sh
clang++ -O2 -std=c++20 -march=native -Iinclude \
        -Ipeers/robin-map-master/include -Ipeers/ankerl-unordered_dense/include/ankerl \
        bench/longkeys.cpp -o temp/longkeys.exe
./temp/longkeys.exe --readme        # the table rows below, 200k keys
./temp/longkeys.exe --readme -n 16000   # same cells at 16k keys
```

#### comparison : 200k keys

Reads/s and inserts/s. Sizes in bytes.

| keys   | value    | MTHT               | robin        | udense        | umap         |
|--------|----------|--------------------|--------------|---------------|--------------|
| 4–24 B | 16–128 B | **67.5M / 18.6M**  | 37.4M / 11.7M | 43.4M / 11.7M | 25.3M / 7.2M |
| 8 B    | 256 B    | **123.4M / 14.0M** | 53.2M / 5.8M  | 60.8M / 5.9M  | 34.8M / 4.9M |
| 256 B  | 8 B      | **12.0M / 7.8M**   | 6.8M / 3.6M   | 8.1M / 4.4M   | 6.5M / 2.6M  |
| 64 B   | 64 B     | **41.0M / 12.3M**  | 21.5M / 5.7M  | 26.3M / 6.3M  | 18.8M / 3.7M |
| 256 B  | 256 B    | **8.4M / 5.6M**    | 5.7M / 2.5M   | 7.3M / 3.1M   | 5.7M / 2.1M  |

Compared to udense, MTHT reads are 1.15–1.87x and inserts 1.62–2.38x the speed here, using 0.77–1.06x the memory.

#### comparison : 16k keys

| keys   | value    | MTHT               | robin         | udense        | umap         |
|--------|----------|--------------------|---------------|---------------|--------------|
| 4–24 B | 16–128 B | **88.5M / 28.3M**  | 37.9M / 13.5M | 50.5M / 15.7M | 30.5M / 9.0M |
| 8 B    | 256 B    | **191.2M / 16.6M** | 55.1M / 7.3M  | 77.1M / 7.5M  | 44.6M / 5.9M |
| 256 B  | 8 B      | **19.4M / 12.8M**  | 12.9M / 4.9M  | 14.5M / 5.5M  | 11.6M / 3.4M |
| 64 B   | 64 B     | **47.2M / 27.2M**  | 31.6M / 8.9M  | 43.9M / 10.0M | 29.7M / 5.4M |
| 256 B  | 256 B    | **18.8M / 8.7M**   | 12.9M / 3.7M  | 14.6M / 3.7M  | 11.8M / 2.8M |

Compared to udense, MTHT reads are 1.11–2.17x and inserts 1.47–2.93x the speed here, using 0.74–1.05x the memory.


### with 64-bit int values

#### code

786k keys, 2^20 slots, 8-byte values, one hash for every table.
clang 23, `-O2`, best of 3.
```sh
clang++ -O2 -std=c++20 -march=native -Iinclude \
        -Ipeers/robin-map-master/include -Ipeers/ankerl-unordered_dense/include/ankerl \
        bench/mthtbench.cpp -o temp/mthtbench.exe
./temp/mthtbench.exe --suite=peers --n1=20   # reads and inserts, integer keys
./temp/mthtbench.exe --suite=strings         # reads and inserts, string keys
./temp/mthtbench.exe --suite=loads --n1=20   # reads across load
./temp/mthtbench.exe --suite=iterate         # iter() vs peers' iteration
```
`--n1=20` is a 2^20-slot table. `--no-log` skips the run log.


#### string keys

Keys are mixed length: 70% at 1–16 B, 27% at 17–64 B, 3% at 65–256 B.

|                         | reads/s   | inserts/s | bytes/key |
|-------------------------|-----------|-----------|-----------|
| MTHT                    | **35.0M** | 18.2M     | **58.4**  |
| tsl::robin_map          | 18.2M     | 14.0M     | 85.3      |
| ankerl::unordered_dense | 27.3M     | **18.5M** | 70.1      |
| std::unordered_map      | 16.3M     | 7.3M      | 78.0      |

#### int keys

|                         | reads/s    | inserts/s  | bytes/key |
|-------------------------|------------|------------|-----------|
| MTHT                    | 103.0M     | 47.2M      | 24.0      |
| tsl::robin_map          | 82.0M      | 60.6M      | 32.0      |
| ankerl::unordered_dense | **199.9M** | **106.7M** | **23.3**  |
| std::unordered_map      | 119.9M     | 17.0M      | 42.7      |

#### load sweep (ints)

| load  | reads/s | probes/read |
|-------|---------|-------------|
| 0.083 | 208.1M  | 1.04        |
| 0.411 | 136.9M  | 1.25        |
| 0.688 | 96.7M   | 1.54        |

### iteration

Iteration over every live key-value pair. String keys and string values.

|                         | items/s  |
|-------------------------|----------|
| MTHT `iter()`           | **396M** |
| tsl::robin_map          | 133M     |
| ankerl::unordered_dense | 281M     |



