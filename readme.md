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
table.insert(key, key_len, val, sizeof val);
void* entry = table.find(key, key_len);
std::string_view k = mtht::read_key(entry);   // the stored key
std::string_view v = mtht::read_str(entry);   // its value
int answer = mtht::read_as<int>(table.find("answer"));  // 0 if absent
table.erase(key, key_len);

// get a pointer to a stored item (outlives an erase)
void* held = table.find(key, key_len);
if (mtht::ptr_is_live(held)) { /* value still valid */ }

// iterate every live entry, in insertion order
auto cur = table.iter_start();
while (void* entry = table.iter(cur)) {
    // iter gives the same item pointers as find()
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
| 4–24 B | 16–128 B | **43.9M / 15.3M**  | 32.7M / 9.2M | 39.0M / 9.6M  | 23.1M / 6.3M |
| 8 B    | 256 B    | **114.0M / 13.4M** | 58.7M / 5.5M | 65.3M / 5.8M  | 36.2M / 4.8M |
| 256 B  | 8 B      | **8.9M / 8.1M**    | 5.9M / 3.7M  | 7.2M / 4.5M   | 5.8M / 2.8M  |
| 64 B   | 64 B     | **38.1M / 10.3M**  | 16.2M / 5.4M | 21.8M / 5.9M  | 14.4M / 3.7M |
| 256 B  | 256 B    | **7.7M / 5.6M**    | 5.3M / 2.6M  | 6.6M / 3.1M   | 5.4M / 2.1M  |

Compared to udense, MTHT reads are 1.12–1.75x and inserts 1.39–1.80x the speed here, using 0.78–1.06x the memory.

#### comparison : 16k keys

| keys   | value    | MTHT               | robin         | udense        | umap         |
|--------|----------|--------------------|---------------|---------------|--------------|
| 4–24 B | 16–128 B | **49.2M / 21.8M**  | 35.7M / 12.7M | 45.3M / 13.9M | 28.7M / 8.0M |
| 8 B    | 256 B    | **181.4M / 21.4M** | 63.6M / 6.6M  | 86.1M / 7.5M  | 48.2M / 5.8M |
| 256 B  | 8 B      | **19.4M / 12.8M**  | 16.0M / 5.5M  | 18.4M / 5.9M  | 14.4M / 3.8M |
| 64 B   | 64 B     | **49.1M / 25.6M**  | 33.3M / 9.3M  | 45.4M / 9.9M  | 31.5M / 5.6M |
| 256 B  | 256 B    | **18.4M / 8.5M**   | 15.7M / 3.7M  | 17.9M / 3.9M  | 14.3M / 2.8M |

Compared to udense, MTHT reads are 1.03–2.11x and inserts 1.43–2.59x the speed here, using 0.70–1.05x the memory.


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
| MTHT                    | **25.3M** | 16.4M     | **58.4**  |
| tsl::robin_map          | 17.0M     | 13.2M     | 85.3      |
| ankerl::unordered_dense | 24.3M     | **17.5M** | 70.1      |
| std::unordered_map      | 16.0M     | 7.1M      | 78.0      |

#### int keys

|                         | reads/s    | inserts/s  | bytes/key |
|-------------------------|------------|------------|-----------|
| MTHT                    | 93.0M      | 41.6M      | 24.0      |
| tsl::robin_map          | 81.9M      | 60.0M      | 32.0      |
| ankerl::unordered_dense | **199.6M** | **106.3M** | **23.3**  |
| std::unordered_map      | 119.2M     | 16.9M      | 42.7      |

#### load sweep (ints)

| load  | reads/s | probes/read |
|-------|---------|-------------|
| 0.083 | 205.9M  | 1.04        |
| 0.411 | 129.3M  | 1.25        |
| 0.686 | 87.8M   | 1.54        |

### iteration

Iteration over every live key-value pair. String keys and string values.

|                         | items/s  |
|-------------------------|----------|
| MTHT `iter()`           | **400M** |
| tsl::robin_map          | 149M     |
| ankerl::unordered_dense | 288M     |



