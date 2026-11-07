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
| 4–24 B | 16–128 B | **44.3M / 16.0M**  | 33.5M / 8.9M | 39.4M / 11.2M | 23.4M / 6.1M |
| 8 B    | 256 B    | **111.6M / 14.0M** | 60.5M / 5.7M | 65.2M / 6.0M  | 36.8M / 4.7M |
| 256 B  | 8 B      | **9.4M / 8.1M**    | 6.1M / 3.9M  | 7.5M / 4.8M   | 6.0M / 2.9M  |
| 64 B   | 64 B     | **39.7M / 12.8M**  | 16.3M / 5.7M | 21.8M / 7.1M  | 15.3M / 4.1M |
| 256 B  | 256 B    | **7.9M / 5.7M**    | 5.2M / 2.7M  | 6.7M / 3.1M   | 5.5M / 2.2M  |

Compared to udense, MTHT reads are 1.12–1.82x and inserts 1.43–2.33x the speed here, using 0.77–1.06x the memory.

#### comparison : 16k keys

| keys   | value    | MTHT               | robin         | udense        | umap         |
|--------|----------|--------------------|---------------|---------------|--------------|
| 4–24 B | 16–128 B | **51.4M / 24.2M**  | 35.9M / 13.5M | 46.2M / 15.4M | 29.0M / 8.6M |
| 8 B    | 256 B    | **183.1M / 21.5M** | 65.2M / 8.0M  | 88.4M / 7.4M  | 49.4M / 6.1M |
| 256 B  | 8 B      | **19.4M / 13.3M**  | 16.1M / 5.9M  | 18.6M / 5.8M  | 14.4M / 3.8M |
| 64 B   | 64 B     | **48.4M / 24.8M**  | 33.3M / 9.5M  | 44.8M / 9.4M  | 31.8M / 5.7M |
| 256 B  | 256 B    | **18.6M / 10.6M**  | 15.9M / 4.1M  | 18.1M / 4.1M  | 14.4M / 2.8M |

Compared to udense, MTHT reads are 1.03–2.07x and inserts 1.57–2.91x the speed here, using 0.74–1.05x the memory.


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
```
`--n1=20` is a 2^20-slot table. `--no-log` skips the run log.


#### string keys

Keys are mixed length: 70% at 1–16 B, 27% at 17–64 B, 3% at 65–256 B.

|                         | reads/s   | inserts/s | bytes/key |
|-------------------------|-----------|-----------|-----------|
| MTHT                    | **26.3M** | 17.1M     | **58.4**  |
| tsl::robin_map          | 17.2M     | 13.5M     | 85.3      |
| ankerl::unordered_dense | 25.9M     | **19.2M** | 70.1      |
| std::unordered_map      | 16.3M     | 7.7M      | 78.0      |

#### int keys

|                         | reads/s    | inserts/s  | bytes/key |
|-------------------------|------------|------------|-----------|
| MTHT                    | 92.3M      | 43.5M      | 24.0      |
| tsl::robin_map          | 81.8M      | 60.8M      | 32.0      |
| ankerl::unordered_dense | **200.7M** | **107.3M** | **23.3**  |
| std::unordered_map      | 121.0M     | 17.2M      | 42.7      |

#### load sweep (ints)

| load  | reads/s | probes/read |
|-------|---------|-------------|
| 0.083 | 197.9M  | 1.04        |
| 0.411 | 125.5M  | 1.25        |
| 0.686 | 87.3M   | 1.54        |


