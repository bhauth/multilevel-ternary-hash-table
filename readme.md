# multilevel ternary hash tables


## about

This is a hashtable which:
- uses a novel design
- has good performance when keys or values are >8 byte strings
- uses a relatively small amount of code
- has stable references to keys and values

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

// n1 slots, and store_bytes, the key store's initial size in bytes (0 = 4 KB,
// the default). Size it for the keys and values you expect. T2 opens at n1/8.
mtht::Table t(1u << 20, 64u << 20);

// Integer key.
uint64_t k = 12345;
MyValue v{};
t.insert(k, &v, sizeof v);
MyValue* got = static_cast<MyValue*>(t.find(k));
t.erase(k);

// String key: std::string, string literal, or string_view. The bytes are not
// owned; keep them alive for the call.
std::string name = "hello";
t.insert(name, &v, sizeof v);
MyValue* s = static_cast<MyValue*>(t.find(name));
t.erase(name);

// Raw byte-span key: key bytes then key length. Use when the key is neither an
// integer nor a std::string.
const char raw[6] = {'h','e','l','l','o','!'};
t.insert(raw, sizeof raw, &v, sizeof v);
MyValue* r = static_cast<MyValue*>(t.find(raw, sizeof raw));
t.erase(raw, sizeof raw);
```

`insert` returns false for an empty key, a key longer than 256 bytes, or a full overflow region. `find` returns nullptr on a miss.

Build: `clang++ -O2 -std=c++20 -march=native -Iinclude your_app.cpp`. Needs C++20.


## benchmarks


### long keys and values

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

| keys   | value    | MTHT                | robin        | udense            | umap         |
|--------|----------|---------------------|--------------|-------------------|--------------|
| 4–24 B | 16–128 B | **43.4M / 14.8M**   | 34.0M / 9.2M | 39.6M / 10.2M    | 23.6M / 6.5M |
| 8 B    | 256 B    | **122.2M / 12.8M**  | 61.0M / 5.8M | 66.0M / 5.8M     | 36.1M / 4.8M |
| 256 B  | 8 B      | **9.3M / 8.5M**     | 7.1M / 4.3M  | 7.6M / 4.7M      | 6.4M / 2.9M  |
| 64 B   | 64 B     | **37.8M / 9.9M**    | 17.7M / 5.7M | 30.6M / 6.8M     | 20.4M / 4.0M |
| 256 B  | 256 B    | **7.5M / 5.6M**     | 5.8M / 2.5M  | 6.7M / 3.0M      | 5.6M / 2.1M  |

Compared to udense, MTHT reads are 1.10–1.85x and inserts 1.45–2.21x the speed here, using 0.77–1.06x the memory.

#### comparison : 16k keys

| keys   | value    | MTHT                | robin        | udense            | umap         |
|--------|----------|---------------------|--------------|-------------------|--------------|
| 4–24 B | 16–128 B | **50.0M / 18.2M**   | 35.6M / 13.0M | 44.7M / 14.2M    | 28.0M / 8.3M |
| 8 B    | 256 B    | **194.4M / 20.9M**  | 62.5M / 7.3M | 86.8M / 7.8M     | 48.8M / 5.9M |
| 256 B  | 8 B      | **19.3M / 14.1M**   | 15.7M / 5.5M | 18.0M / 5.6M     | 14.5M / 3.9M |
| 64 B   | 64 B     | **48.2M / 22.8M**   | 33.3M / 9.5M | 44.8M / 9.2M     | 31.4M / 5.7M |
| 256 B  | 256 B    | **17.8M / 8.5M**    | 15.4M / 4.0M | 17.6M / 3.8M     | 14.2M / 2.9M |

Compared to udense, MTHT reads are 1.01–2.24x and inserts 1.28–2.68x the speed here, using 0.74–1.05x the memory.


### short strings and 64-bit ints

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


#### strings

Keys are mixed length: 70% at 1–16 B, 27% at 17–64 B, 3% at 65–256 B.

|                         | reads/s   | inserts/s | bytes/key |
|-------------------------|-----------|-----------|-----------|
| MTHT                    | 28.3M     | 17.3M     | **55.1**  |
| tsl::robin_map          | 20.0M     | 15.0M     | 85.3      |
| ankerl::unordered_dense | **30.1M** | **19.1M** | 70.1      |
| std::unordered_map      | 19.3M     | 8.2M      | 78.0      |

#### ints

|                         | reads/s    | inserts/s  | bytes/key |
|-------------------------|------------|------------|-----------|
| MTHT                    | 90.9M      | 39.2M      | 24.0      |
| tsl::robin_map          | 79.2M      | 60.5M      | 32.0      |
| ankerl::unordered_dense | **198.9M** | **105.8M** | **23.3**  |
| std::unordered_map      | 126.2M     | 17.6M      | 42.7      |

#### load sweep (ints)

| load | reads/s | probes/read |
|---|---|---|
| 0.083 | 219.8M | 1.04 |
| 0.410 | 131.7M | 1.25 |
| 0.686 | 88.4M  | 1.54 |


