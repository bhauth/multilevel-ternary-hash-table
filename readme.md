# Multilevel Ternary Hash Tables (MTHTs)

## Purpose

This is a description and proof of concept of a new type of hash table. The goal is to have low average probe counts for searches at a high load factor, with relatively good average memory locality.

The implementation here has some limitations:
- it only accepts 64-bit integer keys and a pointer to data
- element deletion and table resizing are not included
- it's not properly tested and should not be used seriously
- it uses some Visual Studio C++/x86 intrinsics

A simple hash function is included.

## Performance

This implementation has the following average probe counts:
load factor | successful searches | unsuccessful searches
--- | --- | ---
0.65 | ~1.67 | ~2.36
0.7 | ~1.68 | ~2.52
0.75 | ~1.80 | ~2.59
0.8 | ~2.03 | ~3.35

Those are net load factors, relative to the combined size of T1 and T2.

For comparison, linear probing has the following average probe counts:
load factor | successful searches | unsuccessful searches
--- | --- | ---
0.5 | 1.5 | 2.5
0.6 | 1.75 | 3.6
0.7 | 2.17 | 6.0
0.8 | 3.0 | 13

## Multilevel Linear Hash Tables

Linear probing hash tables (LPHTs) have good performance at 0.5 load factor.

Suppose we have a LPHT "T1" with a high load factor, but after a small number of collisions we move to a second LPHT "T2" which has a low load factor. Insertions would then have the good performance of T2 with some constant overhead from checking T1 first.

For good read performance, we want a good chance of finding results in the first table checked. Let's say T2 is 1/4 the size of T1.

That gives a better load factor, but every read from T2 will obviously require at least 5 probes, and a lot of the data is in T2. So, the average number of probes is relatively high.

Obviously reads would require less probes at lower load factors, but can we do better on a conceptual level?

## Ternary Hash Tables

Suppose T2.size = T1.size/8 and we aim for no more than 3 probes in T1 before going to T2. To get a good load factor, insertions must check more than 3 slots for empty spaces in T1 before going to T2. This is possible by shifting existing elements to either the right or left to make space for insertions, while keeping offsets in the range (-1 to +1).

Add a 1-byte header to each table element, containing its offset from its original hash position. This header value means:
- 0: empty element
- 1: offset -1
- 2: offset 0
- 3: offset 1

When inserting an element in an occupied slot, push other elements out of the way to make room for it.

Initially, every insertion has an offset of 0. When a table is mostly full, about half of elements will have been pushed an even number of times, so about half the elements will have an offset of 0.

Note that if the initial probe offset is -1 or +1, we only need to check in one direction from it.

---

Headers only require 2 bits. We can use the other 6 bits to hold the highest offset of elements pushed from that position to T2, and search from that position backwards in T2.

---

For more details, see the source code.
