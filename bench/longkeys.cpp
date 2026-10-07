// Where MTHT's design pays off: long keys and/or fat values.
//
// The committed bench (bench/mthtbench.cpp) fixes the string value at a
// uint64_t pointer and generates keys 70% short (1..16 B) -- udense's SSO sweet
// spot and MTHT's worst case. This sweeps the two axes that actually separate
// the designs:
//
//   key band    short 1..16 | medium 17..64 | long 65..256 | all long 32..200
//   value size  8 B (pointer) | 64 B | 256 B
//
// For each cell: reads/ms, inserts/ms and bytes/key for MTHT, tsl::robin_map,
// ankerl::unordered_dense and std::unordered_map. Every peer is given MTHT's
// hash so the comparison is table design, not hash quality.
// Timing is in milliseconds of the best rep, so reads/ms and ins/ms are keys
// per millisecond -- multiply by 1000 for keys/second.
//
// Build:
//   clang++ -O2 -std=c++20 -march=native -Iinclude \
//     -Ipeers/robin-map-master/include -Ipeers/ankerl-unordered_dense \
//     temp/longkeys.cpp -o temp/longkeys.exe
#include "mtht.hpp"
#include <unordered_map>
#include <tsl/robin_map.h>
#include <unordered_dense.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

using StringKeys = std::vector<std::string>;
using Value = std::vector<unsigned char>; // the stored payload

static uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static StringKeys gen_band(std::size_t n, std::size_t lo, std::size_t hi, uint64_t seed) {
    static const char alnum[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    StringKeys v;
    v.reserve(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix64(s);
        const std::size_t len = lo + (s >> 8) % (hi - lo + 1);
        std::string str;
        str.resize(len);
        for (std::size_t j = 0; j < len; ++j) {
            s = splitmix64(s);
            str[j] = alnum[s % (sizeof(alnum) - 1)];
        }
        v.push_back(std::move(str));
    }
    return v;
}

struct mtht_key_hash {
    std::size_t operator()(const std::string& s) const {
        return static_cast<std::size_t>(mtht::hash_bytes(s.data(), s.size()));
    }
};

struct Row {
    const char* name = "";
    double reads = 0, ins = 0, bpk = 0;
};

static double best_reads(double t, std::size_t n) { return (double)n / t; }

// ---------- MTHT ----------
static Row run_mtht(const StringKeys& keys, std::size_t valbytes, std::size_t /*N1*/, std::size_t N2) {
    std::size_t N1 = 4;
    while (N1 < keys.size() / 0.70) N1 <<= 1;
    mtht::BasicTable<mtht::Hash> t(N1, 1u << 20);

    std::vector<Value> vals(keys.size(), Value(valbytes));
    for (auto& v : vals)
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = (unsigned char)(i * 31 + 7);

    std::size_t ins_ok = 0;
    double t_ins = 1e18;
    for (int r = 0; r < 2; ++r) {
        if (r) t.clear();
        auto ti = Clock::now();
        ins_ok = 0;
        for (std::size_t i = 0; i < keys.size(); ++i)
            if (t.insert(keys[i].data(), keys[i].size(), vals[i].data(), valbytes)) ++ins_ok;
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }

    volatile uint64_t sink = 0;
    double best = 1e18;
    std::size_t hit = 0;
    for (int r = 0; r < 5; ++r) {
        auto t0 = Clock::now();
        hit = 0;
        for (const std::string& k : keys) {
            const void* p = t.find(k.data(), k.size());
            if (p) { sink += *(const unsigned char*)p; ++hit; }
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    if (hit != keys.size()) printf("  !! MTHT found %zu/%zu\n", hit, keys.size());

    const double bytes = (double)(t.n1() + t.n2()) * 8.0 + (double)t.key_store_bytes();
    return {"MTHT", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
}

// ---------- robin ----------
static Row run_robin(const StringKeys& keys, std::size_t valbytes, std::size_t, std::size_t) {
    tsl::robin_map<std::string, Value, mtht_key_hash> m;
    m.max_load_factor(0.75f);
    m.reserve(keys.size());
    mtht_key_hash h;
    std::vector<Value> vals(keys.size(), Value(valbytes));
    double t_ins = 1e18;
    for (int r = 0; r < 2; ++r) {
        if (r) m.clear();
        auto ti = Clock::now();
        for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    (void)h;
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 5; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = m.find(k);
            if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0];
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // 24 B bucket + std::string object (32 B) + heap bytes past SSO + value inline
    // in the pair. std::vector<unsigned char> of valbytes is 24 B object + heap.
    double per = 24.0 + 32.0 + 24.0 + (double)valbytes;
    double heap = 0;
    for (const std::string& k : keys)
        if (k.size() > 15) heap += (double)k.size() + 1;
    const double bytes = per * keys.size() + heap;
    return {"robin", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
}

// ---------- udense ----------
static Row run_udense(const StringKeys& keys, std::size_t valbytes, std::size_t, std::size_t) {
    ankerl::unordered_dense::map<std::string, Value, mtht_key_hash> m;
    m.max_load_factor(0.75f);
    m.reserve(keys.size());
    std::vector<Value> vals(keys.size(), Value(valbytes));
    double t_ins = 1e18;
    for (int r = 0; r < 2; ++r) {
        if (r) m.clear();
        auto ti = Clock::now();
        for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 5; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = m.find(k);
            if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0];
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // value vector: pair<std::string, std::vector<uchar>> = 32 + 24 = 56 B/elt,
    // index array 5.5 B/slot at 0.75 load -> ~7.33 B/key, + string heap past SSO
    // + value heap.
    double heap = 0;
    for (const std::string& k : keys)
        if (k.size() > 15) heap += (double)k.size() + 1;
    const double bytes = 56.0 * keys.size() + 7.33 * keys.size() + heap + (double)valbytes * keys.size();
    return {"udense", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
}

// ---------- umap ----------
static Row run_umap(const StringKeys& keys, std::size_t valbytes, std::size_t, std::size_t) {
    std::unordered_map<std::string, Value, mtht_key_hash> m;
    m.max_load_factor(0.75f);
    m.reserve(keys.size());
    std::vector<Value> vals(keys.size(), Value(valbytes));
    double t_ins = 1e18;
    for (int r = 0; r < 2; ++r) {
        if (r) m.clear();
        auto ti = Clock::now();
        for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 5; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = m.find(k);
            if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0];
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // node: next ptr (8) + hash (8) + pair<string, vector<uchar>> (32+24) + alloc overhead
    double per = 16.0 + 32.0 + 24.0 + (double)valbytes + 16.0;
    double heap = 0;
    for (const std::string& k : keys)
        if (k.size() > 15) heap += (double)k.size() + 1;
    const double bytes = per * keys.size() + heap;
    return {"umap", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
}

// ---------- 8-byte keys, long string value ----------
// The vector<unsigned char> paths above carry a 24-byte container object, so an
// "8 B value" there is not an inline 8-byte payload. This cell stores a real
// std::string value instead, which is the shape a string-to-string map has.
static void cell_str8(std::size_t valbytes, std::size_t nkeys) {
    const StringKeys keys = gen_band(nkeys, 8, 8, 0xA5A5A5A5ULL ^ valbytes);
    std::vector<std::string> vals(keys.size(), std::string(valbytes, 'v'));
    for (std::size_t i = 0; i < vals.size(); ++i) vals[i][0] = (char)(i & 0x7f);

    printf("== keys 8 B  value string(%zu)  n=%zu ==\n", valbytes, keys.size());

    Row rows[4];

    { // MTHT
        std::size_t n1 = 4; while (n1 < keys.size() / 0.70) n1 <<= 1;
        mtht::BasicTable<mtht::Hash> t(n1, 1u << 24);
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) t.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i)
                t.insert(keys[i].data(), keys[i].size(), vals[i].data(), vals[i].size());
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18; std::size_t hit = 0;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now(); hit = 0;
            for (const std::string& k : keys) {
                const void* p = t.find(k.data(), k.size());
                if (p) { sink += *(const unsigned char*)p; ++hit; }
            }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        if (hit != keys.size()) printf("  !! MTHT found %zu/%zu\n", hit, keys.size());
        double bytes = (double)(t.n1() + t.n2()) * 8.0 + (double)t.key_store_bytes();
        rows[0] = {"MTHT", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // robin
        tsl::robin_map<std::string, std::string, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += (unsigned char)it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double per = 24.0 + 32.0 + 32.0; // bucket + key string obj + value string obj
        double heap = 0;
        for (const std::string& k : keys) if (k.size() > 15) heap += (double)k.size() + 1;
        heap += (double)valbytes * keys.size(); // string values are heap past SSO
        double bytes = per * keys.size() + heap;
        rows[1] = {"robin", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // udense
        ankerl::unordered_dense::map<std::string, std::string, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += (unsigned char)it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double per = 32.0 + 32.0 + 7.33; // pair<string,string> objs + index array
        double heap = 0;
        for (const std::string& k : keys) if (k.size() > 15) heap += (double)k.size() + 1;
        heap += (double)valbytes * keys.size();
        double bytes = per * keys.size() + heap;
        rows[2] = {"udense", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // umap
        std::unordered_map<std::string, std::string, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += (unsigned char)it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double per = 16.0 + 32.0 + 32.0 + 16.0; // node + key obj + value obj + overhead
        double heap = 0;
        for (const std::string& k : keys) if (k.size() > 15) heap += (double)k.size() + 1;
        heap += (double)valbytes * keys.size();
        double bytes = per * keys.size() + heap;
        rows[3] = {"umap", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }

    printf("  %-8s %12s %12s %10s\n", "map", "reads/ms", "ins/ms", "B/key");
    for (const Row& r : rows)
        printf("  %-8s %12.0f %12.0f %10.1f\n", r.name, r.reads, r.ins, r.bpk);
    for (int i = 1; i < 4; ++i)
        printf("  MTHT vs %-8s  reads %.2fx  bytes %.2fx\n", rows[i].name, rows[0].reads / rows[i].reads,
               rows[0].bpk / rows[i].bpk);
    printf("\n");
    fflush(stdout);
}

// ---------- realistic JSON shape ----------
// Word-length keys, values longer than the keys: the shape a JSON document's
// field map actually has. Both sides are drawn from a range rather than fixed,
// so this is one row rather than a band x value-size sweep. Unlike gen_band
// (uniform random characters), keys here are word-shaped runs joined by '_'.
static StringKeys gen_words(std::size_t n, std::size_t lo, std::size_t hi, uint64_t seed) {
    StringKeys v;
    v.reserve(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix64(s);
        const std::size_t len = lo + (s >> 8) % (hi - lo + 1);
        std::string str;
        str.resize(len);
        std::size_t j = 0;
        while (j < len) {
            s = splitmix64(s);
            const std::size_t word = 1 + (s >> 8) % 5; // 1..5 letters
            for (std::size_t w = 0; w < word && j < len; ++w, ++j) {
                s = splitmix64(s);
                str[j] = (char)('a' + (s >> 8) % 26);
            }
            if (j < len && len - j > 2) str[j++] = '_'; // leave room for a tail word
        }
        v.push_back(std::move(str));
    }
    return v;
}

// Value payloads of drawn length in [lo, hi], pseudorandom bytes so no value is
// all-zero or repeats.
static std::vector<Value> gen_vals(std::size_t n, std::size_t lo, std::size_t hi, uint64_t seed) {
    std::vector<Value> v;
    v.reserve(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix64(s);
        const std::size_t len = lo + (s >> 8) % (hi - lo + 1);
        Value val(len);
        for (std::size_t j = 0; j < len; ++j) {
            s = splitmix64(s);
            val[j] = (unsigned char)(s >> 24);
        }
        v.push_back(std::move(val));
    }
    return v;
}

// The peers' B/key here is 8-byte hash + string object + value object + heap
// past SSO + value heap. MTHT's is slot arrays + one record per key: the key
// bytes, the value bytes, the shim fmt and the state byte.
static void cell_realistic(std::size_t nkeys) {
    const StringKeys keys = gen_words(nkeys, 4, 24, 0x5DEECE66DULL);
    const std::vector<Value> vals = gen_vals(nkeys, 16, 128, 0x1234ABCDULL);
    printf("== realistic JSON: keys 4-24 B, values 16-128 B  n=%zu ==\n", keys.size());

    Row rows[4];
    { // MTHT
        std::size_t n1 = 4; while (n1 < keys.size() / 0.70) n1 <<= 1;
        mtht::BasicTable<mtht::Hash> t(n1, 1u << 22);
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) t.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i)
                t.insert(keys[i].data(), keys[i].size(), vals[i].data(), vals[i].size());
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18; std::size_t hit = 0;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now(); hit = 0;
            for (const std::string& k : keys) {
                const void* p = t.find(k.data(), k.size());
                if (p) { sink += *(const unsigned char*)p; ++hit; }
            }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        if (hit != keys.size()) printf("  !! MTHT found %zu/%zu\n", hit, keys.size());
        double bytes = (double)(t.n1() + t.n2()) * 8.0;
        for (std::size_t i = 0; i < keys.size(); ++i)
            bytes += 8.0 + 1.0 + (double)keys[i].size() + (double)vals[i].size();
        rows[0] = {"MTHT", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // robin
        tsl::robin_map<std::string, Value, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double bytes = (8.0 + 32.0 + 24.0) * keys.size(); // hash + key string obj + value vector obj
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].size() > 15) bytes += (double)keys[i].size() + 1;
            bytes += (double)vals[i].size() + 16.0;        // value heap + alloc header
        }
        rows[1] = {"robin", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // udense
        ankerl::unordered_dense::map<std::string, Value, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double bytes = (32.0 + 24.0) * keys.size(); // pair<string, vector>
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].size() > 15) bytes += (double)keys[i].size() + 1;
            bytes += (double)vals[i].size() + 16.0;
        }
        rows[2] = {"udense", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }
    { // umap
        std::unordered_map<std::string, Value, mtht_key_hash> m;
        m.max_load_factor(0.75f); m.reserve(keys.size());
        double t_ins = 1e18;
        for (int r = 0; r < 2; ++r) {
            if (r) m.clear();
            auto ti = Clock::now();
            for (std::size_t i = 0; i < keys.size(); ++i) m.insert({keys[i], vals[i]});
            t_ins = std::min(t_ins, ms_since(ti));
        }
        volatile uint64_t sink = 0; double best = 1e18;
        for (int r = 0; r < 5; ++r) {
            auto t0 = Clock::now();
            for (const std::string& k : keys) { auto it = m.find(k); if (it != m.end()) sink += it->second.empty() ? 0 : it->second[0]; }
            best = std::min(best, ms_since(t0));
        }
        (void)sink;
        double bytes = (16.0 + 32.0 + 24.0 + 16.0) * keys.size(); // node + key obj + value obj + overhead
        for (std::size_t i = 0; i < keys.size(); ++i) {
            if (keys[i].size() > 15) bytes += (double)keys[i].size() + 1;
            bytes += (double)vals[i].size() + 16.0;
        }
        rows[3] = {"umap", best_reads(best, keys.size()), (double)keys.size() / t_ins, bytes / keys.size()};
    }

    printf("  %-8s %12s %12s %10s\n", "map", "reads/ms", "ins/ms", "B/key");
    for (const Row& r : rows)
        printf("  %-8s %12.0f %12.0f %10.1f\n", r.name, r.reads, r.ins, r.bpk);
    for (int i = 1; i < 4; ++i)
        printf("  MTHT vs %-8s  reads %.2fx  ins %.2fx  bytes %.2fx\n", rows[i].name,
               rows[0].reads / rows[i].reads, rows[0].ins / rows[i].ins, rows[0].bpk / rows[i].bpk);
    printf("\n");
    fflush(stdout);
}

static void cell(const char* kband, std::size_t lo, std::size_t hi,
                 std::size_t valbytes, std::size_t nkeys) {
    const StringKeys keys = gen_band(nkeys, lo, hi, 0x9E3779B9ULL ^ (hi * 131 + lo));
    printf("== keys %s  value %zu B  n=%zu ==\n", kband, valbytes, keys.size());
    const Row rows[] = {run_mtht(keys, valbytes, 0, 0), run_robin(keys, valbytes, 0, 0),
                        run_udense(keys, valbytes, 0, 0), run_umap(keys, valbytes, 0, 0)};
    printf("  %-8s %12s %12s %10s\n", "map", "reads/ms", "ins/ms", "B/key");
    for (const Row& r : rows)
        printf("  %-8s %12.0f %12.0f %10.1f\n", r.name, r.reads, r.ins, r.bpk);
    const Row& m = rows[0];
    for (int i = 1; i < 4; ++i)
        printf("  MTHT vs %-8s  reads %.2fx  bytes %.2fx\n", rows[i].name, m.reads / rows[i].reads,
               m.bpk / rows[i].bpk);
    printf("\n");
    fflush(stdout);
}

// README table rows: the exact key/value combos the README reports. Fixed key
// bands and value sizes, printed one cell at a time with the same row order as
// the committed main() grid, so the README numbers can be regenerated verbatim.
static void cell_readme(std::size_t n) {
    cell_realistic(n);               // 4-24 B keys, 16-128 B values
    cell_str8(256, n);               // 8 B keys, 256 B string value
    cell("all 256", 256, 256, 8, n); // 256 B keys, 8 B value
    cell("long 64", 64, 64, 64, n);  // 64 B keys, 64 B value
    cell("all 256x256", 256, 256, 256, n); // 256 B keys, 256 B value
}

// Accepts -n <keys> or --n=<keys>. Returns the parsed count, or def when absent.
static std::size_t parse_n(int argc, char** argv, std::size_t def) {
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        const char* v = nullptr;
        if (std::strcmp(a, "-n") == 0 && i + 1 < argc) v = argv[i + 1];
        else if (std::strncmp(a, "--n=", 4) == 0) v = a + 4;
        if (!v) continue;
        char* end = nullptr;
        const unsigned long long parsed = std::strtoull(v, &end, 10);
        if (end && *end == '\0' && parsed > 0) return static_cast<std::size_t>(parsed);
        std::fprintf(stderr, "bad -n value: '%s'\n", v);
        std::exit(1);
    }
    return def;
}

int main(int argc, char** argv) {
    const std::size_t n = parse_n(argc, argv, 200000);
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--readme") == 0) { cell_readme(n); return 0; }
    // key bands x value sizes
    struct Band { const char* name; std::size_t lo, hi; };
    const Band bands[] = {{"short 1-16", 1, 16}, {"medium 17-64", 17, 64},
                          {"long 65-256", 65, 256}, {"long 32-200", 32, 200}};
    const std::size_t vals[] = {8, 64, 256};
    for (const Band& b : bands)
        for (std::size_t v : vals)
            cell(b.name, b.lo, b.hi, v, n);
    cell_str8(256, n); // 8-byte keys, 256-byte string values
    cell_realistic(n); // realistic JSON shape: short word keys, longer values
    return 0;
}
