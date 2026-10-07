// MTHT vs std::unordered_map -- CPU time on generated key distributions. Not
// part of the correctness suite; run it by hand:
//
//   clang++ -O2 -std=c++20 -march=native -Iinclude bench/bench.cpp -o temp/bench.exe
//   ./temp/bench.exe [n1_shift] [t2_div] [insert_pct]
//
// IMPORTANT: the generators emit RAW keys. They are not pre-mixed, so the
// table's own hash_key/fast_map pipeline does all the mixing -- sequential and
// strided keys stay adversarial, which is the point of measuring them. Only
// `random` uses splitmix, since a random distribution is by definition mixed.

#include "mtht.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static inline uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// Key generation.
//
// RANGE IS HELD CONSTANT: every generator draws from the same span of the
// 64-bit space, so the only variable between cases is the SHAPE of the draw,
// not the magnitude of the keys. Keys are never pre-mixed (except where the
// distribution is "random" by definition), so the table's own hash_key does the
// mixing and structured shapes stay adversarial.
//
// The span is RANGE keys wide, which is much larger than the table, so the
// residential clustering of any shape is what the table sees -- not the
// compactness of small integers.
static constexpr uint64_t RANGE = 0x1000000000000000ULL; // 2^60 wide span
static constexpr uint64_t SPAN_STRIDE =
    RANGE / 1000000000ULL; // ~1.15e9, keeps low bits varied

// Sequential: dense consecutive keys, then mapped into the top of the range so
// the magnitude matches every other case. Low bits are consecutive.
static std::vector<uint64_t> gen_sequential(std::size_t n) {
    std::vector<uint64_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0xF000000000000000ULL + static_cast<uint64_t>(i);
    return v;
}

// Random: uniform across the same span. The distribution is the mixing, so
// splitmix is legitimate here; the low bits are masked to the same span width.
static std::vector<uint64_t> gen_random(std::size_t n, uint64_t seed) {
    std::vector<uint64_t> v(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix(s);
        v[i] = 0xF000000000000000ULL | (s & ((1ULL << 56) - 1));
    }
    return v;
}

// Strided: every 64th key, so the low six bits are always zero. Same span.
static std::vector<uint64_t> gen_strided(std::size_t n) {
    std::vector<uint64_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0xF000000000000000ULL + static_cast<uint64_t>(i) * 64;
    return v;
}

// Clustered: 16 tight runs spread across the span, so keys are locally
// consecutive but jump between clusters.
static std::vector<uint64_t> gen_clustered(std::size_t n) {
    std::vector<uint64_t> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        uint64_t cluster = static_cast<uint64_t>(i % 16);
        uint64_t within = static_cast<uint64_t>(i / 16);
        v[i] = cluster * SPAN_STRIDE + within;
    }
    return v;
}

// High-bits only: keys spread across the top of the span with the low 24 bits
// zero, so the hash must reach high for entropy.
static std::vector<uint64_t> gen_highbits(std::size_t n) {
    std::vector<uint64_t> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = static_cast<uint64_t>(i) << 40;
    return v;
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

int main(int argc, char** argv) {
    std::size_t n1_shift = (argc > 1) ? static_cast<std::size_t>(std::atoi(argv[1])) : 20;
    std::size_t t2_div   = (argc > 2) ? static_cast<std::size_t>(std::atoi(argv[2])) : 8;
    std::size_t ins_pct  = (argc > 3) ? static_cast<std::size_t>(std::atoi(argv[3])) : 70;
    const std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t N2 = N1 / t2_div;

    const std::size_t target = N1 * ins_pct / 100;
    const uint64_t value = 0xABCDEF01ULL;

    struct Case { const char* name; std::vector<uint64_t> keys; };
    std::vector<Case> cases;
    cases.push_back({"sequential", gen_sequential(target)});
    cases.push_back({"random",     gen_random(target, 0x12345678ULL)});
    cases.push_back({"strided-64", gen_strided(target)});
    cases.push_back({"highbits",   gen_highbits(target)});
    cases.push_back({"clustered",  gen_clustered(target)});

    std::printf("MTHT t1=%zu t2=%zu (t2 = t1/%zu), keys/case=%zu (target t1 load %.2f),"
                " value=8 bytes, RAW keys\n",
                N1, N2, t2_div, target, (double)ins_pct / 100.0);
    std::printf("%-11s %9s %8s %8s %9s %-13s\n",
                "case", "ins_ms", "find_ms", "mtht_k/s", "umap_k/s", "t1/t2 load");
    std::printf("%s\n", std::string(96, '-').c_str());

    for (Case& c : cases) {
        mtht::Table t(N1, sizeof(uint64_t));
        std::unordered_map<uint64_t, uint64_t> um;
        // Match load factors. libstdc++'s reserve(n) sizes buckets to n exactly
        // (max_load_factor 1.0), so a bare reserve(n) would run umap at load
        // 1.00 while MTHT runs at t1 0.65. Ask for n / combined_load buckets so
        // both sit at the same occupancy; the actual figures are printed below.
        const double combined_load = (double)c.keys.size() / (double)N1;
        um.reserve((std::size_t)((double)c.keys.size() / combined_load) + 1);

        auto t0 = Clock::now();
        for (uint64_t k : c.keys) t.insert(k, &value, sizeof value);
        double t_ins_a = ms_since(t0);

        t0 = Clock::now();
        for (uint64_t k : c.keys) um.insert({k, value});
        double t_ins_b = ms_since(t0);
        (void)t_ins_b;

        volatile uint64_t sink = 0;

        t0 = Clock::now();
        for (uint64_t k : c.keys) {
            const void* p = t.find(k);
            if (p) sink += *static_cast<const uint64_t*>(p);
        }
        double t_find_a = ms_since(t0);
        (void)sink;

        t0 = Clock::now();
        for (uint64_t k : c.keys) {
            auto it = um.find(k);
            if (it != um.end()) sink += it->second;
        }
        double t_find_b = ms_since(t0);
        (void)sink;

        char loads[64];
        std::snprintf(loads, sizeof loads, "%.3f/%.3f umap %.3f",
                      t.load1(), t.load2(),
                      (double)um.size() / (double)um.bucket_count());
        std::printf("%-11s %9.2f %8.2f %8.1f %9.1f %-13s\n",
                    c.name, t_ins_a, t_find_a,
                    (double)c.keys.size() / t_find_a,
                    (double)c.keys.size() / t_find_b,
                    loads);
    }
    return 0;
}
