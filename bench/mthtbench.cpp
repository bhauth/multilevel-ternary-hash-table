// mthtbench -- the committed benchmark tool.
//
// HOW TO BUILD
//   clang++ -O2 -std=c++20 -march=native -Iinclude \
//           -Ipeers/robin-map-master/include -Ipeers/ankerl-unordered_dense \
//           bench/mthtbench.cpp -o temp/mthtbench.exe
//   (temp/ is git-ignored; the exe is a scratch artifact. -march=native matters:
//    the fast_map path uses 128-bit multiplies, so the numbers depend on it.)
//
// HOW TO RUN
//   ./temp/mthtbench.exe                # every suite, writes bench/mthtbench.log
//   ./temp/mthtbench.exe --quick        # small tables (~2^16 keys) for a smoke test
//
//   Suite selection -- --suite= picks exactly one (default is all):
//     --suite=correctness   verify placement + every key reads back, then stop
//     --suite=mtht          MTHT alone across key distributions (time, load)
//     --suite=loads         MTHT alone while the fill level rises, to the point
//                           it auto-resizes (time vs achieved load)
//     --suite=peers         MTHT vs other maps on random keys (time, bytes/key)
//     --suite=strings       MTHT (string keys) vs string-keyed maps (time, B/key)
//
//   Other flags:
//     --peers=NAME,...      peer columns for --suite=peers and --suite=strings
//     --list-peers          print the peer registry and exit
//     --log=PATH            write the log somewhere else
//     --no-log              console only, write no log
//     --n1=20               log2 of T1 slots (default 20 -> ~1M)
//     --t2div=8             T2 slots = T1/this (default 8)
//     --fill=75             percent of N1 used as the key count (default 75)
//
//   Typical use: run plain for the full picture, --suite=loads when tuning the
//   growth point, --suite=peers --peers=robin,umap for the memory comparison.
//
// OUTPUT LOG
//   Console output is mirrored to the log file (default bench/mthtbench.log),
//   truncated at the start of each run so the file is exactly that run. The log
//   is untracked (see .gitignore) -- a run record, not a source artifact.
//
// SUITES
//   correctness  -- builds each distribution, verifies every key reads back,
//                   then walks the raw slot arrays to check the placement
//                   invariant. A perf number from a table that failed this is
//                   worthless, so timing is skipped if this fails.
//   mtht         -- MTHT alone across the key distributions: insert/find time,
//                   throughput and occupancy.
//   loads        -- MTHT alone, one row per fill step from N1/12 up to N1, with
//                   the table free to auto-resize. Shows find cost against the
//                   achieved load and marks the row where the resize fires.
//   peers        -- MTHT against the selected peers on random keys: throughput,
//                   bytes of table memory, and bytes per key.
//   strings      -- the same comparison for string keys: MTHT storing the whole
//                   key in the record vs string-keyed std::string containers.
//                   Keys are mixed length so the record store is exercised.
//
// Quiet-by-default output: a passing row prints its numbers; only failures print
// a [FAIL] line. Comments that restate an obvious column are not printed -- the
// caveats that remain are the ones a reader could otherwise get wrong.

// MTHT itself is measured through the shipping header (include/mtht.hpp): one
// traversal, named find(). That header has no probe or spill counters, so the
// bench reports time, throughput and memory -- what the table exposes.
#include "mtht.hpp"

#if __has_include("tsl/robin_map.h")
#include "tsl/robin_map.h"
#define HAVE_ROBIN 1
#else
#define HAVE_ROBIN 0
#endif

#if __has_include("unordered_dense.h")
#include "unordered_dense.h"
#define HAVE_UDENSE 1
#else
#define HAVE_UDENSE 0
#endif

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <unordered_map>
#include <vector>

// ---------------------------------------------------------------------------
// Output: console + log file (truncated each run, so the log is this run's
// record and never stacks the output of earlier runs).
// ---------------------------------------------------------------------------
namespace out {

static FILE* log = nullptr;

void open(const std::string& path) {
    log = std::fopen(path.c_str(), "w");
    if (!log) {
        std::printf("!! cannot open log %s (console only)\n", path.c_str());
    }
}

void line(const char* fmt, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    std::fputs(buf, stdout);
    std::fputc('\n', stdout);
    if (log) {
        std::fputs(buf, log);
        std::fputc('\n', log);
        std::fflush(log);
    }
}

void close() {
    if (log) {
        std::fclose(log);
        log = nullptr;
    }
}

} // namespace out

// ---------------------------------------------------------------------------
// Timing + run identity.
// ---------------------------------------------------------------------------
using Clock = std::chrono::steady_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static std::string now_string() {
    std::time_t t = std::time(nullptr);
    char buf[64];
    std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

static std::string cpu_string() {
    // /proc/cpuinfo under MSYS; fall back to an env hint on other hosts.
    if (FILE* f = std::fopen("/proc/cpuinfo", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f)) {
            if (std::strncmp(line, "model name", 10) == 0) {
                std::fclose(f);
                char* colon = std::strchr(line, ':');
                std::string s = colon ? colon + 2 : line;
                while (!s.empty() && (s.back() == '\n' || s.back() == ' ')) s.pop_back();
                return s;
            }
        }
        std::fclose(f);
    }
    const char* env = std::getenv("PROCESSOR_IDENTIFIER");
    return env ? env : "unknown";
}

// ---------------------------------------------------------------------------
// Key generation. RANGE is held constant across distributions so the only
// variable is the SHAPE of the draw, never the magnitude of the keys. Keys are
// raw (not pre-mixed) except for `random`, where the distribution is itself the
// mixing -- so structured shapes stay adversarial to the table's own hash.
// ---------------------------------------------------------------------------
static constexpr uint64_t RANGE = 0x1000000000000000ULL; // 2^60 span
static constexpr uint64_t SPAN_STRIDE = RANGE / 1000000000ULL;

static inline uint64_t splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

using Keys = std::vector<uint64_t>;

static Keys gen_sequential(std::size_t n) {
    Keys v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0xF000000000000000ULL + (uint64_t)i;
    return v;
}
static Keys gen_random(std::size_t n, uint64_t seed) {
    Keys v(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix(s);
        v[i] = 0xF000000000000000ULL | (s & ((1ULL << 56) - 1));
    }
    return v;
}
static Keys gen_strided(std::size_t n) {
    Keys v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = 0xF000000000000000ULL + (uint64_t)i * 64;
    return v;
}
static Keys gen_clustered(std::size_t n) {
    Keys v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = (uint64_t)(i % 16) * SPAN_STRIDE + (uint64_t)(i / 16);
    return v;
}
static Keys gen_highbits(std::size_t n) {
    Keys v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = (uint64_t)i << 40;
    return v;
}
static Keys gen_absent(std::size_t n, uint64_t seed) {
    Keys v(n);
    uint64_t s = splitmix(seed);
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix(s);
        v[i] = s ^ 0xFFFF000000000000ULL; // disjoint from the generated keys
    }
    return v;
}

// ---------------------------------------------------------------------------
// String keys. A realistic mix: short keys (1-16 B), medium (17-64 B) and a
// few long ones (up to 256 B), each distinct. The bytes are pseudo-random so no
// two keys share a prefix, and every key carries enough entropy that its hash is
// well spread -- otherwise a structured string set would flatter whichever table
// happens to hash it best.
// ---------------------------------------------------------------------------
using StringKeys = std::vector<std::string>;

static StringKeys gen_strings(std::size_t n, uint64_t seed) {
    static const char alnum[] =
        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    StringKeys v;
    v.reserve(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        // Length bands: mostly short, some medium, a thin tail of long keys,
        // so the average sits near a typical string-map workload.
        s = splitmix(s);
        const uint64_t r = s;
        std::size_t len;
        if ((r % 100) < 70)      len = 1 + (r >> 8) % 16;    // 1..16 B
        else if ((r % 100) < 97) len = 17 + (r >> 8) % 48;   // 17..64 B
        else                     len = 65 + (r >> 8) % 192;  // 65..256 B
        std::string str;
        str.resize(len);
        for (std::size_t j = 0; j < len; ++j) {
            s = splitmix(s);
            str[j] = alnum[s % (sizeof(alnum) - 1)];
        }
        v.push_back(std::move(str));
    }
    return v;
}

struct Dist { const char* name; Keys keys; };

static std::vector<Dist> distributions(std::size_t n) {
    return {{"sequential", gen_sequential(n)},
            {"random", gen_random(n, 0x12345678ULL)},
            {"strided-64", gen_strided(n)},
            {"highbits", gen_highbits(n)},
            {"clustered", gen_clustered(n)}};
}

// ---------------------------------------------------------------------------
// Placement invariant: no live T1 entry may sit more than 1 slot from its home
// (that is what guarantees <=2 probes on offsets -1/+1).
// ---------------------------------------------------------------------------
struct Invariant { std::size_t live = 0; std::size_t violations = 0; std::size_t badtag = 0; };

static Invariant check_invariant(mtht::Table& t) {
    Invariant r;
    const mtht::Entry* base = t.raw_t1();
    for (std::size_t i = 0; i < t.n1(); ++i) {
        const mtht::Entry& e = base[i];
        const uint8_t st = mtht::get_state(e);
        if (st == mtht::ST_EMPTY) continue;
        ++r.live;
        const uint64_t h = mtht::fast_map(
            mtht::hash_bytes(mtht::key_ptr_at(mtht::decode_pointer(e.word)),
                             mtht::key_len_at(mtht::decode_pointer(e.word))),
            static_cast<uint64_t>(t.n1()));
        // Distance from home, wrapping.
        const long long n = (long long)t.n1();
        long long d = (long long)i - (long long)h;
        if (d > n / 2) d -= n;
        if (d < -n / 2) d += n;
        if (d < -1 || d > 1) ++r.violations;
        // Tag must agree with the actual offset.
        const uint8_t want = (d == 0) ? mtht::ST_OFF_ZERO
                                      : (d == -1 ? mtht::ST_OFF_NEG : mtht::ST_OFF_POS);
        if (st != want) ++r.badtag;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Suite: correctness. Inserts every key, verifies readback, checks the
// invariant, and confirms absent keys miss. Must pass before any timing is
// trusted.
// ---------------------------------------------------------------------------
static bool suite_correctness(std::size_t n1_shift, std::size_t t2_div,
                              std::size_t ins_pct, int* checks_out) {
    const std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t N2 = N1 / t2_div;
    const std::size_t nkeys = N1 * ins_pct / 100;
    int checks = 0, fails = 0;

    out::line("== correctness ==");
    out::line("  N1=%zu N2=%zu keys=%zu", N1, N2, nkeys);

    for (Dist& d : distributions(nkeys)) {
        mtht::Table t(N1, sizeof(uint64_t));

        // Distinct-key check, so a duplicate cannot silently inflate `ok`.
        Keys mirror = d.keys;
        std::sort(mirror.begin(), mirror.end());
        const bool distinct = std::adjacent_find(mirror.begin(), mirror.end()) == mirror.end();

        std::size_t inserted = 0;
        for (uint64_t k : d.keys)
            if (t.insert(k, &k, sizeof k)) ++inserted;

        std::size_t found = 0, wrong_val = 0;
        for (uint64_t k : d.keys) {
            const void* p = t.find(k);
            if (!p) continue;
            ++found;
            if (*static_cast<const uint64_t*>(p) != k) ++wrong_val;
        }
        const Keys absent = gen_absent(nkeys, 0xFACEULL);
        std::size_t false_pos = 0;
        for (uint64_t k : absent)
            if (t.find(k)) ++false_pos;

        const Invariant inv = check_invariant(t);

        checks += 4;
        if (!distinct) { ++fails; out::line("  [FAIL] %-11s generator produced duplicates", d.name); }
        if (inserted != nkeys) { ++fails; out::line("  [FAIL] %-11s inserted %zu/%zu", d.name, inserted, nkeys); }
        if (found != inserted || wrong_val) { ++fails; out::line("  [FAIL] %-11s found %zu/%zu wrong_val=%zu", d.name, found, inserted, wrong_val); }
        if (inv.violations || inv.badtag) { ++fails; out::line("  [FAIL] %-11s invariant viol=%zu badtag=%zu", d.name, inv.violations, inv.badtag); }

        // One compact row per distribution. A passing row is just its numbers.
        out::line("  %-11s keys=%-7zu found=%-7zu fp=%-3zu inv=%-3zu t1=%.3f t2=%.3f",
                  d.name, inserted, found, false_pos, inv.violations, t.load1(), t.load2());
    }
    out::line("  %d checks, %d failures", checks, fails);
    out::line("");
    if (checks_out) *checks_out = checks;
    return fails == 0;
}

// ---------------------------------------------------------------------------
// Suite: mtht across distributions.
// ---------------------------------------------------------------------------
static void suite_mtht(std::size_t n1_shift, std::size_t t2_div, std::size_t ins_pct) {
    const std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t N2 = N1 / t2_div;
    const std::size_t nkeys = N1 * ins_pct / 100;
    const uint64_t value = 0xABCDEF01ULL;

    out::line("== mtht ==");
    out::line("  N1=%zu N2=%zu (t2=t1/%zu) keys/case=%zu value=8B RAW keys", N1, N2, t2_div, nkeys);
    out::line("  %-11s %9s %11s %9s %9s %-13s", "case", "ins_ms", "find_ms", "Mfind/s",
              "umap/s", "t1/t2 load");
    out::line("  %s", std::string(96, '-').c_str());

    for (Dist& d : distributions(nkeys)) {
        mtht::Table t(N1, sizeof(uint64_t));
        std::unordered_map<uint64_t, uint64_t> um;
        // Match occupancy, not just reserve(n): libstdc++ sizes buckets to n
        // exactly, so a bare reserve runs umap at load 1.00 while MTHT runs at
        // t1 ~0.65. Ask for keys / combined_load buckets so both sit alike.
        const double combined_load = (double)nkeys / (double)N1;
        um.reserve((std::size_t)((double)nkeys / combined_load) + 1);

        auto t0 = Clock::now();
        std::size_t inserted = 0;
        for (uint64_t k : d.keys)
            if (t.insert(k, &value, sizeof value)) ++inserted;
        const double t_ins = ms_since(t0);

        volatile uint64_t sink = 0;
        const int REPS = 5;
        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            auto t1 = Clock::now();
            for (uint64_t k : d.keys) {
                const void* p = t.find(k);
                if (p) sink += *static_cast<const uint64_t*>(p);
            }
            const double dd = ms_since(t1);
            if (dd < best) best = dd;
        }
        const double t_find = best;
        (void)sink;

        auto t2 = Clock::now();
        for (uint64_t k : d.keys) um.insert({k, value});
        const double t_um_ins = ms_since(t2);
        t2 = Clock::now();
        for (uint64_t k : d.keys) {
            auto it = um.find(k);
            if (it != um.end()) sink += it->second;
        }
        const double t_um_find = ms_since(t2);
        (void)sink;

        char loads[64];
        std::snprintf(loads, sizeof loads, "%.3f/%.3f umap %.3f", t.load1(), t.load2(),
                      (double)um.size() / (double)um.bucket_count());
        out::line("  %-11s %9.2f %11.2f %9.1f %9.1f %-13s",
                  d.name, t_ins, t_find, (double)nkeys / t_find,
                  (double)nkeys / t_um_find, loads);
        out::line("     umap ins %.1fms find %.1fms", t_um_ins, t_um_find);
    }
    out::line("");
}

// ---------------------------------------------------------------------------
// Load sweep, MTHT only. One row per fill step, from N1/12 up to N1, with the
// table free to auto-resize: as T2 fills toward T2_GROW_LOAD the table doubles,
// and the row where that happens is marked. Shows find cost against achieved
// load, and how much the resize buys back.
static void suite_mtht_loads(std::size_t n1_shift, std::size_t t2_div,
                             std::size_t ins_pct) {
    const std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t N2 = N1 / t2_div;
    const std::size_t step = N1 / 12;
    const uint64_t value = 0xABCDEF01ULL;

    out::line("== mtht loads ==");
    out::line("  N1=%zu N2=%zu (t2=t1/%zu)  step=%zu keys  random keys  MTHT only",
              N1, N2, t2_div, step);
    out::line("  t1 = keys in T1 / N1; t2 = keys in T2 / N2; when t2 reaches %g the"
              " table doubles", (double)mtht::T2_GROW_LOAD);
    out::line("  each row is a fresh table at N1, so grow = this row's own n1_ / N1.");
    out::line("  %-8s %10s %11s %9s %-9s %s",
              "keys", "t1", "find_ms", "Mfind/s", "t1/t2", "grow");
    out::line("  %s", std::string(96, '-').c_str());

    std::vector<Dist> dists = distributions(N1);
    Dist& d = dists[0]; // "random"
    const int REPS = 5;

    for (std::size_t n = step; n <= N1; n += step) {
        mtht::Table t(N1, sizeof(uint64_t));

        const std::size_t nhave = n < d.keys.size() ? n : d.keys.size();
        bool inserted_all = true;
        for (std::size_t i = 0; i < nhave; ++i) {
            if (!t.insert(d.keys[i], &value, sizeof value)) { inserted_all = false; break; }
        }
        // "grow" reports this row's own n1_ against the N1 the table was built
        // with, so a row that stayed put reads "-" and a row that doubled reads
        // "2x".
        const std::size_t grew = t.n1() / N1;

        volatile uint64_t sink = 0;
        double best = 1e18;
        for (int r = 0; r < REPS; ++r) {
            auto t0 = Clock::now();
            for (std::size_t i = 0; i < nhave; ++i) {
                const void* p = t.find(d.keys[i]);
                if (p) sink += *static_cast<const uint64_t*>(p);
            }
            const double dd = ms_since(t0);
            if (dd < best) best = dd;
        }
        (void)sink;

        char l12[40];
        std::snprintf(l12, sizeof l12, "%.3f/%.3f", t.load1(), t.load2());
        char grow[16];
        if (!inserted_all) std::snprintf(grow, sizeof grow, "refused");
        else if (grew <= 1) std::snprintf(grow, sizeof grow, "-");
        else std::snprintf(grow, sizeof grow, "%zux", grew);
        out::line("  %-8zu %10.3f %11.2f %9.1f %-9s %s",
                  nhave, t.load1(), best, (double)nhave / best, l12, grow);
    }
    out::line("");
}

// ---------------------------------------------------------------------------
// Peers. A peer is one interchangeable function with a uniform signature, so a
// new comparison is a new entry in PEERS -- no other code changes. robin_map is
// the default entry; --peers= selects which to run.
//
// Random keys only: structured generators are degenerate for std::hash<uint64_t>
// (the identity) with a power-of-two mask -- keys with zeroed low bits (i<<40)
// all collide and robin-hood insertion goes quadratic, which is what blew up
// memory on an earlier run.
// ---------------------------------------------------------------------------
// Each peer reports kps (throughput), bytes of table memory, and a note on how
// that byte count was derived. Bytes are the real quantity.
//
// Values are pointed to, not owned: every table here stores key + an 8 B value
// pointer, and the pointed-to value lives in the caller's own storage (it may be
// large, and it is the same for every row). So `bytes` counts table structure
// only, and the rows are directly comparable.
struct PeerResult { double kps, ips, bytes, bpk, load; std::size_t n; };

struct Peer {
    const char* name;                                  // column label
    PeerResult (*run)(const Keys&, std::size_t, std::size_t, uint64_t, float);
    bool available;
};

static PeerResult time_mtht(const Keys& keys, std::size_t N1, std::size_t N2, uint64_t value,
                            float) {
    // Sizing is the caller's: it already turned any target load into a n1/N2
    // pair. The table may still grow on its own T2 rule, which we do not touch.
    // Insert a pointer, like the peers: the Entry is key + value pointer and the
    // pointed-to value is not owned by the table.
    const uint64_t* pv = &value;
    // Inserts, best of 3: each rep clears the table, since inserting the same key
    // twice is a different (replace) path. The final state is what bytes/load and
    // the reads below are measured on.
    mtht::Table t(N1, sizeof(uint64_t));
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) t.clear();
        auto ti = Clock::now();
        for (uint64_t k : keys) t.insert(k, &pv, sizeof pv);
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (uint64_t k : keys) {
            const void* p = t.find(k);
            if (p) sink += **static_cast<const uint64_t* const*>(p);
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // Slot arrays only: 16 B per Entry (64-bit key + 48-bit value pointer + state
    // bits, packed) across the table's *live* T1+T2, filled or not (n1()/n2()
    // report the sizes after any grow). The pointed-to value is the caller's and
    // is not owned by the table, so no value bytes are counted here -- same as the
    // peers, whose payload is a pointer too.
    const double live_n1 = (double)t.n1(), live_n2 = (double)t.n2();
    const double bytes = (live_n1 + live_n2) * 16.0;
    const double load = (double)t.size() / (live_n1 + live_n2);
    return { (double)t.size() / best, (double)t.size() / t_ins, bytes, bytes / (double)t.size(),
             load, t.size() };
}

static PeerResult time_umap(const Keys& keys, std::size_t, std::size_t, uint64_t value,
                            float target_load) {
    // Pointer payload, like MTHT's Entry: the node owns key + pointer, not the
    // pointed-to value.
    const uint64_t* pv = &value;
    // Unlike the robin_map peer (which is given mtht::Hash), this keeps the
    // default std::hash<uint64_t>: on libstdc++ that is the identity, so raw keys
    // arrive with whatever structure they have and nothing is mixed.
    std::unordered_map<uint64_t, const uint64_t*> um;
    // The target load pre-sizes the bucket array: reserve() asks for the count
    // that holds `keys.size()` at that load (umap's default max_load_factor is
    // 1.0, so we set it too). Any later auto-resize is umap's own to make.
    if (target_load > 0.0f) um.max_load_factor(target_load);
    um.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) um.clear();
        auto ti = Clock::now();
        for (uint64_t k : keys) um.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (uint64_t k : keys) {
            auto it = um.find(k);
            if (it != um.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // (bucket_count + 1) pointers + one 32 B node per stored key. Both measured:
    // the node is 32 B (a 16 B pair<key,ptr> plus the chain link), and libstdc++
    // allocates the bucket array as one block of bucket_count+1 pointers. Load is
    // items/buckets in the live table (umap may have resized past the target).
    const double bytes = (double)(um.bucket_count() + 1) * 8.0 + (double)um.size() * 32.0;
    const double load = (double)um.size() / (double)um.bucket_count();
    return { (double)um.size() / best, (double)um.size() / t_ins, bytes, bytes / (double)um.size(),
             load, um.size() };
}

// robin_map defaults to std::hash<uint64_t>, which is the identity on libstdc++:
// it mixes nothing, so raw keys keep whatever structure they arrive with. Give it
// mtht::Hash (splitmix64) instead, the same mixing MTHT pays for, so the two
// columns differ in table design rather than in hash quality.
struct robin_mtht_hash {
    std::size_t operator()(uint64_t k) const { return (std::size_t)mtht::hash_key(k); }
};

// Compiled only where the vendored header exists (see HAVE_ROBIN). The registry
// marks the entry unavailable otherwise, so no #if is needed at the call site.
static PeerResult time_robin(const Keys& keys, std::size_t, std::size_t, uint64_t value,
                             float target_load) {
#if HAVE_ROBIN
    const uint64_t* pv = &value;
    tsl::robin_map<uint64_t, const uint64_t*, robin_mtht_hash> rm;
    // Same rule as umap: the target load pre-sizes the bucket array instead of
    // us disabling growth. reserve() chooses ceil(n / load) buckets, so no
    // rehash happens during insert; a later auto-resize is robin's own to make.
    rm.max_load_factor(target_load > 0.0f ? target_load : 0.75f);
    rm.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) rm.clear();
        auto ti = Clock::now();
        for (uint64_t k : keys) rm.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (uint64_t k : keys) {
            auto it = rm.find(k);
            if (it != rm.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // bucket_entry = 16 B pair<Key,ptr> + a distance/hash byte, padded to 24 B;
    // measured at 24.00 B/bucket. The payload is a pointer, so no value bytes are
    // owned here. Load is items/buckets in the live map (robin may have resized
    // past the target).
    const double bytes = (double)rm.bucket_count() * 24.0;
    const double load = (double)rm.size() / (double)rm.bucket_count();
    return { (double)rm.size() / best, (double)rm.size() / t_ins, bytes, bytes / (double)rm.size(),
             load, rm.size() };
#else
    (void)keys; (void)value; (void)target_load;
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0 };   // never called: registry marks it unavailable
#endif
}

// unordered_dense keeps keys and values in one contiguous vector and an index
// array of 32-bit entries beside it (5.5 bytes per slot: 16 fingerprints in two
// words plus 8 overflow counters per group). No per-key node, so the footprint
// is the vector capacity plus the index, not a node size. Same mtht::Hash as
// robin_map, so the column differs in table design rather than hash quality.
static PeerResult time_udense(const Keys& keys, std::size_t, std::size_t, uint64_t value,
                              float target_load) {
#if HAVE_UDENSE
    const uint64_t* pv = &value;
    ankerl::unordered_dense::map<uint64_t, const uint64_t*, robin_mtht_hash> ud;
    if (target_load > 0.0f) ud.max_load_factor(target_load);
    ud.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) ud.clear();
        auto ti = Clock::now();
        for (uint64_t k : keys) ud.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (uint64_t k : keys) {
            auto it = ud.find(k);
            if (it != ud.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // The value vector holds pair<Key, ptr> = 16 B per element (capacity, not
    // size); the index is 5.5 B per slot. Values are a pointer, so no value bytes
    // are owned. Load is items/slots in the live map.
    const double bytes = (double)ud.values().capacity() * 16.0 + (double)ud.bucket_count() * 5.5;
    const double load = (double)ud.size() / (double)ud.bucket_count();
    return { (double)ud.size() / best, (double)ud.size() / t_ins, bytes, bytes / (double)ud.size(),
             load, ud.size() };
#else
    (void)keys; (void)value; (void)target_load;
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0 };
#endif
}

// The peer registry. Add an entry here (and its time_* above) to compare against
// it; --peers=name selects it. MTHT is always the first column, so it is not a
// peer entry. An entry whose backing header is missing is registered with
// available=false; the suite skips it and --list-peers reports why.
static std::vector<Peer> make_peers() {
    std::vector<Peer> v;
    v.push_back({"tsl::robin_map", time_robin, HAVE_ROBIN != 0});
    v.push_back({"ankerl::unordered_dense", time_udense, HAVE_UDENSE != 0});
    v.push_back({"std::unordered_map", time_umap, true});
    return v;
}

// ---------------------------------------------------------------------------
// String-key peers. Same shape as the integer peers above, but keyed by
// std::string, so the table has to hash and compare variable-length byte spans.
// MTHT stores the whole key with the value; robin_map and unordered_map store a
// std::string in the pair, which owns its bytes on the heap (short keys live
// inline in the SSO buffer, which is why the byte counts are not simply
// length-proportional).
// ---------------------------------------------------------------------------
struct StrPeer {
    const char* name;
    PeerResult (*run)(const StringKeys&, std::size_t, std::size_t, uint64_t, float);
    bool available;
};

static PeerResult time_mtht_strings(const StringKeys& keys, std::size_t N1, std::size_t N2,
                                    uint64_t value, float) {
    const uint64_t* pv = &value;
    mtht::Table t(N1, sizeof(uint64_t));
    std::size_t ins_ok = 0;
    // Inserts, best of 3, clearing between reps (see time_mtht).
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) t.clear();
        ins_ok = 0;
        auto ti = Clock::now();
        for (const std::string& k : keys)
            if (t.insert(k.data(), k.size(), &pv, sizeof pv)) ++ins_ok;
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            const void* p = t.find(k.data(), k.size());
            if (p) sink += **static_cast<const uint64_t* const*>(p);
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;

    // Every key was accepted and reads back as the value it was given. A string
    // key that failed to store or collided in the fingerprint would show here.
    std::size_t found_ok = 0;
    for (const std::string& k : keys) {
        const void* p = t.find(k.data(), k.size());
        if (p && *(const uint64_t* const*)p == pv) ++found_ok;
    }
    if (ins_ok != keys.size())
        out::line("  !! MTHT(strings): %zu/%zu inserts accepted", ins_ok, keys.size());
    if (found_ok != keys.size())
        out::line("  !! MTHT(strings): %zu/%zu keys read back", found_ok, keys.size());

    // Slot arrays (8 B Entry * live T1+T2) plus the record store, which holds the
    // copied keys and values -- for string keys that store is the point of the
    // design, so it is counted rather than assumed away.
    const double live_n1 = (double)t.n1(), live_n2 = (double)t.n2();
    const double bytes = (live_n1 + live_n2) * 8.0 + (double)t.key_store_bytes();
    const double load = (double)t.size() / (live_n1 + live_n2);
    return { (double)t.size() / best, (double)t.size() / t_ins, bytes, bytes / (double)t.size(),
             load, t.size() };
}

struct str_mtht_hash {
    std::size_t operator()(const std::string& s) const {
        return static_cast<std::size_t>(mtht::hash_bytes(s.data(), s.size()));
    }
};

static PeerResult time_robin_strings(const StringKeys& keys, std::size_t, std::size_t,
                                     uint64_t value, float target_load) {
#if HAVE_ROBIN
    const uint64_t* pv = &value;
    tsl::robin_map<std::string, const uint64_t*, str_mtht_hash> rm;
    rm.max_load_factor(target_load > 0.0f ? target_load : 0.75f);
    rm.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) rm.clear();
        auto ti = Clock::now();
        for (const std::string& k : keys) rm.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = rm.find(k);
            if (it != rm.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // 24 B per bucket + the string payload each element owns. A std::string is 32 B
    // (SSO buffer, size, capacity) and keeps short keys inline; longer ones
    // allocate. We report the structure (buckets + 32 B string object) and note the
    // long-key heap bytes separately as a floor, so the number is not understated.
    double heap = 0.0;
    for (const auto& kv : rm)
        if (kv.first.size() > 15) heap += (double)kv.first.size() + 1; // > SSO
    const double bytes = (double)rm.bucket_count() * 24.0 + (double)rm.size() * 32.0 + heap;
    const double load = (double)rm.size() / (double)rm.bucket_count();
    return { (double)rm.size() / best, (double)rm.size() / t_ins, bytes, bytes / (double)rm.size(),
             load, rm.size() };
#else
    (void)keys; (void)value; (void)target_load;
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0 };
#endif
}

static PeerResult time_umap_strings(const StringKeys& keys, std::size_t, std::size_t,
                                    uint64_t value, float target_load) {
    const uint64_t* pv = &value;
    std::unordered_map<std::string, const uint64_t*> um;
    if (target_load > 0.0f) um.max_load_factor(target_load);
    um.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) um.clear();
        auto ti = Clock::now();
        for (const std::string& k : keys) um.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = um.find(k);
            if (it != um.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // Bucket array (bucket_count+1 pointers) + one node per key. The node holds a
    // 32 B std::string (SSO) plus the pointer and chain link; long keys also own
    // heap bytes, added as a floor.
    double heap = 0.0;
    for (const auto& kv : um)
        if (kv.first.size() > 15) heap += (double)kv.first.size() + 1;
    const double bytes =
        (double)(um.bucket_count() + 1) * 8.0 + (double)um.size() * 48.0 + heap;
    const double load = (double)um.size() / (double)um.bucket_count();
    return { (double)um.size() / best, (double)um.size() / t_ins, bytes, bytes / (double)um.size(),
             load, um.size() };
}

// unordered_dense for string keys: same layout as the integer form (values in a
// contiguous vector, index beside it). The value type is pair<std::string,
// const uint64_t*>, and the std::string owns its bytes -- short keys inline (SSO),
// long ones on the heap as a floor.
static PeerResult time_udense_strings(const StringKeys& keys, std::size_t, std::size_t,
                                      uint64_t value, float target_load) {
#if HAVE_UDENSE
    const uint64_t* pv = &value;
    ankerl::unordered_dense::map<std::string, const uint64_t*, str_mtht_hash> ud;
    if (target_load > 0.0f) ud.max_load_factor(target_load);
    ud.reserve(keys.size());
    double t_ins = 1e18;
    for (int r = 0; r < 3; ++r) {
        if (r) ud.clear();
        auto ti = Clock::now();
        for (const std::string& k : keys) ud.insert({k, pv});
        const double d = ms_since(ti);
        if (d < t_ins) t_ins = d;
    }
    volatile uint64_t sink = 0;
    double best = 1e18;
    for (int r = 0; r < 7; ++r) {
        auto t0 = Clock::now();
        for (const std::string& k : keys) {
            auto it = ud.find(k);
            if (it != ud.end()) sink += *it->second;
        }
        const double d = ms_since(t0);
        if (d < best) best = d;
    }
    (void)sink;
    // Value vector = 32 B std::string + 8 B pointer per element (capacity); index
    // 5.5 B per slot; long keys own heap bytes, added as a floor.
    double heap = 0.0;
    for (const auto& kv : ud.values())
        if (kv.first.size() > 15) heap += (double)kv.first.size() + 1;
    const double bytes =
        (double)ud.values().capacity() * 40.0 + (double)ud.bucket_count() * 5.5 + heap;
    const double load = (double)ud.size() / (double)ud.bucket_count();
    return { (double)ud.size() / best, (double)ud.size() / t_ins, bytes, bytes / (double)ud.size(),
             load, ud.size() };
#else
    (void)keys; (void)value; (void)target_load;
    return { 0.0, 0.0, 0.0, 0.0, 0.0, 0 };
#endif
}

static std::vector<StrPeer> make_string_peers() {
    std::vector<StrPeer> v;
    v.push_back({"tsl::robin_map", time_robin_strings, HAVE_ROBIN != 0});
    v.push_back({"ankerl::unordered_dense", time_udense_strings, HAVE_UDENSE != 0});
    v.push_back({"std::unordered_map", time_umap_strings, true});
    return v;
}

static void suite_peers_strings(std::size_t n1_shift, std::size_t t2_div, std::size_t ins_pct,
                                const std::vector<std::string>& want, float target_load = 0.70f) {
    (void)t2_div;
    std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t nkeys_fill = N1 * ins_pct / 100;
    if (target_load > 0.0f) {
        std::size_t want_slots = (std::size_t)((double)nkeys_fill / (double)target_load);
        N1 = 4;
        while (N1 < want_slots) N1 <<= 1;
    }
    const std::size_t N2 = N1 >> 3;
    const std::size_t nkeys = nkeys_fill;
    const uint64_t value = 0xABCDEF01ULL;
    const StringKeys keys = gen_strings(nkeys, 0x12345678ULL);

    std::vector<StrPeer> peers = make_string_peers();
    std::vector<std::string> skipped;

    out::line("== peers (string keys, target load %.2f) ==", (double)target_load);
    out::line("  n=%zu  MTHT t1=%zu t2=%zu", nkeys, N1, N2);

    const PeerResult m = time_mtht_strings(keys, N1, N2, value, 0.0f);

    struct Row { const char* name; PeerResult r; };
    std::vector<Row> rows;
    for (const StrPeer& p : peers) {
        const bool selected = want.empty() ||
            std::find(want.begin(), want.end(), std::string(p.name)) != want.end();
        if (!selected) continue;
        if (!p.available) { skipped.push_back(p.name); continue; }
        rows.push_back({p.name, p.run(keys, N1, N2, value, target_load)});
    }

    out::line("  %-18s %11s %11s %12s %8s %7s", "", "reads/s", "ins/s", "bytes", "B/key", "load");
    out::line("  %s", std::string(74, '-').c_str());
    out::line("  %-18s %11.1f %11.1f %12.0f %8.1f %7.3f", "MTHT", m.kps, m.ips, m.bytes, m.bpk, m.load);
    for (const Row& row : rows)
        out::line("  %-18s %11.1f %11.1f %12.0f %8.1f %7.3f", row.name, row.r.kps, row.r.ips,
                  row.r.bytes, row.r.bpk, row.r.load);
    for (const Row& row : rows)
        out::line("  MTHT vs %-18s %6.2fx", row.name, m.kps / row.r.kps);
    for (const Row& row : rows)
        out::line("  ins  vs %-18s %6.2fx", row.name, m.ips / row.r.ips);
    for (const std::string& s : skipped)
        out::line("  [skip] %s: not built (see --list-peers)", s.c_str());

    out::line("  bytes = table structure + owned key bytes");
    out::line("  B/key = bytes / keys held");
    out::line("  load = keys / slots");
    out::line("");
    out::line("  MTHT stores the whole key beside its value; the peers store a std::string");
    out::line("  in the element, which owns its bytes (short keys inline via SSO).");
    out::line("  tsl::robin_map is given MTHT's hash; std::unordered_map keeps std::hash --");
    out::line("  so its column differs in hash quality as well as table design.");
    out::line("  ankerl::unordered_dense is given MTHT's hash too, and stores keys+values");
    out::line("  in one contiguous vector rather than one node per key.");
    out::line("");
}

static void suite_peers(std::size_t n1_shift, std::size_t t2_div, std::size_t ins_pct,
                        const std::vector<std::string>& want, float target_load = 0.70f) {
    std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t nkeys_fill = N1 * ins_pct / 100;

    // A target load picks n1 from the key count instead of from --n1/--fill.
    // n1 stays a power of two (the table masks with n1-1 and grows by doubling),
    // so round up; T2 = n1/8 as always. The 0.75 T2 grow rule is untouched.
    if (target_load > 0.0f) {
        std::size_t want_slots = (std::size_t)((double)nkeys_fill / (double)target_load);
        N1 = 4;
        while (N1 < want_slots) N1 <<= 1;
    }
    const std::size_t N2 = N1 >> 3;   // SHIFT_RATIO, as in the table itself
    const std::size_t nkeys = nkeys_fill;
    const uint64_t value = 0xABCDEF01ULL;
    const Keys keys = gen_random(nkeys, 0x12345678ULL);

    std::vector<Peer> peers = make_peers();
    std::vector<std::string> skipped;

    if (target_load > 0.0f)
        out::line("== peers (target load %.2f, random keys) ==", (double)target_load);
    else
        out::line("== peers (each at its own default load, random keys) ==");
    out::line("  n=%zu  MTHT t1=%zu t2=%zu", nkeys, N1, N2);

    const PeerResult m = time_mtht(keys, N1, N2, value, 0.0f);

    // Collect the selected peers' results in registry order.
    struct Row { const char* name; PeerResult r; };
    std::vector<Row> rows;
    for (const Peer& p : peers) {
        const bool selected = want.empty() ||
            std::find(want.begin(), want.end(), std::string(p.name)) != want.end();
        if (!selected) continue;
        if (!p.available) { skipped.push_back(p.name); continue; }
        rows.push_back({p.name, p.run(keys, N1, N2, value, target_load)});
    }

    out::line("  %-18s %11s %11s %12s %8s %7s", "", "reads/s", "ins/s", "bytes", "B/key", "load");
    out::line("  %s", std::string(74, '-').c_str());
    out::line("  %-18s %11.1f %11.1f %12.0f %8.1f %7.3f", "MTHT", m.kps, m.ips, m.bytes, m.bpk, m.load);
    for (const Row& row : rows)
        out::line("  %-18s %11.1f %11.1f %12.0f %8.1f %7.3f", row.name, row.r.kps, row.r.ips,
                  row.r.bytes, row.r.bpk, row.r.load);
    for (const Row& row : rows)
        out::line("  MTHT vs %-18s %6.2fx", row.name, m.kps / row.r.kps);
    for (const Row& row : rows)
        out::line("  ins  vs %-18s %6.2fx", row.name, m.ips / row.r.ips);
    for (const std::string& s : skipped)
        out::line("  [skip] %s: not built (see --list-peers)", s.c_str());

    // What each column means, in the terms a reader can check from the numbers.
    out::line("  bytes = what the table allocates");
    out::line("  B/key = bytes / keys held");
    out::line("  load = keys / slots");
    out::line("");
    out::line("  tsl::robin_map and MTHT use the same hash function; umap has no hash.");
    out::line("");
}

// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    std::size_t n1_shift = 20, t2_div = 8, ins_pct = 75;
    std::string log_path = "bench/mthtbench.log";
    bool do_correctness = true, do_mtht = true, do_peers = true, quick = false;
    bool do_loads = true;
    bool do_peers_strings = true;
    bool list_peers = false;
    std::vector<std::string> peer_sel;
    std::vector<std::string> peer_sel_raw;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--suite=", 0) == 0) {
            const std::string s = a.substr(8);
            do_correctness = (s == "all" || s == "correctness");
            do_mtht = (s == "all" || s == "mtht");
            do_peers = (s == "all" || s == "peers");
            do_loads = (s == "all" || s == "loads");
            do_peers_strings = (s == "all" || s == "peers" || s == "strings");
        } else if (a.rfind("--peers=", 0) == 0) {
            // comma-separated list of registry names
            std::string list = a.substr(8);
            std::size_t pos = 0;
            while (pos <= list.size()) {
                const std::size_t comma = list.find(',', pos);
                const std::string tok = list.substr(pos, comma == std::string::npos
                                                            ? std::string::npos
                                                            : comma - pos);
                if (!tok.empty()) peer_sel_raw.push_back(tok);
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        } else if (a == "--list-peers") {
            list_peers = true;
        } else if (a.rfind("--log=", 0) == 0) {
            log_path = a.substr(6);
        } else if (a == "--no-log") {
            log_path.clear();
        } else if (a == "--quick") {
            quick = true;
            n1_shift = 16;
            t2_div = 8;
            ins_pct = 70;
        } else if (a.rfind("--n1=", 0) == 0) {
            n1_shift = (std::size_t)std::atoi(a.c_str() + 5);
        } else if (a.rfind("--t2div=", 0) == 0) {
            t2_div = (std::size_t)std::atoi(a.c_str() + 7);
        } else if (a.rfind("--fill=", 0) == 0) {
            ins_pct = (std::size_t)std::atoi(a.c_str() + 7);
        } else {
            std::printf("unknown arg: %s\n", a.c_str());
            std::printf("usage: %s [--suite=all|correctness|mtht|loads|peers|strings]"
                        " [--peers=NAME,...] [--list-peers]"
                        " [--log=PATH|--no-log] [--quick]"
                        " [--n1=20] [--t2div=8] [--fill=75]\n", argv[0]);
            return 2;
        }
    }

    // --list-peers prints the registry and exits (no log, no timing).
    if (list_peers) {
        std::printf("peers (use --peers=NAME,... to select):\n");
        for (const Peer& p : make_peers())
            std::printf("  %-18s %s\n", p.name, p.available ? "available" : "NOT BUILDABLE");
        std::printf("string-key peers (same names, used by --suite=strings):\n");
        for (const StrPeer& p : make_string_peers())
            std::printf("  %-18s %s\n", p.name, p.available ? "available" : "NOT BUILDABLE");
        return 0;
    }

    if (!log_path.empty()) out::open(log_path);

    out::line("================================================================");
    out::line("mthtbench  %s", now_string().c_str());
    out::line("cpu   : %s", cpu_string().c_str());
    out::line("build : clang %s  %s", __clang_version__, __DATE__);
    out::line("params: n1_shift=%zu t2_div=%zu fill=%zu%%%s", n1_shift, t2_div, ins_pct,
              quick ? " [--quick]" : "");
    if (quick) {
        out::line("        !! --quick uses a small table that fits in cache; k/s is a smoke");
        out::line("        !! test, not a real performance number. Drop --quick to measure.");
    }
    out::line("");

    int checks = 0;
    bool ok = true;
    if (do_correctness) ok = suite_correctness(n1_shift, t2_div, ins_pct, &checks);
    if (!ok) {
        out::line("correctness FAILED -- skipping timing suites.");
        out::close();
        return 1;
    }
    if (do_mtht) suite_mtht(n1_shift, t2_div, ins_pct);
    if (do_loads) suite_mtht_loads(n1_shift, t2_div, ins_pct);
    if (do_peers) suite_peers(n1_shift, t2_div, ins_pct, peer_sel_raw, 0.75f);
    if (do_peers_strings) suite_peers_strings(n1_shift, t2_div, ins_pct, peer_sel_raw, 0.75f);

    out::line("done %s%s", now_string().c_str(), log_path.empty() ? "" : ("  -> " + log_path).c_str());
    out::close();
    return 0;
}
