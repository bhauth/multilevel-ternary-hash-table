// selfbench -- the MTHT-only benchmark tool, for optimization work.
//
// This is bench/mthtbench.cpp with the peers (and their includes) dropped: no
// -Ipeers flags, no vendored headers, no comparison ratios -- so a change to
// mtht.hpp is measured against the previous build of this same tool rather than
// against an external map. A handy property of a self-comparison: every number
// here is MTHT only, so a regression or a gain shows up without a second table
// competing for cache.
//
// HOW TO BUILD
//   clang++ -O2 -std=c++20 -march=native -Iinclude \
//           bench/selfbench.cpp -o temp/selfbench.exe
//   (temp/ is git-ignored; the exe is a scratch artifact. -march=native matters:
//    the fast_map path uses 128-bit multiplies, so the numbers depend on it.)
//
// HOW TO RUN
//   ./temp/selfbench.exe                # every suite, writes bench/selfbench.log
//   ./temp/selfbench.exe --quick        # small tables (~2^14 keys), smoke test
//
//   Suite selection -- --suite= picks exactly one (default is all):
//     --suite=correctness   verify placement + every key reads back, then stop
//     --suite=mtht          MTHT alone across key distributions (time, load)
//     --suite=loads         MTHT alone while the fill level rises, to the point
//                           it auto-resizes (time vs achieved load)
//     --suite=strings       MTHT alone with string keys (time, bytes/key)
//     --suite=longkeys      MTHT alone, the README long-keys shape: word keys
//                           4-24 B, values 16-128 B (time, bytes/key)
//
//   Other flags:
//     --log=PATH            write the log somewhere else
//     --no-log              console only, write no log
//     --n1=N                log2 of T1 slots (default 20); N is the EXPONENT, so
//                           20 is 2^20 = 1048576 slots. Key count follows from
//                           the fill target below, not from this flag.
//     --t2div=D             T2 slots = T1/D (default 8)
//     --fill=F              insert until F percent of the T1 slots are used
//                           (default 75, i.e. a 0.75 load)
//
//   Typical use: --suite=loads when tuning the growth point, --suite=strings or
//   --suite=longkeys for the string-key paths, --suite=mtht for the rest.
//
// OUTPUT LOG
//   Console output is mirrored to the log file (default bench/selfbench.log),
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
//   strings      -- MTHT alone with string keys: the whole key is stored in the
//                   record. Keys are mixed length so the record store is
//                   exercised.
//   longkeys     -- MTHT alone with the README's long-keys shape: word keys
//                   4-24 B and values 16-128 B. Reproduces that README cell.
//
// Quiet-by-default output: a passing row prints its numbers; only failures print
// a [FAIL] line.

// MTHT itself is measured through the shipping header (include/mtht.hpp): one
// traversal, named find(). That header has no probe or spill counters, so the
// bench reports time, throughput and memory -- what the table exposes.
#include "mtht.hpp"

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
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
// well spread.
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

// Word-shaped keys (like a JSON document's field names) and drawn-length
// values. Ported verbatim from longkeys.cpp so the README's "4-24 B keys /
// 16-128 B values" cell is reproduced by this tool too.
using Value = std::vector<unsigned char>;

static StringKeys gen_words(std::size_t n, std::size_t lo, std::size_t hi, uint64_t seed) {
    StringKeys v;
    v.reserve(n);
    uint64_t s = seed;
    for (std::size_t i = 0; i < n; ++i) {
        s = splitmix(s);
        const std::size_t len = lo + (s >> 8) % (hi - lo + 1);
        std::string str;
        str.resize(len);
        std::size_t j = 0;
        while (j < len) {
            s = splitmix(s);
            const std::size_t word = 1 + (s >> 8) % 5; // 1..5 letters
            for (std::size_t w = 0; w < word && j < len; ++w, ++j) {
                s = splitmix(s);
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
        s = splitmix(s);
        const std::size_t len = lo + (s >> 8) % (hi - lo + 1);
        Value val(len);
        for (std::size_t j = 0; j < len; ++j) {
            s = splitmix(s);
            val[j] = (unsigned char)(s >> 24);
        }
        v.push_back(std::move(val));
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
// Suite: mtht across distributions. MTHT alone: the umap column the committed
// tool carries is gone, so each row is the table's own insert/find cost.
// ---------------------------------------------------------------------------
static void suite_mtht(std::size_t n1_shift, std::size_t t2_div, std::size_t ins_pct) {
    const std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t N2 = N1 / t2_div;
    const std::size_t nkeys = N1 * ins_pct / 100;
    const uint64_t value = 0xABCDEF01ULL;

    out::line("== mtht ==");
    out::line("  N1=%zu N2=%zu (t2=t1/%zu) keys/case=%zu value=8B RAW keys", N1, N2, t2_div, nkeys);
    out::line("  %-11s %9s %11s %9s %-9s", "case", "ins_ms", "find_ms", "Mfind/s", "t1/t2 load");
    out::line("  %s", std::string(96, '-').c_str());

    for (Dist& d : distributions(nkeys)) {
        mtht::Table t(N1, sizeof(uint64_t));

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

        char loads[40];
        std::snprintf(loads, sizeof loads, "%.3f/%.3f", t.load1(), t.load2());
        out::line("  %-11s %9.2f %11.2f %9.1f %-9s",
                  d.name, t_ins, t_find, (double)nkeys / t_find, loads);
        if (inserted != nkeys) out::line("     !! inserted %zu/%zu", inserted, nkeys);
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
    Dist& d = dists[1]; // "random"
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
// Suite: MTHT alone, the README long-keys cell: word keys 4-24 B and values
// 16-128 B, the shape a JSON document's field map has. Ported from longkeys.cpp's
// cell_realistic MTHT block -- sizing (n1 for a 0.70 load, a 4 MB key store) and
// the bytes/key accounting both match, so this row lines up with the README's
// "4-24 B / 16-128 B" cell. Reads/s and ins/s are keys per second (best-rep ms).
// ---------------------------------------------------------------------------
static void suite_mtht_longkeys(std::size_t nkeys) {
    const StringKeys keys = gen_words(nkeys, 4, 24, 0x5DEECE66DULL);
    const std::vector<Value> vals = gen_vals(nkeys, 16, 128, 0x1234ABCDULL);

    out::line("== mtht longkeys: keys 4-24 B, values 16-128 B ==");
    out::line("  n=%zu  random word keys, drawn-length values", keys.size());

    std::size_t n1 = 4;
    while (n1 < keys.size() / 0.70) n1 <<= 1;
    mtht::Table t(n1, 1u << 22);

    double t_ins = 1e18;
    for (int r = 0; r < 2; ++r) {
        if (r) t.clear();
        auto ti = Clock::now();
        for (std::size_t i = 0; i < keys.size(); ++i)
            t.insert(keys[i].data(), keys[i].size(), vals[i].data(), vals[i].size());
        t_ins = std::min(t_ins, ms_since(ti));
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
        best = std::min(best, ms_since(t0));
    }
    (void)sink;

    // Slot arrays plus one record per key: the key bytes, the value bytes, the
    // 8 B shim, the length byte and the state byte -- the same accounting the
    // README cell uses.
    double bytes = (double)(t.n1() + t.n2()) * 8.0;
    for (std::size_t i = 0; i < keys.size(); ++i)
        bytes += 8.0 + 1.0 + (double)keys[i].size() + (double)vals[i].size();

    out::line("  %-12s %11s %11s %10s", "", "reads/s", "ins/s", "B/key");
    out::line("  %s", std::string(48, '-').c_str());
    out::line("  %-12s %11.0f %11.0f %10.1f", "MTHT",
              (double)keys.size() / best, (double)keys.size() / t_ins, bytes / keys.size());
    if (hit != keys.size())
        out::line("  !! %zu/%zu keys read back", hit, keys.size());
    out::line("  reads/s and ins/s = keys/s (best-rep ms; x1000 for reads/ms)");
    out::line("");
}

// ---------------------------------------------------------------------------
// Suite: MTHT alone with string keys. The whole key is stored in the record
// beside the value; bytes/key counts the slot arrays plus that record store,
// since for string keys the store is the point of the design -- not assumed
// away. Inserts best of 3, each rep after a clear (re-inserting a key is a
// different, replace path); reads best of 7, with a readback check on the final
// state.
// ---------------------------------------------------------------------------
static void suite_mtht_strings(std::size_t n1_shift, std::size_t t2_div, std::size_t ins_pct,
                               float target_load = 0.70f) {
    std::size_t N1 = std::size_t(1) << n1_shift;
    const std::size_t nkeys_fill = N1 * ins_pct / 100;
    // A target load picks n1 from the key count; n1 stays a power of two.
    if (target_load > 0.0f) {
        std::size_t want_slots = (std::size_t)((double)nkeys_fill / (double)target_load);
        N1 = 4;
        while (N1 < want_slots) N1 <<= 1;
    }
    const std::size_t N2 = N1 >> 3;
    const std::size_t nkeys = nkeys_fill;
    const uint64_t value = 0xABCDEF01ULL;
    const StringKeys keys = gen_strings(nkeys, 0x12345678ULL);

    out::line("== mtht strings (target load %.2f) ==", (double)target_load);
    out::line("  n=%zu  MTHT t1=%zu t2=%zu  value=8B pointer  mixed-length keys",
              nkeys, N1, N2);

    const uint64_t* pv = &value;
    mtht::Table t(N1, sizeof(uint64_t));
    std::size_t ins_ok = 0;
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

    std::size_t found_ok = 0;
    for (const std::string& k : keys) {
        const void* p = t.find(k.data(), k.size());
        if (p && *(const uint64_t* const*)p == pv) ++found_ok;
    }

    // Slot arrays (8 B Entry * live T1+T2) plus the record store, which holds the
    // copied keys and values.
    const double live_n1 = (double)t.n1(), live_n2 = (double)t.n2();
    const double bytes = (live_n1 + live_n2) * 8.0 + (double)t.key_store_bytes();
    const double load = (double)t.size() / (live_n1 + live_n2);

    out::line("  %-12s %11s %11s %12s %8s %7s",
              "", "reads/s", "ins/s", "bytes", "B/key", "load");
    out::line("  %s", std::string(66, '-').c_str());
    out::line("  %-12s %11.1f %11.1f %12.0f %8.1f %7.3f",
              "MTHT", (double)t.size() / best, (double)t.size() / t_ins, bytes,
              bytes / (double)t.size(), load);
    if (ins_ok != keys.size())
        out::line("  !! %zu/%zu inserts accepted", ins_ok, keys.size());
    if (found_ok != keys.size())
        out::line("  !! %zu/%zu keys read back", found_ok, keys.size());

    out::line("  bytes = slot arrays + owned key/value record store");
    out::line("  B/key = bytes / keys held");
    out::line("  load = keys / slots");
    out::line("");
}

// ---------------------------------------------------------------------------
// Kept in one place so the header comment and --help cannot drift apart.
static void print_help(const char* argv0) {
    std::printf(
        "selfbench -- MTHT-only benchmark tool (no peer tables).\n"
        "\n"
        "usage: %s [--suite=all|correctness|mtht|loads|strings|longkeys]\n"
        "       %*s [--log=PATH | --no-log] [--quick]\n"
        "       %*s [--n1=20] [--t2div=8] [--fill=75]\n"
        "\n"
        "  --suite=NAME   run exactly one suite; default is all of them.\n"
        "                   correctness  verify placement + every key reads back\n"
        "                   mtht         MTHT across key distributions\n"
        "                   loads        MTHT vs rising fill level, to resize\n"
        "                   strings      MTHT with mixed-length string keys\n"
        "                   longkeys     README shape: keys 4-24 B, values 16-128 B\n"
        "  --quick        small tables (~2^14 keys) for a fast smoke test\n"
        "  --n1=N         log2 of T1 slots (default 20 -> 2^20 = 1048576 slots)\n"
        "  --t2div=D      T2 slots = T1/D (default 8)\n"
        "  --fill=F       insert until F%% of the T1 slots are used (default 75)\n"
        "  --log=PATH     write the run log elsewhere (default bench/selfbench.log)\n"
        "  --no-log       console only, write no log\n"
        "  --help         print this text\n",
        argv0, (int)std::strlen(argv0), "", (int)std::strlen(argv0), "");
}

int main(int argc, char** argv) {
    std::size_t n1_shift = 20, t2_div = 8, ins_pct = 75;
    std::string log_path = "bench/selfbench.log";
    bool do_correctness = true, do_mtht = true, do_loads = true, do_strings = true;
    bool do_longkeys = true;
    bool quick = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--suite=", 0) == 0) {
            const std::string s = a.substr(8);
            do_correctness = (s == "all" || s == "correctness");
            do_mtht = (s == "all" || s == "mtht");
            do_loads = (s == "all" || s == "loads");
            do_strings = (s == "all" || s == "strings");
            do_longkeys = (s == "all" || s == "longkeys");
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
        } else if (a == "--help" || a == "-h") {
            print_help(argv[0]);
            return 0;
        } else {
            std::printf("unknown arg: %s\n\n", a.c_str());
            print_help(argv[0]);
            return 2;
        }
    }

    if (!log_path.empty()) out::open(log_path);

    out::line("================================================================");
    out::line("selfbench  %s", now_string().c_str());
    out::line("cpu   : %s", cpu_string().c_str());
    out::line("build : clang %s  %s", __clang_version__, __DATE__);
    out::line("params: n1_shift=%zu t2_div=%zu fill=%zu%%%s", n1_shift, t2_div, ins_pct,
              quick ? " [--quick]" : "");
    out::line("purpose: MTHT only (no peers) -- compare against an earlier build of"
              " this tool.");
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
    if (do_strings) suite_mtht_strings(n1_shift, t2_div, ins_pct);
    if (do_longkeys) suite_mtht_longkeys(std::size_t(1) << n1_shift);

    out::line("done %s%s", now_string().c_str(), log_path.empty() ? "" : ("  -> " + log_path).c_str());
    out::close();
    return 0;
}
