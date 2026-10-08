// Multilevel Ternary Hash Table (MTHT)
//
// Layout: T1 is an open-addressing table whose keys live in a ternary
// neighborhood {h-1, h, h+1} of their home bucket. Displaced keys are tagged
// with a 2-bit state in the high bits of val_ptr. Keys that cannot fit spill
// into T2, whose per-bucket displacement is bounded by a 6-bit D_max stored in
// the home slot of T1.
#ifndef MTHT_HPP
#define MTHT_HPP

#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <functional>
#include <utility>
#include <memory>
#include <new>
#include <vector>

namespace mtht {

// Entry for both tables, exactly 8 bytes so that 8 fit one 64-byte cache line.
// The entry holds a fingerprint, not the key: the key lives once in the value
// record the pointer reaches, and a fingerprint match is confirmed against it.
struct alignas(8) Entry {
    // [47..0] pointer to the value record, [55..48] 8-bit key fingerprint,
    // [63..56] header byte: [57..56] 2-bit state, [63..58] 6-bit D_max.
    uint64_t word;
};

static_assert(sizeof(Entry) == 8, "Entry must be 8 bytes");

// 2-bit displacement / occupancy state, at bits [57..56] of the word.
enum : uint8_t {
    ST_EMPTY = 0,  // 00: empty element
    ST_OFF_NEG = 1, // 01: offset -1 (home is slot_index + 1)
    ST_OFF_ZERO = 2, // 10: offset 0 (home == slot_index)
    ST_OFF_POS = 3, // 11: offset +1 (home is slot_index - 1)
};

// T2-local state value. A T2 slot is only ever placed at ST_OFF_ZERO (a spill
// has no ternary neighborhood), so the other two codes are free there; this one
// marks a slot whose entry was erased. It keeps the slot from reading as
// ST_EMPTY, so a forward T2 scan passes over it instead of stopping, and insert
// may still reuse it. Not a displacement tag -- do not use it on a T1 slot.
static constexpr uint8_t ST_TOMBSTONE = ST_OFF_NEG;

// D_max sentinel: offset >= 63, lookup falls back to forward probing in T2.
static constexpr uint8_t DMAX_SENTINEL = 63;

// Shift between a T1 home bucket and its T2 base bucket: T2.size = T1.size >> 3.
static constexpr int SHIFT_RATIO = 3;

// T2 load at which the whole table doubles. Checked on the spill path only.
static constexpr double T2_GROW_LOAD = 0.75;

static constexpr uint64_t PTR_MASK = 0x0000FFFFFFFFFFFFULL;
static constexpr uint64_t PTR_BITS = 48; // address bits kept from a pointer
static constexpr uint64_t FP_SHIFT = 48;
static constexpr uint64_t FP_MASK = 0xFFULL;
static constexpr uint64_t STATE_SHIFT = 56;
static constexpr uint64_t STATE_MASK = 0x03ULL;
static constexpr uint64_t DMAX_SHIFT = 58;
static constexpr uint64_t DMAX_MASK = 0x3FULL;

// Hash to bucket, using the high bits of the 128-bit product (Lemire).
inline uint64_t fast_map(uint64_t hash, uint64_t capacity) {
    return static_cast<uint64_t>(
        (static_cast<__uint128_t>(hash) * static_cast<__uint128_t>(capacity)) >> 64);
}

// Bitfield accessors for the words above. `state` is a value from the ST_*
// enum (or ST_TOMBSTONE); `dmax` the bound from get_t2_max_offset.
inline uint8_t get_state(const Entry& e) {
    return static_cast<uint8_t>((e.word >> STATE_SHIFT) & STATE_MASK);
}

inline void set_state(Entry& e, uint8_t state) {
    e.word = (e.word & ~(STATE_MASK << STATE_SHIFT)) |
             (static_cast<uint64_t>(state & 0x03) << STATE_SHIFT);
}

inline uint8_t get_t2_max_offset(const Entry& e) {
    return static_cast<uint8_t>((e.word >> DMAX_SHIFT) & DMAX_MASK);
}

inline void set_t2_max_offset(Entry& e, uint8_t dmax) {
    e.word = (e.word & ~(DMAX_MASK << DMAX_SHIFT)) |
             (static_cast<uint64_t>(dmax & 0x3F) << DMAX_SHIFT);
}

inline void* decode_pointer(uint64_t word) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(word & PTR_MASK));
}

inline uint64_t encode_pointer(const void* p) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(p);
    // x86-64 user addresses fit in 48 bits. Warn loudly rather than silently
    // corrupting the pointer if that ever stops holding.
    if (addr > PTR_MASK) {
        std::fprintf(stderr, "mtht: pointer above 48 bits: %p\n", p);
        std::abort();
    }
    return static_cast<uint64_t>(addr);
}

// splitmix64 finalizer. Avalanches all 64 input bits, so keys 0 and ~0ULL are
// ordinary keys rather than degenerate ones.
inline uint64_t hash_key(uint64_t k) {
    k ^= k >> 30;
    k *= 0xbf58476d1ce4e5b9ULL;
    k ^= k >> 27;
    k *= 0x94d049bb133111ebULL;
    k ^= k >> 31;
    return k;
}

// An integer key is its own 8 bytes, so it never needs a span.
inline uint64_t hash_int(uint64_t k) { return hash_key(k); }

// The value record stores the whole key, so a fingerprint match is confirmed
// exactly and no collision can return another key's value:
//
//     [ key_bytes : len ][ len - 1 : 1 ][ value... ]
//
// The table hands out a pointer to the value; the key is read back from that
// pointer at fixed offsets from it. The length byte holds `len - MIN_KEY_BYTES`
// so the 8-bit field spans 1..256 bytes (8..2048 bits).
static constexpr std::size_t MIN_KEY_BYTES = 1;
static constexpr std::size_t MAX_KEY_BYTES = 256; // MIN + 255

// Stored length byte for an actual key length.
inline uint8_t encode_key_len(std::size_t len) {
    return static_cast<uint8_t>(len - MIN_KEY_BYTES);
}

// Actual key length from the byte stored with the value at `val`.
inline std::size_t key_len_at(const void* val) {
    uint8_t n = 0;
    std::memcpy(&n, static_cast<const unsigned char*>(val) - 1, 1);
    return static_cast<std::size_t>(n) + MIN_KEY_BYTES;
}

// Pointer to the key bytes for the value at `val`.
inline const unsigned char* key_ptr_at(const void* val) {
    return static_cast<const unsigned char*>(val) - 1 - key_len_at(val);
}

// Are the key bytes stored with `val` equal to the probe key?
inline bool key_at_equals(const void* val, const void* key, std::size_t len) {
    const auto* base = static_cast<const unsigned char*>(val);
    // 8-byte key (every integer key): the stored key ends immediately before the
    // value, so it is one 8-byte load at val-9, and an equal compare already
    // proves the stored key is 8 bytes -- no length byte needed.
    if (len == 8) {
        uint64_t a, b;
        std::memcpy(&a, base - 9, 8);
        std::memcpy(&b, key, 8);
        return a == b;
    }
    // Other lengths: the length byte sits at val-1, the key bytes before it.
    if (base[-1] != encode_key_len(len)) return false;
    return std::memcmp(base - 1 - len, key, len) == 0;
}

// 8-bit fingerprint of a key: the filter that stands in for the key in a slot.
//
// The slot index is fast_map(hash, n1) -- the high bits of the 128-bit product,
// which for a large n1 is the high end of the hash. Taking the fingerprint from
// those same high bits would make it agree exactly for keys that already share
// a home slot, which is the one case the filter exists to reject. So it comes
// from the low end instead, passed through its own mix.
inline uint8_t fingerprint_of_hash(uint64_t h) {
    uint64_t k = h & 0xFFULL;
    k *= 0x9E3779B97F4A7C15ULL; // h mixes the low byte up to byte 7
    return static_cast<uint8_t>(k >> 56);
}

// Spans up to 8 bytes become one word and go straight through hash_key, so they
// pay the finalizer only; longer spans go to std::hash<string_view> first and
// are finalised after.
inline uint64_t hash_bytes(const void* p, std::size_t len) {
    if (len == 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        return hash_key(w);           // plain integer key
    }
    if (len < 8) {
        uint64_t w = 0;
        std::memcpy(&w, p, len);
        return hash_key(w + (static_cast<uint64_t>(len) << 56));
    }
    std::string_view sv(static_cast<const char*>(p), len);
    return hash_key(static_cast<uint64_t>(std::hash<std::string_view>{}(sv)));
}

inline uint8_t fingerprint_bytes(const void* p, std::size_t len) {
    return fingerprint_of_hash(hash_bytes(p, len));
}

inline uint8_t get_fingerprint(const Entry& e) {
    return static_cast<uint8_t>((e.word >> FP_SHIFT) & FP_MASK);
}

// Write an entry into a slot: fingerprint and pointer, with the given state,
// preserving any existing D_max bound (an erased slot may still own elements
// in T2). The key itself is not written here -- it is already in the record.
inline void write_slot(Entry& e, uint8_t fp, const void* val, uint8_t state) {
    uint64_t dmax = e.word & (DMAX_MASK << DMAX_SHIFT);
    e.word = encode_pointer(val) | (static_cast<uint64_t>(fp) << FP_SHIFT) | dmax |
             (static_cast<uint64_t>(state & 0x03) << STATE_SHIFT);
}

// Move src into dst with a new state, clearing src's occupancy but keeping its
// D_max bound. The fingerprint and pointer travel; the key does not move.
inline void shift_entry(Entry& src, Entry& dst, uint8_t new_state) {
    uint64_t dmax = dst.word & (DMAX_MASK << DMAX_SHIFT);
    dst.word = (src.word & (PTR_MASK | (FP_MASK << FP_SHIFT))) | dmax |
               (static_cast<uint64_t>(new_state & 0x03) << STATE_SHIFT);
    src.word &= ~(STATE_MASK << STATE_SHIFT);
}

// Does `e` hold the probe span, at the already-computed fingerprint `fp`?
inline bool slot_holds_fp(const Entry& e, const void* key, std::size_t len, uint8_t fp) {
    if (get_state(e) == ST_EMPTY) return false;
    if (get_fingerprint(e) != fp) return false;
    return key_at_equals(decode_pointer(e.word), key, len);
}

inline bool slot_holds(const Entry& e, const void* key, std::size_t len) {
    return slot_holds_fp(e, key, len, fingerprint_bytes(key, len));
}

// T2 counterpart of slot_holds_fp. A T2 tombstone keeps the erased key's
// fingerprint and pointer, so the shared predicate would match it; a tombstone's
// key is gone and must never read as live. T1 has no tombstones, so this check
// lives here and not in slot_holds_fp.
inline bool t2_holds_fp(const Entry& e, const void* key, std::size_t len, uint8_t fp) {
    if (get_state(e) == ST_TOMBSTONE) return false;
    return slot_holds_fp(e, key, len, fp);
}

// Default hash policy: std::hash<uint64_t> is the identity, and the table
// takes its index from a power-of-two mask, so an identity degenerates on keys
// whose low bits repeat.
struct Hash {
    std::size_t operator()(const void* p, std::size_t len) const {
        return static_cast<std::size_t>(hash_bytes(p, len));
    }
    std::size_t operator()(uint64_t k) const {
        return static_cast<std::size_t>(hash_key(k));
    }
};


class KeyStore {
public:
    // `initial_bytes` is the size of the first arena block, in bytes: the store
    // opens with that much and doubles on each later block. Zero or a value
    // below one block selects the 4 KB default.
    explicit KeyStore(std::size_t initial_bytes) {
        if (initial_bytes > kMinBlock) {
            block_bytes_ = round_up_16(initial_bytes);
        }
        open_block(block_bytes_);
    }

    ~KeyStore() { clear(); }

    KeyStore(const KeyStore&) = delete;
    KeyStore& operator=(const KeyStore&) = delete;

    // Copy a key span and `bytes` into the store as one record and return a
    // stable pointer to the value. The record is
    //
    //     [ key : len ][ len - MIN : 1 ][ value... ]
    //
    // padded to a whole number of 16-byte slots, so records sit at a fixed
    // stride and four fill a cache line. A zero-length value still takes one
    // slot. Returns nullptr for a key outside [MIN_KEY_BYTES, MAX_KEY_BYTES].
    void* store(const void* key, std::size_t len, const void* bytes,
                std::size_t bytes_len) {
        if (len < MIN_KEY_BYTES || len > MAX_KEY_BYTES) return nullptr;

        std::size_t head = len + 1;                      // key bytes + length byte
        std::size_t n = round_up_16(head + (bytes_len ? bytes_len : 1));
        unsigned char* rec = reserve(n);

        std::memcpy(rec, key, len);

        unsigned char* lp = rec + len;
        *lp = encode_key_len(len);

        unsigned char* p = lp + 1;                       // value starts here
        if (bytes_len) std::memcpy(p, bytes, bytes_len);
        std::size_t written = head + bytes_len;
        if (written < n) {
            std::memset(rec + written, 0, n - written);
        }
        return p;
    }

    void clear() {
        for (void* p : blocks_) std::free(p);
        blocks_.clear();
        bump_ = nullptr;
        bump_left_ = 0;
        bytes_used_ = 0;
        block_bytes_ = kMinBlock;
    }

    // Bytes currently held in allocated blocks (rounded to whole slots): the
    // memory the store costs beyond the slot arrays.
    std::size_t bytes_used() const { return bytes_used_; }

private:
    // Round up to the next slot boundary; slots are 16 bytes.
    static std::size_t round_up_16(std::size_t n) {
        return (n + 15) & ~static_cast<std::size_t>(15);
    }

    // Allocate a fresh block of `bytes` and make it the bump region. Any space
    // left in the old block is abandoned, but pointers into it stay valid.
    void open_block(std::size_t bytes) {
        void* raw = std::malloc(bytes);
        if (!raw) {
            std::fprintf(stderr, "mtht: out of memory in KeyStore\n");
            std::abort();
        }
        blocks_.push_back(raw);
        bytes_used_ += bytes;
        bump_ = static_cast<unsigned char*>(raw);
        bump_left_ = bytes;
        block_bytes_ = bytes;
    }

    // Reserve `bytes` of bump space and return it. Callers pass a whole number
    // of slots, so the pointer and the remaining capacity stay slot-aligned.
    unsigned char* reserve(std::size_t bytes) {
        if (!bump_ || bytes > bump_left_) {
            std::size_t block = bytes > block_bytes_ ? bytes : block_bytes_;
            open_block(round_up_16(block));
        }
        unsigned char* p = bump_;
        bump_ += bytes;
        bump_left_ -= bytes;
        return p;
    }

    static constexpr std::size_t kMinBlock = 4096; // 256 slots
    unsigned char* bump_ = nullptr;
    std::size_t bump_left_ = 0;
    std::size_t bytes_used_ = 0;
    std::size_t block_bytes_ = kMinBlock; // doubles on each later block
    std::vector<void*> blocks_;
};

// The table. n1 (T1 slot count) and n2 (T2 slot count) are powers of two.
// Hash is a template parameter so a lookup calls it directly, with no indirect
// call per probe.
template <class Hash = mtht::Hash>
class BasicTable {
public:
    static constexpr std::size_t DEFAULT_N1 = 1u << 20; // 1,048,576 slots, 16 MB
    static constexpr std::size_t DEFAULT_N2 = DEFAULT_N1 >> SHIFT_RATIO;

    // n2 is the initial T2 size only: growth recomputes it as n1 >> SHIFT_RATIO,
    // so n2 == 0 selects the default, which is also what every table settles on.
    //
    // store_bytes is the initial allocation size of the key store, in bytes;
    // zero selects the 4 KB default.
    explicit BasicTable(std::size_t n1 = DEFAULT_N1, std::size_t store_bytes = 0,
                        Hash hash = Hash{})
        : hash_(hash), key_store_(store_bytes) {
        if (n1 < 4) n1 = 4;
        n1 = ceil_pow2(n1);
        n2_ = n1 >> SHIFT_RATIO;
        if (n2_ < 4) n2_ = 4;
        n2_ = ceil_pow2(n2_);
        n1_ = n1;
        mask1_ = n1 - 1;
        mask2_ = n2_ - 1;
        t1_ = static_cast<Entry*>(std::calloc(n1, sizeof(Entry)));
        t2_ = static_cast<Entry*>(std::calloc(n2_, sizeof(Entry)));
        if (!t1_ || !t2_) {
            std::fprintf(stderr, "mtht: out of memory for tables\n");
            std::abort();
        }
    }

    ~BasicTable() {
        std::free(t1_);
        std::free(t2_);
    }

    BasicTable(const BasicTable&) = delete;
    BasicTable& operator=(const BasicTable&) = delete;

    std::size_t size() const { return count_; }
    std::size_t n1() const { return n1_; }
    std::size_t n2() const { return n2_; }
    std::size_t t1_count() const { return count_ - t2_count_; }
    std::size_t t2_count() const { return t2_count_; }
    double load1() const { return static_cast<double>(t1_count()) / static_cast<double>(n1_); }
    double load2() const { return static_cast<double>(t2_count_) / static_cast<double>(n2_); }

    std::size_t key_store_bytes() const { return key_store_.bytes_used(); }

    // Raw slot arrays, for tests that verify the bit-level layout.
    const Entry* raw_t1() const { return t1_; }
    const Entry* raw_t2() const { return t2_; }
    Entry* raw_t1() { return t1_; }
    Entry* raw_t2() { return t2_; }

    // Erase every element, keeping capacity.
    void clear() {
        std::memset(t1_, 0, n1_ * sizeof(Entry));
        std::memset(t2_, 0, n2_ * sizeof(Entry));
        count_ = t2_count_ = 0;
        key_store_.clear();
    }

    // Re-inserting a key points the slot at a new record carrying the new
    // value. The three overloads take an integer key, a string (through the
    // string_view overload), or a raw key span; all reach insert_key().
    bool insert(uint64_t key, const void* bytes, std::size_t bytes_len) {
        return insert_key(&key, sizeof key, bytes, bytes_len);
    }
    bool insert(std::string_view key, const void* bytes, std::size_t bytes_len) {
        return insert_key(key.data(), key.size(), bytes, bytes_len);
    }
    bool insert(const void* key, std::size_t key_len, const void* bytes,
                std::size_t bytes_len) {
        return insert_key(key, key_len, bytes, bytes_len);
    }

    // The shared insert body: the key arrives as a span.
    bool insert_key(const void* key, std::size_t key_len, const void* bytes,
                    std::size_t bytes_len) {
        void* val = key_store_.store(key, key_len, bytes, bytes_len);
        if (!val) return false; // key outside [MIN_KEY_BYTES, MAX_KEY_BYTES]

        const uint64_t hh = hash_bytes(key, key_len);
        const uint64_t h = fast_map(hh, static_cast<uint64_t>(n1_));
        const uint8_t fp = fingerprint_of_hash(hh);

        // Existing key: keep the fingerprint, D_max and state bits; only the
        // pointer changes. slot_for_key finds it in T1 or T2 wherever it lives.
        bool in_t1 = false;
        if (Entry* e = slot_for_key(key, key_len, h, fp, in_t1)) {
            uint64_t tag = e->word & ~PTR_MASK;
            e->word = tag | encode_pointer(val);
            return true;
        }

        return place_into(key, key_len, hh, h, val);
    }

    // The live slot holding the key span, or nullptr -- the probe order find()
    // uses. `h` is the key's home bucket and `fp` its fingerprint, both already
    // computed by the caller. `in_t1` reports which table the hit came from, so
    // erase() knows whether pull_neighbors_home applies.
    Entry* slot_for_key(const void* key, std::size_t key_len, uint64_t h, uint8_t fp,
                        bool& in_t1) {
        Entry& home = t1_[h];
        if (get_state(home) == ST_OFF_ZERO && slot_holds_fp(home, key, key_len, fp)) {
            in_t1 = true;
            return &home;
        }

        const int first = +1;
        Entry& n1 = t1_[(h + static_cast<uint64_t>(first)) & mask1_];
        if (slot_holds_fp(n1, key, key_len, fp)) { in_t1 = true; return &n1; }
        Entry& n2 = t1_[(h + static_cast<uint64_t>(-first)) & mask1_];
        if (slot_holds_fp(n2, key, key_len, fp)) { in_t1 = true; return &n2; }

        uint8_t max_offset = get_t2_max_offset(home);
        if (max_offset == 0) return nullptr;
        Entry* e = slot_for_key_T2(h, key, key_len, fp, max_offset);
        if (e) in_t1 = false;
        return e;
    }

    // T2 half of the probe set: the matching Entry, or nullptr. Mirrors
    // find_in_T2's two scan modes.
    Entry* slot_for_key_T2(uint64_t h, const void* key, std::size_t key_len, uint8_t fp,
                           uint8_t max_offset) {
        uint64_t base_t2 = (h >> SHIFT_RATIO) & mask2_;

        if (max_offset < DMAX_SENTINEL) {
            for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
                Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
                if (t2_holds_fp(e, key, key_len, fp)) return &e;
            }
        } else {
            // Saturated: forward sweep, stopping only at an ST_EMPTY slot.
            for (uint64_t off = 0; off < n2_; ++off) {
                Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == ST_EMPTY) break;
                if (t2_holds_fp(e, key, key_len, fp)) return &e;
            }
        }
        return nullptr;
    }

    // The placement ladder, steps 1-5 below.
    bool place_into(const void* key, std::size_t key_len, uint64_t hh, uint64_t h, void* val) {
        const uint8_t fp = fingerprint_of_hash(hh);
        Entry& home = t1_[h];
        uint8_t state = get_state(home);

        // 1. Direct placement in the home slot.
        if (state == ST_EMPTY) {
            write_slot(home, fp, val, ST_OFF_ZERO);
            ++count_;
            return true;
        }

        // 2. First free neighbor, biased to keep the entry in h's cache line
        // (h+1 unless h is the last slot of its line).
        int cache_slot = static_cast<int>(h & 3);
        int first_pref = (cache_slot == 3) ? -1 : +1;
        int second_pref = -first_pref;
        uint64_t first_idx = (h + first_pref) & mask1_;
        uint64_t second_idx = (h + second_pref) & mask1_;
        uint8_t first_state = (first_pref == +1) ? ST_OFF_POS : ST_OFF_NEG;
        uint8_t second_state = (second_pref == +1) ? ST_OFF_POS : ST_OFF_NEG;

        if (get_state(t1_[first_idx]) == ST_EMPTY) {
            write_slot(t1_[first_idx], fp, val, first_state);
            ++count_;
            return true;
        }
        if (get_state(t1_[second_idx]) == ST_EMPTY) {
            write_slot(t1_[second_idx], fp, val, second_state);
            ++count_;
            return true;
        }

        // 3. On the right cache-line edge, shift h-1 to h-2 to free h-1 for the
        //    home slot's key. The mover must be at offset 0 (it keeps its own
        //    home) and h-2 free; the shifted tag is then ST_OFF_NEG by
        //    construction.
        uint64_t left_idx = (h - 1) & mask1_;
        uint64_t far_left = (h - 2) & mask1_;
        if (cache_slot == 3 && get_state(t1_[left_idx]) == ST_OFF_ZERO &&
            get_state(t1_[far_left]) == ST_EMPTY) {
            shift_entry(t1_[left_idx], t1_[far_left], ST_OFF_NEG);
            write_slot(t1_[left_idx], fp, val, ST_OFF_NEG);
            ++count_;
            return true;
        }

        // 4. Greedy rightward ripple: open a hole at h+1 by shifting the run
        //    right to the first empty slot. Before anything moves, every entry
        //    in the run is checked against its own home (not its tag) -- it must
        //    be able to move right and still land in {home-1, home, home+1}.
        //    One entry failing the test abandons the whole ripple.
        //
        for (uint64_t k = h + 1; k < h + 5; ++k) {
            uint64_t idx = k & mask1_;
            uint8_t k_state = get_state(t1_[idx]);
            if (k_state == ST_EMPTY) {
                for (uint64_t j = idx; j != ((h + 1) & mask1_); j = (j - 1) & mask1_) {
                    uint64_t prev = (j - 1) & mask1_;
                    shift_entry(t1_[prev], t1_[j],
                                shifted_right_tag(get_state(t1_[prev])));
                }
                write_slot(t1_[(h + 1) & mask1_], fp, val, ST_OFF_POS);
                ++count_;
                return true;
            }
            if (!can_shift_right_into(t1_[idx], idx + 1)) {
                break; // blocked: spill to T2, leaving the run in place
            }
        }

        // 5. Fallback: spill into T2.
        return insert_into_T2(hh, h, key, key_len, val);
    }

    // Returns the stored bytes for the key, or nullptr. Three key forms, as in
    // insert().
    void* find(uint64_t key) const { return find_key(&key, sizeof key); }
    void* find(std::string_view key) const { return find_key(key.data(), key.size()); }
    void* find(const void* key, std::size_t key_len) const { return find_key(key, key_len); }

    void* find_key(const void* key, std::size_t key_len) const {
        const uint64_t hh = hash_bytes(key, key_len);
        const uint64_t h = fast_map(hh, static_cast<uint64_t>(n1_));
        const uint8_t fp = fingerprint_of_hash(hh);
        const Entry& home = t1_[h];

        // --- PROBE 1: home slot ---
        if (get_state(home) == ST_OFF_ZERO && slot_holds_fp(home, key, key_len, fp)) {
            return decode_pointer(home.word);
        }

        // --- PROBE 2: the key's own neighbour ---
        //
        // A key's offset is confined to {-1, 0, +1} by construction, so when it
        // is not in the home slot it is in exactly one neighbour -- never both,
        // never a third.
        //
        // Probe order: +1 first -- the ripple shifts runs rightward, so a
        // displaced key sits at h+1 more often than at h-1.
        const int first = +1;
        const Entry& n1 = t1_[(h + (uint64_t)first) & mask1_];
        if (slot_holds_fp(n1, key, key_len, fp)) {
            return decode_pointer(n1.word);
        }
        const Entry& n2 = t1_[(h + (uint64_t)-first) & mask1_];
        if (slot_holds_fp(n2, key, key_len, fp)) {
            return decode_pointer(n2.word);
        }

        // --- spillover check: T2 ---
        uint8_t max_offset = get_t2_max_offset(home);
        if (max_offset == 0) return nullptr;
        return find_in_T2(h, key, key_len, fp, max_offset);
    }

    void* find_in_T2(uint64_t h, const void* key, std::size_t key_len, uint8_t fp,
                     uint8_t max_offset) const {
        uint64_t base_t2 = (h >> SHIFT_RATIO) & mask2_;

        if (max_offset < DMAX_SENTINEL) {
            // The stored bound is (furthest offset + 1); scan down to 0 and
            // stop there. Empty and tombstoned slots are expected below the
            // bound.
            for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
                const Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
                if (t2_holds_fp(e, key, key_len, fp)) {
                    return decode_pointer(e.word);
                }
            }
        } else {
            // D_max saturated: sweep forward from the base, stopping only at an
            // ST_EMPTY slot.
            for (uint64_t off = 0; off < n2_; ++off) {
                const Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == ST_EMPTY) break;
                if (t2_holds_fp(e, key, key_len, fp)) return decode_pointer(e.word);
            }
        }
        return nullptr;
    }

    // Remove a key. A T1 slot clears to EMPTY but keeps its D_max (the home
    // bucket may still own T2 spills); a T2 slot becomes a tombstone, so a
    // forward T2 scan does not stop short of the live entries past it. Three
    // key forms, as in insert().
    bool erase(uint64_t key) { return erase_key(&key, sizeof key); }
    bool erase(std::string_view key) { return erase_key(key.data(), key.size()); }
    bool erase(const void* key, std::size_t key_len) { return erase_key(key, key_len); }

    bool erase_key(const void* key, std::size_t key_len) {
        const uint64_t hh = hash_bytes(key, key_len);
        const uint64_t h = fast_map(hh, static_cast<uint64_t>(n1_));
        const uint8_t fp = fingerprint_of_hash(hh);

        bool in_t1 = false;
        Entry* e = slot_for_key(key, key_len, h, fp, in_t1);
        if (!e) return false;

        if (in_t1) {
            e->word &= ~(STATE_MASK << STATE_SHIFT); // state -> EMPTY, D_max kept
            pull_neighbors_home(static_cast<uint64_t>(e - t1_));
        } else {
            e->word &= ~(STATE_MASK << STATE_SHIFT);
            e->word |= static_cast<uint64_t>(ST_TOMBSTONE) << STATE_SHIFT;
            --t2_count_;
        }
        --count_;
        return true;
    }

private:
    // Smallest power of two >= n, so slot indexing is `h & mask`. n <= 1
    // returns 1; callers clamp to a floor first.
    static std::size_t ceil_pow2(std::size_t n) {
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    // Can the entry in `src` move one slot right, into `dest`, and still lie
    // within its own ternary neighbourhood? Taken from the key's real home, not
    // the stored tag.
    bool can_shift_right_into(const Entry& src, uint64_t dest_slot) const {
        uint8_t st = get_state(src);
        if (st == ST_EMPTY) return true;
        if (st == ST_OFF_POS) return false; // already at home+1
        uint64_t home = home_slot_of(decode_pointer(src.word));
        uint64_t delta = (dest_slot - home) & mask1_;
        return delta == 0 || delta == 1 || delta == mask1_;
    }

    // The tag an entry carries after moving one slot right, closing one slot of
    // the gap to its home: NEG (home is one to the right) becomes ZERO, ZERO
    // becomes POS. POS never reaches here -- can_shift_right_into rejects it.
    static uint8_t shifted_right_tag(uint8_t old_state) {
        return (old_state == ST_OFF_NEG) ? ST_OFF_ZERO : ST_OFF_POS;
    }

    // After `slot` frees, pull a key at `slot ± 1` whose own home is `slot`
    // back into it. The freed gap can then travel one more step, bounded at
    // two. The <= 2-probe bound in find() comes from insert confining offsets
    // to {-1, 0, +1} -- this only keeps the displaced case rare under churn.
    void pull_neighbors_home(uint64_t slot) {
        uint64_t gap = slot;
        for (int step = 0; step < 2; ++step) {
            bool moved = false;
            for (int dir = -1; dir <= 1; dir += 2) {
                if (dir == 0) continue;
                uint64_t from = (gap + (uint64_t)dir) & mask1_;
                const Entry& src = t1_[from];
                if (get_state(src) == ST_EMPTY) continue;
                if (home_slot_of(decode_pointer(src.word)) != gap) continue;
                shift_entry(t1_[from], t1_[gap], ST_OFF_ZERO);
                gap = from;
                moved = true;
                break;
            }
            if (!moved) break;
        }
    }

public:
    // The home bucket of a key in T1: the only slot where it may live without
    // displacement. Exposed so tests can verify the ternary-neighborhood
    // invariant bit for bit.
    uint64_t home_slot(uint64_t key) const { return home_slot_of(&key, sizeof key); }
    uint64_t home_slot(std::string_view key) const { return home_slot_of(key.data(), key.size()); }
    uint64_t home_slot(const void* key, std::size_t key_len) const {
        return home_slot_of(key, key_len);
    }
    static uint8_t state_of(const Entry& e) { return get_state(e); }
    static uint8_t dmax_of(const Entry& e) { return get_t2_max_offset(e); }

private:
    // Home bucket of the key in the record at `val` -- for callers that hold
    // only the record pointer.
    uint64_t home_slot_of(const void* val) const {
        return home_slot_of(key_ptr_at(val), key_len_at(val));
    }
    uint64_t home_slot_of(const void* key, std::size_t key_len) const {
        return fast_map(hash_bytes(key, key_len), static_cast<uint64_t>(n1_));
    }

    bool insert_into_T2(uint64_t hh, uint64_t h, const void* key, std::size_t key_len,
                        void* val) {
        // Double the table when T2 reaches three quarters full, so the spill
        // path below always has room.
        if (t2_count_ >= (n2_ >> 2) * 3) {
            if (!grow_to(n1_ * 2)) {
                return false;
            }
            h = fast_map(hh, static_cast<uint64_t>(n1_)); // n1_ changed with the growth
        }

        uint64_t base_t2 = (h >> SHIFT_RATIO) & mask2_;

        // First slot that is not a live entry -- ST_EMPTY or a tombstone --
        // takes the spill.
        for (uint64_t off = 0; off < n2_; ++off) {
            uint64_t idx = (base_t2 + off) & mask2_;
            uint8_t st = get_state(t2_[idx]);
            if (st == ST_EMPTY || st == ST_TOMBSTONE) {
                write_slot(t2_[idx], fingerprint_of_hash(hh), val, ST_OFF_ZERO);
                ++count_;
                ++t2_count_;

                // The bound is stored as (offset + 1), so offset 0 is
                // distinguishable from "nothing spilled here".
                uint8_t cur_dmax = get_t2_max_offset(t1_[h]);
                uint8_t new_off = (off >= DMAX_SENTINEL - 1)
                                      ? DMAX_SENTINEL
                                      : static_cast<uint8_t>(off + 1);
                if (new_off > cur_dmax) set_t2_max_offset(t1_[h], new_off);
                return true;
            }
        }
        return false; // n2_ slots all live
    }

    // Replace both slot arrays with larger ones and re-place every live entry
    // through place_into. T2-grown state does not need to carry over: the
    // entries are re-placed from scratch against the new arrays.
    bool grow_to(std::size_t new_n1) {
        if (new_n1 <= n1_) return false;
        new_n1 = ceil_pow2(new_n1); // mask = n1 - 1 requires a power of two
        const std::size_t old_n1 = n1_, old_n2 = n2_;
        Entry* old_t1 = t1_;
        Entry* old_t2 = t2_;

        Entry* new_t1 = static_cast<Entry*>(std::calloc(new_n1, sizeof(Entry)));
        Entry* new_t2 =
            static_cast<Entry*>(std::calloc(new_n1 >> SHIFT_RATIO, sizeof(Entry)));
        if (!new_t1 || !new_t2) {
            std::free(new_t1);
            std::free(new_t2);
            return false; // table untouched
        }

        t1_ = new_t1;
        t2_ = new_t2;
        n1_ = new_n1;
        n2_ = new_n1 >> SHIFT_RATIO;
        mask1_ = n1_ - 1;
        mask2_ = n2_ - 1;
        count_ = t2_count_ = 0;

        // T1 entries first, then T2: re-placing T1 while the table is still
        // empty keeps its neighbourhoods sparse, so fewer entries spill.
        for (std::size_t i = 0; i < old_n1; ++i) {
            const Entry& e = old_t1[i];
            if (get_state(e) == ST_EMPTY) continue;
            void* val = decode_pointer(e.word);
            std::size_t klen = key_len_at(val);
            const void* k = key_ptr_at(val);
            const uint64_t hh = hash_bytes(k, klen);
            place_into(k, klen, hh, fast_map(hh, static_cast<uint64_t>(n1_)), val);
        }
        for (std::size_t i = 0; i < old_n2; ++i) {
            const Entry& e = old_t2[i];
            uint8_t st = get_state(e);
            if (st == ST_EMPTY || st == ST_TOMBSTONE) continue;
            void* val = decode_pointer(e.word);
            std::size_t klen = key_len_at(val);
            const void* k = key_ptr_at(val);
            const uint64_t hh = hash_bytes(k, klen);
            place_into(k, klen, hh, fast_map(hh, static_cast<uint64_t>(n1_)), val);
        }

        std::free(old_t1);
        std::free(old_t2);
        return true;
    }

    Entry* t1_ = nullptr;
    Entry* t2_ = nullptr;
    std::size_t n1_ = 0;
    std::size_t n2_ = 0;
    std::size_t mask1_ = 0;
    std::size_t mask2_ = 0;
    std::size_t count_ = 0;
    std::size_t t2_count_ = 0;
    Hash hash_{};
    KeyStore key_store_;
};

using Table = BasicTable<mtht::Hash>;

} // namespace mtht

#endif // MTHT_HPP
