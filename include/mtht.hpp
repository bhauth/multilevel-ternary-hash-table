// Multilevel Ternary Hash Table (MTHT)
#ifndef MTHT_HPP
#define MTHT_HPP

#include <cstdint>
#include <cstddef>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <string>
#include <functional>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace mtht {

struct alignas(8) Entry {
    uint64_t word;
};

static_assert(sizeof(Entry) == 8, "Entry must be 8 bytes");

// A slot's word packs two kinds of thing. Bits [56..0] describe the item in
// the slot -- where its record is, its fingerprint, and whether the slot is
// occupied -- so they travel with the item when it moves. Bits [63..57] are
// owned by the slot's own home and stay put: the 3-bit state says which of the
// three slots at [-1, 0, +1] hold an item homed here, and D_max is the reach of
// this home's spill into T2.
//
//     [63..61] state   [60..57] D_max   [56..49] fingerprint
//     [48]     occupied                [47..0]  pointer
static constexpr uint64_t PTR_MASK = 0x0000FFFFFFFFFFFFULL;
static constexpr int OCC_SHIFT = 48;
static constexpr uint64_t OCC_BIT = 1ULL << OCC_SHIFT;
static constexpr uint64_t FP_SHIFT = 49;
static constexpr uint64_t FP_MASK = 0xFFULL;
static constexpr uint64_t DMAX_SHIFT = 57;
static constexpr uint64_t DMAX_MASK = 0x0FULL;
static constexpr uint64_t STATE_SHIFT = 61;
static constexpr uint64_t STATE_MASK = 0x07ULL;

// Everything an item carries. Clearing the home's own fields leaves this.
static constexpr uint64_t ITEM_MASK = OCC_BIT | (FP_MASK << FP_SHIFT) | PTR_MASK;

// 3-bit ownership mask in a slot's state field. Bit i is set when slot
// (slot - 1 + i) holds an item whose home is this slot.
enum : uint8_t {
    MASK_MINUS = 1, // 001: slot - 1 holds an item homed here
    MASK_SELF = 2,  // 010: this slot holds an item homed here
    MASK_PLUS = 4,  // 100: slot + 1 holds an item homed here
};

// T2 slots reuse the state field for a marker on the item in that slot.
static constexpr uint8_t T2_EMPTY = 0;
static constexpr uint8_t T2_LIVE = 1;
static constexpr uint8_t T2_TOMBSTONE = 4;

static constexpr uint8_t DMAX_SENTINEL = 15;

static constexpr int SHIFT_RATIO = 3;

inline uint8_t get_state(const Entry& e) {
    return static_cast<uint8_t>(e.word >> STATE_SHIFT);
}

// Replace the state field. Every caller sets it to a value it computed, never
// accumulates into it, so a tombstone (100) cannot survive into a live (001).
inline void set_state(Entry& e, uint8_t s) {
    e.word = (e.word & ~(static_cast<uint64_t>(7) << STATE_SHIFT)) |
             (static_cast<uint64_t>(s) << STATE_SHIFT);
}

inline bool is_occupied(const Entry& e) { return (e.word & OCC_BIT) != 0; }

// The mask bit that names `slot_idx` from the home at `home_idx`. Compared
// modulo the table size, so home + 1 wraps to slot 0 at the last slot.
inline uint8_t mask_bit_for(uint64_t slot_idx, uint64_t home_idx, uint64_t mask1) {
    if (slot_idx == home_idx) return MASK_SELF;
    if (slot_idx == ((home_idx + 1) & mask1)) return MASK_PLUS;
    return MASK_MINUS;
}

inline uint8_t get_t2_max_offset(const Entry& e) {
    return static_cast<uint8_t>((e.word >> DMAX_SHIFT) & DMAX_MASK);
}

inline void set_t2_max_offset(Entry& e, uint8_t dmax) {
    e.word = (e.word & ~(DMAX_MASK << DMAX_SHIFT)) |
             (static_cast<uint64_t>(dmax & DMAX_MASK) << DMAX_SHIFT);
}

inline void* decode_pointer(uint64_t word) {
    return reinterpret_cast<void*>(static_cast<uintptr_t>(word & PTR_MASK));
}

inline uint64_t encode_pointer(const void* p) {
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p));
}

inline void check_pointer_range(const void* p) {
    if (reinterpret_cast<uintptr_t>(p) > PTR_MASK) {
        std::fprintf(stderr, "mtht: pointer above 48 bits: %p\n", p);
        std::abort();
    }
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

inline uint64_t read_word(const void* p) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    return w;
}

// The value record is
//
//     [ data_len : 32 | key_len : 31 + live : 1 ][ key... ][ data... ]
//
// with the key first, so a find that has matched a fingerprint reads the key
// bytes at the pointer it returns, and the record header sits behind it.
//
//     bits 63..32 data size, in bytes
//     bits 31..1  key size, in bytes
//     bit 0       0 while the key is live, 1 once it is erased
//
// Erasing flips bit 0 only: it is in the word that holds both lengths, so the
// header stays exactly one load behind the key.
//
// A key length needs 32 bits to hold 2^31 itself, so the header spends its top
// bit on liveness instead: key lengths are stored doubled in bits 31..1, which
// holds 0 .. 2^30-1.
// static constexpr std::size_t MAX_KEY_BYTES = (std::size_t(1) << 30) - 1;

static constexpr std::size_t PL_HEAD_BYTES = 8;

static constexpr uint64_t PL_DEAD = 1ULL;
static constexpr int PL_DATA_SHIFT = 32;
static constexpr int PL_KEY_SHIFT = 1;
static constexpr uint64_t PL_KEY_MASK = 0x7FFFFFFFULL; // 31 bits

inline constexpr uint64_t pack_len(std::size_t key_len, std::size_t data_len) {
    return (static_cast<uint64_t>(data_len) << PL_DATA_SHIFT) |
           (static_cast<uint64_t>(key_len) << PL_KEY_SHIFT);
}

inline std::size_t packed_key_len(uint64_t packed) {
    return static_cast<std::size_t>((packed & PL_KEY_MASK) >> PL_KEY_SHIFT);
}

inline std::size_t packed_data_len(uint64_t packed) {
    return static_cast<std::size_t>(packed >> PL_DATA_SHIFT);
}

inline bool packed_is_live(uint64_t packed) { return (packed & PL_DEAD) == 0; }

inline uint64_t read_packed_len(const void* rec) {
    uint64_t p;
    std::memcpy(&p, rec, sizeof p);
    return p;
}

// ptr is what find and iter hand out: the key, with the record header 8 bytes
// before it and the value just past the key.
inline std::size_t key_len_at(const void* ptr) {
    return packed_key_len(read_packed_len(static_cast<const unsigned char*>(ptr) - PL_HEAD_BYTES));
}

inline const unsigned char* data_ptr_at(const void* ptr) {
    return static_cast<const unsigned char*>(ptr) + key_len_at(ptr);
}

// Is the item a find or iter handed out still live? False once it is erased,
// false for nullptr.
inline bool ptr_is_live(const void* ptr) {
    if (!ptr) return false;
    uint64_t packed = 0;
    std::memcpy(&packed, static_cast<const unsigned char*>(ptr) - PL_HEAD_BYTES,
                sizeof packed);
    return packed_is_live(packed);
}

// The key a find or iter pointer refers to.
inline std::string_view read_key(const void* ptr) {
    if (!ptr) return {};
    return std::string_view(static_cast<const char*>(ptr), key_len_at(ptr));
}

// The value a find or iter pointer refers to.
inline std::string_view read_str(const void* ptr) {
    if (!ptr) return {};
    uint64_t packed = 0;
    std::memcpy(&packed, static_cast<const unsigned char*>(ptr) - PL_HEAD_BYTES,
                sizeof packed);
    return std::string_view(reinterpret_cast<const char*>(data_ptr_at(ptr)),
                            packed_data_len(packed));
}

// The value a find or iter pointer refers to, copied out as a fixed-size type.
// T{} if ptr is null. Same caveat as find_as: T has to be a type whose byte
// image is the whole value.
template <class T>
inline T read_as(const void* ptr) {
    if (!ptr) return T{};
    T v{};
    std::memcpy(&v, data_ptr_at(ptr), sizeof v);
    return v;
}

// 8-bit fingerprint of a key: the filter that stands in for the key in a slot.
//
// The bits of h are spoken for elsewhere, so the fingerprint takes a byte clear
// of all of them rather than mixing a shared one:
//   [15..0]  the low half, so a fingerprint there would agree exactly for keys
//            that already share a home slot -- the one case the filter exists
//            to reject.
//   [47..16] the T2 band index
//   [63..56] the top byte, clear of the pointer, the band index and the low
//            half. The header of an entry is [63..62] state, [61..54]
//            fingerprint, [53..48] D_max; the fingerprint rides in [61..54], so
//            its tag has to be the byte at that position.
// hash_key avalanches all 64 bits before this point.
inline uint8_t fingerprint_of_hash(uint64_t h) {
    return static_cast<uint8_t>(h >> 54);
}

inline uint64_t hash_bytes(const void* p, std::size_t len) {
    if (len == 8) {
        uint64_t w;
        std::memcpy(&w, p, 8);
        return hash_key(w);
    }
    if (len < 8) {
        uint64_t w = 0;
        std::memcpy(&w, p, len);
        return hash_key(w + (static_cast<uint64_t>(len) << 56));
    }
    std::string_view sv(static_cast<const char*>(p), len);
    return hash_key(static_cast<uint64_t>(std::hash<std::string_view>{}(sv)));
}

inline uint8_t get_fingerprint(const Entry& e) {
    return static_cast<uint8_t>((e.word >> FP_SHIFT) & FP_MASK);
}

// The item in a slot: its record pointer, whether the slot is occupied, and the
// fingerprint. This is the whole of ITEM_MASK, so setting it leaves the slot's
// own state and D_max alone.
inline void set_item(Entry& e, uint8_t fp, void* val) {
    e.word = (e.word & ~ITEM_MASK) | OCC_BIT | encode_pointer(val) |
             (static_cast<uint64_t>(fp) << FP_SHIFT);
}

// Place an item in slot `dst`, in the slot `bit` names in its home's mask. The
// destination's state belongs to whatever already owns items here and is left
// alone; the slot is marked occupied. The caller knows the bit: it is placing
// at the home, the +1 neighbour, or the -1 neighbour.
inline void* write_slot(Entry& dst, Entry& home_slot, uint8_t bit, uint8_t fp, void* val) {
    home_slot.word |= static_cast<uint64_t>(bit) << STATE_SHIFT;

    set_item(dst, fp, val);
    return val;
}

// Move the item in src to dst. Bits [56..0] -- pointer, occupied, fingerprint --
// travel whole; dst keeps its own state and D_max; src is left free.
inline void shift_entry(Entry& src, Entry& dst) {
    dst.word = (dst.word & ~ITEM_MASK) | (src.word & ITEM_MASK);
    src.word &= ~ITEM_MASK;
}

// The pointer in a slot is the record's key, so the key bytes sit right at it
// and the header one load behind. Callers gate on the home's mask, so the slot
// is known occupied; a mask bit exists iff the named slot holds an item.
inline bool slot_holds_fp_8(const Entry& e, uint64_t key_val, uint8_t fp) {
    if (get_fingerprint(e) != fp) return false;
    const auto* k = static_cast<const unsigned char*>(decode_pointer(e.word));
    uint64_t packed;
    std::memcpy(&packed, k - PL_HEAD_BYTES, sizeof packed);
    if (packed_key_len(packed) != 8) return false;
    return read_word(k) == key_val;
}

inline bool slot_holds_fp(const Entry& e, const void* key, std::size_t len, uint8_t fp) {
    if (get_fingerprint(e) != fp) return false;
    const auto* k = static_cast<const unsigned char*>(decode_pointer(e.word));
    // A stored key of another length can never equal the probe, so compare the
    // stored length before touching the key bytes.
    uint64_t packed;
    std::memcpy(&packed, k - PL_HEAD_BYTES, sizeof packed);
    if (packed_key_len(packed) != len) return false;
    return std::memcmp(k, key, len) == 0;
}

// A T2 tombstone keeps the erased key's fingerprint and pointer, so it would
// match the shared predicate. T2 slots hold only EMPTY (0), LIVE (1) or
// TOMBSTONE (4), so one compare against live rejects both dead cases.
inline bool t2_holds_fp_8(const Entry& e, uint64_t key_val, uint8_t fp) {
    if (get_state(e) != T2_LIVE) return false;
    if (get_fingerprint(e) != fp) return false;
    const auto* k = static_cast<const unsigned char*>(decode_pointer(e.word));
    uint64_t packed;
    std::memcpy(&packed, k - PL_HEAD_BYTES, sizeof packed);
    if (packed_key_len(packed) != 8) return false;
    return read_word(k) == key_val;
}

inline bool t2_holds_fp(const Entry& e, const void* key, std::size_t len, uint8_t fp) {
    if (get_state(e) != T2_LIVE) return false;
    if (get_fingerprint(e) != fp) return false;
    const auto* k = static_cast<const unsigned char*>(decode_pointer(e.word));
    uint64_t packed;
    std::memcpy(&packed, k - PL_HEAD_BYTES, sizeof packed);
    if (packed_key_len(packed) != len) return false;
    return std::memcmp(k, key, len) == 0;
}

// Default hash policy, and the shape a custom one takes:
//
//     struct MyHash {
//         uint64_t operator()(const void* p, std::size_t len) const; // key span of len bytes
//         uint64_t operator()(const void* p) const;                  // 8-byte key
//     };
//
// The span form is only called with len != 8 -- an 8-byte key is routed to the
// 8-byte form first -- so its length-8 case is free to differ.
struct Hash {
    uint64_t operator()(const void* p, std::size_t len) const {
        return hash_bytes(p, len);
    }
    uint64_t operator()(const void* p) const {
        return hash_key(read_word(p));
    }
};

class KeyStore {
public:
    static std::size_t round_up_8(std::size_t n) {
        return (n + 7) & ~static_cast<std::size_t>(7);
    }

    explicit KeyStore(std::size_t initial_bytes) {
        block_bytes_ = round_up_8(std::max(initial_bytes, kMinBlock));
        open_block(block_bytes_);
    }

    ~KeyStore() { clear(); }

    KeyStore(const KeyStore&) = delete;
    KeyStore& operator=(const KeyStore&) = delete;

    // Stores [header][key][data] and returns the address of the key -- the
    // pointer the table keeps and find hands out.
    void* store(const void* key, std::size_t len, const void* bytes,
                std::size_t bytes_len) {
        //  if (len == 0 || len > MAX_KEY_BYTES) return nullptr;   ??

        unsigned char* rec = reserve(round_up_8(PL_HEAD_BYTES + len + bytes_len));
        const uint64_t packed = pack_len(len, bytes_len);
        std::memcpy(rec, &packed, sizeof packed);
        std::memcpy(rec + PL_HEAD_BYTES, key, len);
        if (bytes_len) std::memcpy(rec + PL_HEAD_BYTES + len, bytes, bytes_len);
        return rec + PL_HEAD_BYTES;
    }

    // ptr is the key address store returned.
    void mark_dead(void* ptr) {
        unsigned char* rec = static_cast<unsigned char*>(ptr) - PL_HEAD_BYTES;
        uint64_t packed = 0;
        std::memcpy(&packed, rec, sizeof packed);
        packed |= PL_DEAD;
        std::memcpy(rec, &packed, sizeof packed);
    }

    // Undoes mark_dead. ptr is the key address store returned.
    void mark_live(void* ptr) {
        unsigned char* rec = static_cast<unsigned char*>(ptr) - PL_HEAD_BYTES;
        uint64_t packed = 0;
        std::memcpy(&packed, rec, sizeof packed);
        packed &= ~PL_DEAD;
        std::memcpy(rec, &packed, sizeof packed);
    }

    void clear() {
        for (void* p : blocks_) std::free(p);
        blocks_.clear();
        block_items_.clear();
        bump_ = nullptr;
        bump_left_ = 0;
        bytes_used_ = 0;
        items_ = 0;
        block_bytes_ = kMinBlock;
    }

    std::size_t bytes_used() const { return bytes_used_; }

private:
    void open_block(std::size_t bytes) {
        void* raw = std::malloc(bytes);
        if (!raw) {
            std::fprintf(stderr, "mtht: out of memory in KeyStore\n");
            std::abort();
        }
        if (!blocks_.empty()) block_items_.push_back(items_);
        blocks_.push_back(raw);
        check_pointer_range(raw);
        bytes_used_ += bytes;
        bump_ = static_cast<unsigned char*>(raw);
        bump_left_ = bytes;
        block_bytes_ = bytes;
    }

    unsigned char* reserve(std::size_t bytes) {
        if (bytes > bump_left_) {
            open_block(std::max(bytes, block_bytes_));
        }
        unsigned char* p = bump_;
        bump_ += bytes;
        bump_left_ -= bytes;
        ++items_;
        return p;
    }

public:
    std::size_t block_end(std::size_t block) const {
        return block < block_items_.size() ? block_items_[block] : items_;
    }

    // Records in a block. block_items_ holds each closed block's cumulative
    // item count, so a block's own size is the step from the previous one.
    std::size_t block_size(std::size_t block) const {
        const std::size_t end = block < block_items_.size() ? block_items_[block] : items_;
        const std::size_t begin = block == 0 ? 0 : block_items_[block - 1];
        return end - begin;
    }

    std::size_t items() const { return items_; }
    std::size_t block_count() const { return blocks_.size(); }

    const unsigned char* block_start(std::size_t block) const {
        return static_cast<const unsigned char*>(blocks_[block]);
    }

    const unsigned char* next_record(const unsigned char* rec) const {
        const uint64_t packed = read_packed_len(rec);
        return rec + round_up_8(PL_HEAD_BYTES + packed_key_len(packed) + packed_data_len(packed));
    }

    const unsigned char* first_record() const {
        return items_ == 0 ? nullptr : static_cast<const unsigned char*>(blocks_[0]);
    }

private:
    static constexpr std::size_t kMinBlock = 4096; // 256 slots
    unsigned char* bump_ = nullptr;
    std::size_t bump_left_ = 0;
    std::size_t bytes_used_ = 0;
    std::size_t block_bytes_ = kMinBlock; // doubles on each later block
    std::size_t items_ = 0;
    std::vector<void*> blocks_;
    std::vector<std::size_t> block_items_;
};

// A walk position, filled in by iter_start. The caller owns it: the table
// keeps no reference, so copies of one are independent walks.
struct IterState {
    const unsigned char* rec = nullptr;
    std::size_t block = 0;
    std::size_t items = 0;
};

// The table. n1 (T1 slot count) and n2 (T2 slot count) are powers of two.
template <class Hash = mtht::Hash>
class BasicTable {
public:
    static constexpr std::size_t DEFAULT_N1 = 1u << 20; // 1,048,576 slots, 16 MB

    explicit BasicTable(std::size_t n1 = DEFAULT_N1, std::size_t store_bytes = 0)
        : key_store_(store_bytes) {
        if (n1 < 8) n1 = 8;
        n1 = ceil_pow2(n1);
        n2_ = n1 >> SHIFT_RATIO;
        n1_ = n1;
        mask1_ = n1 - 1;
        mask2_ = n2_ - 1;
        t1_ = static_cast<Entry*>(std::calloc(n1 + n2_, sizeof(Entry)));
        if (!t1_) {
            std::fprintf(stderr, "mtht: out of memory for tables\n");
            std::abort();
        }
        t2_ = t1_ + n1_;
    }

    ~BasicTable() { std::free(t1_); }

    BasicTable(const BasicTable&) = delete;
    BasicTable& operator=(const BasicTable&) = delete;

    std::size_t size() const { return key_store_.items() - deleted_; }
    std::size_t n1() const { return n1_; }
    std::size_t n2() const { return n2_; }
    std::size_t t1_count() const { return size() - t2_count_; }
    std::size_t t2_count() const { return t2_count_; }
    double load1() const { return static_cast<double>(t1_count()) / static_cast<double>(n1_); }
    double load2() const { return static_cast<double>(t2_count_) / static_cast<double>(n2_); }

    std::size_t key_store_bytes() const { return key_store_.bytes_used(); }

    void clear() {
        std::memset(t1_, 0, (n1_ + n2_) * sizeof(Entry));
        t2_count_ = 0;
        deleted_ = 0;
        key_store_.clear();
    }

    void* insert(uint64_t key, const void* bytes, std::size_t bytes_len) {
        return insert_key_8(&key, bytes, bytes_len);
    }
    // An int-int record: 8-byte key, 8-byte value.
    void* insert(uint64_t key, uint64_t value) {
        return insert_key_8(&key, &value, sizeof value);
    }
    void* insert(std::string_view key, const void* bytes, std::size_t bytes_len) {
        if (key.size() == 8) return insert_key_8(key.data(), bytes, bytes_len);
        return insert_key(key.data(), key.size(), bytes, bytes_len);
    }
    void* insert(const void* key, std::size_t key_len, const void* bytes,
                 std::size_t bytes_len) {
        if (key_len == 8) return insert_key_8(key, bytes, bytes_len);
        return insert_key(key, key_len, bytes, bytes_len);
    }
    void* insert(std::string_view key, const std::string& value) {
        if (key.size() == 8) return insert_key_8_str(key.data(), value);
        return insert_key_str(key.data(), key.size(), value);
    }

    void* insert_key(const void* key, std::size_t key_len, const void* bytes,
                     std::size_t bytes_len) {
        void* rec = key_store_.store(key, key_len, bytes, bytes_len);
        if (!rec) return nullptr;

        const uint64_t hh = hash_(key, key_len);
        return insert_place(key, key_len, hh, rec, true);
    }
    void* insert_key_8(const void* key, const void* bytes, std::size_t bytes_len) {
        const uint64_t hh = hash_(key);
        const uint64_t key_val = read_word(key);

        void* rec = key_store_.store(key, 8, bytes, bytes_len);
        if (!rec) return nullptr;

        return insert_place_8(key_val, hh, rec, true);
    }

    void* insert_key_str(const void* key, std::size_t key_len, const std::string& value) {
        return insert_key(key, key_len, value.data(), value.size());
    }

    void* insert_key_8_str(const void* key, const std::string& value) {
        return insert_key_8(key, value.data(), value.size());
    }

    Entry* slot_for_key(const void* key, std::size_t key_len, uint64_t hh, uint8_t fp,
                        bool& in_t1) {
        Entry& home = t1_[hh & mask1_];
        const uint8_t s_home = get_state(home);
        if ((s_home & MASK_SELF) && slot_holds_fp(home, key, key_len, fp)) {
            in_t1 = true;
            return &home;
        }

        // A neighbour can only hold this key if home's mask says so.
        if (s_home & MASK_PLUS) {
            Entry& n1 = t1_[(hh + 1) & mask1_];
            if (slot_holds_fp(n1, key, key_len, fp)) { in_t1 = true; return &n1; }
        }
        if (s_home & MASK_MINUS) {
            Entry& n2 = t1_[(hh - 1) & mask1_];
            if (slot_holds_fp(n2, key, key_len, fp)) { in_t1 = true; return &n2; }
        }

        uint8_t max_offset = get_t2_max_offset(home);
        if (max_offset == 0) return nullptr;
        Entry* e = slot_for_key_T2(hh, key, key_len, fp, max_offset);
        if (e) in_t1 = false;
        return e;
    }

    Entry* slot_for_key_T2(uint64_t hh, const void* key, std::size_t key_len, uint8_t fp,
                           uint8_t max_offset) {
        uint64_t base_t2 = (hh >> 32) & mask2_;

        for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
            Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
            if (t2_holds_fp(e, key, key_len, fp)) return &e;
        }
        if (max_offset == DMAX_SENTINEL) {
            for (uint64_t off = max_offset; off < n2_; ++off) {
                Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == T2_EMPTY) break;
                if (t2_holds_fp(e, key, key_len, fp)) return &e;
            }
        }
        return nullptr;
    }

    __attribute__((noinline))
    Entry* slot_for_key_T2_8(uint64_t hh, uint64_t key_val, uint8_t fp, uint8_t max_offset) {
        uint64_t base_t2 = (hh >> 32) & mask2_;

        for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
            Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
            if (t2_holds_fp_8(e, key_val, fp)) return &e;
        }
        if (max_offset == DMAX_SENTINEL) {
            for (uint64_t off = max_offset; off < n2_; ++off) {
                Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == T2_EMPTY) break;
                if (t2_holds_fp_8(e, key_val, fp)) return &e;
            }
        }
        return nullptr;
    }

    void* insert_place(const void* key, std::size_t key_len, uint64_t hh, void* val,
                       bool check_existing) {
        const uint8_t fp = fingerprint_of_hash(hh);
        Entry& home = t1_[hh & mask1_];
        Entry& right = t1_[(hh + 1) & mask1_];
        Entry& left = t1_[(hh - 1) & mask1_];
        
        {
        const uint8_t s_home = get_state(home);
        Entry* hit = nullptr;

        if (!is_occupied(home) && !get_t2_max_offset(home)) {
            return write_slot(home, home, MASK_SELF, fp, val);
        }

        if (check_existing) {
            // pull_neighbors_home never leaves an offset entry when home is empty.
            if ((s_home & MASK_SELF) && slot_holds_fp(home, key, key_len, fp)) {
                hit = &home;
            } else if ((s_home & MASK_PLUS) && slot_holds_fp(right, key, key_len, fp)) {
                hit = &right;
            } else if ((s_home & MASK_MINUS) && slot_holds_fp(left, key, key_len, fp)) {
                hit = &left;
            } else {
                const uint8_t max_offset = get_t2_max_offset(home);
                if (max_offset) {
                    hit = slot_for_key_T2(hh, key, key_len, fp, max_offset);
                } else { goto check_sides; }
            }
            if (hit) {
                key_store_.mark_dead(decode_pointer(hit->word));
                ++deleted_;
                hit->word = (hit->word & ~PTR_MASK) | encode_pointer(val);
                return val;
            }
            
            if (!is_occupied(home)) {
                return write_slot(home, home, MASK_SELF, fp, val);
            }
        }
        }

    check_sides:
        if (!is_occupied(right)) {
            return write_slot(right, home, MASK_PLUS, fp, val);
        }
        if (!is_occupied(left)) {
            return write_slot(left, home, MASK_MINUS, fp, val);
        }

        // 3. Both neighbors taken: open h+1 by shifting the entry there into
        //    h+2, if that entry sits at its own home. An entry homed at h-1
        //    (named by home's MASK_PLUS) cannot go further out, and an entry
        //    already pushed cannot move again. Only an entry at offset 0 can
        //    be pushed, and it lands at offset +1 of its own home -- still
        //    inside its window. The new key then takes h+1 at offset +1, the
        //    slot the probe checks first.
        if (get_state(right) & MASK_SELF) {
            const uint64_t far_right = (hh + 2) & mask1_;
            Entry& next_right = t1_[far_right];
            if (!is_occupied(next_right)) {
                // right is homed at right_idx: its mask bit moves self -> plus.
                const uint8_t rs = get_state(right);
                set_state(right, (rs & ~MASK_SELF) | MASK_PLUS);
                shift_entry(right, next_right);
                return write_slot(right, home, MASK_PLUS, fp, val);
            }
        }

        // 4. Mirror on the left: same reasoning.
        const uint8_t s_left = get_state(left);
        if (s_left & MASK_SELF) {
            const uint64_t far_left = (hh - 2) & mask1_;
            Entry& next_left = t1_[far_left];
            if (!is_occupied(next_left)) {
                // If left also names an item at h, that item is homed at left
                // and h-2 is offset -1 of its home: move it instead, and the
                // new key takes h at offset 0 rather than h-1. Otherwise the
                // item homed at left moves and the new key takes h-1.
                const bool plus = s_left & MASK_PLUS;
                shift_entry(plus ? home : left, next_left);
                if (plus) {
                    set_state(left, (s_left & ~MASK_PLUS) | MASK_MINUS);
                    return write_slot(home, home, MASK_SELF, fp, val);
                }
                // left is its own home; its mask bit moves self -> minus.
                set_state(left, (s_left & ~MASK_SELF) | MASK_MINUS);
                return write_slot(left, home, MASK_MINUS, fp, val);
            }
        }

        return insert_into_T2(hh, val);
    }
    void* insert_place_8(uint64_t key_val, uint64_t hh, void* val, bool check_existing) {
        const uint8_t fp = fingerprint_of_hash(hh);
        Entry& home = t1_[hh & mask1_];
        Entry& right = t1_[(hh + 1) & mask1_];
        Entry& left = t1_[(hh - 1) & mask1_];

        {
        const uint8_t s_home = get_state(home);
        Entry* hit = nullptr;

        if (!is_occupied(home) && !get_t2_max_offset(home)) {
            return write_slot(home, home, MASK_SELF, fp, val);
        }

        if (check_existing) {
            // Same as insert_place.
            if ((s_home & MASK_SELF) && slot_holds_fp_8(home, key_val, fp)) {
                hit = &home;
            } else if ((s_home & MASK_PLUS) && slot_holds_fp_8(right, key_val, fp)) {
                hit = &right;
            } else if ((s_home & MASK_MINUS) && slot_holds_fp_8(left, key_val, fp)) {
                hit = &left;
            } else {
                const uint8_t max_offset = get_t2_max_offset(home);
                if (max_offset) {
                    hit = slot_for_key_T2_8(hh, key_val, fp, max_offset);
                } else { goto check_sides; }
            }
            if (hit) {
                key_store_.mark_dead(decode_pointer(hit->word));
                ++deleted_;
                hit->word = (hit->word & ~PTR_MASK) | encode_pointer(val);
                return val;
            }
            
            if (!is_occupied(home)) {
                return write_slot(home, home, MASK_SELF, fp, val);
            }
        }
        }

    check_sides:
        if (!is_occupied(right)) {
            return write_slot(right, home, MASK_PLUS, fp, val);
        }
        if (!is_occupied(left)) {
            return write_slot(left, home, MASK_MINUS, fp, val);
        }
        if (get_state(right) & MASK_SELF) {
            const uint64_t far_right = (hh + 2) & mask1_;
            Entry& next_right = t1_[far_right];
            if (!is_occupied(next_right)) {
                const uint8_t rs = get_state(right);
                set_state(right, (rs & ~MASK_SELF) | MASK_PLUS);
                shift_entry(right, next_right);
                return write_slot(right, home, MASK_PLUS, fp, val);
            }
        }
        const uint8_t s_left = get_state(left);
        if (s_left & MASK_SELF) {
            const uint64_t far_left = (hh - 2) & mask1_;
            Entry& next_left = t1_[far_left];
            if (!is_occupied(next_left)) {
                // Same as insert_place: an item at h homed at left moves to h-2.
                const bool plus = s_left & MASK_PLUS;
                shift_entry(plus ? home : left, next_left);
                if (plus) {
                    set_state(left, (s_left & ~MASK_PLUS) | MASK_MINUS);
                    return write_slot(home, home, MASK_SELF, fp, val);
                }
                set_state(left, (s_left & ~MASK_SELF) | MASK_MINUS);
                return write_slot(left, home, MASK_MINUS, fp, val);
            }
        }
        return insert_into_T2(hh, val);
    }


    // returns pointer to key; everything can be recovered from that
    void* find(uint64_t key) const { return find_key_8_val(key); }
    void* find(std::string_view key) const {
        if (key.size() == 8) return find_key_8(key.data());
        return find_key(key.data(), key.size());
    }
    void* find(const void* key, std::size_t key_len) const {
        if (key_len == 8) return find_key_8(key);
        return find_key(key, key_len);
    }

    // returns data as string view
    std::string find_str(std::string_view key) const {
        void* ptr = key.size() == 8 ? find_key_8(key.data())
                                    : find_key(key.data(), key.size());
        if (!ptr) return {};
        const std::size_t n = packed_data_len(read_packed_len(
            static_cast<const unsigned char*>(ptr) - PL_HEAD_BYTES));
        return std::string(reinterpret_cast<const char*>(data_ptr_at(ptr)), n);
    }

    // The value, copied out as a fixed-size type: T is read from the stored
    // bytes, so it has to be a type whose byte image is the whole value
    // (integers, floats, raw pointers, plain structs). T{} if the key is absent.
    template <class T>
    T find_as(std::string_view key) const {
        const std::size_t n = key.size();
        const void* p = n == 8 ? find_key_8(key.data()) : find_key(key.data(), n);
        if (!p) return T{};
        T v{};
        std::memcpy(&v, data_ptr_at(p), sizeof v);
        return v;
    }



    void* find_key_8(const void* key) const {
        return find_key_8_val(read_word(key));
    }

    void* find_key_8_val(uint64_t key_val) const {
        const uint64_t hh = hash_key(key_val);
        const uint8_t fp = fingerprint_of_hash(hh);
        const Entry& home = t1_[hh & mask1_];
        const uint8_t s_home = get_state(home);

        if ((s_home & MASK_SELF) && slot_holds_fp_8(home, key_val, fp)) {
            return decode_pointer(home.word);
        }

        if (s_home & MASK_PLUS) {
            const Entry& n1 = t1_[(hh + 1) & mask1_];
            if (slot_holds_fp_8(n1, key_val, fp)) return decode_pointer(n1.word);
        }
        if (s_home & MASK_MINUS) {
            const Entry& n2 = t1_[(hh - 1) & mask1_];
            if (slot_holds_fp_8(n2, key_val, fp)) return decode_pointer(n2.word);
        }

        uint8_t max_offset = get_t2_max_offset(home);
        if (max_offset == 0) return nullptr;
        return find_in_T2_8(hh, key_val, fp, max_offset);
    }

    __attribute__((noinline))
    void* find_in_T2_8(uint64_t hh, uint64_t key_val, uint8_t fp, uint8_t max_offset) const {
        uint64_t base_t2 = (hh >> 32) & mask2_;

        for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
            const Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
            if (t2_holds_fp_8(e, key_val, fp)) return decode_pointer(e.word);
        }
        if (max_offset == DMAX_SENTINEL) {
            for (uint64_t off = max_offset; off < n2_; ++off) {
                const Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == T2_EMPTY) break;
                if (t2_holds_fp_8(e, key_val, fp)) return decode_pointer(e.word);
            }
        }
        return nullptr;
    }

    void* find_key(const void* key, std::size_t key_len) const {
        const uint64_t hh = hash_(key, key_len);
        const uint8_t fp = fingerprint_of_hash(hh);
        const Entry& home = t1_[hh & mask1_];
        const uint8_t s_home = get_state(home);

        if ((s_home & MASK_SELF) && slot_holds_fp(home, key, key_len, fp)) {
            return decode_pointer(home.word);
        }

        if (s_home & MASK_PLUS) {
            const Entry& n1 = t1_[(hh + 1) & mask1_];
            if (slot_holds_fp(n1, key, key_len, fp)) {
                return decode_pointer(n1.word);
            }
        }
        if (s_home & MASK_MINUS) {
            const Entry& n2 = t1_[(hh - 1) & mask1_];
            if (slot_holds_fp(n2, key, key_len, fp)) {
                return decode_pointer(n2.word);
            }
        }

        uint8_t max_offset = get_t2_max_offset(home);
        if (max_offset == 0) return nullptr;
        return find_in_T2(hh, key, key_len, fp, max_offset);
    }

    // ---- insertion-order iteration -------------------------------------

    // Starts a walk and returns its state. The returned struct is the caller's:
    // copy it to fork a walk, keep it as long as the walk runs. An empty table
    // starts exhausted, so the first iter call returns nullptr.
    IterState iter_start() const {
        IterState state;
        state.rec = key_store_.first_record();
        return state;
    }

    // Advances a walk and returns the next live value pointer, or nullptr at
    // the end. The state comes from iter_start; a state at the end stays at it.
    void* iter(IterState& state) {
        IterState& c = state;
        while (c.rec) {
            const unsigned char* rec = c.rec;
            const uint64_t packed = read_packed_len(rec);

            if (++c.items < key_store_.block_end(c.block)) {
                c.rec = key_store_.next_record(rec);
            } else if (++c.block >= key_store_.block_count()) {
                c.rec = nullptr;
            } else {
                c.rec = key_store_.block_start(c.block);
            }

            if (packed_is_live(packed)) {
                return const_cast<unsigned char*>(rec) + PL_HEAD_BYTES; // the key
            }
        }
        return nullptr;
    }

    // Advances a walk and returns the next pair in the order its key was first
    // stored, or nullptr at the end. Slower than iter(): a tombstoned record
    // costs a lookup, so a key erased and put back is yielded where it first
    // appeared rather than where it was put back.
    //
    // A tombstone's key still reads back, so it can be looked up. The live
    // record that lookup finds is this key's first appearance: it is yielded
    // here, at the tombstone's position, and marked dead. Reaching it later,
    // the walk sees it dead, finds itself, marks it live again and yields
    // nothing -- so the pair is visited once, where it belongs.
    void* iter_first(IterState& state) {
        IterState& c = state;
        while (c.rec) {
            const unsigned char* rec = c.rec;
            const uint64_t packed = read_packed_len(rec);

            if (++c.items < key_store_.block_end(c.block)) {
                c.rec = key_store_.next_record(rec);
            } else if (++c.block >= key_store_.block_count()) {
                c.rec = nullptr;
            } else {
                c.rec = key_store_.block_start(c.block);
            }

            if (packed_is_live(packed)) {
                return const_cast<unsigned char*>(rec) + PL_HEAD_BYTES;
            }

            // Dead: look the key up and act on what is found.
            unsigned char* const key = const_cast<unsigned char*>(rec) + PL_HEAD_BYTES;
            void* const found = find(key, packed_key_len(packed));
            if (!found) continue;

            const unsigned char* const found_rec =
                static_cast<const unsigned char*>(found) - PL_HEAD_BYTES;
            if (packed_is_live(read_packed_len(found_rec))) {
                key_store_.mark_dead(found); // first appearance, served here
                return found;
            }
            // Dead already. If it is this record, an earlier tombstone served
            // its position, so put the bit back.
            if (found == key) key_store_.mark_live(found);
        }
        return nullptr;
    }

    // Calls f(key, value) for every live pair in insertion order. Both spans
    // borrow from the table and are valid for the call; f must not insert or
    // erase. The header is decoded once per record, not once per span.
    //
    // If f returns void, the walk is all it does. Otherwise it allocates R[n],
    // where n is the live pair count known before the walk, and returns it as
    // a unique_ptr<R[]> filled in insertion order. The caller owns the array.
    template <class F>
    auto map(F&& f) const {
        using R = std::decay_t<std::invoke_result_t<F&, std::string_view,
                                                    std::string_view>>;
        if constexpr (std::is_void_v<R>) {
            map_walk([&](std::string_view k, std::string_view v) { f(k, v); });
        } else {
            const std::size_t n = size();
            std::unique_ptr<R[]> out(new R[n]);
            R* p = out.get();
            map_walk([&](std::string_view k, std::string_view v) {
                *p++ = f(k, v);
            });
            return out;
        }
    }

private:
    // The walk map() runs, once per live pair, in insertion order.
    template <class F>
    void map_walk(F&& f) const {
        std::size_t block = 0;
        const unsigned char* rec = key_store_.first_record();
        std::size_t left = key_store_.block_size(0);

    walk:
        while (left != 0) {
            const unsigned char* cur = rec;
            const uint64_t packed = read_packed_len(cur);
            const std::size_t key_len = packed_key_len(packed);
            const std::size_t data_len = packed_data_len(packed);
            rec = cur + KeyStore::round_up_8(PL_HEAD_BYTES + key_len + data_len);
            --left;
            if (packed_is_live(packed)) {
                const char* key = reinterpret_cast<const char*>(cur) + PL_HEAD_BYTES;
                f(std::string_view(key, key_len),
                  std::string_view(key + key_len, data_len));
            }
        }

        if (++block == key_store_.block_count()) return;
        rec = key_store_.block_start(block);
        left = key_store_.block_size(block);
        goto walk;
    }

    void* find_in_T2(uint64_t hh, const void* key, std::size_t key_len, uint8_t fp,
                     uint8_t max_offset) const {
        uint64_t base_t2 = (hh >> 32) & mask2_;

        for (int off = static_cast<int>(max_offset) - 1; off >= 0; --off) {
            const Entry& e = t2_[(base_t2 + static_cast<uint64_t>(off)) & mask2_];
            if (t2_holds_fp(e, key, key_len, fp)) {
                return decode_pointer(e.word);
            }
        }

        if (max_offset == DMAX_SENTINEL) {
            for (uint64_t off = max_offset; off < n2_; ++off) {
                const Entry& e = t2_[(base_t2 + off) & mask2_];
                if (get_state(e) == T2_EMPTY) break;
                if (t2_holds_fp(e, key, key_len, fp)) return decode_pointer(e.word);
            }
        }
        return nullptr;
    }

public:
    bool erase(uint64_t key) { return erase_key(&key, sizeof key); }
    bool erase(std::string_view key) { return erase_key(key.data(), key.size()); }
    bool erase(const void* key, std::size_t key_len) { return erase_key(key, key_len); }


private:
    bool erase_key(const void* key, std::size_t key_len) {
        const uint64_t hh = hash_(key, key_len);
        const uint8_t fp = fingerprint_of_hash(hh);

        bool in_t1 = false;
        Entry* e = slot_for_key(key, key_len, hh, fp, in_t1);
        if (!e) return false;

        key_store_.mark_dead(decode_pointer(e->word));
        ++deleted_;
        if (in_t1) {
            // The slot's state belongs to its own home; only the item bits go.
            e->word &= ~ITEM_MASK;
            // Its home's mask still names this slot. Clear that bit, then let a
            // neighbouring item homed at the freed slot move in.
            Entry& home = t1_[hh & mask1_];
            const uint64_t home_i = hh & mask1_;
            const uint8_t bit = mask_bit_for(static_cast<uint64_t>(e - t1_), home_i, mask1_);
            home.word &= ~(static_cast<uint64_t>(bit) << STATE_SHIFT);
            pull_neighbors_home(static_cast<uint64_t>(e - t1_));
        } else {
            // T2 slots carry no mask; the three state bits hold only the marker,
            // so replace them wholesale and leave the item bits for reuse.
            set_state(*e, T2_TOMBSTONE);
            --t2_count_;
        }
        return true;
    }

private:
    static std::size_t ceil_pow2(std::size_t n) {
        std::size_t p = 1;
        while (p < n) p <<= 1;
        return p;
    }

    // After `slot` frees, pull in a key at `slot ± 1` whose own home is `slot`.
    // The gap can travel two steps at most; find()'s 2-probe bound comes from
    // insert confining offsets to {-1, 0, +1}.
    void pull_neighbors_home(uint64_t slot) {
        uint64_t gap = slot;
        for (int step = 0; step < 2; ++step) {
            bool moved = false;
            for (int dir = -1; dir <= 1; dir += 2) {
                uint64_t from = (gap + (uint64_t)dir) & mask1_;
                // An item in `from` homed at `gap` is one step from its home;
                // `gap`'s mask says so, and the item itself is in the neighbour.
                const uint8_t bit = (dir < 0) ? MASK_MINUS : MASK_PLUS;
                const uint8_t gs = get_state(t1_[gap]);
                if (!(gs & bit)) continue;
                // Move the item in. Its home is `gap`, so the mask bit travels
                // from the far side to MASK_SELF in that same home word.
                shift_entry(t1_[from], t1_[gap]);
                set_state(t1_[gap], (gs & ~bit) | MASK_SELF);
                gap = from;
                moved = true;
                break;
            }
            if (!moved) break;
        }
    }

    void* insert_into_T2(uint64_t hh, void* val) {
        if (t2_count_ >= (n2_ >> 2) * 3) {
            if (!grow_to(n1_ * 2)) {
                return nullptr;
            }
        }

        uint64_t base_t2 = (hh >> 32) & mask2_;

        for (uint64_t off = 0; off < n2_; ++off) {
            uint64_t idx = (base_t2 + off) & mask2_;
            uint8_t st = get_state(t2_[idx]);
            if (st != T2_LIVE) {
                ++t2_count_;
                Entry& d_home = t1_[hh & mask1_];

                // Stored as offset + 1, so offset 0 is not "nothing spilled".
                const uint8_t dmax = std::min<uint8_t>(off + 1, DMAX_SENTINEL);
                set_t2_max_offset(d_home, std::max(get_t2_max_offset(d_home), dmax));
                // A T2 slot belongs to no T1 home, so no mask bit is set; its
                // state is the marker alone, so replace the field rather than
                // OR into it (a tombstone is 100, live is 001).
                Entry& d = t2_[idx];
                set_state(d, T2_LIVE);
                set_item(d, fingerprint_of_hash(hh), val);
                return val;
            }
        }
        return nullptr;
    }

    bool grow_to(std::size_t new_n1) {
        if (new_n1 <= n1_) return false;
        new_n1 = ceil_pow2(new_n1); // mask = n1 - 1 needs a power of two
        const std::size_t old_n1 = n1_, old_n2 = n2_;
        Entry* old_t1 = t1_;

        const std::size_t new_n2 = new_n1 >> SHIFT_RATIO;
        Entry* new_t1 = static_cast<Entry*>(std::calloc(new_n1 + new_n2, sizeof(Entry)));
        if (!new_t1) return false;

        t1_ = new_t1;
        t2_ = new_t1 + new_n1;
        n1_ = new_n1;
        n2_ = new_n2;
        mask1_ = n1_ - 1;
        mask2_ = n2_ - 1;
        t2_count_ = 0;

        // T1 first, then T2: re-placing T1 into an empty table keeps its
        // neighbourhoods sparse, so fewer entries spill.
        for (std::size_t i = 0; i < old_n1; ++i) {
            const Entry& e = old_t1[i];
            if (!is_occupied(e)) continue;
            void* val = decode_pointer(e.word);
            std::size_t klen = key_len_at(val);
            const void* k = val;
            const uint64_t hh = hash_(k, klen);
            insert_place(k, klen, hh, val, false);
        }
        const Entry* old_t2 = old_t1 + old_n1;
        for (std::size_t i = 0; i < old_n2; ++i) {
            const Entry& e = old_t2[i];
            uint8_t st = get_state(e);
            if (st != T2_LIVE) continue;
            void* val = decode_pointer(e.word);
            std::size_t klen = key_len_at(val);
            const void* k = val;
            const uint64_t hh = hash_(k, klen);
            insert_place(k, klen, hh, val, false);
        }

        std::free(old_t1);
        return true;
    }

    Entry* t1_ = nullptr;
    Entry* t2_ = nullptr;
    std::size_t n1_ = 0;
    std::size_t n2_ = 0;
    std::size_t mask1_ = 0;
    std::size_t mask2_ = 0;
    std::size_t deleted_ = 0;
    std::size_t t2_count_ = 0;
    [[no_unique_address]] Hash hash_{};
    KeyStore key_store_;
};

using Table = BasicTable<mtht::Hash>;

} // namespace mtht

#endif // MTHT_HPP
