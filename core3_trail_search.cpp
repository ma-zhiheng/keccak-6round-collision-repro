#include "core3_search_data.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

using State = std::array<std::uint64_t, 25>;

static constexpr std::array<std::uint64_t, 24> ROUND_CONSTANTS = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

static constexpr int ROTATION[5][5] = {
    {0, 36, 3, 41, 18},
    {1, 44, 10, 45, 2},
    {62, 6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39, 8, 14},
};

inline int idx(int x, int y) { return x + 5 * y; }

inline std::uint64_t rotl64(std::uint64_t value, int amount) {
    amount &= 63;
    if (amount == 0) return value;
    return (value << amount) | (value >> (64 - amount));
}

void theta(State& state) {
    std::uint64_t columns[5] = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) columns[x] ^= state[idx(x, y)];
    }
    std::uint64_t delta[5];
    for (int x = 0; x < 5; ++x) {
        delta[x] = columns[(x + 4) % 5] ^ rotl64(columns[(x + 1) % 5], 1);
    }
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) state[idx(x, y)] ^= delta[x];
    }
}

void rho_pi(State& state) {
    State moved = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            int nx = y;
            int ny = (2 * x + 3 * y) % 5;
            moved[idx(nx, ny)] = rotl64(state[idx(x, y)], ROTATION[x][y]);
        }
    }
    state = moved;
}

void chi(State& state) {
    for (int y = 0; y < 5; ++y) {
        std::uint64_t row[5];
        for (int x = 0; x < 5; ++x) row[x] = state[idx(x, y)];
        for (int x = 0; x < 5; ++x) {
            state[idx(x, y)] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
    }
}

void round(State& state, int round_number) {
    theta(state);
    rho_pi(state);
    chi(state);
    state[0] ^= ROUND_CONSTANTS[round_number];
}

State xor_state(const State& a, const State& b) {
    State out{};
    for (std::size_t i = 0; i < out.size(); ++i) out[i] = a[i] ^ b[i];
    return out;
}

bool equal_state(const State& a, const State& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

bool digest_zero(const State& diff, int digest_bits) {
    int full_words = digest_bits / 64;
    int rem = digest_bits % 64;
    for (int i = 0; i < full_words; ++i) {
        if (diff[i] != 0) return false;
    }
    if (rem == 0) return true;
    return (diff[full_words] & ((1ULL << rem) - 1ULL)) == 0;
}

State sample_message(std::mt19937_64& rng) {
    State message = core3_data::PARTICULAR_MESSAGE;
    std::uint64_t bits = 0;
    int available = 0;
    for (std::size_t basis = 0; basis < core3_data::BASIS_SIZE; ++basis) {
        if (available == 0) {
            bits = rng();
            available = 64;
        }
        bool take = (bits & 1ULL) != 0;
        bits >>= 1;
        --available;
        if (!take) continue;
        for (std::size_t lane = 0; lane < 25; ++lane) {
            message[lane] ^= core3_data::BASIS_MESSAGE[basis][lane];
        }
    }
    return message;
}

void print_state_hex(const char* label, const State& state) {
    std::cout << label;
    for (int lane = 24; lane >= 0; --lane) {
        std::cout << std::hex << std::setw(16) << std::setfill('0') << state[lane];
    }
    std::cout << std::dec << "\n";
}

struct EvalResult {
    bool alpha3 = false;
    bool alpha4 = false;
    bool digest = false;
};

EvalResult evaluate(const State& message1, const State& message2, int digest_bits) {
    EvalResult result;
    State s1 = message1;
    State s2 = message2;
    for (int r = 0; r < 3; ++r) {
        round(s1, r);
        round(s2, r);
    }
    result.alpha3 = equal_state(xor_state(s1, s2), core3_data::ALPHA3);
    round(s1, 3);
    round(s2, 3);
    result.alpha4 = equal_state(xor_state(s1, s2), core3_data::ALPHA4);
    round(s1, 4);
    round(s2, 4);
    round(s1, 5);
    round(s2, 5);
    result.digest = digest_zero(xor_state(s1, s2), digest_bits);
    return result;
}

std::uint64_t parse_u64_arg(int argc, char** argv, const std::string& name, std::uint64_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) return std::strtoull(argv[i + 1], nullptr, 10);
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) return true;
    }
    return false;
}

struct Totals {
    std::uint64_t samples = 0;
    std::uint64_t alpha3 = 0;
    std::uint64_t alpha4 = 0;
    std::uint64_t digest = 0;
};

int main(int argc, char** argv) {
    std::uint64_t samples = parse_u64_arg(argc, argv, "--samples", 1000000);
    std::uint64_t seed = parse_u64_arg(argc, argv, "--seed", core3_data::CONNECTOR_SEED);
    std::uint64_t report = parse_u64_arg(argc, argv, "--report", samples / 10 ? samples / 10 : 1);
    int threads = static_cast<int>(parse_u64_arg(argc, argv, "--threads", 1));
    int digest_bits = static_cast<int>(parse_u64_arg(argc, argv, "--digest-bits", 160));
    bool self_test = has_flag(argc, argv, "--self-test");

    std::cout << "C++ core3 Keccak[1440,160,6,160] trail search\n";
    std::cout << "  samples=" << samples << ", seed=" << seed
              << ", basis=" << core3_data::BASIS_SIZE
              << ", threads=" << threads << "\n";

    if (self_test) {
        State m1 = core3_data::PARTICULAR_MESSAGE;
        State m2 = xor_state(m1, core3_data::ALPHA0);
        EvalResult result = evaluate(m1, m2, digest_bits);
        std::cout << "self-test alpha3=" << result.alpha3
                  << " alpha4=" << result.alpha4
                  << " digest=" << result.digest << "\n";
        if (!result.alpha3) return 2;
    }

    std::atomic<std::uint64_t> next{0};
    std::atomic<std::uint64_t> done{0};
    std::atomic<std::uint64_t> alpha3{0};
    std::atomic<std::uint64_t> alpha4{0};
    std::atomic<std::uint64_t> digest{0};
    std::atomic<bool> stop{false};
    std::mutex output;
    auto start = std::chrono::steady_clock::now();
    static constexpr std::uint64_t CHUNK = 4096;
    std::vector<std::thread> workers;
    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::mt19937_64 rng(seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(tid + 1)));
            while (!stop.load(std::memory_order_relaxed)) {
                std::uint64_t begin = next.fetch_add(CHUNK, std::memory_order_relaxed);
                if (begin >= samples) break;
                std::uint64_t end = std::min(samples, begin + CHUNK);
                for (std::uint64_t i = begin; i < end; ++i) {
                    State m1 = sample_message(rng);
                    State m2 = xor_state(m1, core3_data::ALPHA0);
                    EvalResult result = evaluate(m1, m2, digest_bits);
                    alpha3.fetch_add(result.alpha3 ? 1 : 0, std::memory_order_relaxed);
                    alpha4.fetch_add(result.alpha4 ? 1 : 0, std::memory_order_relaxed);
                    if (result.digest) {
                        digest.fetch_add(1, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(output);
                        std::cout << "  FOUND digest sample=" << (i + 1) << " thread=" << tid << "\n";
                        print_state_hex("    M1: ", m1);
                        print_state_hex("    M2: ", m2);
                        stop.store(true, std::memory_order_relaxed);
                        break;
                    }
                }
                std::uint64_t completed = done.fetch_add(end - begin, std::memory_order_relaxed) + (end - begin);
                if (report && completed / report != (completed - (end - begin)) / report) {
                    auto now = std::chrono::steady_clock::now();
                    double elapsed = std::chrono::duration<double>(now - start).count();
                    std::lock_guard<std::mutex> lock(output);
                    std::cout << "  completed=" << completed
                              << ", elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                              << ", rate=" << std::setprecision(2) << (static_cast<double>(completed) / elapsed / 1000000.0) << " M/s"
                              << ", alpha3=" << alpha3.load(std::memory_order_relaxed)
                              << ", alpha4=" << alpha4.load(std::memory_order_relaxed)
                              << ", digest=" << digest.load(std::memory_order_relaxed) << "\n";
                }
            }
        });
    }
    for (auto& worker : workers) worker.join();
    auto finish = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(finish - start).count();
    std::uint64_t completed = done.load(std::memory_order_relaxed);
    std::cout << "summary\n";
    std::cout << "  samples=" << completed << "\n";
    std::cout << "  alpha3_hits=" << alpha3.load(std::memory_order_relaxed) << "\n";
    std::cout << "  alpha4_hits=" << alpha4.load(std::memory_order_relaxed) << "\n";
    std::cout << "  digest_hits=" << digest.load(std::memory_order_relaxed) << "\n";
    std::cout << "  elapsed=" << std::fixed << std::setprecision(3) << elapsed << "s\n";
    if (elapsed > 0.0) {
        std::cout << "  rate=" << std::setprecision(2) << (static_cast<double>(completed) / elapsed / 1000000.0) << " M/s\n";
    }
    return 0;
}
