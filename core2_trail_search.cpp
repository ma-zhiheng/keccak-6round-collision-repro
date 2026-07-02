#include "core2_search_data.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include <atomic>
#include <mutex>

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

inline int idx(int x, int y) {
    return x + 5 * y;
}

inline std::uint64_t rotl64(std::uint64_t value, int amount) {
    amount &= 63;
    if (amount == 0) {
        return value;
    }
    return (value << amount) | (value >> (64 - amount));
}

void theta(State& state) {
    std::uint64_t columns[5] = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            columns[x] ^= state[idx(x, y)];
        }
    }
    std::uint64_t delta[5];
    for (int x = 0; x < 5; ++x) {
        delta[x] = columns[(x + 4) % 5] ^ rotl64(columns[(x + 1) % 5], 1);
    }
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            state[idx(x, y)] ^= delta[x];
        }
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
        for (int x = 0; x < 5; ++x) {
            row[x] = state[idx(x, y)];
        }
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
    State out;
    for (std::size_t i = 0; i < out.size(); ++i) {
        out[i] = a[i] ^ b[i];
    }
    return out;
}

bool equal_state(const State& a, const State& b) {
    for (std::size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

bool digest_zero(const State& diff, int digest_bits) {
    int full_words = digest_bits / 64;
    int rem = digest_bits % 64;
    for (int i = 0; i < full_words; ++i) {
        if (diff[i] != 0) {
            return false;
        }
    }
    if (rem == 0) {
        return true;
    }
    std::uint64_t mask = (1ULL << rem) - 1ULL;
    return (diff[full_words] & mask) == 0;
}

State sample_message(std::mt19937_64& rng) {
    State message = core2_data::PARTICULAR_MESSAGE;
    std::uint64_t bits = 0;
    int available = 0;
    for (std::size_t basis = 0; basis < core2_data::BASIS_SIZE; ++basis) {
        if (available == 0) {
            bits = rng();
            available = 64;
        }
        bool take = (bits & 1ULL) != 0;
        bits >>= 1;
        --available;
        if (!take) {
            continue;
        }
        for (std::size_t lane = 0; lane < 25; ++lane) {
            message[lane] ^= core2_data::BASIS_MESSAGE[basis][lane];
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
    bool alpha2 = false;
    bool alpha3 = false;
    bool alpha4 = false;
    bool digest = false;
};

struct SearchTotals {
    std::uint64_t samples = 0;
    std::uint64_t hits_alpha2 = 0;
    std::uint64_t hits_alpha3 = 0;
    std::uint64_t hits_alpha4 = 0;
    std::uint64_t hits_digest = 0;
};

struct AtomicSearchTotals {
    std::atomic<std::uint64_t> samples{0};
    std::atomic<std::uint64_t> hits_alpha2{0};
    std::atomic<std::uint64_t> hits_alpha3{0};
    std::atomic<std::uint64_t> hits_alpha4{0};
    std::atomic<std::uint64_t> hits_digest{0};
};

struct SearchOptions {
    std::uint64_t samples = 1000000;
    std::uint64_t seed = core2_data::CONNECTOR_SEED;
    int digest_bits = 160;
    std::uint64_t report = 100000;
    int threads = 1;
    std::uint64_t max_alpha3_print = 1;
    std::string candidate_file;
    bool self_test = false;
};

EvalResult evaluate(const State& message1, const State& message2, int digest_bits) {
    EvalResult result;
    State s1 = message1;
    State s2 = message2;

    round(s1, 0);
    round(s2, 0);
    round(s1, 1);
    round(s2, 1);
    State diff2 = xor_state(s1, s2);
    result.alpha2 = equal_state(diff2, core2_data::ALPHA2);
    if (!result.alpha2) {
        return result;
    }

    round(s1, 2);
    round(s2, 2);
    State diff3 = xor_state(s1, s2);
    result.alpha3 = equal_state(diff3, core2_data::ALPHA3);
    if (!result.alpha3) {
        return result;
    }

    round(s1, 3);
    round(s2, 3);
    State diff4 = xor_state(s1, s2);
    result.alpha4 = equal_state(diff4, core2_data::ALPHA4);
    if (!result.alpha4) {
        return result;
    }

    round(s1, 4);
    round(s2, 4);
    State diff5 = xor_state(s1, s2);

    result.digest = digest_zero(diff5, digest_bits);
    return result;
}

std::uint64_t parse_u64_arg(int argc, char** argv, const std::string& name, std::uint64_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::strtoull(argv[i + 1], nullptr, 10);
        }
    }
    return fallback;
}

std::string parse_string_arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

bool has_flag(int argc, char** argv, const std::string& name) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == name) {
            return true;
        }
    }
    return false;
}

void write_candidate(
    std::ostream& out,
    const std::string& label,
    std::uint64_t sample,
    int thread_id,
    const State& message1,
    const State& message2
) {
    out << label << " sample=" << sample;
    if (thread_id >= 0) {
        out << " thread=" << thread_id;
    }
    out << "\n";
    print_state_hex("    M1: ", message1);
    print_state_hex("    M2: ", message2);
}

void write_candidate_file(
    const std::string& path,
    const std::string& label,
    std::uint64_t sample,
    int thread_id,
    const State& message1,
    const State& message2
) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "warning: could not open candidate file: " << path << "\n";
        return;
    }
    out << label << " sample=" << sample;
    if (thread_id >= 0) {
        out << " thread=" << thread_id;
    }
    out << "\n";
    out << "M1 ";
    for (int lane = 24; lane >= 0; --lane) {
        out << std::hex << std::setw(16) << std::setfill('0') << message1[lane];
    }
    out << "\nM2 ";
    for (int lane = 24; lane >= 0; --lane) {
        out << std::hex << std::setw(16) << std::setfill('0') << message2[lane];
    }
    out << std::dec << "\n";
}

SearchTotals run_serial(const SearchOptions& options) {
    const std::uint64_t samples = options.samples;
    std::mt19937_64 rng(options.seed);
    SearchTotals totals;
    std::uint64_t printed_alpha3 = 0;
    auto start = std::chrono::steady_clock::now();

    for (std::uint64_t i = 1; i <= samples; ++i) {
        State message1 = sample_message(rng);
        State message2 = xor_state(message1, core2_data::ALPHA0);
        EvalResult result = evaluate(message1, message2, options.digest_bits);
        totals.samples = i;
        totals.hits_alpha2 += result.alpha2 ? 1 : 0;
        totals.hits_alpha3 += result.alpha3 ? 1 : 0;
        totals.hits_alpha4 += result.alpha4 ? 1 : 0;
        totals.hits_digest += result.digest ? 1 : 0;

        if (result.alpha3 && printed_alpha3 < options.max_alpha3_print) {
            ++printed_alpha3;
            write_candidate(std::cout, "  alpha3 hit", i, -1, message1, message2);
        }
        if (result.alpha4 && result.digest) {
            write_candidate(std::cout, "  FOUND alpha4/digest candidate", i, -1, message1, message2);
            write_candidate_file(options.candidate_file, "FOUND alpha4/digest candidate", i, -1, message1, message2);
            break;
        }
        if (options.report && i % options.report == 0) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            std::cout << "  sampled=" << i
                      << ", elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                      << ", rate=" << std::setprecision(2) << (static_cast<double>(i) / elapsed / 1000000.0) << " M/s"
                      << ", alpha3_hits=" << totals.hits_alpha3
                      << ", alpha4_hits=" << totals.hits_alpha4
                      << ", digest_hits=" << totals.hits_digest << "\n";
        }
    }
    return totals;
}

SearchTotals run_parallel(
    const SearchOptions& options
) {
    const std::uint64_t samples = options.samples;
    const int threads = options.threads;
    std::atomic<std::uint64_t> next_sample{0};
    std::atomic<bool> stop{false};
    std::atomic<std::uint64_t> reported{0};
    std::atomic<std::uint64_t> alpha3_prints{0};
    std::mutex output_mutex;
    AtomicSearchTotals totals;
    auto start = std::chrono::steady_clock::now();
    static constexpr std::uint64_t CHUNK = 4096;

    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));

    for (int tid = 0; tid < threads; ++tid) {
        workers.emplace_back([&, tid]() {
            std::mt19937_64 rng(options.seed ^ (0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(tid + 1)));
            SearchTotals local;
            while (!stop.load(std::memory_order_relaxed)) {
                std::uint64_t begin = next_sample.fetch_add(CHUNK, std::memory_order_relaxed);
                if (begin >= samples) {
                    break;
                }
                std::uint64_t end = std::min(samples, begin + CHUNK);
                for (std::uint64_t i = begin; i < end && !stop.load(std::memory_order_relaxed); ++i) {
                    State message1 = sample_message(rng);
                    State message2 = xor_state(message1, core2_data::ALPHA0);
                    EvalResult result = evaluate(message1, message2, options.digest_bits);
                    local.samples += 1;
                    local.hits_alpha2 += result.alpha2 ? 1 : 0;
                    local.hits_alpha3 += result.alpha3 ? 1 : 0;
                    local.hits_alpha4 += result.alpha4 ? 1 : 0;
                    local.hits_digest += result.digest ? 1 : 0;

                    if (result.alpha3 && alpha3_prints.load(std::memory_order_relaxed) < options.max_alpha3_print) {
                        std::uint64_t previous = alpha3_prints.fetch_add(1, std::memory_order_relaxed);
                        if (previous < options.max_alpha3_print) {
                            std::lock_guard<std::mutex> lock(output_mutex);
                            write_candidate(std::cout, "  alpha3 hit", i + 1, tid, message1, message2);
                        }
                    }
                    if (result.alpha4 && result.digest) {
                        stop.store(true, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(output_mutex);
                        write_candidate(std::cout, "  FOUND alpha4/digest candidate", i + 1, tid, message1, message2);
                        write_candidate_file(options.candidate_file, "FOUND alpha4/digest candidate", i + 1, tid, message1, message2);
                        break;
                    }
                }

                totals.samples.fetch_add(local.samples, std::memory_order_relaxed);
                totals.hits_alpha2.fetch_add(local.hits_alpha2, std::memory_order_relaxed);
                totals.hits_alpha3.fetch_add(local.hits_alpha3, std::memory_order_relaxed);
                totals.hits_alpha4.fetch_add(local.hits_alpha4, std::memory_order_relaxed);
                totals.hits_digest.fetch_add(local.hits_digest, std::memory_order_relaxed);
                local = SearchTotals{};

                if (options.report) {
                    std::uint64_t done = std::min(samples, next_sample.load(std::memory_order_relaxed));
                    std::uint64_t previous = reported.load(std::memory_order_relaxed);
                    if (done / options.report > previous / options.report
                        && reported.compare_exchange_strong(previous, done, std::memory_order_relaxed)) {
                        auto now = std::chrono::steady_clock::now();
                        double elapsed = std::chrono::duration<double>(now - start).count();
                        std::uint64_t completed = totals.samples.load(std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lock(output_mutex);
                        std::cout << "  scheduled=" << done
                                  << ", completed=" << completed
                                  << ", elapsed=" << std::fixed << std::setprecision(1) << elapsed << "s"
                                  << ", rate=" << std::setprecision(2) << (static_cast<double>(completed) / elapsed / 1000000.0) << " M/s"
                                  << ", alpha3_hits=" << totals.hits_alpha3.load(std::memory_order_relaxed)
                                  << ", alpha4_hits=" << totals.hits_alpha4.load(std::memory_order_relaxed)
                                  << ", digest_hits=" << totals.hits_digest.load(std::memory_order_relaxed) << "\n";
                    }
                }
            }
            if (local.samples) {
                totals.samples.fetch_add(local.samples, std::memory_order_relaxed);
                totals.hits_alpha2.fetch_add(local.hits_alpha2, std::memory_order_relaxed);
                totals.hits_alpha3.fetch_add(local.hits_alpha3, std::memory_order_relaxed);
                totals.hits_alpha4.fetch_add(local.hits_alpha4, std::memory_order_relaxed);
                totals.hits_digest.fetch_add(local.hits_digest, std::memory_order_relaxed);
            }
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    SearchTotals snapshot;
    snapshot.samples = totals.samples.load(std::memory_order_relaxed);
    snapshot.hits_alpha2 = totals.hits_alpha2.load(std::memory_order_relaxed);
    snapshot.hits_alpha3 = totals.hits_alpha3.load(std::memory_order_relaxed);
    snapshot.hits_alpha4 = totals.hits_alpha4.load(std::memory_order_relaxed);
    snapshot.hits_digest = totals.hits_digest.load(std::memory_order_relaxed);
    return snapshot;
}

int main(int argc, char** argv) {
    SearchOptions options;
    options.samples = parse_u64_arg(argc, argv, "--samples", options.samples);
    options.seed = parse_u64_arg(argc, argv, "--seed", options.seed);
    options.digest_bits = static_cast<int>(parse_u64_arg(argc, argv, "--digest-bits", options.digest_bits));
    options.report = parse_u64_arg(argc, argv, "--report", options.samples / 10 ? options.samples / 10 : 1);
    options.threads = static_cast<int>(parse_u64_arg(argc, argv, "--threads", options.threads));
    options.max_alpha3_print = parse_u64_arg(argc, argv, "--max-alpha3-print", options.max_alpha3_print);
    options.candidate_file = parse_string_arg(argc, argv, "--candidate-file", "");
    options.self_test = has_flag(argc, argv, "--self-test");

    std::cout << "C++ Table 7 core No. 2 trail search\n";
    std::cout << "  samples=" << options.samples << ", seed=" << options.seed
              << ", digest_bits=" << options.digest_bits
              << ", basis=" << core2_data::BASIS_SIZE
              << ", threads=" << options.threads
              << ", max_alpha3_print=" << options.max_alpha3_print << "\n";
    if (!options.candidate_file.empty()) {
        std::cout << "  candidate_file=" << options.candidate_file << "\n";
    }

    if (options.self_test) {
        State message1 = core2_data::PARTICULAR_MESSAGE;
        State message2 = xor_state(message1, core2_data::ALPHA0);
        EvalResult result = evaluate(message1, message2, options.digest_bits);
        std::cout << "self-test\n";
        std::cout << "  alpha2=" << result.alpha2 << "\n";
        std::cout << "  alpha3=" << result.alpha3 << "\n";
        std::cout << "  alpha4=" << result.alpha4 << "\n";
        std::cout << "  digest=" << result.digest << "\n";
        if (!result.alpha2) {
            std::cerr << "self-test failed: exported connector base does not reach alpha2\n";
            return 2;
        }
    }

    auto start = std::chrono::steady_clock::now();
    SearchTotals totals = options.threads <= 1
        ? run_serial(options)
        : run_parallel(options);
    auto stop = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(stop - start).count();

    std::cout << "summary\n";
    std::cout << "  samples=" << totals.samples << "\n";
    std::cout << "  alpha2_hits=" << totals.hits_alpha2 << "\n";
    std::cout << "  alpha3_hits=" << totals.hits_alpha3 << "\n";
    std::cout << "  alpha4_hits=" << totals.hits_alpha4 << "\n";
    std::cout << "  digest_hits=" << totals.hits_digest << "\n";
    std::cout << "  elapsed=" << std::fixed << std::setprecision(3) << elapsed << "s\n";
    if (elapsed > 0.0) {
        std::cout << "  rate=" << std::setprecision(2) << (static_cast<double>(totals.samples) / elapsed / 1000000.0) << " M/s\n";
    }
    return 0;
}
