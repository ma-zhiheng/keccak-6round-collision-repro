#include "core3_search_data.hpp"

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

static constexpr int LANES = 25;
static constexpr int BASIS_SIZE = static_cast<int>(core3_data::BASIS_SIZE);

__constant__ std::uint64_t D_PARTICULAR[LANES];
__constant__ std::uint64_t D_ALPHA0[LANES];
__constant__ std::uint64_t D_ALPHA2[LANES];
__constant__ std::uint64_t D_ALPHA3[LANES];
__constant__ std::uint64_t D_ALPHA4[LANES];
__constant__ std::uint64_t D_BASIS[BASIS_SIZE * LANES];

static constexpr std::array<std::uint64_t, 24> ROUND_CONSTANTS_HOST = {
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

__constant__ std::uint64_t D_ROUND_CONSTANTS[24];

using HostState = std::array<std::uint64_t, LANES>;

struct DeviceTotals {
    unsigned long long samples;
    unsigned long long alpha2_hits;
    unsigned long long alpha3_hits;
    unsigned long long alpha4_hits;
    unsigned long long digest_hits;
    unsigned long long first_alpha3_sample;
    unsigned long long found_sample;
    unsigned long long stored_alpha3_hits;
    int first_alpha3_written;
    int found_written;
};

struct HostTotals {
    std::uint64_t samples = 0;
    std::uint64_t alpha2_hits = 0;
    std::uint64_t alpha3_hits = 0;
    std::uint64_t alpha4_hits = 0;
    std::uint64_t digest_hits = 0;
};

struct DeviceResult {
    HostTotals totals;
    bool has_alpha3 = false;
    bool has_found = false;
    std::uint64_t first_alpha3_sample = 0;
    std::uint64_t found_sample = 0;
    std::array<std::uint64_t, LANES> alpha3_m1{};
    std::array<std::uint64_t, LANES> alpha3_m2{};
    std::array<std::uint64_t, LANES> found_m1{};
    std::array<std::uint64_t, LANES> found_m2{};
    std::vector<std::uint64_t> alpha3_samples;
};

struct Options {
    std::uint64_t samples = 1000000;
    std::uint64_t seed = core3_data::CONNECTOR_SEED;
    int digest_bits = 160;
    int threads_per_block = 512;
    int blocks_per_sm = 8;
    bool verify_alpha2 = false;
    bool alpha3_only = false;
    std::uint64_t max_alpha3_print = 1;
    std::uint64_t max_stored_alpha3 = 1048576;
    std::string devices = "0";
    std::string candidate_file;
};

__device__ __forceinline__ int idx(int x, int y) {
    return x + 5 * y;
}

__device__ __forceinline__ std::uint64_t rotl64(std::uint64_t value, int amount) {
    amount &= 63;
    if (amount == 0) {
        return value;
    }
    return (value << amount) | (value >> (64 - amount));
}

__device__ __forceinline__ std::uint64_t splitmix64_next(std::uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

__device__ void theta(std::uint64_t state[LANES]) {
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

__device__ void rho_pi(std::uint64_t state[LANES]) {
    constexpr int rotation[5][5] = {
        {0, 36, 3, 41, 18},
        {1, 44, 10, 45, 2},
        {62, 6, 43, 15, 61},
        {28, 55, 25, 21, 56},
        {27, 20, 39, 8, 14},
    };
    std::uint64_t moved[LANES] = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            int nx = y;
            int ny = (2 * x + 3 * y) % 5;
            moved[idx(nx, ny)] = rotl64(state[idx(x, y)], rotation[x][y]);
        }
    }
    for (int i = 0; i < LANES; ++i) {
        state[i] = moved[i];
    }
}

__device__ void chi(std::uint64_t state[LANES]) {
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

__device__ void keccak_round(std::uint64_t state[LANES], int round_number) {
    theta(state);
    rho_pi(state);
    chi(state);
    state[0] ^= D_ROUND_CONSTANTS[round_number];
}

__device__ __forceinline__ bool equal_to_constant(const std::uint64_t diff[LANES], const std::uint64_t target[LANES]) {
    for (int i = 0; i < LANES; ++i) {
        if (diff[i] != target[i]) {
            return false;
        }
    }
    return true;
}

__device__ bool digest_zero(const std::uint64_t diff[LANES], int digest_bits) {
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

__device__ void sample_message(std::uint64_t global_sample, std::uint64_t seed, std::uint64_t message[LANES]) {
    for (int lane = 0; lane < LANES; ++lane) {
        message[lane] = D_PARTICULAR[lane];
    }

    std::uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (global_sample + 1));
    std::uint64_t selectors[3] = {
        splitmix64_next(state),
        splitmix64_next(state),
        splitmix64_next(state),
    };

    for (int basis = 0; basis < BASIS_SIZE; ++basis) {
        bool take = ((selectors[basis >> 6] >> (basis & 63)) & 1ULL) != 0;
        if (!take) {
            continue;
        }
        const int base = basis * LANES;
        for (int lane = 0; lane < LANES; ++lane) {
            message[lane] ^= D_BASIS[base + lane];
        }
    }
}

__device__ void sample_state_pair(
    std::uint64_t global_sample,
    std::uint64_t seed,
    std::uint64_t s1[LANES],
    std::uint64_t s2[LANES]
) {
    for (int lane = 0; lane < LANES; ++lane) {
        s1[lane] = D_PARTICULAR[lane];
    }

    std::uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (global_sample + 1));
    std::uint64_t selectors[3] = {
        splitmix64_next(state),
        splitmix64_next(state),
        splitmix64_next(state),
    };

    for (int basis = 0; basis < BASIS_SIZE; ++basis) {
        bool take = ((selectors[basis >> 6] >> (basis & 63)) & 1ULL) != 0;
        if (!take) {
            continue;
        }
        const int base = basis * LANES;
        for (int lane = 0; lane < LANES; ++lane) {
            s1[lane] ^= D_BASIS[base + lane];
        }
    }

    for (int lane = 0; lane < LANES; ++lane) {
        s2[lane] = s1[lane] ^ D_ALPHA0[lane];
    }
}

__device__ __forceinline__ bool diff_equal_constant(
    const std::uint64_t s1[LANES],
    const std::uint64_t s2[LANES],
    const std::uint64_t target[LANES]
) {
    for (int lane = 0; lane < LANES; ++lane) {
        if ((s1[lane] ^ s2[lane]) != target[lane]) {
            return false;
        }
    }
    return true;
}

__device__ __forceinline__ bool diff_digest_zero(
    const std::uint64_t s1[LANES],
    const std::uint64_t s2[LANES],
    int digest_bits
) {
    int full_words = digest_bits / 64;
    int rem = digest_bits % 64;
    for (int lane = 0; lane < full_words; ++lane) {
        if ((s1[lane] ^ s2[lane]) != 0) {
            return false;
        }
    }
    if (rem == 0) {
        return true;
    }
    std::uint64_t mask = (1ULL << rem) - 1ULL;
    return ((s1[full_words] ^ s2[full_words]) & mask) == 0;
}

__device__ void evaluate_sample(
    std::uint64_t global_sample,
    std::uint64_t seed,
    int digest_bits,
    bool verify_alpha2,
    bool& alpha2,
    bool& alpha3,
    bool& alpha4,
    bool& digest
) {
    std::uint64_t s1[LANES];
    std::uint64_t s2[LANES];
    sample_state_pair(global_sample, seed, s1, s2);

    keccak_round(s1, 0);
    keccak_round(s2, 0);
    keccak_round(s1, 1);
    keccak_round(s2, 1);
    if (verify_alpha2) {
        alpha2 = diff_equal_constant(s1, s2, D_ALPHA2);
        if (!alpha2) {
            return;
        }
    } else {
        alpha2 = true;
    }

    keccak_round(s1, 2);
    keccak_round(s2, 2);
    alpha3 = diff_equal_constant(s1, s2, D_ALPHA3);
    if (!alpha3) {
        return;
    }

    keccak_round(s1, 3);
    keccak_round(s2, 3);
    alpha4 = diff_equal_constant(s1, s2, D_ALPHA4);

    keccak_round(s1, 4);
    keccak_round(s2, 4);
    keccak_round(s1, 5);
    keccak_round(s2, 5);
    digest = diff_digest_zero(s1, s2, digest_bits);
}

__device__ bool evaluate_alpha3_sample(
    std::uint64_t global_sample,
    std::uint64_t seed,
    bool verify_alpha2,
    bool& alpha2
) {
    std::uint64_t s1[LANES];
    std::uint64_t s2[LANES];
    sample_state_pair(global_sample, seed, s1, s2);

    keccak_round(s1, 0);
    keccak_round(s2, 0);
    keccak_round(s1, 1);
    keccak_round(s2, 1);
    if (verify_alpha2) {
        alpha2 = diff_equal_constant(s1, s2, D_ALPHA2);
        if (!alpha2) {
            return false;
        }
    } else {
        alpha2 = true;
    }

    keccak_round(s1, 2);
    keccak_round(s2, 2);
    return diff_equal_constant(s1, s2, D_ALPHA3);
}

__global__ void search_kernel(
    std::uint64_t offset,
    std::uint64_t count,
    std::uint64_t seed,
    int digest_bits,
    bool verify_alpha2,
    DeviceTotals* totals
) {
    std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    std::uint64_t local_samples = 0;
    std::uint64_t local_alpha2 = 0;
    std::uint64_t local_alpha3 = 0;
    std::uint64_t local_alpha4 = 0;
    std::uint64_t local_digest = 0;

    for (std::uint64_t i = tid; i < count; i += stride) {
        std::uint64_t sample_index = offset + i;
        bool alpha2 = false;
        bool alpha3 = false;
        bool alpha4 = false;
        bool digest = false;
        evaluate_sample(sample_index, seed, digest_bits, verify_alpha2, alpha2, alpha3, alpha4, digest);

        local_samples += 1;
        local_alpha2 += alpha2 ? 1 : 0;
        local_alpha3 += alpha3 ? 1 : 0;
        local_alpha4 += alpha4 ? 1 : 0;
        local_digest += digest ? 1 : 0;

        if (alpha3 && atomicCAS(&totals->first_alpha3_written, 0, 1) == 0) {
            totals->first_alpha3_sample = sample_index + 1;
        }

        if (digest && atomicCAS(&totals->found_written, 0, 1) == 0) {
            totals->found_sample = sample_index + 1;
        }
    }

    atomicAdd(&totals->samples, static_cast<unsigned long long>(local_samples));
    atomicAdd(&totals->alpha2_hits, static_cast<unsigned long long>(local_alpha2));
    atomicAdd(&totals->alpha3_hits, static_cast<unsigned long long>(local_alpha3));
    atomicAdd(&totals->alpha4_hits, static_cast<unsigned long long>(local_alpha4));
    atomicAdd(&totals->digest_hits, static_cast<unsigned long long>(local_digest));
}

__global__ void search_alpha3_kernel(
    std::uint64_t offset,
    std::uint64_t count,
    std::uint64_t seed,
    bool verify_alpha2,
    std::uint64_t* alpha3_samples,
    std::uint64_t alpha3_capacity,
    DeviceTotals* totals
) {
    std::uint64_t tid = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;

    std::uint64_t local_samples = 0;
    std::uint64_t local_alpha2 = 0;
    std::uint64_t local_alpha3 = 0;

    for (std::uint64_t i = tid; i < count; i += stride) {
        std::uint64_t sample_index = offset + i;
        bool alpha2 = false;
        bool alpha3 = evaluate_alpha3_sample(sample_index, seed, verify_alpha2, alpha2);

        local_samples += 1;
        local_alpha2 += alpha2 ? 1 : 0;
        local_alpha3 += alpha3 ? 1 : 0;

        if (alpha3) {
            unsigned long long position = atomicAdd(&totals->stored_alpha3_hits, 1ULL);
            if (position < alpha3_capacity) {
                alpha3_samples[position] = sample_index + 1;
            }
            if (atomicCAS(&totals->first_alpha3_written, 0, 1) == 0) {
                totals->first_alpha3_sample = sample_index + 1;
            }
        }
    }

    atomicAdd(&totals->samples, static_cast<unsigned long long>(local_samples));
    atomicAdd(&totals->alpha2_hits, static_cast<unsigned long long>(local_alpha2));
    atomicAdd(&totals->alpha3_hits, static_cast<unsigned long long>(local_alpha3));
}

std::uint64_t parse_u64_arg(int argc, char** argv, const std::string& name, std::uint64_t fallback) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return std::strtoull(argv[i + 1], nullptr, 10);
        }
    }
    return fallback;
}

int parse_int_arg(int argc, char** argv, const std::string& name, int fallback) {
    return static_cast<int>(parse_u64_arg(argc, argv, name, static_cast<std::uint64_t>(fallback)));
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

void check_cuda(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        std::cerr << label << ": " << cudaGetErrorString(status) << "\n";
        std::exit(2);
    }
}

void print_state_hex(const char* label, const std::array<std::uint64_t, LANES>& state) {
    std::cout << label;
    for (int lane = LANES - 1; lane >= 0; --lane) {
        std::cout << std::hex << std::setw(16) << std::setfill('0') << state[lane];
    }
    std::cout << std::dec << "\n";
}

std::uint64_t host_splitmix64_next(std::uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void reconstruct_cuda_sample(
    std::uint64_t sample_number,
    std::uint64_t seed,
    HostState& m1,
    HostState& m2
) {
    std::uint64_t global_sample = sample_number - 1;
    m1 = core3_data::PARTICULAR_MESSAGE;

    std::uint64_t state = seed ^ (0xd1b54a32d192ed03ULL * (global_sample + 1));
    std::uint64_t selectors[3] = {
        host_splitmix64_next(state),
        host_splitmix64_next(state),
        host_splitmix64_next(state),
    };

    for (int basis = 0; basis < BASIS_SIZE; ++basis) {
        bool take = ((selectors[basis >> 6] >> (basis & 63)) & 1ULL) != 0;
        if (!take) {
            continue;
        }
        for (int lane = 0; lane < LANES; ++lane) {
            m1[lane] ^= core3_data::BASIS_MESSAGE[static_cast<std::size_t>(basis)][lane];
        }
    }

    for (int lane = 0; lane < LANES; ++lane) {
        m2[lane] = m1[lane] ^ core3_data::ALPHA0[lane];
    }
}

int h_idx(int x, int y) {
    return x + 5 * y;
}

std::uint64_t h_rotl64(std::uint64_t value, int amount) {
    amount &= 63;
    if (amount == 0) {
        return value;
    }
    return (value << amount) | (value >> (64 - amount));
}

void h_theta(HostState& state) {
    std::uint64_t columns[5] = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            columns[x] ^= state[h_idx(x, y)];
        }
    }
    std::uint64_t delta[5];
    for (int x = 0; x < 5; ++x) {
        delta[x] = columns[(x + 4) % 5] ^ h_rotl64(columns[(x + 1) % 5], 1);
    }
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            state[h_idx(x, y)] ^= delta[x];
        }
    }
}

void h_rho_pi(HostState& state) {
    static constexpr int rotation[5][5] = {
        {0, 36, 3, 41, 18},
        {1, 44, 10, 45, 2},
        {62, 6, 43, 15, 61},
        {28, 55, 25, 21, 56},
        {27, 20, 39, 8, 14},
    };
    HostState moved = {};
    for (int x = 0; x < 5; ++x) {
        for (int y = 0; y < 5; ++y) {
            int nx = y;
            int ny = (2 * x + 3 * y) % 5;
            moved[h_idx(nx, ny)] = h_rotl64(state[h_idx(x, y)], rotation[x][y]);
        }
    }
    state = moved;
}

void h_chi(HostState& state) {
    for (int y = 0; y < 5; ++y) {
        std::uint64_t row[5];
        for (int x = 0; x < 5; ++x) {
            row[x] = state[h_idx(x, y)];
        }
        for (int x = 0; x < 5; ++x) {
            state[h_idx(x, y)] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
        }
    }
}

void h_round(HostState& state, int round_number) {
    h_theta(state);
    h_rho_pi(state);
    h_chi(state);
    state[0] ^= ROUND_CONSTANTS_HOST[static_cast<std::size_t>(round_number)];
}

bool h_diff_equal(const HostState& s1, const HostState& s2, const HostState& target) {
    for (int lane = 0; lane < LANES; ++lane) {
        if ((s1[lane] ^ s2[lane]) != target[lane]) {
            return false;
        }
    }
    return true;
}

bool h_digest_zero(const HostState& s1, const HostState& s2, int digest_bits) {
    int full_words = digest_bits / 64;
    int rem = digest_bits % 64;
    for (int lane = 0; lane < full_words; ++lane) {
        if ((s1[lane] ^ s2[lane]) != 0) {
            return false;
        }
    }
    if (rem == 0) {
        return true;
    }
    std::uint64_t mask = (1ULL << rem) - 1ULL;
    return ((s1[full_words] ^ s2[full_words]) & mask) == 0;
}

struct HostEvalResult {
    bool alpha3 = false;
    bool alpha4 = false;
    bool digest = false;
};

HostEvalResult h_evaluate_full(const HostState& m1, const HostState& m2, int digest_bits) {
    HostState s1 = m1;
    HostState s2 = m2;
    h_round(s1, 0);
    h_round(s2, 0);
    h_round(s1, 1);
    h_round(s2, 1);
    h_round(s1, 2);
    h_round(s2, 2);

    HostEvalResult result;
    result.alpha3 = h_diff_equal(s1, s2, core3_data::ALPHA3);
    if (!result.alpha3) {
        return result;
    }

    h_round(s1, 3);
    h_round(s2, 3);
    result.alpha4 = h_diff_equal(s1, s2, core3_data::ALPHA4);

    h_round(s1, 4);
    h_round(s2, 4);
    h_round(s1, 5);
    h_round(s2, 5);
    result.digest = h_digest_zero(s1, s2, digest_bits);
    return result;
}

void append_candidate_file(
    const std::string& path,
    const std::string& label,
    const std::array<std::uint64_t, LANES>& m1,
    const std::array<std::uint64_t, LANES>& m2
) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "warning: could not open candidate file: " << path << "\n";
        return;
    }
    out << label << "\nM1 ";
    for (int lane = LANES - 1; lane >= 0; --lane) {
        out << std::hex << std::setw(16) << std::setfill('0') << m1[lane];
    }
    out << "\nM2 ";
    for (int lane = LANES - 1; lane >= 0; --lane) {
        out << std::hex << std::setw(16) << std::setfill('0') << m2[lane];
    }
    out << std::dec << "\n";
}

std::vector<int> parse_devices(const std::string& text) {
    int count = 0;
    check_cuda(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count <= 0) {
        std::cerr << "no CUDA devices found\n";
        std::exit(2);
    }
    if (text == "all") {
        std::vector<int> devices;
        for (int device = 0; device < count; ++device) {
            devices.push_back(device);
        }
        return devices;
    }

    std::vector<int> devices;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) {
            continue;
        }
        int device = std::stoi(item);
        if (device < 0 || device >= count) {
            std::cerr << "invalid CUDA device " << device << ", available range: 0.." << (count - 1) << "\n";
            std::exit(2);
        }
        devices.push_back(device);
    }
    if (devices.empty()) {
        devices.push_back(0);
    }
    return devices;
}

void copy_constants_to_device() {
    check_cuda(cudaMemcpyToSymbol(D_PARTICULAR, core3_data::PARTICULAR_MESSAGE.data(), sizeof(std::uint64_t) * LANES), "copy particular");
    check_cuda(cudaMemcpyToSymbol(D_ALPHA0, core3_data::ALPHA0.data(), sizeof(std::uint64_t) * LANES), "copy alpha0");
    check_cuda(cudaMemcpyToSymbol(D_ALPHA2, core3_data::ALPHA2.data(), sizeof(std::uint64_t) * LANES), "copy alpha2");
    check_cuda(cudaMemcpyToSymbol(D_ALPHA3, core3_data::ALPHA3.data(), sizeof(std::uint64_t) * LANES), "copy alpha3");
    check_cuda(cudaMemcpyToSymbol(D_ALPHA4, core3_data::ALPHA4.data(), sizeof(std::uint64_t) * LANES), "copy alpha4");
    check_cuda(cudaMemcpyToSymbol(D_ROUND_CONSTANTS, ROUND_CONSTANTS_HOST.data(), sizeof(std::uint64_t) * ROUND_CONSTANTS_HOST.size()), "copy round constants");
    check_cuda(
        cudaMemcpyToSymbol(D_BASIS, &core3_data::BASIS_MESSAGE[0][0], sizeof(std::uint64_t) * BASIS_SIZE * LANES),
        "copy basis"
    );
}

DeviceResult run_device(const Options& options, int device, std::uint64_t offset, std::uint64_t count) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    copy_constants_to_device();

    cudaDeviceProp prop;
    check_cuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    int blocks = prop.multiProcessorCount * options.blocks_per_sm;

    DeviceTotals* d_totals = nullptr;
    check_cuda(cudaMalloc(&d_totals, sizeof(DeviceTotals)), "cudaMalloc totals");
    check_cuda(cudaMemset(d_totals, 0, sizeof(DeviceTotals)), "cudaMemset totals");

    std::uint64_t* d_alpha3_samples = nullptr;
    std::uint64_t alpha3_capacity = options.alpha3_only ? options.max_stored_alpha3 : 0;
    if (alpha3_capacity) {
        check_cuda(cudaMalloc(&d_alpha3_samples, sizeof(std::uint64_t) * alpha3_capacity), "cudaMalloc alpha3 samples");
    }

    if (options.alpha3_only) {
        search_alpha3_kernel<<<blocks, options.threads_per_block>>>(
            offset,
            count,
            options.seed,
            options.verify_alpha2,
            d_alpha3_samples,
            alpha3_capacity,
            d_totals
        );
        check_cuda(cudaGetLastError(), "launch search_alpha3_kernel");
    } else {
        search_kernel<<<blocks, options.threads_per_block>>>(
            offset,
            count,
            options.seed,
            options.digest_bits,
            options.verify_alpha2,
            d_totals
        );
        check_cuda(cudaGetLastError(), "launch search_kernel");
    }
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    DeviceTotals host{};
    check_cuda(cudaMemcpy(&host, d_totals, sizeof(DeviceTotals), cudaMemcpyDeviceToHost), "copy totals");

    std::vector<std::uint64_t> alpha3_samples;
    if (options.alpha3_only && d_alpha3_samples) {
        std::uint64_t stored = std::min<std::uint64_t>(host.stored_alpha3_hits, alpha3_capacity);
        alpha3_samples.resize(static_cast<std::size_t>(stored));
        if (stored) {
            check_cuda(
                cudaMemcpy(alpha3_samples.data(), d_alpha3_samples, sizeof(std::uint64_t) * stored, cudaMemcpyDeviceToHost),
                "copy alpha3 samples"
            );
        }
        check_cuda(cudaFree(d_alpha3_samples), "cudaFree alpha3 samples");
    }
    check_cuda(cudaFree(d_totals), "cudaFree totals");

    DeviceResult result;
    result.totals.samples = host.samples;
    result.totals.alpha2_hits = host.alpha2_hits;
    result.totals.alpha3_hits = host.alpha3_hits;
    result.totals.alpha4_hits = host.alpha4_hits;
    result.totals.digest_hits = host.digest_hits;
    result.has_alpha3 = host.first_alpha3_written != 0;
    result.has_found = host.found_written != 0;
    result.first_alpha3_sample = host.first_alpha3_sample;
    result.found_sample = host.found_sample;
    if (result.has_alpha3) {
        reconstruct_cuda_sample(result.first_alpha3_sample, options.seed, result.alpha3_m1, result.alpha3_m2);
    }
    if (result.has_found) {
        reconstruct_cuda_sample(result.found_sample, options.seed, result.found_m1, result.found_m2);
    }
    result.alpha3_samples = std::move(alpha3_samples);
    return result;
}

int main(int argc, char** argv) {
    Options options;
    options.samples = parse_u64_arg(argc, argv, "--samples", options.samples);
    options.seed = parse_u64_arg(argc, argv, "--seed", options.seed);
    options.digest_bits = parse_int_arg(argc, argv, "--digest-bits", options.digest_bits);
    options.threads_per_block = parse_int_arg(argc, argv, "--threads-per-block", options.threads_per_block);
    options.blocks_per_sm = parse_int_arg(argc, argv, "--blocks-per-sm", options.blocks_per_sm);
    options.devices = parse_string_arg(argc, argv, "--devices", options.devices);
    options.candidate_file = parse_string_arg(argc, argv, "--candidate-file", "");
    options.verify_alpha2 = has_flag(argc, argv, "--verify-alpha2");
    options.alpha3_only = has_flag(argc, argv, "--alpha3-only");
    options.max_alpha3_print = parse_u64_arg(argc, argv, "--max-alpha3-print", options.max_alpha3_print);
    options.max_stored_alpha3 = parse_u64_arg(argc, argv, "--max-stored-alpha3", options.max_stored_alpha3);

    std::vector<int> devices = parse_devices(options.devices);
    std::cout << "CUDA core3 Keccak[1440,160,6,160] trail search\n";
    std::cout << "  samples=" << options.samples << ", seed=" << options.seed
              << ", digest_bits=" << options.digest_bits
              << ", basis=" << BASIS_SIZE
              << ", devices=" << options.devices
              << ", threads_per_block=" << options.threads_per_block
              << ", blocks_per_sm=" << options.blocks_per_sm
              << ", verify_alpha2=" << options.verify_alpha2
              << ", alpha3_only=" << options.alpha3_only
              << ", max_alpha3_print=" << options.max_alpha3_print << "\n";

    std::vector<DeviceResult> results(devices.size());
    std::vector<std::thread> workers;
    workers.reserve(devices.size());
    std::mutex result_mutex;
    auto start = std::chrono::steady_clock::now();

    std::uint64_t base = 0;
    for (std::size_t i = 0; i < devices.size(); ++i) {
        std::uint64_t remaining = options.samples - base;
        std::uint64_t count = remaining / (devices.size() - i);
        std::uint64_t offset = base;
        base += count;
        workers.emplace_back([&, i, offset, count]() {
            DeviceResult result = run_device(options, devices[i], offset, count);
            std::lock_guard<std::mutex> lock(result_mutex);
            results[i] = result;
        });
    }

    for (std::thread& worker : workers) {
        worker.join();
    }

    auto stop = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(stop - start).count();

    HostTotals totals;
    std::uint64_t stored_alpha3_total = 0;
    std::uint64_t printed_alpha3 = 0;
    HostTotals post_totals;
    for (std::size_t i = 0; i < results.size(); ++i) {
        DeviceResult& result = results[i];
        totals.samples += result.totals.samples;
        totals.alpha2_hits += result.totals.alpha2_hits;
        totals.alpha3_hits += result.totals.alpha3_hits;
        totals.alpha4_hits += result.totals.alpha4_hits;
        totals.digest_hits += result.totals.digest_hits;
        stored_alpha3_total += result.alpha3_samples.size();

        if (result.has_alpha3 && printed_alpha3 < options.max_alpha3_print) {
            ++printed_alpha3;
            std::cout << "  device " << devices[i] << " first alpha3 sample=" << result.first_alpha3_sample << "\n";
            print_state_hex("    M1: ", result.alpha3_m1);
            print_state_hex("    M2: ", result.alpha3_m2);
        }
        if (result.has_found) {
            std::ostringstream label;
            label << "FOUND digest candidate sample=" << result.found_sample << " device=" << devices[i];
            std::cout << "  " << label.str() << "\n";
            print_state_hex("    M1: ", result.found_m1);
            print_state_hex("    M2: ", result.found_m2);
            append_candidate_file(options.candidate_file, label.str(), result.found_m1, result.found_m2);
        }

        if (options.alpha3_only) {
            for (std::uint64_t sample_number : result.alpha3_samples) {
                HostState m1;
                HostState m2;
                reconstruct_cuda_sample(sample_number, options.seed, m1, m2);
                HostEvalResult eval = h_evaluate_full(m1, m2, options.digest_bits);
                post_totals.alpha3_hits += eval.alpha3 ? 1 : 0;
                post_totals.alpha4_hits += eval.alpha4 ? 1 : 0;
                post_totals.digest_hits += eval.digest ? 1 : 0;
                if (eval.digest) {
                    std::ostringstream label;
                    label << "FOUND digest candidate sample=" << sample_number << " device=" << devices[i];
                    std::cout << "  " << label.str() << "\n";
                    print_state_hex("    M1: ", m1);
                    print_state_hex("    M2: ", m2);
                    append_candidate_file(options.candidate_file, label.str(), m1, m2);
                }
            }
        }
    }

    if (options.alpha3_only) {
        totals.alpha4_hits = post_totals.alpha4_hits;
        totals.digest_hits = post_totals.digest_hits;
    }

    std::cout << "summary\n";
    std::cout << "  samples=" << totals.samples << "\n";
    std::cout << "  alpha2_hits=" << totals.alpha2_hits << "\n";
    std::cout << "  alpha3_hits=" << totals.alpha3_hits << "\n";
    std::cout << "  alpha4_hits=" << totals.alpha4_hits << "\n";
    std::cout << "  digest_hits=" << totals.digest_hits << "\n";
    if (options.alpha3_only) {
        std::cout << "  stored_alpha3_hits=" << stored_alpha3_total << "\n";
        std::cout << "  post_alpha3_verified=" << post_totals.alpha3_hits << "\n";
        if (stored_alpha3_total < totals.alpha3_hits) {
            std::cout << "  warning=alpha3 hit buffer full; increase --max-stored-alpha3 for complete post-processing\n";
        }
    }
    std::cout << "  elapsed=" << std::fixed << std::setprecision(3) << elapsed << "s\n";
    if (elapsed > 0.0) {
        std::cout << "  rate=" << std::setprecision(2) << (static_cast<double>(totals.samples) / elapsed / 1000000.0) << " M/s\n";
    }
    return 0;
}
