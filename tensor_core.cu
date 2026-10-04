
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace nvcuda;

constexpr int N = 64;
constexpr int TILE = 16;
constexpr int NUM_TILES = N / TILE;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << '\n'; \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void tensor_core_matmul(const half* A, const half* B, float* C) {
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;

    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half,
                   wmma::row_major> a_frag;

    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half,
                   wmma::row_major> b_frag;

    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int k_tile = 0; k_tile < NUM_TILES; ++k_tile) {
        const half* a_ptr =
            A + tile_row * TILE * N + k_tile * TILE;

        const half* b_ptr =
            B + k_tile * TILE * N + tile_col * TILE;

        wmma::load_matrix_sync(a_frag, a_ptr, N);
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr =
        C + tile_row * TILE * N + tile_col * TILE;

    wmma::store_matrix_sync(
        c_ptr, c_frag, N, wmma::mem_row_major
    );
}

int main() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << '\n';

    if (prop.major < 7) {
        std::cerr
            << "This program requires a GPU with Tensor Core / WMMA support "
               "(compute capability 7.0 or newer).\n";

        return EXIT_FAILURE;
    }

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C(N * N, 0.0f);

    std::vector<half> h_A_half(N * N);
    std::vector<half> h_B_half(N * N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);

        h_A_half[i] = __float2half(h_A[i]);
        h_B_half[i] = __float2half(h_B[i]);
    }

    half *d_A = nullptr;
    half *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_A),
        N * N * sizeof(half)
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_B),
        N * N * sizeof(half)
    ));

    CUDA_CHECK(cudaMalloc(
        reinterpret_cast<void**>(&d_C),
        N * N * sizeof(float)
    ));

    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A_half.data(),
        N * N * sizeof(half),
        cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B_half.data(),
        N * N * sizeof(half),
        cudaMemcpyHostToDevice
    ));

    dim3 grid(NUM_TILES, NUM_TILES);
    dim3 block(32);

    tensor_core_matmul<<<grid, block>>>(d_A, d_B, d_C);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C.data(),
        d_C,
        N * N * sizeof(float),
        cudaMemcpyDeviceToHost
    ));

    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    bool passed = true;

    constexpr double ABS_TOL = 2e-2;
    constexpr double REL_TOL = 2e-2;

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {

            float reference = 0.0f;

            for (int k = 0; k < N; ++k) {
                reference +=
                    __half2float(h_A_half[row * N + k]) *
                    __half2float(h_B_half[k * N + col]);
            }

            const double got = h_C[row * N + col];

            const double abs_error =
                std::abs(got - reference);

            const double rel_error =
                abs_error /
                std::max(
                    std::abs(static_cast<double>(reference)),
                    1e-6
                );

            max_abs_error =
                std::max(max_abs_error, abs_error);

            max_rel_error =
                std::max(max_rel_error, rel_error);

            if (abs_error > ABS_TOL &&
                rel_error > REL_TOL) {
                passed = false;
            }
        }
    }

    std::cout << "\nMatrix dimensions: "
              << N << " x " << N << '\n';

    std::cout << "Tile dimensions:   "
              << TILE << " x " << TILE << '\n';

    std::cout << "Tiles per matrix:  "
              << NUM_TILES * NUM_TILES << '\n';

    std::cout << "Output tiles:      "
              << NUM_TILES * NUM_TILES << '\n';

    std::cout << "WMMA operations:   "
              << NUM_TILES * NUM_TILES * NUM_TILES << '\n';

    std::cout << std::scientific
              << std::setprecision(6);

    std::cout << "Maximum absolute error: "
              << max_abs_error << '\n';

    std::cout << "Maximum relative error: "
              << max_rel_error << '\n';

    std::cout << "Validation: "
              << (passed ? "PASSED" : "FAILED")
              << '\n';

    std::cout << "\nTop-left 4x4 values of C:\n";

    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            std::cout
                << std::fixed
                << std::setprecision(4)
                << h_C[row * N + col]
                << ' ';
        }

        std::cout << '\n';
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
