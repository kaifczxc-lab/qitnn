/*
Readable CUDA QITNN Code
This is a refactored version of the kernel used in the experiments. For the exact original code check qitnn/cuda-kernels/qitnnRaw
This code can be run with the command
nvcc -O3 -std=c++17 qitnnRefactored -o qitnnRefactored
in the X64 Native Tools Command Prompt for VS 2022
I tested the code with the following results:
===================================================
qitnn cuda core tests
Q1 pure|+1> | E=1.0000 ref=1.0000 | P=(0.0000, 0.0000, 1.0000) sum=1.000000 PASS
Q2 pure|-1> | E=-1.0000 ref=-1.0000 | P=(1.0000, 0.0000, 0.0000) sum=1.000000 PASS
Q3 pure|0> | E=0.0000 ref=0.0000 | P=(0.0000, 1.0000, 0.0000) sum=1.000000 PASS
Q4 sup(3,0,4) | E=0.2800 ref=0.2800 | P=(0.3600, 0.0000, 0.6400) sum=1.000000 PASS
Q5 equal(1,1,1) | E=0.0000 ref=0.0000 | P=(0.3333, 0.3333, 0.3333) sum=1.000000 PASS
Q6 sign(5,0,-5) | E=0.0000 ref=0.0000 | P=(0.5000, 0.0000, 0.5000) sum=1.000000 PASS
Q7 bias(1,2,10) | E=0.9429 ref=0.9429 | P=(0.0095, 0.0381, 0.9524) sum=1.000000 PASS
results: ok
<(^-^)>
===================================================
Disclaimer: The code may be unprocessed and may not run on other video cards or PCs
*/

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
constexpr int kThreads1D = 256;
constexpr int kTile = 16;
constexpr float kBornEps = 1.0e-12f;
constexpr float kNormEps = 1.0e-6f;

inline void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(
            stderr,
            "CUDA error: %s\n  expr: %s\n  file: %s:%d\n",
            cudaGetErrorString(err),
            expr,
            file,
            line);
        std::fflush(stderr);
        std::abort();
    }
}

#define QTS_CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)

__device__ __forceinline__ unsigned int xorshift32(unsigned int x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void rand_fill_kernel(float* out, int n, unsigned int seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int s = seed ^ static_cast<unsigned int>(i * 2654435761u);
    s = xorshift32(s);
    s = xorshift32(s);

    out[i] = static_cast<float>(s & 0xFFFFu) * (1.0f / 32768.0f) - 1.0f;
}

__global__ void rand_ternary_kernel(float* out, int n, unsigned int seed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    unsigned int s = seed ^ static_cast<unsigned int>(i * 2654435761u);
    s = xorshift32(s);
    s = xorshift32(s);

    out[i] = static_cast<float>(static_cast<int>(s % 3u) - 1);
}

__global__ void normalize_amplitudes_kernel(float* a_neg, float* a_zero, float* a_pos, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float an = a_neg[i];
    const float az = a_zero[i];
    const float ap = a_pos[i];

    const float norm = sqrtf(an * an + az * az + ap * ap);
    const float inv = (norm > kNormEps) ? (1.0f / norm) : 0.57735026919f;

    a_neg[i] = an * inv;
    a_zero[i] = az * inv;
    a_pos[i] = ap * inv;
}

__global__ void ternary_to_amplitudes_kernel(
    const float* ternary,
    float* a_neg,
    float* a_zero,
    float* a_pos,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float v = ternary[i];
    a_neg[i] = (v < -0.5f) ? 1.0f : 0.0f;
    a_zero[i] = (v > -0.5f && v < 0.5f) ? 1.0f : 0.0f;
    a_pos[i] = (v > 0.5f) ? 1.0f : 0.0f;
}

__global__ void transpose_kernel(const float* in, float* out, int rows, int cols) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = rows * cols;
    if (i >= n) {
        return;
    }

    const int r = i / cols;
    const int c = i % cols;
    out[c * rows + r] = in[r * cols + c];
}


__global__ void sgd_kernel(float* w, const float* g, float lr, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] -= lr * g[i];
    }
}

__global__ void scale_kernel(float* w, float factor, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        w[i] *= factor;
    }
}

__global__ void zero_fill_kernel(float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 0.0f;
    }
}

__global__ void naive_gemm_kernel(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

__global__ void born_normalize_kernel(
    const float* c_neg,
    const float* c_zero,
    const float* c_pos,
    float* p_neg,
    float* p_zero,
    float* p_pos,
    float* ev,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float an = c_neg[i];
    const float az = c_zero[i];
    const float ap = c_pos[i];

    float qn = an * an;
    float qz = az * az;
    float qp = ap * ap;

    const float z = qn + qz + qp;
    const float inv_z = (z > kBornEps) ? (1.0f / z) : 0.0f;

    qn *= inv_z;
    qz *= inv_z;
    qp *= inv_z;

    p_neg[i] = qn;
    p_zero[i] = qz;
    p_pos[i] = qp;
    ev[i] = qp - qn;
}

__global__ void backward_born_kernel(
    const float* d_ev,
    const float* c_neg,
    const float* c_zero,
    const float* c_pos,
    float* d_c_neg,
    float* d_c_zero,
    float* d_c_pos,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float g = d_ev[i];
    const float a = c_neg[i];
    const float b = c_zero[i];
    const float c = c_pos[i];

    const float a2 = a * a;
    const float b2 = b * b;
    const float c2 = c * c;
    const float z = a2 + b2 + c2;
    const float inv_z2 = (z > kBornEps) ? (1.0f / (z * z)) : 0.0f;

    d_c_neg[i] = g * (-2.0f * a * (b2 + 2.0f * c2)) * inv_z2;
    d_c_zero[i] = g * (-2.0f * b * (c2 - a2)) * inv_z2;
    d_c_pos[i] = g * ( 2.0f * c * (2.0f * a2 + b2)) * inv_z2;
}

__global__ void mse_grad_kernel(
    const float* pred,
    const float* target,
    float* grad,
    float* loss_buf,
    int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float d = pred[i] - target[i];
    grad[i] = d;
    loss_buf[i] = d * d;
}

__global__ void sum_reduce_atomic_kernel(const float* in, float* out, int n) {
    __shared__ float scratch[kThreads1D];

    const int tid = threadIdx.x;
    const int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    float local = 0.0f;
    for (int i = idx; i < n; i += stride) {
        local += in[i];
    }

    scratch[tid] = local;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            scratch[tid] += scratch[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, scratch[0]);
    }
}

inline int blocks_1d(int n) {
    return (n + kThreads1D - 1) / kThreads1D;
}

inline dim3 gemm_block() {
    return dim3(kTile, kTile);
}

inline dim3 gemm_grid(int M, int N) {
    return dim3((N + kTile - 1) / kTile, (M + kTile - 1) / kTile);
}

extern "C" void qts_forward_device(
    const float* d_x,
    const float* d_a_neg,
    const float* d_a_zero,
    const float* d_a_pos,
    float* d_c_neg,
    float* d_c_zero,
    float* d_c_pos,
    float* d_p_neg,
    float* d_p_zero,
    float* d_p_pos,
    float* d_ev,
    int M,
    int K,
    int N) {
    const dim3 block = gemm_block();
    const dim3 grid = gemm_grid(M, N);

    naive_gemm_kernel<<<grid, block>>>(d_x, d_a_neg, d_c_neg, M, N, K);
    naive_gemm_kernel<<<grid, block>>>(d_x, d_a_zero, d_c_zero, M, N, K);
    naive_gemm_kernel<<<grid, block>>>(d_x, d_a_pos, d_c_pos, M, N, K);
    born_normalize_kernel<<<blocks_1d(M * N), kThreads1D>>>(
        d_c_neg,
        d_c_zero,
        d_c_pos,
        d_p_neg,
        d_p_zero,
        d_p_pos,
        d_ev,
        M * N);

    QTS_CUDA_CHECK(cudaGetLastError());
}

extern "C" void qts_init_random_amplitudes_device(
    float* d_a_neg,
    float* d_a_zero,
    float* d_a_pos,
    int n,
    unsigned int seed_base) {
    rand_fill_kernel<<<blocks_1d(n), kThreads1D>>>(d_a_neg, n, seed_base + 0u);
    rand_fill_kernel<<<blocks_1d(n), kThreads1D>>>(d_a_zero, n, seed_base + 1u);
    rand_fill_kernel<<<blocks_1d(n), kThreads1D>>>(d_a_pos, n, seed_base + 2u);
    normalize_amplitudes_kernel<<<blocks_1d(n), kThreads1D>>>(d_a_neg, d_a_zero, d_a_pos, n);

    QTS_CUDA_CHECK(cudaGetLastError());
}

extern "C" double qts_train_demo(int M, int K, int N, int epochs) {
    const int x_elems = M * K;
    const int w_elems = K * N;
    const int y_elems = M * N;

    const size_t x_bytes = static_cast<size_t>(x_elems) * sizeof(float);
    const size_t w_bytes = static_cast<size_t>(w_elems) * sizeof(float);
    const size_t y_bytes = static_cast<size_t>(y_elems) * sizeof(float);

    float* d_x = nullptr;
    float* d_w_true = nullptr;
    float* d_y = nullptr;

    float* d_a_neg = nullptr;
    float* d_a_zero = nullptr;
    float* d_a_pos = nullptr;

    float* d_c_neg = nullptr;
    float* d_c_zero = nullptr;
    float* d_c_pos = nullptr;
    float* d_p_neg = nullptr;
    float* d_p_zero = nullptr;
    float* d_p_pos = nullptr;
    float* d_ev = nullptr;

    float* d_d_ev = nullptr;
    float* d_d_c_neg = nullptr;
    float* d_d_c_zero = nullptr;
    float* d_d_c_pos = nullptr;

    float* d_g_neg = nullptr;
    float* d_g_zero = nullptr;
    float* d_g_pos = nullptr;

    float* d_x_t = nullptr;
    float* d_loss_buf = nullptr;
    float* d_loss_sum = nullptr;

    float* d_a_neg_true = nullptr;
    float* d_a_zero_true = nullptr;
    float* d_a_pos_true = nullptr;

    QTS_CUDA_CHECK(cudaMalloc(&d_x, x_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_w_true, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_y, y_bytes));

    QTS_CUDA_CHECK(cudaMalloc(&d_a_neg, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_a_zero, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_a_pos, w_bytes));

    QTS_CUDA_CHECK(cudaMalloc(&d_c_neg, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_c_zero, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_c_pos, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_neg, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_zero, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_pos, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_ev, y_bytes));

    QTS_CUDA_CHECK(cudaMalloc(&d_d_ev, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_d_c_neg, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_d_c_zero, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_d_c_pos, y_bytes));

    QTS_CUDA_CHECK(cudaMalloc(&d_g_neg, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_g_zero, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_g_pos, w_bytes));

    QTS_CUDA_CHECK(cudaMalloc(&d_x_t, static_cast<size_t>(K) * M * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_loss_buf, y_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_loss_sum, sizeof(float)));

    QTS_CUDA_CHECK(cudaMalloc(&d_a_neg_true, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_a_zero_true, w_bytes));
    QTS_CUDA_CHECK(cudaMalloc(&d_a_pos_true, w_bytes));

    rand_fill_kernel<<<blocks_1d(x_elems), kThreads1D>>>(d_x, x_elems, 0xCAFE0001u);
    rand_ternary_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_w_true, w_elems, 0xBEEF0001u);

    ternary_to_amplitudes_kernel<<<blocks_1d(w_elems), kThreads1D>>>(
        d_w_true,
        d_a_neg_true,
        d_a_zero_true,
        d_a_pos_true,
        w_elems);

    qts_forward_device(
        d_x,
        d_a_neg_true,
        d_a_zero_true,
        d_a_pos_true,
        d_c_neg,
        d_c_zero,
        d_c_pos,
        d_p_neg,
        d_p_zero,
        d_p_pos,
        d_y,
        M,
        K,
        N);

    qts_init_random_amplitudes_device(d_a_neg, d_a_zero, d_a_pos, w_elems, 0xFACE0001u);
    transpose_kernel<<<blocks_1d(x_elems), kThreads1D>>>(d_x, d_x_t, M, K);

    QTS_CUDA_CHECK(cudaGetLastError());
    QTS_CUDA_CHECK(cudaDeviceSynchronize());

    float lr = 0.001f;
    float host_loss = 0.0f;

    const dim3 block = gemm_block();
    const dim3 grid_fwd = gemm_grid(M, N);
    const dim3 grid_wgr = gemm_grid(K, N);
    const int loss_blocks = std::max(1, std::min(blocks_1d(y_elems), 1024));

    std::fprintf(stderr, "QTS demo: M=%d K=%d N=%d epochs=%d\n", M, K, N, epochs);

    for (int ep = 0; ep < epochs; ++ep) {
        naive_gemm_kernel<<<grid_fwd, block>>>(d_x, d_a_neg, d_c_neg, M, N, K);
        naive_gemm_kernel<<<grid_fwd, block>>>(d_x, d_a_zero, d_c_zero, M, N, K);
        naive_gemm_kernel<<<grid_fwd, block>>>(d_x, d_a_pos, d_c_pos, M, N, K);
        born_normalize_kernel<<<blocks_1d(y_elems), kThreads1D>>>(
            d_c_neg,
            d_c_zero,
            d_c_pos,
            d_p_neg,
            d_p_zero,
            d_p_pos,
            d_ev,
            y_elems);

        mse_grad_kernel<<<blocks_1d(y_elems), kThreads1D>>>(d_ev, d_y, d_d_ev, d_loss_buf, y_elems);
        zero_fill_kernel<<<1, 1>>>(d_loss_sum, 1);
        sum_reduce_atomic_kernel<<<loss_blocks, kThreads1D>>>(d_loss_buf, d_loss_sum, y_elems);

        backward_born_kernel<<<blocks_1d(y_elems), kThreads1D>>>(
            d_d_ev,
            d_c_neg,
            d_c_zero,
            d_c_pos,
            d_d_c_neg,
            d_d_c_zero,
            d_d_c_pos,
            y_elems);

        naive_gemm_kernel<<<grid_wgr, block>>>(d_x_t, d_d_c_neg, d_g_neg, K, N, M);
        naive_gemm_kernel<<<grid_wgr, block>>>(d_x_t, d_d_c_zero, d_g_zero, K, N, M);
        naive_gemm_kernel<<<grid_wgr, block>>>(d_x_t, d_d_c_pos, d_g_pos, K, N, M);

        sgd_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_neg, d_g_neg, lr, w_elems);
        sgd_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_zero, d_g_zero, lr, w_elems);
        sgd_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_pos, d_g_pos, lr, w_elems);

        scale_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_neg, 0.999f, w_elems);
        scale_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_zero, 0.999f, w_elems);
        scale_kernel<<<blocks_1d(w_elems), kThreads1D>>>(d_a_pos, 0.999f, w_elems);

        QTS_CUDA_CHECK(cudaGetLastError());

        const int log_step = (epochs / 10 > 0) ? (epochs / 10) : 1;
        if (ep % log_step == 0 || ep == epochs - 1) {
            QTS_CUDA_CHECK(cudaDeviceSynchronize());
            QTS_CUDA_CHECK(cudaMemcpy(&host_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
            host_loss /= static_cast<float>(y_elems);
            std::fprintf(stderr, "  [ep %4d/%d] mse=%.6f\n", ep, epochs, host_loss);
        }

        if (ep == epochs * 7 / 10) {
            lr *= 0.3f;
            std::fprintf(stderr, "  lr=%.6f\n", lr);
        }
        if (ep == epochs * 9 / 10) {
            lr *= 0.3f;
            std::fprintf(stderr, "  lr=%.6f\n", lr);
        }
    }

    QTS_CUDA_CHECK(cudaDeviceSynchronize());
    QTS_CUDA_CHECK(cudaMemcpy(&host_loss, d_loss_sum, sizeof(float), cudaMemcpyDeviceToHost));
    host_loss /= static_cast<float>(y_elems);

    std::vector<float> h_a_neg(w_elems);
    std::vector<float> h_a_zero(w_elems);
    std::vector<float> h_a_pos(w_elems);
    std::vector<float> h_w_true(w_elems);

    QTS_CUDA_CHECK(cudaMemcpy(h_a_neg.data(), d_a_neg, w_bytes, cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_a_zero.data(), d_a_zero, w_bytes, cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_a_pos.data(), d_a_pos, w_bytes, cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_w_true.data(), d_w_true, w_bytes, cudaMemcpyDeviceToHost));

    int match = 0;
    int cnt_neg = 0;
    int cnt_zero = 0;
    int cnt_pos = 0;

    for (int i = 0; i < w_elems; ++i) {
        const float an2 = h_a_neg[i] * h_a_neg[i];
        const float az2 = h_a_zero[i] * h_a_zero[i];
        const float ap2 = h_a_pos[i] * h_a_pos[i];

        int collapsed = 0;
        if (an2 >= az2 && an2 >= ap2) {
            collapsed = -1;
            ++cnt_neg;
        } else if (ap2 >= az2 && ap2 >= an2) {
            collapsed = 1;
            ++cnt_pos;
        } else {
            collapsed = 0;
            ++cnt_zero;
        }

        if (collapsed == static_cast<int>(h_w_true[i])) {
            ++match;
        }
    }

    const float accuracy = 100.0f * static_cast<float>(match) / static_cast<float>(w_elems);

    std::fprintf(stderr, "Final mse=%.6f\n", host_loss);
    std::fprintf(stderr, "Collapse accuracy: %d/%d = %.1f%%\n", match, w_elems, accuracy);
    std::fprintf(
        stderr,
        "Collapsed distribution: neg=%d zero=%d pos=%d\n",
        cnt_neg,
        cnt_zero,
        cnt_pos);

    const int preview = std::min(8, w_elems);
    std::fprintf(stderr, "First %d weights:\n", preview);
    for (int i = 0; i < preview; ++i) {
        const float an2 = h_a_neg[i] * h_a_neg[i];
        const float az2 = h_a_zero[i] * h_a_zero[i];
        const float ap2 = h_a_pos[i] * h_a_pos[i];
        const float z = an2 + az2 + ap2;

        const float p_neg = (z > 0.0f) ? (an2 / z) : 0.0f;
        const float p_zero = (z > 0.0f) ? (az2 / z) : 0.0f;
        const float p_pos = (z > 0.0f) ? (ap2 / z) : 0.0f;

        const int collapsed =
            (an2 >= az2 && an2 >= ap2) ? -1 :
            (ap2 >= az2 && ap2 >= an2) ? 1 : 0;

        std::fprintf(
            stderr,
            "  w[%d] true=%+d coll=%+d | P(-1)=%.3f P(0)=%.3f P(+1)=%.3f\n",
            i,
            static_cast<int>(h_w_true[i]),
            collapsed,
            p_neg,
            p_zero,
            p_pos);
    }

    cudaFree(d_x);
    cudaFree(d_w_true);
    cudaFree(d_y);

    cudaFree(d_a_neg);
    cudaFree(d_a_zero);
    cudaFree(d_a_pos);

    cudaFree(d_c_neg);
    cudaFree(d_c_zero);
    cudaFree(d_c_pos);
    cudaFree(d_p_neg);
    cudaFree(d_p_zero);
    cudaFree(d_p_pos);
    cudaFree(d_ev);

    cudaFree(d_d_ev);
    cudaFree(d_d_c_neg);
    cudaFree(d_d_c_zero);
    cudaFree(d_d_c_pos);

    cudaFree(d_g_neg);
    cudaFree(d_g_zero);
    cudaFree(d_g_pos);

    cudaFree(d_x_t);
    cudaFree(d_loss_buf);
    cudaFree(d_loss_sum);

    cudaFree(d_a_neg_true);
    cudaFree(d_a_zero_true);
    cudaFree(d_a_pos_true);

    return static_cast<double>(accuracy);
}

extern "C" int qts_verify_born_normalize() {
    constexpr int kTests = 7;

    const float h_c_neg[kTests]  = {0.0f, 1.0f, 0.0f, 3.0f, 1.0f, 5.0f, 1.0f};
    const float h_c_zero[kTests] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 2.0f};
    const float h_c_pos[kTests]  = {1.0f, 0.0f, 0.0f, 4.0f, 1.0f, -5.0f, 10.0f};

    const float ref_ev[kTests]     = {1.0f, -1.0f, 0.0f, 0.28f, 0.0f, 0.0f, 99.0f / 105.0f};
    const float ref_p_neg[kTests]  = {0.0f, 1.0f, 0.0f, 9.0f / 25.0f, 1.0f / 3.0f, 0.5f, 1.0f / 105.0f};
    const float ref_p_zero[kTests] = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f / 3.0f, 0.0f, 4.0f / 105.0f};
    const float ref_p_pos[kTests]  = {1.0f, 0.0f, 0.0f, 16.0f / 25.0f, 1.0f / 3.0f, 0.5f, 100.0f / 105.0f};

    const char* names[kTests] = {
        "pure|+1>",
        "pure|-1>",
        "pure|0>",
        "sup(3,0,4)",
        "equal(1,1,1)",
        "sign(5,0,-5)",
        "bias(1,2,10)"
    };

    float* d_c_neg  = nullptr;
    float* d_c_zero = nullptr;
    float* d_c_pos  = nullptr;
    float* d_p_neg  = nullptr;
    float* d_p_zero = nullptr;
    float* d_p_pos  = nullptr;
    float* d_ev     = nullptr;

    QTS_CUDA_CHECK(cudaMalloc(&d_c_neg,  kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_c_zero, kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_c_pos,  kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_neg,  kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_zero, kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_p_pos,  kTests * sizeof(float)));
    QTS_CUDA_CHECK(cudaMalloc(&d_ev,     kTests * sizeof(float)));

    QTS_CUDA_CHECK(cudaMemcpy(d_c_neg,  h_c_neg,  kTests * sizeof(float), cudaMemcpyHostToDevice));
    QTS_CUDA_CHECK(cudaMemcpy(d_c_zero, h_c_zero, kTests * sizeof(float), cudaMemcpyHostToDevice));
    QTS_CUDA_CHECK(cudaMemcpy(d_c_pos,  h_c_pos,  kTests * sizeof(float), cudaMemcpyHostToDevice));

    born_normalize_kernel<<<1, kThreads1D>>>(
        d_c_neg,
        d_c_zero,
        d_c_pos,
        d_p_neg,
        d_p_zero,
        d_p_pos,
        d_ev,
        kTests
    );

    QTS_CUDA_CHECK(cudaGetLastError());
    QTS_CUDA_CHECK(cudaDeviceSynchronize());

    float h_p_neg[kTests]  = {};
    float h_p_zero[kTests] = {};
    float h_p_pos[kTests]  = {};
    float h_ev[kTests]     = {};

    QTS_CUDA_CHECK(cudaMemcpy(h_p_neg,  d_p_neg,  kTests * sizeof(float), cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_p_zero, d_p_zero, kTests * sizeof(float), cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_p_pos,  d_p_pos,  kTests * sizeof(float), cudaMemcpyDeviceToHost));
    QTS_CUDA_CHECK(cudaMemcpy(h_ev,     d_ev,     kTests * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;

    std::printf("qitnn cuda core tests\n");

    for (int i = 0; i < kTests; ++i) {
        const float d_ev_abs = fabsf(h_ev[i] - ref_ev[i]);
        const float d_n = fabsf(h_p_neg[i] - ref_p_neg[i]);
        const float d_z = fabsf(h_p_zero[i] - ref_p_zero[i]);
        const float d_p = fabsf(h_p_pos[i] - ref_p_pos[i]);
        const float sum = h_p_neg[i] + h_p_zero[i] + h_p_pos[i];

        const bool ok =
            d_ev_abs < 0.002f &&
            d_n < 0.002f &&
            d_z < 0.002f &&
            d_p < 0.002f &&
            sum > 0.999f &&
            sum < 1.001f;

        if (!ok) {
            ++errors;
        }

        std::printf(
            "Q%d %s | E=%.4f ref=%.4f | P=(%.4f, %.4f, %.4f) sum=%.6f %s\n",
            i + 1,
            names[i],
            h_ev[i],
            ref_ev[i],
            h_p_neg[i],
            h_p_zero[i],
            h_p_pos[i],
            sum,
            ok ? "PASS" : "FAIL"
        );
    }

    std::printf("results: %s\n", errors == 0 ? "ok" : "fail");
    std::printf("<(^-^)>\n");

    cudaFree(d_c_neg);
    cudaFree(d_c_zero);
    cudaFree(d_c_pos);
    cudaFree(d_p_neg);
    cudaFree(d_p_zero);
    cudaFree(d_p_pos);
    cudaFree(d_ev);

    return errors;
}
int main() {
    return qts_verify_born_normalize();
}