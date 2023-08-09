#include "Fractal.h"

#include <vector_functions.h>

#include <math.h>

namespace fractal
{

struct Params
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    int maxIters;
    uchar4 colors[256];
};

struct Complex
{
    float re;
    float im;

    __device__ __forceinline__ Complex &operator=(const Complex &rhs)
    {
        re = rhs.re;
        im = rhs.im;
        return *this;
    }
};

__device__ __forceinline__ Complex operator+(const Complex &lhs, const Complex &rhs)
{
    return {lhs.re + rhs.re, lhs.im + rhs.im};
}

__device__ __forceinline__ Complex operator-(const Complex &lhs, const Complex &rhs)
{
    return {lhs.re - rhs.re, lhs.im - rhs.im};
}

__device__ __forceinline__ Complex operator*(const Complex &lhs, const Complex &rhs)
{
    return {lhs.re * rhs.re - lhs.im * rhs.im, lhs.re * rhs.im + lhs.im * rhs.re};
}

__device__ __forceinline__ float magSquared(const Complex &arg)
{
    return arg.re*arg.re + arg.im*arg.im;
}

__device__ __forceinline__ int iterate( float cx, float cy, int maxIters )
{
    Complex     z{cx, cy};
    Complex     c{cx, cy};
    int         n = 0;
    const float bailOut = 4.0f;

    for (n = 0; n < maxIters; ++n)
    {
        if (magSquared(z) > bailOut)
            break;
        z = z*z + c;
    }

    return n;
}

extern "C" __global__ static void colorPixel(int width, int height, uchar4 *pixels, const Params params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx >= width * height )
        return;

    int j = idx / width;
    int i = idx - (j * width);

    float dr = (static_cast<float>(i) + 0.5f) / width;
    float re = params.xmin * (1.0f - dr) + params.xmax * dr;

    float di = (static_cast<float>(j) + 0.5f) / height;
    float im = params.ymin * (1.0f - di) + params.ymax * di;
    const int iters = iterate(re, im, params.maxIters) % 256;
    pixels[idx] = params.colors[iters];
}

__host__ void render(int width, int height, uchar4 *pixels)
{
    const unsigned int totalThreads = width * height;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(numBLocks, 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    Params params{-2.0f, 1.f, -1.5f, 1.5f, 8192};
    for (int i = 0; i < 256; ++i)
    {
        params.colors[i] = make_uchar4(static_cast<unsigned char>(i), static_cast<unsigned char>(255 - i),
                                       static_cast<unsigned char>(i), 255U);
    }
    colorPixel<<<grid, block, 0U>>>(width, height, pixels, params);
}

} // namespace fractal
