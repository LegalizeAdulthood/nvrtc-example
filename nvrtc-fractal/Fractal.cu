#include "Fractal.h"

#include <vector_functions.h>

#include <math.h>
#include <stdio.h>

namespace fractal
{

struct Params
{
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    int maxIters;
    int numColors;
    uchar4 *colors;
};

struct Complex
{
    float2 val;
};

__device__ __forceinline__ Complex operator+(const Complex &lhs, const Complex &rhs)
{
    return {make_float2(lhs.val.x + rhs.val.x, lhs.val.y + rhs.val.y)};
}

__device__ __forceinline__ Complex operator-(const Complex &lhs, const Complex &rhs)
{
    return {make_float2(lhs.val.x - rhs.val.x, lhs.val.y - rhs.val.y)};
}

__device__ __forceinline__ Complex operator*(const Complex &lhs, const Complex &rhs)
{
    return {make_float2(lhs.val.x * rhs.val.x - lhs.val.y * rhs.val.y, lhs.val.x * rhs.val.y + lhs.val.y * rhs.val.x)};
}

__device__ __forceinline__ float magSquared(const Complex &arg)
{
    return arg.val.x*arg.val.x + arg.val.y*arg.val.y;
}

__host__ __device__ __forceinline__ int mandelbrotColor( float cx, float cy, const Params& params )
{
    float2      z{cx, cy};
    int         n = 0;
    const float bailOut = 4.0f;

    for (n = 0; n < params.maxIters; ++n)
    {
        if (z.x * z.x + z.y * z.y > bailOut)
            break;
        z = {(cx + z.x * z.x - z.y * z.y), (cy + z.x * z.y + z.y * z.x)};
    }

    return n;
}

__global__ static void iterate(int width, int height, uchar4 *pixels, const Params params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if( idx >= width * height )
        return;

    int j = idx / width;
    int i = idx - (j * width);

    float dx = (static_cast<float>(i) + 0.5f) / width;
    float x = params.xmin * (1.0f - dx) + params.xmax * dx;

    float dy = (static_cast<float>(j) + 0.5f) / height;
    float y = params.ymin * (1.0f - dy) + params.ymax * dy;
    const int iters = mandelbrotColor(x, y, params) % params.numColors;
    pixels[idx] = params.colors[iters];
}

__host__ void render(int width, int height, uchar4 *pixels)
{
    const unsigned int totalThreads = width * height;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(numBLocks, 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    static uchar4 *devColors{};
    if (devColors == nullptr)
    {
        cudaMalloc(reinterpret_cast<void **>(&devColors), 256 * sizeof(uchar4));
        uchar4 colors[256];
        for (int i = 0; i < 256; ++i)
        {
            colors[i] = make_uchar4(static_cast<unsigned char>(i), static_cast<unsigned char>(255 - i),
                                    static_cast<unsigned char>(i), 255U);
        }
        cudaMemcpy(devColors, colors, 256 * sizeof(uchar4), cudaMemcpyHostToDevice);
    }
    Params params{-1.5f, 1.5f, -1.5f, 1.5f, 8192, 256, devColors};
    iterate<<<grid, block, 0U>>>(width, height, pixels, params);
}

} // namespace fractal
