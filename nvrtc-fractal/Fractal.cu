#include "Fractal.h"
#include "Complex.cuh"
#include "Iterate.cuh"

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
    uchar4 colors[6];
};

__global__ static void colorPixel(int width, int height, uchar4 *pixels, const Params params)
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

    const int iters = iterate(re, im, params.maxIters);
    if (iters == params.maxIters)
        pixels[idx] = make_uchar4(0, 0, 0, 255U);
    else
        pixels[idx] = params.colors[iters % 6];
}

__host__ void render(int width, int height, uchar4 *pixels)
{
    const unsigned int totalThreads = width * height;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(numBLocks, 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    Params params{-2.0f, 1.f, -1.5f, 1.5f, 8192};
    static const uchar4 colors[6] = {
        make_uchar4(255, 0, 0, 255),
        make_uchar4(0, 255, 0, 255),
        make_uchar4(0, 0, 255, 255),
        make_uchar4(255, 0, 255, 255),
        make_uchar4(0, 255, 255, 255),
        make_uchar4(255, 0, 255, 255),
    };
    for (int i = 0; i < 6; ++i)
    {
        params.colors[i] = colors[i];
    }
    colorPixel<<<grid, block, 0U>>>(width, height, pixels, params);
}

} // namespace fractal
