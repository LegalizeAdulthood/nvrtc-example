#include "Dynamic.h"
#include "Params.h"

#include <vector_functions.h>
#include <cuda.h>
#include <nvrtc.h>

namespace fractal
{

void render(int width, int height, uchar4 *pixels, const char *const formula)
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
    //colorPixel<<<grid, block, 0U>>>(width, height, pixels, params);
}

}
