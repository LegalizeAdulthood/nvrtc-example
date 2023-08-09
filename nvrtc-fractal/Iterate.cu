#include "Iterate.cuh"
#include "Complex.cuh"

namespace fractal
{

extern "C" __device__ int iterate( float cx, float cy, int maxIters )
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

} // namespace fractal
