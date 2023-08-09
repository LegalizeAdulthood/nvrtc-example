#include "Iterate.cuh"
#include "Complex.cuh"
#include "Formula.cuh"

#include <vector_functions.h>

namespace fractal
{

__device__ int iterate( float cx, float cy, int maxIters )
{
    Complex     z{cx, cy};
    Complex     c{cx, cy};
    int         n = 0;
    const float bailOut = 4.0f;

    for (n = 0; n < maxIters; ++n)
    {
        if (magSquared(z) > bailOut)
            break;
        z = FORMULA;
    }

    return n;
}

} // namespace fractal
