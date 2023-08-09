#ifndef PARAMS_H
#define PARAMS_H

#include <vector_types.h>

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

} // namespace fractal

#endif
