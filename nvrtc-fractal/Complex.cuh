#ifndef COMPLEX_CUH
#define COMPLEX_CUH

namespace fractal
{

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

} // namespace fractal

#endif
