#ifndef NVJITLINK_ERROR_CHECK_H
#define NVJITLINK_ERROR_CHECK_H

#include <nvJitLink.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

namespace otk {
namespace error {

/// Specializations for JIT link error names.
template <>
inline std::string getErrorName(nvJitLinkResult value)
{
    switch (value)
    {
    case NVJITLINK_SUCCESS:
        return "NVJITLINK_SUCCESS";
    case NVJITLINK_ERROR_UNRECOGNIZED_OPTION:
        return "NVJITLINK_ERROR_UNRECOGNIZED_OPTION";
    case NVJITLINK_ERROR_MISSING_ARCH:
        return "NVJITLINK_ERROR_MISSING_ARCH";
    case NVJITLINK_ERROR_INVALID_INPUT:
        return "NVJITLINK_ERROR_INVALID_INPUT";
    case NVJITLINK_ERROR_PTX_COMPILE:
        return "NVJITLINK_ERROR_PTX_COMPILE";
    case NVJITLINK_ERROR_NVVM_COMPILE:
        return "NVJITLINK_ERROR_NVVM_COMPILE";
    case NVJITLINK_ERROR_INTERNAL:
        return "NVJITLINK_ERROR_INTERNAL";
    }
    return {};
}

/// Specializations for JIT link error messages.
template <>
inline std::string getErrorMessage( nvJitLinkResult value )
{
    return {};
}

}  // namespace error
}  // namespace otk

#endif
