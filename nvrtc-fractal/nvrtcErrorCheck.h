#ifndef NVRTC_ERROR_CHECK_H
#define NVRTC_ERROR_CHECK_H

#include <nvrtc.h>

#include <OptiXToolkit/Error/ErrorCheck.h>

namespace otk {
namespace error {

/// Specializations for CUDA runtime error names.
template <>
inline std::string getErrorName(nvrtcResult value)
{
    switch (value)
    {
    case NVRTC_SUCCESS:
        return "NVRTC_SUCCESS";
    case NVRTC_ERROR_OUT_OF_MEMORY:
        return "NVRTC_ERROR_OUT_OF_MEMORY";
    case NVRTC_ERROR_PROGRAM_CREATION_FAILURE:
        return "NVRTC_ERROR_PROGRAM_CREATION_FAILURE";
    case NVRTC_ERROR_INVALID_INPUT:
        return "NVRTC_ERROR_INVALID_INPUT";
    case NVRTC_ERROR_INVALID_PROGRAM:
        return "NVRTC_ERROR_INVALID_PROGRAM";
    case NVRTC_ERROR_INVALID_OPTION:
        return "NVRTC_ERROR_INVALID_OPTION";
    case NVRTC_ERROR_COMPILATION:
        return "NVRTC_ERROR_COMPILATION";
    case NVRTC_ERROR_BUILTIN_OPERATION_FAILURE:
        return "NVRTC_ERROR_BUILTIN_OPERATION_FAILURE";
    case NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION:
        return "NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION";
    case NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION:
        return "NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION";
    case NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID:
        return "NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID";
    case NVRTC_ERROR_INTERNAL_ERROR:
        return "NVRTC_ERROR_INTERNAL_ERROR";
    }
    return {};
}

/// Specializations for CUDA runtime error messages.
template <>
inline std::string getErrorMessage( nvrtcResult value )
{
    return ::nvrtcGetErrorString( value );
}

}  // namespace error
}  // namespace otk

#endif
