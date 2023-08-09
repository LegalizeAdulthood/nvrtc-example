#include "Dynamic.h"
#include "Params.h"
#include "SourceDir.h"
#include "nvrtcErrorCheck.h"
#include "nvJitLinkErrorCheck.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>
#include <nvJitLink.h>
#include <nvrtc.h>
#include <vector_functions.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace fractal
{

static std::string fileContents(const char *file)
{
    std::ifstream str(file);
    if (!str)
        throw std::runtime_error(std::string{"Couldn't read "} + file);
    std::stringstream buff;
    buff << str.rdbuf();
    return buff.str();
}

struct Header
{
    std::string name;
    std::string contents;
};

std::vector<Header>       g_headers;
std::vector<const char *> g_headerContentsPtrs;
std::vector<const char *> g_headerNamePtrs;

static std::string sourcePath(const char *name)
{
    return std::string{g_sourceDir} + "/" + name;
}

static Header getHeader(const char *name)
{
    Header result;
    result.name = name;
    result.contents = fileContents(sourcePath(name).c_str());
    return result;
}

static const char *const g_formulaPrefix = R"text(#ifndef FORMULA_H
#define FORMULA_H

#define FORMULA )text";
static const char *const g_formulaSuffix = R"text(

#endif
)text";

static void readHeaders(const char *formula)
{
    if (!g_headers.empty())
        return;

    for (const char *header : {"Params.h", "Iterate.cuh", "Complex.cuh"})
    {
        g_headers.emplace_back(getHeader(header));
    }
    Header formulaHeader;
    formulaHeader.name = "Formula.cuh";
    formulaHeader.contents = std::string{g_formulaPrefix} + formula + g_formulaSuffix;
    g_headers.emplace_back(std::move(formulaHeader));
    g_headers.emplace_back(Header{"vector_types.h", ""});
    g_headers.emplace_back(Header{"vector_functions.h", ""});
    g_headers.emplace_back(Header{"Fractal.h", ""});
    std::transform(g_headers.begin(), g_headers.end(), std::back_inserter(g_headerContentsPtrs),
                   [](const Header &header) { return header.contents.c_str(); });
    std::transform(g_headers.begin(), g_headers.end(), std::back_inserter(g_headerNamePtrs),
                   [](const Header &header) { return header.name.c_str(); });
}

static std::vector<std::string> g_ptx;

std::string getArchOption()
{
    CUdevice device{};
    OTK_ERROR_CHECK(cuDeviceGet(&device, 0));
    int major{};
    OTK_ERROR_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    int minor{};
    OTK_ERROR_CHECK(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    return "-arch=sm_" + std::to_string(major*10 + minor);
}

void createProgramFromText(const char *text, const char *file)
{
    nvrtcProgram program{};
    OTK_ERROR_CHECK(nvrtcCreateProgram(&program, text, file, g_headerContentsPtrs.size(),
                                       g_headerContentsPtrs.data(), g_headerNamePtrs.data()));
    std::string archOption{getArchOption()};
    const char *options[] = {"-dc", "-DUSE_LAUNCHER=0", archOption.c_str()};
    if (const nvrtcResult status = nvrtcCompileProgram(program, sizeof(options) / sizeof(options[0]), options))
    {
        std::string log;
        size_t      logSize{};
        OTK_ERROR_CHECK(nvrtcGetProgramLogSize(program, &logSize));
        log.resize(logSize);
        OTK_ERROR_CHECK(nvrtcGetProgramLog(program, &log[0]));
        std::cout << log << '\n';
        OTK_ERROR_CHECK(status);
    }
    std::string ptx;
    size_t      size{};
    OTK_ERROR_CHECK(nvrtcGetPTXSize(program, &size));
    ptx.resize(size);
    OTK_ERROR_CHECK(nvrtcGetPTX(program, &ptx[0]));
    OTK_ERROR_CHECK(nvrtcDestroyProgram(&program));
    g_ptx.emplace_back(std::move(ptx));
}

void createProgram(const char *file)
{
    return createProgramFromText(fileContents(sourcePath(file).c_str()).c_str(), file);
}

void createSingleProgram()
{
    std::string text{fileContents(sourcePath("Iterate.cu").c_str()) + fileContents(sourcePath("Fractal.cu").c_str())};
    return createProgramFromText(text.c_str(), "Dynamic.cu");
}

// Set this to 1 to use separate JIT compilation and linking.
#define USE_NVJITLINK 0

#if USE_NVJITLINK
static CUfunction getEntryPoint()
{
    std::string     arch{getArchOption()};
    const char     *options[] = {arch.c_str()};
    nvJitLinkHandle handle{};
    OTK_ERROR_CHECK(nvJitLinkCreate(&handle, sizeof(options) / sizeof(options[0]), options));
    for (const std::string &ptx : g_ptx)
    {
        OTK_ERROR_CHECK(nvJitLinkAddData(handle, NVJITLINK_INPUT_PTX, ptx.c_str(), ptx.size(), ""));
    }
    OTK_ERROR_CHECK(nvJitLinkComplete(handle));
    size_t cubinSize{};
    OTK_ERROR_CHECK(nvJitLinkGetLinkedCubinSize(handle, &cubinSize));
    std::vector<std::uint8_t> cubin;
    cubin.resize(cubinSize);
    OTK_ERROR_CHECK(nvJitLinkGetLinkedCubin(handle, cubin.data()));
    size_t logSize{};
    OTK_ERROR_CHECK(nvJitLinkGetErrorLogSize(handle, &logSize));
    if(logSize > 1)
    {
        std::string log;
        log.resize(logSize);
        OTK_ERROR_CHECK(nvJitLinkGetErrorLog(handle, &log[0]));
        std::cout << log << '\n';
    }
    OTK_ERROR_CHECK(nvJitLinkDestroy(&handle));

    CUmodule module{};
    OTK_ERROR_CHECK(cuModuleLoadData(&module, cubin.data()));
    CUfunction entryPoint{};
    OTK_ERROR_CHECK(cuModuleGetFunction(&entryPoint, module, "colorPixel"));
    return entryPoint;
}
#else
CUmodule g_module{};

CUfunction getEntryPoint()
{
    OTK_ERROR_CHECK(cuModuleLoadData(&g_module, g_ptx[0].c_str()));
    CUfunction entryPoint{};
    OTK_ERROR_CHECK(cuModuleGetFunction(&entryPoint, g_module, "colorPixel"));
    return entryPoint;
}
#endif

void render(int width, int height, uchar4 *pixels, const char *const formula)
{
    readHeaders(formula);
#if USE_NVJITLINK
    createProgram("Iterate.cu");
    createProgram("Fractal.cu");
#else
    createSingleProgram();
#endif
    CUfunction entryPoint = getEntryPoint();

    const unsigned int totalThreads = width * height;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    Params              params{-2.0f, 1.f, -1.5f, 1.5f, 8192};
    static const uchar4 colors[6] = {
        make_uchar4(255, 0, 0, 255),   make_uchar4(0, 255, 0, 255),   make_uchar4(0, 0, 255, 255),
        make_uchar4(255, 0, 255, 255), make_uchar4(0, 255, 255, 255), make_uchar4(255, 0, 255, 255),
    };
    for (int i = 0; i < 6; ++i)
    {
        params.colors[i] = colors[i];
    }

    // call function
    void *args[] = {&width, &height, &pixels, &params};
    OTK_ERROR_CHECK(cuLaunchKernel(entryPoint, numBlocks, 1, 1, threadsPerBlock, 1, 1, 0, nullptr, args, nullptr));
}

} // namespace fractal
