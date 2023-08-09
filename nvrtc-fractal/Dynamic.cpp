#include "Dynamic.h"
#include "Params.h"
#include "SourceDir.h"
#include "nvrtcErrorCheck.h"

#include <OptiXToolkit/Error/cuErrorCheck.h>

#include <cuda.h>
#include <nvrtc.h>
#include <vector_functions.h>

#include <algorithm>
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

static std::vector<std::string> g_programs;
static std::vector<CUmodule> g_modules;

void createProgramFromText(const char *text, const char *file)
{
    nvrtcProgram program{};
    OTK_ERROR_CHECK(nvrtcCreateProgram(&program, text, file, g_headerContentsPtrs.size(),
                                       g_headerContentsPtrs.data(), g_headerNamePtrs.data()));
    const char *options[] = {"-dc", "-DUSE_LAUNCHER=0"};
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
    g_programs.emplace_back(std::move(ptx));
}

void createProgram(const char *file)
{
    return createProgramFromText(fileContents(sourcePath(file).c_str()).c_str(), file);
}

void render(int width, int height, uchar4 *pixels, const char *const formula)
{
    readHeaders(formula);
    createProgram("Iterate.cu");
    createProgram("Fractal.cu");
    
    OTK_ERROR_CHECK(cuInit(0));
    CUdevice device{};
    OTK_ERROR_CHECK(cuDeviceGet(&device, 0));
    CUcontext context{};
    OTK_ERROR_CHECK(cuCtxCreate(&context, 0, device));
    CUmodule module{};
    OTK_ERROR_CHECK(cuModuleLoadDataEx(&module, g_programs[0].c_str(), 0, nullptr, nullptr));

    const unsigned int totalThreads = width * height;
    const unsigned int threadsPerBlock = 64;
    const unsigned int numBLocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    dim3 grid(numBLocks, 1, 1);
    dim3 block(threadsPerBlock, 1, 1);

    Params              params{-2.0f, 1.f, -1.5f, 1.5f, 8192};
    static const uchar4 colors[6] = {
        make_uchar4(255, 0, 0, 255),   make_uchar4(0, 255, 0, 255),   make_uchar4(0, 0, 255, 255),
        make_uchar4(255, 0, 255, 255), make_uchar4(0, 255, 255, 255), make_uchar4(255, 0, 255, 255),
    };
    for (int i = 0; i < 6; ++i)
    {
        params.colors[i] = colors[i];
    }

    // colorPixel<<<grid, block, 0U>>>(width, height, pixels, params);
}

} // namespace fractal
