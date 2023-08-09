#include "Dynamic.h"
#include "Params.h"
#include "SourceDir.h"
#include "nvrtcErrorCheck.h"

#include <cuda.h>
#include <nvrtc.h>
#include <vector_functions.h>

#include <algorithm>
#include <fstream>
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

static void readHeaders( const char *formula )
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
    std::transform(g_headers.begin(), g_headers.end(), std::back_inserter(g_headerContentsPtrs),
                   [](const Header &header) { return header.contents.c_str(); });
    std::transform(g_headers.begin(), g_headers.end(), std::back_inserter(g_headerNamePtrs),
                   [](const Header &header) { return header.name.c_str(); });
}

nvrtcProgram createProgram(const char *file)
{
    std::string  text(fileContents(sourcePath(file).c_str()));
    nvrtcProgram program{};
    OTK_ERROR_CHECK(nvrtcCreateProgram(&program, text.c_str(), file, g_headerContentsPtrs.size(),
                                       g_headerContentsPtrs.data(), g_headerNamePtrs.data()));
    return program;
}

void compileProgram(nvrtcProgram program)
{
    const char *options[] = {"-rdc", "-I", g_sourceDir};
    OTK_ERROR_CHECK(nvrtcCompileProgram(program, sizeof(options) / sizeof(options[0]), options));
}

void createPrograms()
{
    nvrtcProgram iterate = createProgram("Iterate.cu");
    nvrtcProgram fractal = createProgram("Fractal.cu");
    OTK_ERROR_CHECK(nvrtcDestroyProgram(&fractal));
    OTK_ERROR_CHECK(nvrtcDestroyProgram(&iterate));
}

void render(int width, int height, uchar4 *pixels, const char *const formula)
{
    readHeaders(formula);
    createPrograms();

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
