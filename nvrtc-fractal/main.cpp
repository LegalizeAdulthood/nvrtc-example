#include <OptiXToolkit/Error/cudaErrorCheck.h>
#include <OptiXToolkit/Gui/BufferMapper.h>
#include <OptiXToolkit/Gui/CUDAOutputBuffer.h>

#include <nvrtc.h>

#include <cuda.h>

#include "Fractal.h"

#include <stdexcept>

namespace
{

int render(char *program)
{
    OTK_ERROR_CHECK(cudaFree(nullptr));

    int                           width = 512;
    int                           height = 512;
    otk::CUDAOutputBuffer<uchar4> output(otk::CUDAOutputBufferType::CUDA_DEVICE, width, height);

    uchar4 *pixels = output.map();
    fractal::render(width, height, pixels);
    output.unmap();

    otk::ImageBuffer buffer;
    buffer.data = output.getHostPointer();
    buffer.width = width;
    buffer.height = height;
    buffer.pixel_format = otk::BufferImageFormat::UNSIGNED_BYTE4;
    displayBufferWindow(program, buffer);

    return 0;
}

} // namespace

int main(int argc, char *argv[])
{
    try
    {
        return render(argv[0]);
    }
    catch (const std::exception &bang)
    {
        std::cerr << bang.what();
        return 1;
    }
    catch (...)
    {
        return 2;
    }
}
