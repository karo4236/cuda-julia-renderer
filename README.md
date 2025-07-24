# cuda-julia-renderer
# Julia Set Fractal Renderer with CUDA

This is a GPU-accelerated fractal renderer that visualizes the Julia set using CUDA. It computes complex-number iterations in parallel across the image grid and outputs a high-resolution bitmap (BMP) with custom color gradients to highlight edge dynamics and iteration depth.

## Overview

The Julia set is computed pixel-by-pixel across a complex plane, with each pixel representing a point whose behavior under iteration reveals intricate fractal patterns. This project offloads the computation to the GPU using CUDA, enabling high-speed rendering and smooth coloring.

- CUDA-parallelized fractal computation
- BMP image output using EasyBMP (minimal BMP image writer)
- Custom color mapping to highlight escape-time and boundaries
- Optional tuning of parameters: resolution, zoom, color palettes

## Build & Run

Compile the CUDA file:

```bash
nvcc Julia_set.cu -o julia
