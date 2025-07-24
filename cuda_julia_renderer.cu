#include <stdio.h>
#include "EasyBMP.h"
#include "EasyBMP.cpp"
#include <cuda_runtime.h>
#include <math.h>

void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        printf("CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Complex number definition
struct Complex {
    float x; // Real part (x-axis in output image)
    float y; // Imaginary part (y-axis in output image)
};

__device__ Complex d_add(Complex c1, Complex c2) {
    return { c1.x + c2.x, c1.y + c2.y };
}

__device__ Complex d_mul(Complex c1, Complex c2) {
    return { c1.x * c2.x - c1.y * c2.y, c1.x * c2.y + c2.x * c1.y };
}

__device__ float d_mag(Complex c){
    return sqrtf(c.x * c.x + c.y * c.y);
}

__device__ uchar4 hsv_to_rgb(float h, float s, float v) {
    float r, g, b;
    int i = int(h * 6);
    float f = h * 6 - i;
    float p = v * (1 - s);
    float q = v * (1 - f * s);
    float t = v * (1 - (1 - f) * s);
    
    switch (i % 6) {
        case 0: r = v, g = t, b = p; break;
        case 1: r = q, g = v, b = p; break;
        case 2: r = p, g = v, b = t; break;
        case 3: r = p, g = q, b = v; break;
        case 4: r = t, g = p, b = v; break;
        case 5: r = v, g = p, b = q; break;
    }
    return make_uchar4(r * 255, g * 255, b * 255, 255);
}

__device__ uchar4 color_map(int n, int max_iterations) {
    if (n == max_iterations) {
        return make_uchar4(0, 0, 0, 255); 
    }
    float h = (float)n / max_iterations; 
    float s = 1.0f;                     
    float v = 1.0f;                      
    return hsv_to_rgb(h, s, v);
}

// CUDA Kernel 
__global__ void kernel(uchar4* pixels, int width, int height, int infinity, int max_iterations, float x_min, float x_incr, float y_min, float y_incr, Complex c) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= height || col >= width) { return; }

    Complex z;
    z.x = x_min + col * x_incr;
    z.y = y_min + row * y_incr;

    int n = 0;
    do {
        z = d_add(d_mul(z, z), c);
    } while (d_mag(z) < infinity && n++ < max_iterations);
    
    pixels[col + row * width] = color_map(n, max_iterations);
}

void compute_julia(const char*, int, int);
void save_image(uchar4*, const char*, int, int);
Complex add(Complex, Complex);
Complex mul(Complex, Complex);
float mag(Complex);

int main(void) {
    const char* name = "test.bmp";
    compute_julia(name, 3000, 3000);
    printf("Finished creating %s.\n", name);
    return 0;
}
void save_image(uchar4* pixels, const char* filename, int width, int height) {
    BMP output;
    output.SetSize(width, height);
    output.SetBitDepth(24);
    // Save each pixel to output image
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            uchar4 color = pixels[col + row * width];
            output(col, row)->Red = color.x;
            output(col, row)->Green = color.y;
            output(col, row)->Blue = color.z;
        }
    }
    output.WriteToFile(filename);
}


void compute_julia(const char* filename, int width, int height) {
    double t = clock();
    uchar4 *pixels = (uchar4*)malloc(width * height * sizeof(uchar4));
    
    int max_iterations = 400;
    int infinity = 20;
    Complex c = { 0.285, 0.01 };
    float w = 4;
    float h = w * height / width;
    float x_min = -w / 2, y_min = -h / 2;
    float x_incr = w / width, y_incr = h / height;
    
    int npixels = width * height;
    dim3 blocksize(32,32);
    int nblocks_row=(width-1)/32+1;
    int nblocks_col=(height-1)/32+1;
    dim3 gridsize(nblocks_row,nblocks_col);
    
    uchar4 *d_p;
    checkCudaError(cudaMalloc(&d_p, npixels * sizeof(uchar4)), "cudaMalloc failed");
    checkCudaError(cudaMemset(d_p, 0, npixels * sizeof(uchar4)), "cudaMemset failed");
    
    kernel<<<gridsize, blocksize>>>(d_p, width, height, infinity, max_iterations, x_min, x_incr, y_min, y_incr, c);
    
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    checkCudaError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    
    checkCudaError(cudaMemcpy(pixels, d_p, npixels * sizeof(uchar4), cudaMemcpyDeviceToHost), "cudaMemcpy D2H failed");
    checkCudaError(cudaFree(d_p), "cudaFree failed");
    
    save_image(pixels, filename, width, height);
    
    free(pixels);
    t = (clock() - t) / CLOCKS_PER_SEC;
    printf("Finished processing in %.7f seconds.\n", t);
}
