#include <unistd.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#define WARP_SIZE 16
#define DEBUG false

/* ---------------- [[HELPER FUNCTIONS FOR GLOBAL MEMORY]] ---------------- */

float *_copyHostDevice(float *src, int src_size) {
    float *src_d;
    cudaMalloc((void**)&src_d, sizeof(float) * src_size);
    cudaMemcpy(src_d, src, sizeof(float) * src_size, cudaMemcpyHostToDevice);
    return src_d;
}

float *_copyDeviceHost(float *src, int src_size, float *dst) {
    float *target;
    if (dst == NULL) {
        target = (float*)malloc(sizeof(float) * src_size);
    } else {
        target = dst;
    }
    
    cudaMemcpy(target, src, sizeof(float) * src_size, cudaMemcpyDeviceToHost);
    return target;
}

/* ---------------- [[HELPER FUNCTIONS FOR TILING]] ---------------- */

typedef struct {
    int x;
    int y;
} GlobalDim;

__device__ GlobalDim getGlobalDim(dim3 blockDim, dim3 blockIdx, dim3 threadIdx) {
    GlobalDim gd;
    gd.x = blockDim.x * blockIdx.x + threadIdx.x;
    gd.y = blockDim.y * blockIdx.y + threadIdx.y;
    return gd;
}

dim3 getGridBasedOnBlockSize(int width, int height, int block_size) {
    int gridX = (int)ceil((float)width / block_size);
    int gridY = (int)ceil((float)height / block_size);
    dim3 gridXY(gridX, gridY);
    return gridXY;
}

/* ---------------- [[HELPER FUNCTIONS FOR DEBUGGING]] ---------------- */

void _sleep(int n) {
    usleep(n*1000000);
}

void drawMatrix(float *m, int width, int height) {
    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%f ", m[i * width + j]);
        }
        printf("\n");
    }
}

