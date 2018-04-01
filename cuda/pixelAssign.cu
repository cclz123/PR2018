#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <random>
#define ImageSize 300

__global__ void mexFunction_kernel(double *result, int imageSize, double *LocalSize, const double *input1, const double * input2,double *Par)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= imageSize || j >= imageSize)
		return;
    int localSize=(int)LocalSize[0];
	double totalW = 0, Rc, Gc, Bc;
	double ColorCM, weightM, SaliencyPath;
	double myRes = 0;

	for (int k = max(i - (localSize), 0); k<min(i + 1 + localSize, imageSize); k++)
		for (int z = max(j - localSize, 0); z<min(j + 1 + localSize, imageSize); z++)
		{
		SaliencyPath = input1[k*imageSize + z];
		Rc = input2[(k*imageSize + z) ] - input2[(i*imageSize + j) ];
		Gc = input2[(k*imageSize + z) +ImageSize*ImageSize] - input2[(i*imageSize + j)+ImageSize*ImageSize];
		Bc = input2[(k*imageSize + z) + 2*ImageSize*ImageSize] - input2[(i*imageSize + j) +2*ImageSize*ImageSize];
		ColorCM = abs(Rc) + abs(Gc) + abs(Bc);
		weightM = exp(-(ColorCM) * (*Par));
		totalW += weightM;
		myRes += weightM*SaliencyPath;
		}
	result[i*imageSize + j] = myRes / totalW;
}

void pixelAssign(double * result, const double * input1, const double * input2, double * localSize,double *Par)
{
	double * dev_result;
	double *dev_input1, *dev_input2;
    double *dev_localsize,*dev_Par;

	cudaMalloc((void **)&dev_result, sizeof(double)* ImageSize * ImageSize);
    cudaMalloc((void **)&dev_localsize, sizeof(double));
    cudaMalloc((void **)&dev_Par, sizeof(double));
	cudaMalloc((void**)& dev_input1, sizeof(double) * ImageSize * ImageSize );
	cudaMalloc((void**)& dev_input2, sizeof(double) * ImageSize * ImageSize * 3);

	cudaMemcpy(dev_input1, input1, sizeof(double)* ImageSize* ImageSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_localsize, localSize, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Par, Par, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_input2, input2, sizeof(double)* ImageSize* ImageSize * 3, cudaMemcpyHostToDevice);
	dim3 threads(16, 16);
	dim3 grids(ImageSize / threads.x + 1, ImageSize / threads.y + 1);
	mexFunction_kernel << <grids, threads >> >(dev_result, ImageSize, dev_localsize, dev_input1, dev_input2,dev_Par);

	cudaMemcpy(result, dev_result, sizeof(double)* ImageSize* ImageSize, cudaMemcpyDeviceToHost);

	cudaFree(dev_result);
	cudaFree(dev_input1);
	cudaFree(dev_input2);
    cudaFree(dev_localsize);
    cudaFree(dev_Par);

}




