#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>


__global__ void Transfer_kernel(double *Result, double* Result2,double * Result3,double * Result4,double *ModelNum,double* ModelSm,double *SpNum,double *ColorD,double *SumG,double *MaxDim,double* MainColor)
{
	int z = threadIdx.x;
	int MNum = (int)(*ModelNum), SNum = (int)(*SpNum),MDim=(int)(*MaxDim);
	if (threadIdx.x >= SNum)
		return;

	for (int k = 0; k < MNum; k++)
	{
		if (ModelSm[k] != 0&&ColorD[z+MDim*k]!=1000)
		{
			Result[z] = Result[z] + ModelSm[k] *exp(-ColorD[z+MDim*k])*SumG[z+MDim*k];//
            Result2[z+MDim*k]=exp(-MainColor[z+MDim*k]);
            Result4[z+MDim*k]=ModelSm[k] *exp(-ColorD[z+MDim*k])*SumG[z+MDim*k];
            Result3[z+MDim*k]=ModelSm[k] *exp(-ColorD[z+MDim*k])*SumG[z+MDim*k];
		}
	}
	return;

}
void ModelT(double *Result, double* Result2,double * Result3,double * Result4,double *ModelNum,double* ModelSm,double *SpNum,double *MaxDim,double *ColorD,double *SumG,double* MainColor)
{
	double * dev_Result,*dev_Result2,*dev_Result3,*dev_Result4;
	double *dev_ModelNum;
	double *dev_ModelSm;
	double *dev_SpNum,*dev_MaxDim;
    double *dev_ColorD, *dev_SumG,*dev_MainColor;
	int MDim = (int)(*MaxDim);
	int MNum = (int)(*ModelNum);
	int Spnum = (int)(*SpNum);

	cudaMalloc((void **)&dev_Result, sizeof(double)* MDim);
    cudaMalloc((void **)&dev_Result2, sizeof(double)* MDim*MNum);
    cudaMalloc((void **)&dev_Result3, sizeof(double)* MDim*MNum);
    cudaMalloc((void **)&dev_Result4, sizeof(double)* MDim*MNum);
	cudaMalloc((void **)&dev_ModelSm, sizeof(double)* MNum);
	cudaMalloc((void **)&dev_SpNum, sizeof(double));
    cudaMalloc((void **)&dev_MaxDim, sizeof(double));
	cudaMalloc((void **)&dev_ModelNum, sizeof(double));
    cudaMalloc((void **)&dev_ColorD, sizeof(double)* MDim*MNum);
	cudaMalloc((void **)&dev_SumG, sizeof(double)* MDim*MNum);
    cudaMalloc((void **)&dev_MainColor, sizeof(double)* MDim*MNum);

	cudaMemcpy(dev_ModelNum, ModelNum, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_SpNum, SpNum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_MaxDim, MaxDim, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ModelSm, ModelSm, sizeof(double)* MNum, cudaMemcpyHostToDevice);

    cudaMemcpy(dev_ColorD, ColorD, sizeof(double)* MDim*MNum, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_SumG, SumG, sizeof(double)* MDim*MNum, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_MainColor, MainColor, sizeof(double)* MDim*MNum, cudaMemcpyHostToDevice);

	dim3 threads(Spnum);
	Transfer_kernel << <1, threads >> >(dev_Result, dev_Result2, dev_Result3,dev_Result4, dev_ModelNum, dev_ModelSm, dev_SpNum, dev_ColorD,dev_SumG,dev_MaxDim,dev_MainColor);
	cudaMemcpy(Result, dev_Result, sizeof(double)*MDim, cudaMemcpyDeviceToHost);
    cudaMemcpy(Result2, dev_Result2, sizeof(double)* MDim*MNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(Result3, dev_Result3, sizeof(double)* MDim*MNum, cudaMemcpyDeviceToHost);
    cudaMemcpy(Result4, dev_Result4, sizeof(double)* MDim*MNum, cudaMemcpyDeviceToHost);
	cudaFree(dev_ModelSm);
	cudaFree(dev_ModelNum);
	cudaFree(dev_Result);
	cudaFree(dev_Result2);
	cudaFree(dev_Result3);
    cudaFree(dev_Result4);
	cudaFree(dev_SpNum);
	cudaFree(dev_Result);
    cudaFree(dev_ColorD);
	cudaFree(dev_SumG);
    cudaFree(dev_MainColor);

}




