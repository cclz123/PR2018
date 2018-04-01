#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>

__global__ void computec_kernel(double *MSaliencyM,double * CSaliencyM, double* midP,double *motion1,double * motion2,double * color11,double * color12,double * color13,double * ring,double *K1,double *N1,double *Par,double *spnum)
{
    int K=(int)(*K1),N=(int)(*N1);
	
	int i,j,k;
	i=blockIdx.x;
	j=threadIdx.x;
	if (blockIdx.x >= N || threadIdx.x >= spnum[i])
		return;

	double Lmotion1,Lmotion2,Lcolor1,Lcolor2,Lcolor3,Llocation1,Llocation2;
	double Rmotion1,Rmotion2,Rcolor1,Rcolor2,Rcolor3,Rlocation1,Rlocation2;
	double LDist,CDist,MDist,MScale,CScale;


	MScale=0,CScale=0;
	Lmotion1 = motion1[i*K+j];
	Lmotion2 = motion2[i*K+j];
	Lcolor1  = color11[i*K+j];
	Lcolor2  = color12[i*K+j];
	Lcolor3  = color13[i*K+j];
    Llocation1 = midP[(K)*5*i+j],Llocation2 = midP[(K)*5*i+(K)*1+j];
		
	for (k=0;k<spnum[i];k++)
	{
		Rmotion1 = motion1[i*K+k];
		Rmotion2 = motion2[i*K+k];
		Rcolor1 =  color11[i*K+k];
		Rcolor2 =  color12[i*K+k];
		Rcolor3 =  color13[i*K+k];
        Rlocation1 = midP[(K)*5*i+k],Rlocation2 = midP[(K)*5*i+(K)*1+k];
		
		MDist=abs(Rmotion1-Lmotion1)+abs(Rmotion2-Lmotion2);
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3));
		if (LDist>ring[i*K+k]&&LDist<(*Par))
		{
                MScale=MScale + MDist/(LDist+1.0);
                CScale=CScale + CDist/(LDist+1.0);
        }

	 }
	 MSaliencyM[i*K+j]=MScale;
	 CSaliencyM[i*K+j]=CScale;
     return;
		
}
void cudacomputec(double *MSaliencyM,double * CSaliencyM, double* midP,double *motion1, double * motion2,double * color11,double * color12,double * color13,double * ring,double *K1,double *N1,double *Par,double *spnum)
{
	double * dev_MSaliencyM,* dev_CSaliencyM;
	double *dev_mid, *dev_motion1,*dev_motion2,*dev_color11,*dev_color12,*dev_color13,*dev_ring;
    double *dev_K1,*dev_N1,*dev_Par,*dev_spnum;
    int K=(int)(*K1),N=(int)(*N1);

	cudaMalloc((void **)&dev_mid, sizeof(double)* (K) * 5 *N);
	cudaMalloc((void **)&dev_motion1, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_motion2, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_color11, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_color12, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_color13, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_ring, sizeof(double)* K * N);
    cudaMalloc((void **)&dev_K1, sizeof(double));
    cudaMalloc((void **)&dev_N1, sizeof(double));
    cudaMalloc((void **)&dev_Par, sizeof(double));
	cudaMalloc((void **)&dev_MSaliencyM, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_CSaliencyM, sizeof(double)* K * N);
    cudaMalloc((void **)&dev_spnum, sizeof(double)*N);

	cudaMemcpy(dev_mid, midP, sizeof(double)* (K)*5 *N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_motion1, motion1, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_motion2, motion2, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_color11, color11, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_color12, color12, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_color13, color13, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_K1, K1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N1, N1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par, Par, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ring, ring, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spnum, spnum, sizeof(double)*N, cudaMemcpyHostToDevice);


	dim3 threads(K);
	dim3 grids(N);
	computec_kernel << <grids, threads >> >(dev_MSaliencyM,dev_CSaliencyM, dev_mid,dev_motion1,dev_motion2,dev_color11,dev_color12,dev_color13,dev_ring,dev_K1,dev_N1,dev_Par,dev_spnum);

	cudaMemcpy(MSaliencyM, dev_MSaliencyM, sizeof(double)*K*N, cudaMemcpyDeviceToHost);
	cudaMemcpy(CSaliencyM, dev_CSaliencyM, sizeof(double)*K*N, cudaMemcpyDeviceToHost);

	cudaFree(dev_mid);
	cudaFree(dev_motion1);
	cudaFree(dev_motion2);
	cudaFree(dev_color11);
	cudaFree(dev_color12);
	cudaFree(dev_color13);
    cudaFree(dev_ring);
    cudaFree(dev_K1);
    cudaFree(dev_N1);
	cudaFree(dev_MSaliencyM);
	cudaFree(dev_CSaliencyM);
    cudaFree(dev_Par);
    cudaFree(dev_spnum);

}	




