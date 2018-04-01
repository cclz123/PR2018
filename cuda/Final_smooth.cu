#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>


__global__ void smooth_kernel(double *FinS,double* midP,double * ring,double *MSaliencyM ,double *K1,double *N1,double *Par,double *spnum ,double *Par1,double *Par2)
{
	int i=blockIdx.x;
	int j=threadIdx.x;
    int K=(int)(*K1),N=(int)(*N1);
	if (blockIdx.x >= N || threadIdx.x >= spnum[i])
		return;

	double Lcolor1,Lcolor2,Lcolor3,Llocation1,Llocation2,Lcolor4,Lcolor5;
	double Rcolor1,Rcolor2,Rcolor3,Rlocation1,Rlocation2,Rcolor4,Rcolor5;
	double Tweight=0,weight1=0;
	double MSS=0;
	double CDist=0,LDist=0;
    Llocation1 = midP[(K)*11*i+j],Llocation2 = midP[(K)*11*i+(K)*1+j],Lcolor1 = midP[(K)*11*i+(K)*2+j], Lcolor2 = midP[(K)*11*i+(K)*3+j],Lcolor3 = midP[(K)*11*i+(K)*4+j],Lcolor4 = midP[(K)*11*i+(K)*9+j],Lcolor5 = midP[(K)*11*i+(K)*10+j];

	for (int k=0;k<spnum[i];k++)
	{
		Rlocation1 = midP[(K)*11*i+k],Rlocation2 = midP[(K)*11*i+(K)*1+k],Rcolor1 = midP[(K)*11*i+(K)*2+k], Rcolor2 = midP[(K)*11*i+(K)*3+k],Rcolor3 = midP[(K)*11*i+(K)*4+k],Rcolor4 = midP[(K)*11*i+(K)*9+k],Rcolor5 = midP[(K)*11*i+(K)*10+k];
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
       // CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
		if (LDist<min(max(ring[i*K+k],*Par1),*Par2))
		{
			weight1=exp(-CDist*(*Par));
			Tweight+=weight1;
			MSS=MSS+MSaliencyM[K*i+k]*weight1;
		}
	}
    if(i<N-1)
    for (int k=0;k<spnum[i+1];k++)
	{
		Rlocation1 = midP[(K)*11*(i+1)+k],Rlocation2 = midP[(K)*11*(i+1)+(K)*1+k],Rcolor1 = midP[(K)*11*(i+1)+(K)*2+k], Rcolor2 = midP[(K)*11*(i+1)+(K)*3+k],Rcolor3 = midP[(K)*11*(i+1)+(K)*4+k],Rcolor4 = midP[(K)*11*(i+1)+(K)*9+k],Rcolor5 = midP[(K)*11*(i+1)+(K)*10+k];
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
        //CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
		if (LDist<min(max(ring[(i+1)*K+k],*Par1),*Par2))
		{
			weight1=exp(-CDist*(*Par));
			Tweight+=weight1;
			MSS=MSS+MSaliencyM[K*(i+1)+k]*weight1;
		}
	}
    if (i>0)
    for (int k=0;k<spnum[i-1];k++)
	{
		Rlocation1 = midP[(K)*11*(i-1)+k],Rlocation2 = midP[(K)*11*(i-1)+(K)*1+k],Rcolor1 = midP[(K)*11*(i-1)+(K)*2+k], Rcolor2 = midP[(K)*11*(i-1)+(K)*3+k],Rcolor3 = midP[(K)*11*(i-1)+(K)*4+k],Rcolor4 = midP[(K)*11*(i-1)+(K)*9+k],Rcolor5 = midP[(K)*11*(i-1)+(K)*10+k];
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
        //CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
		if (LDist<min(max(ring[(i-1)*K+k],*Par1),*Par2))
		{
			weight1=exp(-CDist*(*Par));
			Tweight+=weight1;
			MSS=MSS+MSaliencyM[K*(i-1)+k]*weight1;
		}
	}
	MSS=MSS/Tweight;
	FinS[K*i+j]=MSS;
	return;

}
void Final_smooth(double *FinS,double* midP,double * ring,double *MSaliencyM,double *K1,double *N1,double *Par,double *spnum,double *Par1,double *Par2)
{
	double * dev_FinS;
	double *dev_mid,*dev_ring,*dev_MSaliencyM;
    double *dev_K1,*dev_N1, *dev_Par,*dev_spnum,*dev_Par1,*dev_Par2;
    int K=(int)(*K1),N=(int)(*N1);

	cudaMalloc((void **)&dev_mid, sizeof(double)* (K) * 11 * N);
	cudaMalloc((void **)&dev_ring, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_MSaliencyM, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_FinS, sizeof(double)* K * N);
    cudaMalloc((void **)&dev_K1, sizeof(double));
    cudaMalloc((void **)&dev_N1, sizeof(double));
    cudaMalloc((void **)&dev_Par, sizeof(double));
    cudaMalloc((void **)&dev_Par1, sizeof(double));
    cudaMalloc((void **)&dev_Par2, sizeof(double));
    cudaMalloc((void **)&dev_spnum, sizeof(double)*N);

	cudaMemcpy(dev_K1, K1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N1, N1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mid, midP, sizeof(double)* (K) * 11 * N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ring, ring, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MSaliencyM, MSaliencyM, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par, Par, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par1, Par1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par2, Par2, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spnum, spnum, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 threads(K);
	dim3 grids(N);
	smooth_kernel << <grids, threads >> >(dev_FinS,dev_mid,dev_ring,dev_MSaliencyM,dev_K1,dev_N1,dev_Par,dev_spnum,dev_Par1,dev_Par2);

	cudaMemcpy(FinS, dev_FinS, sizeof(double)*K*N, cudaMemcpyDeviceToHost);

	cudaFree(dev_mid);
	cudaFree(dev_FinS);
	cudaFree(dev_MSaliencyM);
    cudaFree(dev_ring);
    cudaFree(dev_K1);
    cudaFree(dev_N1);
    cudaFree(dev_Par);
    cudaFree(dev_spnum);
    cudaFree(dev_Par1);
    cudaFree(dev_Par2);
}	




