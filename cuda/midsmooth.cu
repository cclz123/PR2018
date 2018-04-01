#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>


__global__ void smooth_kernel(double *FinS,double* midP,double * ring,double *SaliencyM,double* AfterSaliencyM ,double *K1,double *N1,double *Par,double *spnum ,double *Par1,double *N2,double *Flag,double *Par2)
{
	int j=threadIdx.x,i=(int)(*N1);
    int K=(int)(*K1),N=(int)(*N2);
	if (threadIdx.x >= K)
		return;
	double Lcolor1,Lcolor2,Lcolor3,Lcolor4,Lcolor5,Llocation1,Llocation2;//N1是列标记，N2是列总和
	double Rcolor1,Rcolor2,Rcolor3,Rcolor4,Rcolor5,Rlocation1,Rlocation2;
	double Tweight=0,weight1=0;
	double MSS=0;
	double CDist=0,LDist=0;
    Llocation1 = midP[(K)*11*i+j],Llocation2 = midP[(K)*11*i+(K)*1+j],Lcolor1 = midP[(K)*11*i+(K)*2+j], Lcolor2 = midP[(K)*11*i+(K)*3+j],Lcolor3 = midP[(K)*11*i+(K)*4+j],Lcolor4 = midP[(K)*11*i+(K)*9+j],Lcolor5 = midP[(K)*11*i+(K)*10+j];
	for (int k=0;k<spnum[i];k++)
	{
		Rlocation1 = midP[(K)*11*i+k],Rlocation2 = midP[(K)*11*i+(K)*1+k],Rcolor1 = midP[(K)*11*i+(K)*2+k], Rcolor2 = midP[(K)*11*i+(K)*3+k],Rcolor3 = midP[(K)*11*i+(K)*4+k],Rcolor4 = midP[(K)*11*i+(K)*9+k],Rcolor5 = midP[(K)*11*i+(K)*10+k];
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
        //CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
		if (LDist<min(max(ring[i*K+k],*Par1),*Par2))
		{
			weight1=exp(-CDist*(*Par));
			Tweight+=weight1;
			MSS=MSS+SaliencyM[k]*weight1;
		}
	}
    if(i<N-1&&Flag[i+1]!=0)
        for (int k=0;k<spnum[i+1];k++)
        {
            Rlocation1 = midP[(K)*11*(i+1)+k],Rlocation2 = midP[(K)*11*(i+1)+(K)*1+k],Rcolor1 = midP[(K)*11*(i+1)+(K)*2+k], Rcolor2 = midP[(K)*11*(i+1)+(K)*3+k],Rcolor3 = midP[(K)*11*(i+1)+(K)*4+k],Rcolor4 = midP[(K)*11*(i+1)+(K)*9+k],Rcolor5 = midP[(K)*11*(i+1)+(K)*10+k];
            LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
            //CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
            CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
            if (LDist<min(max(ring[(i+1)*K+k],*Par1),*Par2))
            {
            	weight1=exp(-CDist*(*Par)*2);
                Tweight+=weight1;
            	MSS=MSS+AfterSaliencyM[K*(i+1)+k]*weight1;
            }
        }
    if (i>0&&Flag[i-1]!=0)
    for (int k=0;k<spnum[i-1];k++)
	{
		Rlocation1 = midP[(K)*11*(i-1)+k],Rlocation2 = midP[(K)*11*(i-1)+(K)*1+k],Rcolor1 = midP[(K)*11*(i-1)+(K)*2+k], Rcolor2 = midP[(K)*11*(i-1)+(K)*3+k],Rcolor3 = midP[(K)*11*(i-1)+(K)*4+k],Rcolor4 = midP[(K)*11*(i-1)+(K)*9+k],Rcolor5 = midP[(K)*11*(i-1)+(K)*10+k];
		LDist=abs(Llocation1-Rlocation1)+abs(Llocation2-Rlocation2);
        //CDist=(abs(Lcolor1-Rcolor1)+abs(Lcolor3-Rcolor3)+abs(Lcolor2-Rcolor2)+abs(Lcolor4-Rcolor4)+abs(Lcolor5-Rcolor5));
		CDist=sqrt((Lcolor1-Rcolor1)*(Lcolor1-Rcolor1)+(Lcolor3-Rcolor3)*(Lcolor3-Rcolor3)+(Lcolor2-Rcolor2)*(Lcolor2-Rcolor2)+(Lcolor4-Rcolor4)*(Lcolor4-Rcolor4)+(Lcolor5-Rcolor5)*(Lcolor5-Rcolor5));
		if (LDist<min(max(ring[(i-1)*K+k],*Par1),*Par2))
		{
			weight1=exp(-CDist*(*Par)*2);
			Tweight+=weight1;
			MSS=MSS+AfterSaliencyM[K*(i-1)+k]*weight1;
		}
	}
	FinS[j]=MSS/Tweight;
	return;

}

void midsmooth(double *FinS,double* midP,double * ring,double *SaliencyM,double* AfterSaliencyM,double *K1,double *N1,double *Par,double *spnum,double *Par1,double *N2,double *Flag,double *Par2)
{
	double * dev_FinS;
	double *dev_mid,*dev_ring,*dev_SaliencyM,*dev_AfterSaliencyM;
    double *dev_K1,*dev_N1,*dev_N2, *dev_Par,*dev_spnum,*dev_Par1,*dev_Par2,*dev_Flag;
    int K=(int)(*K1),N=(int)(*N2);
	cudaMalloc((void **)&dev_mid, sizeof(double)* (K) * N *11);
	cudaMalloc((void **)&dev_ring, sizeof(double)* K*N);
	cudaMalloc((void **)&dev_AfterSaliencyM, sizeof(double)* K * N);
	cudaMalloc((void **)&dev_SaliencyM, sizeof(double)* K);
	cudaMalloc((void **)&dev_FinS, sizeof(double)* K);
    cudaMalloc((void **)&dev_N2, sizeof(double));
    cudaMalloc((void **)&dev_K1, sizeof(double));
    cudaMalloc((void **)&dev_N1, sizeof(double));
    cudaMalloc((void **)&dev_Par, sizeof(double));
    cudaMalloc((void **)&dev_Par1, sizeof(double));
    cudaMalloc((void **)&dev_Par2, sizeof(double));
    cudaMalloc((void **)&dev_spnum, sizeof(double)*N);
    cudaMalloc((void **)&dev_Flag, sizeof(double)*N);

	cudaMemcpy(dev_K1, K1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_N2, N2, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_N1, N1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mid, midP, sizeof(double)* (K) * 11*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ring, ring, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_AfterSaliencyM, AfterSaliencyM, sizeof(double)* K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_SaliencyM, SaliencyM, sizeof(double)* K, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par, Par, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par1, Par1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Par2, Par2, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_spnum, spnum, sizeof(double)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Flag, Flag, sizeof(double)*N, cudaMemcpyHostToDevice);

	dim3 threads(K);
	smooth_kernel << <1, threads >> >(dev_FinS,dev_mid,dev_ring,dev_SaliencyM,dev_AfterSaliencyM,dev_K1,dev_N1,dev_Par,dev_spnum,dev_Par1,dev_N2,dev_Flag,dev_Par1);

	cudaMemcpy(FinS, dev_FinS, sizeof(double)*K, cudaMemcpyDeviceToHost);

	cudaFree(dev_mid);
	cudaFree(dev_FinS);
    cudaFree(dev_N2);
    cudaFree(dev_N1);
	cudaFree(dev_AfterSaliencyM);
	cudaFree(dev_SaliencyM);
    cudaFree(dev_ring);
    cudaFree(dev_K1);
    cudaFree(dev_Par);
    cudaFree(dev_spnum);
    cudaFree(dev_Par1);
    cudaFree(dev_Par2);
    cudaFree(dev_Flag);
}	




