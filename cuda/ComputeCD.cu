#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>


__global__ void Transfer_kernel(double * SpNum, double *MidP,double*Color_Dist,double *WeightD,double *Distance,double *ring)
{
	int z = threadIdx.x;
	int SNum = (int)(SpNum[0]);
	if (z >= SNum)
		return;
    Color_Dist[z]=0;
    for(int i=0;i<SNum;i++)
    {
        double LDist=sqrt((MidP[i + SNum * 0]-MidP[z + SNum * 0])*(MidP[i + SNum * 0]-MidP[z + SNum * 0])+(MidP[i + SNum * 1]-MidP[z + SNum * 1])*(MidP[i + SNum * 1]-MidP[z + SNum * 1]));
        if(LDist<Distance[0])//max(min(Distance[0],ring[z]),30.0)
        {
            double r = MidP[z + SNum * 2] - MidP[i + SNum * 2];
            double g = MidP[z + SNum * 3] - MidP[i + SNum * 3];
            double b = MidP[z + SNum * 4] - MidP[i + SNum * 4];
            double s = MidP[z + SNum * 9] - MidP[i + SNum * 9];
            double v = MidP[z + SNum * 10] - MidP[i + SNum * 10];
            double Dist = sqrt(r*r + g*g + b*b + s*s + v*v);
            Color_Dist[z]+=exp(-Dist*WeightD[0]);
        }
    }
	return;
}
void ComputeCD(double * SpNum, double *MidP,double*Color_Dist,double *WeightD,double *Distance,double *ring)
{
	double *dev_Color_Dist,*dev_WeightD,*dev_Distance;
	double *dev_SpNum, *dev_MidP,*dev_ring;


	int Spnum = (int)(SpNum[0]);

	cudaMalloc((void **)&dev_MidP, sizeof(double)* Spnum * 11);
    cudaMalloc((void **)&dev_Distance, sizeof(double));
    cudaMalloc((void **)&dev_Color_Dist, sizeof(double)* Spnum);
	
	cudaMalloc((void **)&dev_WeightD, sizeof(double));
    cudaMalloc((void **)&dev_SpNum, sizeof(double));
	cudaMalloc((void **)&dev_ring, sizeof(double)* Spnum);

    cudaMemcpy(dev_ring, ring, sizeof(double)* Spnum, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_SpNum, SpNum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Distance, Distance, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_WeightD, WeightD, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MidP, MidP, sizeof(double)* Spnum * 11, cudaMemcpyHostToDevice);
	
	//printf("%f %f %f %f %f %f %f %f %f %f %f %f \n", Model_Rgb[0], Model_Rgb[1], Model_Rgb[2], Model_Rgb[3], Model_Rgb[4], Model_Rgb[5], Model_Rgb[6], Model_Rgb[7], Model_Rgb[8], Model_Rgb[9], Model_Rgb[10], Model_Rgb[11]);
   // printf("%f %f %f %f %f %f %f %f\n", Model_Lab[0], Model_Lab[1], Model_Lab[2], Model_Lab[3], Model_Lab[4], Model_Lab[5], Model_Lab[6], Model_Lab[7]);
    printf("%f\n", Distance[0]);

	dim3 threads(Spnum);
	Transfer_kernel << <1, threads >> >(dev_SpNum, dev_MidP,dev_Color_Dist,dev_WeightD,dev_Distance,dev_ring);

    cudaMemcpy(Color_Dist, dev_Color_Dist, sizeof(double)*Spnum, cudaMemcpyDeviceToHost);


	cudaFree(dev_SpNum);
	cudaFree(dev_MidP);
    cudaFree(dev_Color_Dist);
    cudaFree(dev_WeightD);
    cudaFree(dev_Distance);
    cudaFree(dev_ring);
}




