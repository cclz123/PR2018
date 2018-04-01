#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"E:\Program Files\MATLAB\R2012b\extern\include\mex.h"
#include <stdio.h>
#include <algorithm>


__global__ void Transfer_kernel(double *NewSM, double* Model_Trans, double * SpNum, double *MidP, double* Model_Rgb, double *Neighbor, double *Model_Lab, double *Grad_M, double *MaxDim,double*Color_Dist,double *WeightD,double *Model_Pos,double *Distance,double *ring,double* weg1,double *weg2)
{
	int z = threadIdx.x;
	int SNum = (int)(SpNum[0]);
	if (z >= SNum)
		return;

    double LDist=sqrt((Model_Pos[0]-MidP[z + SNum * 0])*(Model_Pos[0]-MidP[z + SNum * 0])+(Model_Pos[1]-MidP[z + SNum * 1])*(Model_Pos[1]-MidP[z + SNum * 1]));
    if(LDist>max(min(Distance[0],ring[z]*1.5),30.0))
     {
        NewSM[z] = 1000;
        Model_Trans[z] = 0;
        return ;
    }
	double r = (MidP[z + SNum * 2] - Model_Rgb[0]);
	double g = (MidP[z + SNum * 3] - Model_Rgb[1]);
	double b = (MidP[z + SNum * 4] - Model_Rgb[2]);
	double s = (MidP[z + SNum * 9] - Model_Lab[0]);
	double v = (MidP[z + SNum * 10] - Model_Lab[1]);
	double Dist = sqrt(r*r + g*g + b*b + s*s + v*v);

    Color_Dist[z]=Dist*WeightD[0];
	int count = 0;
	int S_M[50];
	for (int i = 0; i < 50; i++)
	{
		if (Neighbor[z + SNum*i] != 0)
		{
			S_M[i] = Neighbor[z + SNum * i] - 1;
			count++;
		}
	}
	double minDist[3];
	
	
	minDist[0] = 10;
	int Ind[3];
	Ind[0] = 0; Ind[1] = 0; Ind[2] = 0;
	double temp;
	for (int i = 0; i < count; i++)
	{
		r = (MidP[S_M[i] + SNum * 2] - Model_Rgb[3]);
		g = (MidP[S_M[i] + SNum * 3] - Model_Rgb[4]);
		b = (MidP[S_M[i] + SNum * 4] - Model_Rgb[5]);
		s = (MidP[S_M[i] + SNum * 9] - Model_Lab[2]);
		v = (MidP[S_M[i] + SNum * 10] - Model_Lab[3]);
		temp = sqrt(r*r + g*g + b*b + s*s + v*v);
		if (temp < minDist[0])
		{
			minDist[0] = temp;
			Ind[0] = i;
		}
	}
    minDist[1] = 10;
	for (int i = 0; i < count; i++)
	{
		r = (MidP[S_M[i] + SNum * 2] - Model_Rgb[6]);
		g = (MidP[S_M[i] + SNum * 3] - Model_Rgb[7]);
		b = (MidP[S_M[i] + SNum * 4] - Model_Rgb[8]);
		s = (MidP[S_M[i] + SNum * 9] - Model_Lab[4]);
		v = (MidP[S_M[i] + SNum * 10] - Model_Lab[5]);
		temp = sqrt(r*r + g*g + b*b + s*s + v*v);
		if (temp < minDist[1])
		{
			minDist[1] = temp;
			Ind[1] = i;
		}
	}
    minDist[2] = 10;
	for (int i = 0; i < count; i++)
	{
		r = (MidP[S_M[i] + SNum * 2] - Model_Rgb[9]);
		g = (MidP[S_M[i] + SNum * 3] - Model_Rgb[10]);
		b = (MidP[S_M[i] + SNum * 4] - Model_Rgb[11]);
		s = (MidP[S_M[i] + SNum * 9] - Model_Lab[6]);
		v = (MidP[S_M[i] + SNum * 10] - Model_Lab[7]);
		temp = sqrt(r*r + g*g + b*b + s*s + v*v);
		if (temp < minDist[2])
		{
			minDist[2] = temp;
			Ind[2] = i;
		}
	}





	double pos11 = MidP[z + SNum * 0], pos12 = MidP[z + SNum * 1];
	double SUM[3] = {0};

	for (int jj = 0; jj < 3; jj++)
		if (Ind[jj] != 0)
		{
			double pos21 = MidP[S_M[Ind[jj]] + SNum * 0], pos22 = MidP[S_M[Ind[jj]] + SNum * 1];
			double DIS = sqrt((pos11 - pos21)*(pos11 - pos21) + (pos12 - pos22)*(pos12 - pos22));
			int range_w = min(DIS / 2, 10.0);//10.0
			//double D = min(DIS / 2, 30.0);
            double D = min(DIS / 2, 10.0);
			int pos1 = (pos11 - pos21) * D / DIS + pos21;
			int pos2 = (pos12 - pos22) * D / DIS + pos22;
			double Max = 0;
			for (int k1 = (pos1 - range_w); k1 <= (pos1 + range_w); k1++)
				if (k1 > 0 && k1 <= 300)
					for (int k2 = (pos2 - range_w); k2 <= (pos2 + range_w); k2++)
						if (k2 > 0 && k2 <= 300)
							if ((pos1 - k1)*(pos1 - k1) + (pos2 - k2)*(pos2 - k2) <= range_w*range_w&&Max < Grad_M[k1 - 1 + (k2 - 1) * 300])
								Max = Grad_M[k1 - 1 + (k2 - 1) * 300];
D = min(DIS / 2, 20.0);
            pos1 = (pos11 - pos21) * D / DIS + pos21;
            pos2 = (pos12 - pos22) * D / DIS + pos22;
 
            for (int k1 = (pos1 - range_w); k1 <= (pos1 + range_w); k1++)
                if (k1 > 0 && k1 <= 300)
                    for (int k2 = (pos2 - range_w); k2 <= (pos2 + range_w); k2++)
                        if (k2 > 0 && k2 <= 300)
                            if ((pos1 - k1)*(pos1 - k1) + (pos2 - k2)*(pos2 - k2) <= range_w*range_w&&Max < Grad_M[k1 - 1 + (k2 - 1) * 300])
                                Max = Grad_M[k1 - 1 + (k2 - 1) * 300];
 D = min(DIS / 2, 30.0);
            pos1 = (pos11 - pos21) * D / DIS + pos21;
            pos2 = (pos12 - pos22) * D / DIS + pos22;
 
            for (int k1 = (pos1 - range_w); k1 <= (pos1 + range_w); k1++)
                if (k1 > 0 && k1 <= 300)
                    for (int k2 = (pos2 - range_w); k2 <= (pos2 + range_w); k2++)
                        if (k2 > 0 && k2 <= 300)
                            if ((pos1 - k1)*(pos1 - k1) + (pos2 - k2)*(pos2 - k2) <= range_w*range_w&&Max < Grad_M[k1 - 1 + (k2 - 1) * 300])
                                Max = Grad_M[k1 - 1 + (k2 - 1) * 300];


			SUM[jj] = Max;
		}
	NewSM[z] = weg1[0] * (Dist )+ (minDist[0]*exp(-5*SUM[0]) + minDist[1]*exp(-5*SUM[1]) + minDist[2]*exp(-5*SUM[2])) *weg2[0]/exp((SUM[0]+SUM[1]+SUM[2])*0.3);//
	Model_Trans[z] = (minDist[0]*exp(-5*SUM[0]) + minDist[1]*exp(-5*SUM[1]) + minDist[2]*exp(-5*SUM[2])) *weg2[0]/exp((SUM[0]+SUM[1]+SUM[2])*0.3);
	return;
}
void ModelTransfer(double *NewSM, double* Model_Trans, double * SpNum, double *MidP, double* Model_Rgb, double *Neighbor, double *Model_Lab, double *Grad_M, double *MaxDim,double *Color_Dist,double *WeightD,double *Model_Pos,double *Distance,double *ring,double *weg1,double* weg2)
{
	double * dev_NewSM, *dev_Model_Trans,*dev_Color_Dist,*dev_WeightD,*dev_Model_Pos,*dev_Distance;
	double *dev_SpNum, *dev_MidP, *dev_Model_Rgb, *dev_Neighbor, *dev_Model_Lab, *dev_Grad_M, *dev_MaxDim,*dev_ring,*dev_weg1,*dev_weg2;
    double *dev_temp;
	int MDim = (int)(MaxDim[0]);
	int Spnum = (int)(SpNum[0]);

cudaMalloc((void **)&dev_temp, sizeof(double));

	cudaMalloc((void **)&dev_MidP, sizeof(double)* Spnum * 11);
    cudaMalloc((void **)&dev_Distance, sizeof(double));
    cudaMalloc((void **)&dev_Model_Pos, sizeof(double)* 2);
	cudaMalloc((void **)&dev_NewSM, sizeof(double)* MDim);
    cudaMalloc((void **)&dev_Color_Dist, sizeof(double)* MDim);
	cudaMalloc((void **)&dev_Model_Trans, sizeof(double)* MDim);
	cudaMalloc((void **)&dev_WeightD, sizeof(double));
    cudaMalloc((void **)&dev_SpNum, sizeof(double));
	cudaMalloc((void **)&dev_Model_Rgb, sizeof(double) * 12);
	cudaMalloc((void **)&dev_Model_Lab, sizeof(double) * 8);
	cudaMalloc((void **)&dev_Neighbor, sizeof(double) * Spnum * 50);
	cudaMalloc((void **)&dev_Grad_M, sizeof(double) * 300 * 300);
	cudaMalloc((void **)&dev_MaxDim, sizeof(double));
    cudaMalloc((void **)&dev_ring, sizeof(double)* MDim);
    cudaMalloc((void **)&dev_weg1, sizeof(double));
    cudaMalloc((void **)&dev_weg2, sizeof(double));

	cudaMemcpy(dev_SpNum, SpNum, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ring, ring, sizeof(double)* MDim, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Distance, Distance, sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(dev_weg1, weg1, sizeof(double), cudaMemcpyHostToDevice);
cudaMemcpy(dev_weg2, weg2, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_WeightD, WeightD, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MidP, MidP, sizeof(double)* Spnum * 11, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Model_Rgb, Model_Rgb, sizeof(double) * 12, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Model_Lab, Model_Lab, sizeof(double) * 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_Model_Pos, Model_Pos, sizeof(double) * 2, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Grad_M, Grad_M, sizeof(double) * 300 * 300, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Neighbor, Neighbor, sizeof(double) * Spnum * 50, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_MaxDim, MaxDim, sizeof(double), cudaMemcpyHostToDevice);
	//printf("%f %f %f %f %f %f %f %f %f %f %f %f \n", Model_Rgb[0], Model_Rgb[1], Model_Rgb[2], Model_Rgb[3], Model_Rgb[4], Model_Rgb[5], Model_Rgb[6], Model_Rgb[7], Model_Rgb[8], Model_Rgb[9], Model_Rgb[10], Model_Rgb[11]);
   // printf("%f %f %f %f %f %f %f %f\n", Model_Lab[0], Model_Lab[1], Model_Lab[2], Model_Lab[3], Model_Lab[4], Model_Lab[5], Model_Lab[6], Model_Lab[7]);
    //printf("%f\n", Distance[0]);

	dim3 threads(Spnum);
	Transfer_kernel << <1, threads >> >(dev_NewSM, dev_Model_Trans, dev_SpNum, dev_MidP, dev_Model_Rgb, dev_Neighbor, dev_Model_Lab, dev_Grad_M, dev_MaxDim,dev_Color_Dist,dev_WeightD,dev_Model_Pos,dev_Distance,dev_ring,dev_weg1,dev_weg2);
	cudaMemcpy(NewSM, dev_NewSM, sizeof(double)*MDim, cudaMemcpyDeviceToHost);
	cudaMemcpy(Model_Trans, dev_Model_Trans, sizeof(double)*MDim, cudaMemcpyDeviceToHost);
    cudaMemcpy(Color_Dist, dev_Color_Dist, sizeof(double)*MDim, cudaMemcpyDeviceToHost);
	cudaFree(dev_NewSM);
	cudaFree(dev_Model_Trans);
	cudaFree(dev_SpNum);
	cudaFree(dev_MidP);
	cudaFree(dev_Model_Rgb);
	cudaFree(dev_Neighbor);
	cudaFree(dev_Model_Lab);
	cudaFree(dev_Grad_M);
	cudaFree(dev_MaxDim);
    cudaFree(dev_Color_Dist);
    cudaFree(dev_WeightD);
    cudaFree(dev_Model_Pos);
    cudaFree(dev_Distance);
    cudaFree(dev_ring);
cudaFree(dev_weg1);
cudaFree(dev_weg2);
}




