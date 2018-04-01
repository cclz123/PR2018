#include"mex.h"
#define ImageSize 300
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
extern void FrameTrans(double *NewSM, double* Model_Trans,double * SpNum,double *MidP,double* Model_Rgb,double *Neighbor,double *Model_Lab,double *Grad_M,double *MaxDim,double *Color_Dist,double *WeightD,double *Model_Pos,double *Distance,double *ring,double *weg1,double *weg2);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *SpNum,*MidP,*Model_Rgb,*Neighbor,*Model_Lab,*Grad_M,*MaxDim,*Model_Pos,*Distance;
    double *NewSM,*Model_Trans,*Color_Dist,*WeightD,*ring,*weg1,*weg2;
    
    
	SpNum=mxGetPr(prhs[0]);
	MidP=mxGetPr(prhs[1]);
    Model_Rgb=mxGetPr(prhs[2]);
    Neighbor=mxGetPr(prhs[3]);
    Model_Lab=mxGetPr(prhs[4]);
    Grad_M=mxGetPr(prhs[5]);
    MaxDim=mxGetPr(prhs[6]);
    WeightD=mxGetPr(prhs[7]);
    Model_Pos=mxGetPr(prhs[8]);
    Distance=mxGetPr(prhs[9]);
    ring=mxGetPr(prhs[10]);
    weg1=mxGetPr(prhs[11]);
    weg2=mxGetPr(prhs[12]);
    int N=(int)(*MaxDim);
	plhs[0]=mxCreateDoubleMatrix(N,1,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(N,1,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(N,1,mxREAL);
	NewSM=mxGetPr(plhs[0]);
    Model_Trans=mxGetPr(plhs[1]);
    Color_Dist=mxGetPr(plhs[2]);
	FrameTrans(NewSM,Model_Trans,SpNum,MidP,Model_Rgb,Neighbor,Model_Lab,Grad_M,MaxDim,Color_Dist,WeightD,Model_Pos,Distance,ring,weg1,weg2);
}