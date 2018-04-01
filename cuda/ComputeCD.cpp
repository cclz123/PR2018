#include"mex.h"
#define ImageSize 300
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
extern void ComputeCD(double * SpNum, double *Mid,double*Color_Dist,double *WeightD,double *Distance,double *ring);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    double *SpNum,*MidP,*Distance;
    double *Color_Dist,*WeightD,*ring;
    
    
	SpNum=mxGetPr(prhs[0]);
	MidP=mxGetPr(prhs[1]);
    WeightD=mxGetPr(prhs[2]);
    Distance=mxGetPr(prhs[3]);
    ring=mxGetPr(prhs[4]);
    int N=(int)(*SpNum);
	plhs[0]=mxCreateDoubleMatrix(N,1,mxREAL);
    Color_Dist=mxGetPr(plhs[0]);
	ComputeCD(SpNum,MidP,Color_Dist,WeightD,Distance,ring);
}