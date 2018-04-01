#include"mex.h"
#define ImageSize 300
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
extern void ModelT(double *Result, double* Result2,double * Result3,double * Result4,double *ModelNum,double* ModelSm,double *SpNum,double *MaxDim,double *ColorD,double *SumG,double*MainColor);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *ModelNum,*ModelSm,*SpNum,*MaxDim,*ColorD,*SumG,*MainColor;
	double *Result,*Result2,*Result3,*Result4;
	ModelNum=mxGetPr(prhs[0]);
	ModelSm=mxGetPr(prhs[1]);
    SpNum=mxGetPr(prhs[2]);
    MaxDim=mxGetPr(prhs[3]);
    ColorD=mxGetPr(prhs[4]);
    SumG=mxGetPr(prhs[5]);
    MainColor=mxGetPr(prhs[6]);
    int N=(int)(*MaxDim);
    int K=(int)(*ModelNum);
    
	plhs[0]=mxCreateDoubleMatrix(N,1,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(N,K,mxREAL);
    plhs[2]=mxCreateDoubleMatrix(N,K,mxREAL);
    plhs[3]=mxCreateDoubleMatrix(N,K,mxREAL);
	Result=mxGetPr(plhs[0]);
    Result2=mxGetPr(plhs[1]);
    Result3=mxGetPr(plhs[2]);
    Result4=mxGetPr(plhs[3]);
    //Result[0]=ModelSm[0];
    //printf("%f",ModelNum[0]);
	ModelT(Result,Result2,Result3,Result4,ModelNum,ModelSm,SpNum,MaxDim,ColorD,SumG,MainColor);
}