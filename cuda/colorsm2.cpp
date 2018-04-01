#include"mex.h"
#define ImageSize 300
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
extern void colorsm2(double *Result, double* midP,double * ring,double *K1,double* Ind,double *N1,double *SaA,double *SaA0,double *SaA2,double *Par1,double *Par2,double *spnum);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *midP;
    double *SaA,*SaA0,*SaA2;
	double *Result;
    double *K1,*Ind,*N1,*ring;
    double * Par1,*Par2,*spnum;
	midP=mxGetPr(prhs[0]);
	ring=mxGetPr(prhs[1]);
    Ind=mxGetPr(prhs[2]);
    K1=mxGetPr(prhs[3]);
    SaA=mxGetPr(prhs[4]);
    SaA0=mxGetPr(prhs[5]);
    SaA2=mxGetPr(prhs[6]);
    N1=mxGetPr(prhs[7]);
    Par1=mxGetPr(prhs[8]);
    Par2=mxGetPr(prhs[9]);
    spnum=mxGetPr(prhs[10]);
    int K=(int)(*K1);
    int N=(int)(*N1);
	plhs[0]=mxCreateDoubleMatrix(K,1,mxREAL);
	Result=mxGetPr(plhs[0]);
	colorsm2(Result,midP,ring,K1,Ind,N1,SaA,SaA0,SaA2,Par1,Par2,spnum);



}