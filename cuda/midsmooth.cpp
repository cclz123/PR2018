#include"mex.h"
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
extern void midsmooth(double *FinS,double* midP,double * ring,double *SaliencyM,double* AfterSaliencyM,double *K1,double *N1,double *Par,double *spnum,double *Par1,double *N2,double *Flag,double *Par2);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *midP;
	double *FinS,*ring,*SaliencyM,*AfterSaliencyM;
    double *K1,*N1,*Par,*SPnum,*Par1,*Par2,*N2,*Flag;
	midP=mxGetPr(prhs[0]);
	ring=mxGetPr(prhs[1]);
    SaliencyM=mxGetPr(prhs[2]);
    AfterSaliencyM=mxGetPr(prhs[3]);
    K1=mxGetPr(prhs[4]);
    N1=mxGetPr(prhs[5]);
    Par=mxGetPr(prhs[6]);
    SPnum=mxGetPr(prhs[7]);
    Par1=mxGetPr(prhs[8]);
    N2=mxGetPr(prhs[9]);
    Flag=mxGetPr(prhs[10]);
    Par2=mxGetPr(prhs[11]);
    int K=(int)(*K1);
	plhs[0]=mxCreateDoubleMatrix(K,1,mxREAL);
	FinS=mxGetPr(plhs[0]);
	midsmooth(FinS,midP,ring,SaliencyM,AfterSaliencyM,K1,N1,Par,SPnum,Par1,N2,Flag,Par2);
}