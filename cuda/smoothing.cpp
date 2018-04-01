#include"mex.h"
extern void cudasmooth(double *FinS, double *FinC, double* midP,double * ring,double *MSaliencyM,double *CSaliencyM,double *K1,double *N1,double *Par,double *SPnum,double *Par1);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *midP;
	double *FinS,*ring,*MSaliencyM,*CSaliencyM,* FinC;
    double *K1,*N1,*Par,*SPnum,*Par1;
	midP=mxGetPr(prhs[0]);
	ring   =mxGetPr(prhs[1]);
    MSaliencyM=mxGetPr(prhs[2]);
    CSaliencyM=mxGetPr(prhs[3]);
    K1=mxGetPr(prhs[4]);
    N1=mxGetPr(prhs[5]);
    Par=mxGetPr(prhs[6]);
    SPnum=mxGetPr(prhs[7]);
    Par1=mxGetPr(prhs[8]);
    int K=(int)(*K1);
    int N=(int)(*N1);
	plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
    plhs[1]=mxCreateDoubleMatrix(K,N,mxREAL);
	FinS=mxGetPr(plhs[0]);
    FinC=mxGetPr(plhs[1]);
	cudasmooth(FinS,FinC,midP,ring,MSaliencyM,CSaliencyM,K1,N1,Par,SPnum,Par1);
}