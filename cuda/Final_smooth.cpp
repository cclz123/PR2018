#include"mex.h"
extern void Final_smooth(double *FinS, double* midP,double * ring,double *MSaliencyM,double *K1,double *N1,double *Par,double *SPnum,double *Par1,double *Par2);
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *midP;
	double *FinS,*ring,*MSaliencyM;
    double *K1,*N1,*Par,*SPnum,*Par1,*Par2;
	midP=mxGetPr(prhs[0]);
	ring=mxGetPr(prhs[1]);
    MSaliencyM=mxGetPr(prhs[2]);
    K1=mxGetPr(prhs[3]);
    N1=mxGetPr(prhs[4]);
    Par=mxGetPr(prhs[5]);
    SPnum=mxGetPr(prhs[6]);
    Par1=mxGetPr(prhs[7]);
    Par2=mxGetPr(prhs[8]);
    int K=(int)(*K1);
    int N=(int)(*N1);
	plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
	FinS=mxGetPr(plhs[0]);
	Final_smooth(FinS,midP,ring,MSaliencyM,K1,N1,Par,SPnum,Par1,Par2);
}