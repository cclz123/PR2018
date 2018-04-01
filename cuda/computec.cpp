#include"mex.h"
#include"computec.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *midP;
	double *motion1,*motion2,*color12,*color11,*color13;
	double *MSaliencyM,*CSaliencyM,*ring;
    double *K1,*N1,*Par,*SPnum;
	midP=mxGetPr(prhs[0]);
	motion1=mxGetPr(prhs[1]);
	motion2=mxGetPr(prhs[2]);
	color11=mxGetPr(prhs[3]);
	color12=mxGetPr(prhs[4]);
	color13=mxGetPr(prhs[5]);
	ring=mxGetPr(prhs[6]);
    K1=mxGetPr(prhs[7]);
    N1=mxGetPr(prhs[8]);
    Par=mxGetPr(prhs[9]);
    SPnum=mxGetPr(prhs[10]);
    int K=(int)(*K1);
    int N=(int)(*N1);

	plhs[0]=mxCreateDoubleMatrix(K,N,mxREAL);
	plhs[1]=mxCreateDoubleMatrix(K,N,mxREAL);
	plhs[2]=mxCreateDoubleMatrix(K,N,mxREAL);
	MSaliencyM=mxGetPr(plhs[0]);
	CSaliencyM=mxGetPr(plhs[1]);

	cudacomputec(MSaliencyM,CSaliencyM,midP,motion1,motion2,color11,color12,color13,ring,K1,N1,Par,SPnum);



}