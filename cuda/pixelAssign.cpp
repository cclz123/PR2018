# include "mex.h"
# include "pixelAssign.h"
#pragma comment(linker,"/nodefaultlib:LIBCMT.lib")
#define ImageSize 300
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	double *input1, *input2;
    double *LocalSize,*Par;
	double *result;
	double ColorCM, weightM, SaliencyPath;
	int k, z;
	input1 = mxGetPr(prhs[0]);
	input2 = mxGetPr(prhs[1]);
	LocalSize = (mxGetPr(prhs[2]));
    Par= (mxGetPr(prhs[3]));
	//motionG=mxGetPr(prhs[4]);
	//rate = mxGetPr(prhs[3]);
	plhs[0] = mxCreateDoubleMatrix(ImageSize, ImageSize, mxREAL);
	result = mxGetPr(plhs[0]);
	pixelAssign(result, input1, input2, LocalSize , Par);
}