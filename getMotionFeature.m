function [vx vy] = getMotionFeature(im1,im2)

% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;%0.012
ratio = 0.75;%0.75
minWidth = 20;%20
nOuterFPIterations = 7;%7
nInnerFPIterations = 1;%1
nSORIterations = 30;%30

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);



