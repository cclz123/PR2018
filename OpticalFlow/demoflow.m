addpath('mex');
tic;
% we provide two sequences "car" and "table"
for wy=1:2
    example = 'animal\';
%example = 'car';

% load the two frames
if wy<10 
    wy1=strcat('frame_000',int2str(wy)); 
elseif wy<100 
wy1=strcat('frame_00',int2str(wy));
% else wy1=strcat('0',int2str(wy));
end;
if wy+1<10 
wy2=strcat('frame_000',int2str(wy+1)); 
elseif wy+1<100 
wy2=strcat('frame_00',int2str(wy+1)); 
% else wy2=strcat('0',int2str(wy+1));
end;
% wy1=int2str(wy);
% wy2=int2str(wy+1);
im1 = im2double(imread(strcat(example,wy1, '.jpg')));
im2 = im2double(imread(strcat(example,wy2, '.jpg')));

% im1 = imresize(im1,0.5,'bicubic');
% im2 = imresize(im2,0.5,'bicubic');

% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation
tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc

%figure;imshow(im1);figure;imshow(warpI2);



% output gif
clear volume;
volume(:,:,:,1) = im1;
volume(:,:,:,2) = im2;
if exist('output','dir')~=7
    mkdir('output');
end
%frame2gif(volume,fullfile('output',strcat(example,int2str(wy), '_input.gif')));
volume(:,:,:,2) = warpI2;
%frame2gif(volume,fullfile('output',strcat(example,int2str(wy+1),'_warp.gif')));


% visualize flow field
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
imflow = flowToColor(flow);
toc;
%figure;imshow(imflow);
imwrite(imflow,fullfile('output',strcat(example,int2str(wy), '_flow.jpg')),'quality',100);
end;
