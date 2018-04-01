%%
clear all;
clc;
addpath(genpath('.\boundary'));
addpath(genpath('.\cuda'));
addpath(genpath('.\OpticalFlow'));
addpath(genpath('.\sequences'));
addpath(genpath('.\PROPACK'));
addpath(genpath('.\sp'));

%%
%Parameter Initialization
Mode='girl';
Weg1=5;Weg2=20;
smrg1=15;smrg2=30;
spnum=600;
SmoothStrength=25;
CoarseRegionT=2;
Weight_SmoothS=10;Weight_SmoothS1=10;Weight_SmoothS2=5;
Par_sigmoid=5;
a_Par=0.6;
Energy_Num=3;
PairInitDis=20;
ParR2='';
Weight6=3;
path = ['.\sequences\' Mode '\'];
s=strcat('mkdir ','.\result\', Mode ,'\FinalSaliency',ParR2,'\');system(s);
s=strcat('mkdir ','.\OpticalF\');system(s);
Files = dir(fullfile(path,'*.*'));
LengthFiles = length(Files);
ImgIndex = 3;
BatchInitSpan = 8;
BatchNum = floor((LengthFiles-2)/(BatchInitSpan));
BatchResidual = mod(LengthFiles-2,BatchInitSpan);

%%
%Short-term Batch Decomposition (Algorithm 1)
for i=1:BatchNum
    BatchSize{i} = BatchInitSpan;
end
index = 1;
while(1)
    if(BatchResidual==0)
        break;
    end
    BatchSize{index} = BatchSize{index} + 1;
    BatchResidual = BatchResidual-1;
    index = index + 1;
    if(index>BatchNum)
        index = 1;
    end
end
BatchSize{BatchNum}=BatchSize{BatchNum}-1;
BatchStart{1}=1;
for i=2:BatchNum
    BatchStart{i}=BatchStart{i-1}+BatchSize{i-1};
end

BatchSubIndex = 1;
sigma1 = 0.03;
sigma2 = 0.01;
mu1 = 0.1;
mu2 = 0.05;
sigma_s = 100;sigma_r = 0.2;
BatchIndex = 0;
binNum = 5;
bin = 1/binNum;
AppearanceModelIndex = 1;
AppearanceModelMaxSize = 600;

%%
%Initialization (i.e., SLIC suprpixels)
temp = load('w2crs');%color mapping histogram
w2c = temp.w2crs;
while(ImgIndex<=LengthFiles)
    fprintf('compute the SP structure of images: ');
    fprintf('%d ', ImgIndex-2);
    ImageName = Files(ImgIndex).name;
    ImagePath = [path ImageName];
    ImageIndex=ImgIndex-2;
    ImageNameContainer{ImgIndex} = ImageName;
    temp = imread(ImagePath);
    oW=size(temp,1);oH=size(temp,2);%Preserve ImageSize
    temp = imresize(temp,[300 300]);%Down Sampling
    I{ImageIndex} = im2double(temp);
    IHsv{ImageIndex}=rgb2hsv(I{ImageIndex});
    cform=makecform('srgb2lab');
    ILab{ImageIndex}=applycform(I{ImageIndex},cform);
    ILab{ImageIndex}(:,:,1)=MatrixNormalization(ILab{ImageIndex}(:,:,1));
    ILab{ImageIndex}(:,:,2)=MatrixNormalization(ILab{ImageIndex}(:,:,2));
    ILab{ImageIndex}(:,:,3)=MatrixNormalization(ILab{ImageIndex}(:,:,3));
    ISmoothed{ImageIndex} = RF(im2double(temp), sigma_s, sigma_r);%Image Preprocessing/Smoothing
    I_CN{ImageIndex} = im2c(single(ISmoothed{ImageIndex}*255), w2c, -2);
    temp1=RF(temp, sigma_s, sigma_r);
    %Formulate Superpixel Structure Info
    [SEGMENTS{ImageIndex},SP_all{ImageIndex}] = getSPInfo(temp1,ISmoothed{ImageIndex},IHsv{ImageIndex},ILab{ImageIndex},spnum,I_CN{ImageIndex});
    fprintf('successed\n');
    [W H] = size(I{1});
    ImgIndex=ImgIndex+1;
end

%%
%Superpixel Based Features and ST Gradient
[color_all,motion_all,location_all,vx_all,vy_all,hsv_all,lab_all,vx_copy,vy_copy] = getHistogramFeatures(I,SP_all);
[MotionGradientMatrix_all,GradientMatrix,S_num,Cluster_Gradient] = computeMotionGradientMap(ISmoothed,vx_all,vy_all,SP_all,Mode,oW,oH,vx_copy,vy_copy);%get Contour Information(MotionGradient.*ColorGradient) I,vx,vy,SP

%%
%Compute Low-level Saliency Clues
ImgIndex=3;
while(ImgIndex<=LengthFiles)
        BatchIndex = BatchIndex + 1;
        if(BatchSubIndex<=BatchNum&&BatchIndex==BatchSize{BatchSubIndex})
            N = BatchSize{BatchSubIndex};
            BatchIndex = 0;
            for i=1:N
                SP{i}=SP_all{BatchStart{BatchSubIndex}+i-1};
            end
            K = -inf;
            for i=1:N
                K = max(K,SP{i}.SuperPixelNumber);% Identical Super Pixel Number
                color{i}=color_all{BatchStart{BatchSubIndex}+i-1};
                motion{i}=motion_all{BatchStart{BatchSubIndex}+i-1};
                location{i}=location_all{BatchStart{BatchSubIndex}+i-1};
            end
            K=double(K);
            clear BoundaryMask;
            %Load Input Data
            for i=1:N%Initial Containers
                for j=1:SP{i}.SuperPixelNumber%Normal Nodes
                    BoundaryMask{i}{j} = shiftdim(SP{i}.BoundaryIndex(1,j));
                end
                for j=SP{i}.SuperPixelNumber+1:K%Dummy Nodes
                    color{i}{j} = [1 1 1];
                    motion{i}{j} = [1 1];
                    location{i}{j} = [300 300];
                    BoundaryMask{i}{j} = [0];
                end
            end
            
            MotionGradientMatrix=MotionGradientMatrix_all(:,BatchStart{BatchSubIndex}:BatchStart{BatchSubIndex}+N-1);
            BoundaryMaskMatrix = [];
            for i=1:N
                BoundaryMaskMatrix = [BoundaryMaskMatrix cell2mat(BoundaryMask{i})'];%Supress Boundary Superpixels
            end
            
            %Estimate the smooth range
            SmoothRange = 0;
            for i=1:N
                CenterLocation = mean(MotionGradientMatrix{i},2);
                SmoothRange = SmoothRange + mean(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},CenterLocation))));
            end
            SmoothRange = SmoothRange/N;
            if(exist('lastSmoothRange'))
                SmoothRange = 0.8*lastSmoothRange + 0.2*SmoothRange;
            end
            SmoothRange = max(SmoothRange,20);SmoothRange = min(SmoothRange,50);
            lastSmoothRange = SmoothRange;
            %Compute the Motion Contrast and Color Contrast(Initial Saliency Clues)
            ring = zeros(K,N);
            motion11 = zeros(K,N);
            motion12 = zeros(K,N);
            color1= zeros(K,N);
            color2 = zeros(K,N);
            color3= zeros(K,N);
            SPnum=zeros(N,1);
            for i=1:N
                SPnum(i)=SP{i}.SuperPixelNumber;
            end
            for i=1:N
                for l=1:SP{i}.SuperPixelNumber
                    lLocation = SP{i}.MiddlePoint(l,1:2);
                    ring(l,i) = min(sum(abs(bsxfun(@minus,MotionGradientMatrix{i},lLocation')),1));
                    motion11(l,i)=motion{i}{l}(1);
                    motion12(l,i)=motion{i}{l}(2);
                    color1(l,i)=color{i}{l}(1);
                    color2(l,i)=color{i}{l}(2);
                    color3(l,i)=color{i}{l}(3);
                end
            end
            clear MidTemp;
            for k=1:N
                temp = zeros(K,5);
                for j=1:SP{k}.SuperPixelNumber
                    temp(j,:) = SP{k}.MiddlePoint(j,1:5);
                end
                MidTemp{k}=temp;
            end
            mid=cell2mat(MidTemp);
            [MotionSaliencyMatrix,ColorSaliency] = computec(mid,motion11,motion12,color1,color2,color3,ring,K,N,300,SPnum);%cuda and C program
            MotionSaliencyMatrix = MotionSaliencyMatrix.*BoundaryMaskMatrix;
            oM = MotionSaliencyMatrix.*ColorSaliency;
            TempColorSaliency = ColorSaliency;
            TempMotionSaliencyMatrix = MotionSaliencyMatrix;
            %Normalize ColorSaliency
            for i=1:N
                maxValue = max(TempColorSaliency(:,i));
                minValue = min(TempColorSaliency(:,i));
                TempColorSaliency(:,i) = (TempColorSaliency(:,i)-minValue)./(maxValue-minValue);
            end
            for i=1:N
                maxValue = max(TempMotionSaliencyMatrix(:,i));
                minValue = min(TempMotionSaliencyMatrix(:,i));
                TempMotionSaliencyMatrix(:,i) = (TempMotionSaliencyMatrix(:,i)-minValue)./(maxValue-minValue);
            end
            %Refine ColorSaliency(i.e., saliency adjustment)
            for i=1:N
                for j=1:K
                    if(TempColorSaliency(j,i) - TempMotionSaliencyMatrix(j,i)>0.5)
                        ColorSaliency(j,i) = ColorSaliency(j,i)*0.5;
                    end
                    if(TempMotionSaliencyMatrix(j,i) - TempColorSaliency(j,i)>0.5)
                        ColorSaliency(j,i) = mean(ColorSaliency(:,i))*2;
                    end
                end
            end
            if(exist('AppearanceModel'))
                AM = cell2mat(AppearanceModel);
                BM = cell2mat(BackgroundModel);
                for i=1:N
                    for j=1:K
                        ColorValue = color{i}{j};
                        AContrast = 0; BContrast = 0;
                        minA = inf; minB = inf;
                        AppearanceModelLength = size(AppearanceModel,2);
                        AM_temp = bsxfun(@minus,AM,ColorValue');
                        AM_temp = sqrt(sum(AM_temp.^2));
                        AContrast = sum(AM_temp);
                        minA = min(AM_temp);
                        BM_temp = bsxfun(@minus,BM,ColorValue');
                        BM_temp = sqrt(sum(BM_temp.^2));
                        BContrast = sum(BM_temp);
                        minB = min(BM_temp);
                        %Saliency Adjustment
                        if(minA>minB*1.1 || AContrast/AppearanceModelLength>BContrast/BackgroundModelLength)
                            ColorSaliency(j,i) = ColorSaliency(j,i) * 0.5;
                        end
                    end
                end
            end
            %Spatial-temporal Smoothing (CUDA accelerated)
            [MotionContrastSmoothed,ColorContrastSmoothed] = smoothing(mid,ring,MotionSaliencyMatrix,ColorSaliency,K,N,SmoothStrength,SPnum,SmoothRange);%cuda and C program
            CountImg=BatchStart{BatchSubIndex}+2;
            for i=1:N
                [W H] = size(I{1});
                ColorResult = zeros(W,H/3);
                MotionResult = zeros(W,H/3);
                FusionedResult = zeros(W,H/3);
                for l=1:SP{i}.SuperPixelNumber
                    ClusteringPixelNumber = SP{i}.ClusteringPixelNum(1,l);
                    for j=1:ClusteringPixelNumber%for Test Only
                        XIndex= SP{i}.Clustering(l,j,1);
                        YIndex= SP{i}.Clustering(l,j,2);
                        ColorResult(XIndex,YIndex) = ColorContrastSmoothed(l,i);
                        MotionResult(XIndex,YIndex) = MotionContrastSmoothed(l,i);
                        FusionedResult(XIndex,YIndex) = MotionContrastSmoothed(l,i).*ColorContrastSmoothed(l,i);
                    end
                end
                ColorResult = MatrixNormalization(ColorResult);
                MotionResult = MatrixNormalization(MotionResult);
                FusionedResult = MatrixNormalization(FusionedResult);
                ColorResult = imresize(ColorResult,[oW oH]);
                MotionResult = imresize(MotionResult,[oW oH]);
                FusionedResult = imresize(FusionedResult,[oW oH]);
                CountImg=CountImg+1;
            end
            MotionSaliencyMatrix =  MotionContrastSmoothed.*ColorContrastSmoothed;
            MotionSaliencyMatrix = MotionSaliencyMatrix.*BoundaryMaskMatrix;
            
            [IndexX IndexY] = find(isnan(MotionSaliencyMatrix)==1);%Eliminate Ill SLIC SuperPixels
            MotionSaliencyMatrix(IndexX,IndexY) = 0;
            
            for i=1:N
                maxValue = max(MotionSaliencyMatrix(:,i));
                minValue = min(MotionSaliencyMatrix(:,i));
                MotionSaliencyMatrix(:,i) = (MotionSaliencyMatrix(:,i)-minValue)./(maxValue-minValue);
                temp=MotionSaliencyMatrix(:,i);
                rate = CoarseRegionT;
                count = 0;
                Threshold = mean(temp(:));
                [index  value] = find(temp<max(0.5*Threshold));
                temp(index) = 0;
                Threshold = mean(temp(:));
               %%
                %Adaptive Foreground Segmentation (Algorithm 2)
                while(count<50)
                    count = count + 1;
                    [index value] = find(temp>Threshold*rate);
                    MotionMask{i} = zeros(K,1);
                    MotionMask{i}(index,1) = 1;
                    [index value] = find(MotionMask{i}==1);
                    MotionMaskLength = size(index,1);
                    if(MotionMaskLength==0)
                        rate=rate*0.6;
                        continue;
                    end
                    if(~exist('lastMotionMaskLength'))
                        lastMotionMaskLength = MotionMaskLength;
                    end
                    if(abs(MotionMaskLength-lastMotionMaskLength)<=floor(MotionMaskLength/3))%Minimal Number of Selected Superpixels
                        break;
                    end
                    if(MotionMaskLength<lastMotionMaskLength)
                        rate = rate * 0.95;
                    else
                        rate = rate * 1.05;
                    end
                end
                if(~exist('MeanMotionMaskLength'))
                    MeanMotionMaskLength=MotionMaskLength;
                    MeanMotionMaskLength1=MotionMaskLength;
                end
                lastMotionMaskLength=MotionMaskLength;
                MeanMotionMaskLength1 = (MotionMaskLength+MeanMotionMaskLength1*(i-1))/(i);
                MeanMotionMaskLength = (MotionMaskLength+MeanMotionMaskLength*(ImgIndex-N-2+i-1))/(ImgIndex-N-2+i);
                if i==1
                    CoarseRegionT=(0.2*size(MotionSaliencyMatrix,1)/MeanMotionMaskLength1)*1.25;
                    if CoarseRegionT>10
                        CoarseRegionT=10;
                    elseif CoarseRegionT<2
                        CoarseRegionT=2;
                    end
                end
                CoarseRetemp{ImgIndex-N-2+i}=CoarseRegionT;%
            end
            MotionSaliency_all{BatchSubIndex}=MotionSaliencyMatrix;
            MotionSM{BatchSubIndex}=MotionContrastSmoothed;
            MotionMask_all{BatchSubIndex}=cell2mat(MotionMask);
            clear MotionMask
           %%
            %Update the long-term color model
            BatchSubIndex = BatchSubIndex+1;
            AppearanceModelNum = zeros(binNum,binNum,binNum);%
            FBThreshold = mean(mean(abs(MotionSaliencyMatrix)));
            count1 = 0; count2 = 0;
            for i=1:N
                for j=1:SP{i}.SuperPixelNumber%traval all SP
                    CurrentSaliencyDegree = MotionSaliencyMatrix(j,i);
                    ColorValue = color{i}{j};
                    binIndex = floor(pos((ColorValue-0.000001)./bin))+1;
                    RIndex = binIndex(1,1);GIndex = binIndex(1,2);BIndex = binIndex(1,3);
                    if(CurrentSaliencyDegree>5.0*FBThreshold)%Foreground
                        count1 = count1+1;
                        Index = max(mod(AppearanceModelIndex,AppearanceModelMaxSize),1);
                        AppearanceModel{Index} = ColorValue';
                        AppearanceModelIndex = Index + 1;
                    else%Background
                        count2 = count2+1;
                        BackgroundModel{count2} = ColorValue';
                    end
                end
            end
            BackgroundModelLength = count2;
        end
        ImgIndex = ImgIndex+1;
        fprintf('%d\n ',ImgIndex);
end

%%
%Compute Motion Mask
MaxDim=0;
for i=1:size(MotionSM,2)
    if size(MotionSM{i},1)>MaxDim
        MaxDim=size(MotionSM{i},1);
    end
end
for i=1:size(MotionSM,2)
    tempCon=zeros(MaxDim,size(MotionSM{i},2));
    tempCon(1:size(MotionSM{i},1),1:size(MotionSM{i},2))=MotionSM{i};
    MotionSM{i}=tempCon;
end
MotionM=cell2mat(MotionSM);
MeanMotionMaskLength=MeanMotionMaskLength*0.8;

%Spatial-termporal Smoothing (CUDA acclerated)
MaxDim=0;
for i=1:size(MotionSaliency_all,2)
	if size(MotionSaliency_all{i},1)>MaxDim
    	MaxDim=size(MotionSaliency_all{i},1);
    end
end
for i=1:size(MotionSaliency_all,2)
	tempCon=zeros(MaxDim,size(MotionSaliency_all{i},2));
	tempCon(1:size(MotionSaliency_all{i},1),1:size(MotionSaliency_all{i},2))=MotionSaliency_all{i};
	MotionSaliency_all{i}=tempCon;
end
MaxDim=0;
for i=1:size(MotionMask_all,2)
	if size(MotionMask_all{i},1)>MaxDim
    	MaxDim=size(MotionMask_all{i},1);
    end
end
for i=1:size(MotionMask_all,2)
        tempCon=zeros(MaxDim,size(MotionMask_all{i},2));
        tempCon(1:size(MotionMask_all{i},1),1:size(MotionMask_all{i},2))=MotionMask_all{i};
        MotionMask_all{i}=tempCon;
    end
clear tempCon;
SaliencyM=cell2mat(MotionSaliency_all);

for i=1:LengthFiles-3
	temp = zeros(MaxDim,11);
    for j=1:SP_all{i}.SuperPixelNumber
       temp(j,:) = SP_all{i}.MiddlePoint(j,1:11);
    end
    MidTemp{i}=temp;
end
mid=cell2mat(MidTemp);
SPnum=zeros(LengthFiles-3,1);ring = zeros(MaxDim,LengthFiles-3);
for i=1:LengthFiles-3
	SPnum(i)=SP_all{i}.SuperPixelNumber;
end
for i=1:LengthFiles-3
	for l=1:SP_all{i}.SuperPixelNumber
    	lLocation = SP_all{i}.MiddlePoint(l,1:2);
    	ring(l,i) = min(sum(abs(bsxfun(@minus,MotionGradientMatrix_all{i},lLocation')),1));
	end
end
SaliencyM=Final_smooth(mid,ring,SaliencyM,MaxDim,LengthFiles-3,10.0,SPnum,30.0,50.0);
SaliencyM=Final_smooth(mid,ring,SaliencyM,MaxDim,LengthFiles-3,5.0,SPnum,15.0,30.0);
MotionMaskAll=cell2mat(MotionMask_all);
MotionMaskAll_Copy=zeros(MaxDim,LengthFiles-3);
MotionMaskAll_Copy1=zeros(MaxDim,LengthFiles-3);

%%
%Iterative Labeling/Segmentation
for i=1:LengthFiles-3
	FusionedResult=zeros(300,300);
    for j=1:size(MotionMaskAll,1)
    	if MotionMaskAll(j,i)~=0
        	ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,j);
            for z=1:ClusteringPixelNumber
            	XIndex= SP_all{i}.Clustering(j,z,1);
            	YIndex= SP_all{i}.Clustering(j,z,2);
            	FusionedResult(XIndex,YIndex) = 1;
            end
        end
    end
    se = strel('disk',70);
    FusionedResult_Copy = imdilate(FusionedResult,se);
    FusionedResult_Copy=MatrixNormalization(FusionedResult_Copy);
    for j=1:300
            for z=1:300
                if FusionedResult_Copy(j,z)>0
                    MotionMaskAll_Copy(SEGMENTS{i}(j,z),i)=MotionMaskAll_Copy(SEGMENTS{i}(j,z),i)+1;
                end
            end
        end
    for j=1:SP_all{i}.SuperPixelNumber
            if MotionMaskAll_Copy(j,i)>SP_all{i}.ClusteringPixelNum(1,j)/3
                MotionMaskAll_Copy(j,i)=1;
            else
                MotionMaskAll_Copy(j,i)=0;
            end
        end
    se = strel('disk',20);
    FusionedResult_Copy1 = imdilate(FusionedResult,se);
    FusionedResult_Copy1=MatrixNormalization(FusionedResult_Copy1);
    for j=1:300
            for z=1:300
                if FusionedResult_Copy1(j,z)>0
                    MotionMaskAll_Copy1(SEGMENTS{i}(j,z),i)=MotionMaskAll_Copy1(SEGMENTS{i}(j,z),i)+1;
                end
            end
        end
    for j=1:SP_all{i}.SuperPixelNumber
            if MotionMaskAll_Copy1(j,i)>SP_all{i}.ClusteringPixelNum(1,j)/3
                MotionMaskAll_Copy1(j,i)=1;
            else
                MotionMaskAll_Copy1(j,i)=0;
            end
        end
    FusionedResult=zeros(300,300);
    for j=1:size(MotionMaskAll,1)
    	if MotionMaskAll(j,i)~=0
        	ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,j);
        	for z=1:ClusteringPixelNumber
                    XIndex= SP_all{i}.Clustering(j,z,1);
                    YIndex= SP_all{i}.Clustering(j,z,2);
                    FusionedResult(XIndex,YIndex) = 1;
                end
        end
    end
    FusionedResult = imresize(FusionedResult,[oW oH]);
end

%%
%Low-rank Analysis
[A_hat E_hat A]=TenDH(LengthFiles,I_CN,I,MotionMaskAll,SaliencyM,SP_all,Mode,ImageNameContainer,oW,oH,MaxDim);
AnchorI{1}=1;
E_Hat=sum(abs(E_hat));
E_Hat=MatrixNormalization(E_Hat);
E_Mean=mean(E_Hat);
E_SUM=E_Hat;
E_SUM1=E_Hat;

%Initialize Our Appearance Model
for i=1:size(E_SUM,2)
    if E_SUM(i)>E_Mean*1.5
        E_SUM(i)=E_SUM(i)*0.1+0.2+a_Par;
    else
        E_SUM(i)=E_SUM(i)*0.2+a_Par;
    end
end
if LengthFiles-3<50
    for j=1:BatchNum-1
        Start{j}=int32(BatchStart{j}+BatchStart{j+1})/2;
    end
    Start{BatchNum}=int32(BatchStart{BatchNum}+LengthFiles-3)/2;
    ct=1;
    BatchNum=2*BatchNum;
    for j=1:2:BatchNum
        BStart{j}=BatchStart{ct};
        BStart{j+1}=double(Start{ct});
        ct=ct+1;
    end
    BatchStart=BStart;
    ModelNumber=int32(MeanMotionMaskLength)*10;%Capacity of Our Appearance Model
else
    ModelNumber=int32(MeanMotionMaskLength)*15;
end
i=1;
E_Hat_Copy=E_Hat;

%Locate Anchor Frames
while(i<BatchNum)
    frontInd=BatchStart{i};
    Ind=BatchStart{i+1}-1;
    [value index]=min(E_Hat_Copy(frontInd:Ind));
    AnchorI{i+1}=frontInd+index-1;
    i=i+1;
end
frontInd=BatchStart{i};
Ind=LengthFiles-3;
[value index]=min(E_Hat_Copy(frontInd:Ind));
AnchorI{i+1}=frontInd+index-1;
i=i+1;
AnchorI{i+1}=LengthFiles-3;
AnchorIndex=cell2mat(AnchorI);
SN=cell2mat(S_num);
Mean_SN=mean(SN);
for i=1:LengthFiles-3
    if SN(i)>2*Mean_SN
        SN(i)=Mean_SN;
    end
    if SN(i)<0.3*Mean_SN
        SN(i)=Mean_SN;
    end
end
for i=2:LengthFiles-3
    if abs(SN(i)-SN(i-1))>SN(i-1)/4
        SN(i)=(SN(i)+SN(i-1))/2;
    end
end
for i=1:BatchNum+1
    SUM_MotionMask=0;
    S_NUM=0;
    for j=AnchorIndex(i):AnchorIndex(i+1)
        SUM_MotionMask=SUM_MotionMask+CoarseRetemp{j};
        S_NUM=S_NUM+SN(j);
    end
    MotionMask_Num{i}=SUM_MotionMask/(AnchorIndex(i+1)-AnchorIndex(i)+1);
    SP_NUM{i}=S_NUM/(AnchorIndex(i+1)-AnchorIndex(i)+1)*1.3;
end

%%
%Fomulate Our Structure-aware Descriptor
for i=1:LengthFiles-3
	FusionedResult=zeros(300,300);
    result{i}=FindPair(i,MotionMaskAll,SP_all,Cluster_Gradient);
    for j=1:size(result{i},2)
    	if result{i}{j}{1}(2)~=0;
        	ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,result{i}{j}{1}(1));
            for z=1:ClusteringPixelNumber
            	XIndex= SP_all{i}.Clustering(result{i}{j}{1}(1),z,1);
            	YIndex= SP_all{i}.Clustering(result{i}{j}{1}(1),z,2);
            	FusionedResult(XIndex,YIndex) = 1;
            end
        end
    end
    FusionedResult = imresize(FusionedResult,[oW oH]);
end
result2=result;
PairDistance=mean(SN);%
%Topology Constraints    
for i=1:LengthFiles-3
	[Dis2Um{i},SP_all{i}.Neighbor]=get_neighbor(SP_all{i},MotionMaskAll_Copy1(:,i),mean(SN),PairInitDis);
	FusionedResult=zeros(300,300);
	for j=1:size(result{i},2)
    	if result{i}{j}{1}(2)~=0;
        	ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,result{i}{j}{1}(2));
        	for z=1:ClusteringPixelNumber
            	XIndex= SP_all{i}.Clustering(result{i}{j}{1}(2),z,1);
            	YIndex= SP_all{i}.Clustering(result{i}{j}{1}(2),z,2);
            	FusionedResult(XIndex,YIndex) = 1;
            end
        end
        if result{i}{j}{2}(2)~=0;
        	ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,result{i}{j}{2}(2));
        	for z=1:ClusteringPixelNumber
            	XIndex= SP_all{i}.Clustering(result{i}{j}{2}(2),z,1);
            	YIndex= SP_all{i}.Clustering(result{i}{j}{2}(2),z,2);
            	FusionedResult(XIndex,YIndex) = 1;
            end
        end
        if result{i}{j}{3}(2)~=0;
           ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,result{i}{j}{3}(2));
           for z=1:ClusteringPixelNumber
               XIndex= SP_all{i}.Clustering(result{i}{j}{3}(2),z,1);
               YIndex= SP_all{i}.Clustering(result{i}{j}{3}(2),z,2);
               FusionedResult(XIndex,YIndex) = 1;
           end
        end
    end
    FusionedResult = imresize(FusionedResult,[oW oH]);
end

for i=1:LengthFiles-3
    FusionedResult1=zeros(300,300);
    for j=1:size(MotionMaskAll_Copy1,1)
        if MotionMaskAll_Copy1(j,i)~=0
            ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,j);
            for z=1:ClusteringPixelNumber
                XIndex= SP_all{i}.Clustering(j,z,1);
                YIndex= SP_all{i}.Clustering(j,z,2);
                FusionedResult1(XIndex,YIndex) = 1;
            end
        end
    end
end
AfterSaliencyM=zeros(size(SaliencyM,1),size(SaliencyM,2));
FlagAfter=zeros(1,LengthFiles-3);
AfterSaliencyM2=zeros(size(SaliencyM,1),size(SaliencyM,2));
FlagAfter2=zeros(1,LengthFiles-3);
FlagAfter4=zeros(1,LengthFiles-3);
FlagAfter3=zeros(1,LengthFiles-3);
FlagAfter5=zeros(1,LengthFiles-3);
AfterSaliencyM4=zeros(size(SaliencyM,1),size(SaliencyM,2));
AfterSaliencyM3=zeros(size(SaliencyM,1),size(SaliencyM,2));
AfterSaliencyM5=zeros(size(SaliencyM,1),size(SaliencyM,2));
SaliencyM_Copy1=zeros(size(SaliencyM,1),size(SaliencyM,2));
SaliencyM_Copy2=zeros(size(SaliencyM,1),size(SaliencyM,2));
SaliencySM_MotionM=SaliencyM;

Mean_SN=mean(SN);
DDis=30+70*max(Mean_SN-20,0)/80;
DDis=DDis*1;
ColorRange=30+70*min(max(Mean_SN-20,0),80)/80;
Distance=30+70*max(Mean_SN-20,0)/80;
Distance=Distance*1;
ColorRange=ColorRange*1;
MotionM=Final_smooth(mid,ring,MotionM,MaxDim,LengthFiles-3,5.0,SPnum,smrg1,smrg2);
Weight2=1;
WeightD_M=5;
WeightD_F=5;
ModelNumber=int32(mean(SN)*min(max(5,BatchNum),8));
Model=zeros(ModelNumber,12);
Model_Hsv=zeros(ModelNumber,8);
Model_Sm=zeros(ModelNumber,1);
Model_res=zeros(ModelNumber,1);
Model_Pos=zeros(ModelNumber,2);
Model_Energy=zeros(ModelNumber,1);
count=1;

%%
%Learning/Updating Iterations
for i=2:size(AnchorIndex,2)-1
    for j=1:size(result2{AnchorIndex(i)},2)
        if ~isempty(result2{AnchorIndex(i)}{j})
            if result2{AnchorIndex(i)}{j}{1}(2)~=0&&count<=ModelNumber
                Model(count,:)=[SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{1}(1),3:5) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{1}(2),3:5) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{2}(2),3:5) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{3}(2),3:5)];
                Model_Energy(count)=Energy_Num;
                Model_Pos(count,:)=SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{1}(1),1:2);
                Model_Hsv(count,:)=[SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{1}(1),10:11) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{1}(2),10:11) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{2}(2),10:11) SP_all{AnchorIndex(i)}.MiddlePoint(result2{AnchorIndex(i)}{j}{3}(2),10:11)];
                Model_Sm(count)=SaliencyM(result2{AnchorIndex(i)}{j}{1}(1),AnchorIndex(i));
                Model_res(count)=1;
                count=count+1;
            end
        end
    end
end
for i=1:LengthFiles-3
    ColorWeight{i}=ComputeCD(double(SP_all{i}.SuperPixelNumber),SP_all{i}.MiddlePoint,WeightD_M,ColorRange,ring(:,i));
    ColorWeight2{i}=ComputeCD(double(SP_all{i}.SuperPixelNumber),SP_all{i}.MiddlePoint,WeightD_M,ColorRange,ring(:,i));
end
for i=2:size(AnchorIndex,2)-1
    for j= AnchorIndex(i):-1:AnchorIndex(i-1)+1
        NewSM1=zeros(MaxDim,1);
        NewSM2=zeros(MaxDim,1);
        ColorDist=zeros(MaxDim,size(result{j},2));
        GradDist=zeros(MaxDim,size(result{j},2));
        MainColorD=zeros(MaxDim,size(result{j},2));
        Result_RGB=zeros(size(result{j},2),12);
        Result_LAB=zeros(size(result{j},2),8);
        Result_POS=zeros(size(result{j},2),2);
        tic
        for k=1:size(result{j},2)
            if  result{j}{k}{1}(2)~=0
                Result_RGB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),3:5) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),3:5)];
                Result_LAB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),10:11) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),10:11)];
                Result_POS(k,:)=SP_all{j}.MiddlePoint(result{j}{k}{1}(1),1:2);
            end
        end
        for k=1:size(result{j},2)
            if result{j}{k}{1}(2)~=0
                [Tt,Ss,Ww]=FrameTrans(double(SP_all{j-1}.SuperPixelNumber),SP_all{j-1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j-1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j-1},double(MaxDim),WeightD_F,Result_POS(k,1:2),Distance,ring(:,j-1),Weg1,Weg2);
                ColorDist(:,k)=Tt;
                GradDist(:,k)=Ss;
                MainColorD(:,k)=Ww;
            end
        end
        GradDist=exp(0.6*GradDist);
        GradDist=MatrixNormalization(GradDist);
        GradDist=GradDist*0.3+0.7;
        Trans_W=zeros(MaxDim,1);
        for k=1:size(result{j},2)
            if result{j}{k}{1}(2)~=0
                for z=1:SP_all{j-1}.SuperPixelNumber
                    if(ColorDist(z,k)~=1000)
                        WW=exp(-1*ColorDist(z,k)*Weight2);%
                        NewSM1(z)=NewSM1(z)+SaliencyM(result{j}{k}{1}(1),j)*WW;%
                        Trans_W(z)=Trans_W(z)+exp(-1*MainColorD(z,k)*Weight2);
                    end
                end
            end
        end
        
        for k=1:SP_all{j-1}.SuperPixelNumber
            if ColorWeight2{j-1}(k)~=0
                NewSM1(k)=NewSM1(k)/ColorWeight2{j-1}(k);
            end
        end
        clear Pair;
        MaxCD=0;
        MinCD=1000;
        MaxG=0;
        MinG=1000;
        if j+1<LengthFiles-3&&j~=AnchorIndex(i)
            ColorDist=zeros(MaxDim,size(result{j+1},2));
            GradDist=zeros(MaxDim,size(result{j+1},2));
            MainColorD=zeros(MaxDim,size(result{j+1},2));
            Result_RGB=zeros(size(result{j+1},2),12);
            Result_LAB=zeros(size(result{j+1},2),8);
            Result_POS=zeros(size(result{j+1},2),2);
            for k=1:size(result{j+1},2)
                if  result{j+1}{k}{1}(2)~=0
                    Result_RGB(k,:)=[SP_all{j+1}.MiddlePoint(result{j+1}{k}{1}(1),3:5) SP_all{j+1}.MiddlePoint(result{j+1}{k}{1}(2),3:5) SP_all{j+1}.MiddlePoint(result{j+1}{k}{2}(2),3:5) SP_all{j+1}.MiddlePoint(result{j+1}{k}{3}(2),3:5)];
                    Result_LAB(k,:)=[SP_all{j+1}.MiddlePoint(result{j+1}{k}{1}(1),10:11) SP_all{j+1}.MiddlePoint(result{j+1}{k}{1}(2),10:11) SP_all{j+1}.MiddlePoint(result{j+1}{k}{2}(2),10:11) SP_all{j+1}.MiddlePoint(result{j+1}{k}{3}(2),10:11)];
                    Result_POS(k,:)=SP_all{j+1}.MiddlePoint(result{j+1}{k}{1}(1),1:2);
                end
            end
            for k=1:size(result{j+1},2)
                if result{j+1}{k}{1}(2)~=0
                    [Tt,Ss,Ww]=FrameTrans(double(SP_all{j-1}.SuperPixelNumber),SP_all{j-1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j-1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j-1},double(MaxDim),WeightD_M,Result_POS(k,1:2),Distance,ring(:,j-1),Weg1,Weg2);
                    ColorDist(:,k)=Tt;
                    GradDist(:,k)=Ss;
                    MainColorD(:,k)=Ww;
                end
            end
            GradDist=exp(0.6*GradDist);
            GradDist=MatrixNormalization(GradDist);
            GradDist=GradDist*0.3+0.7;
            Trans_W=zeros(MaxDim,1);
            for k=1:size(result{j+1},2)
                if result{j+1}{k}{1}(2)~=0
                    for z=1:SP_all{j-1}.SuperPixelNumber
                        if(ColorDist(z,k)~=1000)
                            WW=exp(-1*ColorDist(z,k)*Weight2);
                            NewSM2(z)=NewSM2(z)+SaliencyM(result{j+1}{k}{1}(1),j+1)*WW;%
                            Trans_W(z)=Trans_W(z)+exp(-1*MainColorD(z,k)*Weight2);
                        end
                    end
                end
            end
            for k=1:SP_all{j-1}.SuperPixelNumber
                if ColorWeight2{j-1}(k)~=0
                    NewSM2(k)=NewSM2(k)/ColorWeight2{j-1}(k);
                end
            end
        end
        ModelTran=zeros(MaxDim,ModelNumber);
        ModelTran1=zeros(MaxDim,ModelNumber);
        Color_dist=zeros(MaxDim,ModelNumber);
        SUM=zeros(MaxDim,ModelNumber);
        MainColor=zeros(MaxDim,ModelNumber);
        MaxCD=0;
        MinCD=1000;
        
%%
        %Saliency Transferring
        for k=1:ModelNumber
            if Model_Sm(k)~=0
                [Tt,Ss,Ww]=ModelTransfer(double(SP_all{j-1}.SuperPixelNumber),SP_all{j-1}.MiddlePoint,Model(k,1:12),SP_all{j-1}.Neighbor,Model_Hsv(k,1:8),GradientMatrix{j-1},double(MaxDim),WeightD_M,Model_Pos(k,1:2),DDis,ring(:,j-1),Weg1,Weg2);
                Color_dist(:,k)=Tt;
                SUM(:,k)=Ss;
                MainColor(:,k)=Ww;
            end
        end
        SUM=exp(0.6*SUM);
        SUM=MatrixNormalization(SUM);
        SUM=SUM*0+1;
        [NewSM, ModelTran1,S_s,ModelTran]=ModelT(double(ModelNumber),Model_Sm,double(SP_all{j-1}.SuperPixelNumber),MaxDim,Color_dist,SUM,MainColor);
        Sum_ModelTran=sum(ModelTran1,2);
        for k=1:SP_all{j-1}.SuperPixelNumber
            if ColorWeight{j-1}(k)~=0
                NewSM(k)=NewSM(k)/ColorWeight{j-1}(k);
            end
        end
        NewSM1=NewSM1+NewSM2;
        NewSM1=MatrixNormalization(NewSM1);
        par1=3*CoarseRetemp{j-1}*mean(NewSM1);
        for ii=1:size(NewSM1,1)%Sigmoid Funcion
            NewSM1(ii)=1/(1+(exp(-Weight6*(NewSM1(ii)-par1))));
        end
        NewSM1=MatrixNormalization(NewSM1);
        Par1=num2str(i);
        NewSM1=midsmooth(mid,ring,NewSM1,AfterSaliencyM,MaxDim,j-1-1,Weight_SmoothS1,SPnum,smrg1,LengthFiles-3,FlagAfter,smrg2);
        NewSM1=MatrixNormalization(NewSM1);
        AfterSaliencyM(:,j-1)=NewSM1;
        FlagAfter(j-1)=1;
        Par1=num2str(i);
        NewSM=MatrixNormalization(NewSM);
        par2=CoarseRetemp{j-1}*mean(NewSM);
        for ii=1:size(NewSM,1)
            NewSM(ii)=1/(1+(exp(-Weight6*(NewSM(ii)-par2))));
        end
        NewSM=MatrixNormalization(NewSM);
        NewSM=midsmooth(mid,ring,NewSM,AfterSaliencyM2,MaxDim,j-1-1,Weight_SmoothS1,SPnum,smrg1,LengthFiles-3,FlagAfter2,smrg2);
        NewSM=MatrixNormalization(NewSM);
        AfterSaliencyM2(:,j-1)=NewSM;
        FlagAfter2(j-1)=1;
        
%%
        %Low-rank Guided Saliency Fusion
        Temp_SaliencyM1=zeros(MaxDim,1);
        Temp_SaliencyM2=zeros(MaxDim,1);
        for k=1:SP_all{j-1}.SuperPixelNumber
            Temp_SaliencyM1(k)=(NewSM1(k)+NewSM(k))/2;
            Temp_SaliencyM2(k)=(E_SUM(j-1)*NewSM(k)+SaliencySM_MotionM(k,j-1))*(NewSM(k)+(1-E_SUM(j-1))*SaliencySM_MotionM(k,j-1));
        end
        Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
        Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
        
%%
        %Spatial-temporal Smoothing
        Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
        AfterSaliencyM3(:,j-1)=Temp_SaliencyM1;
        FlagAfter3(j-1)=1;
        Par1=num2str(i);
        Temp_SaliencyM2=midsmooth(mid,ring,Temp_SaliencyM2,AfterSaliencyM4,MaxDim,j-1-1,Weight_SmoothS2,SPnum,smrg1,LengthFiles-3,FlagAfter4,smrg2);
        Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
        AfterSaliencyM4(:,j-1)=Temp_SaliencyM2;
        FlagAfter4(j-1)=1;
        Refine_Temp_SaliencyM1=Temp_SaliencyM1*0.3+0.7;
        SaliencyM(:,j-1)=0.5*(MatrixNormalization(E_SUM(j-1)*Temp_SaliencyM1+(1-E_SUM(j-1))*MotionM(:,j-1))).*Refine_Temp_SaliencyM1+SaliencySM_MotionM(:,j-1)*0.5;
        SaliencyM(:,j-1) = 1./(1+exp(-SaliencyM(:,j-1)*Par_sigmoid));
        SaliencyM(:,j-1) = MatrixNormalization(SaliencyM(:,j-1));
        SaliencyM_Copy1(:,j-1)=MatrixNormalization(E_SUM(j-1)*Temp_SaliencyM1+(1-E_SUM(j-1))*MotionM(:,j-1)).*Refine_Temp_SaliencyM1;
        for k=1:SP_all{j-1}.SuperPixelNumber
            SaliencyM_Copy1(k,j-1)=min(SaliencyM_Copy1(k,j-1),1);
        end
        SaliencyM(:,j-1)=midsmooth(mid,ring,SaliencyM(:,j-1),AfterSaliencyM5,MaxDim,j-1-1,Weight_SmoothS,SPnum,smrg1,LengthFiles-3,FlagAfter5,smrg2);
        SaliencyM(:,j-1)=MatrixNormalization(SaliencyM(:,j-1));
        AfterSaliencyM5(:,j-1)=SaliencyM(:,j-1);
        FlagAfter5(j-1)=1;
        if(i-1>1)
            rate = MotionMask_Num{i-1}*0.7+MotionMask_Num{i-2}*0.3;%CoarseRetemp{j-1};
        else
            rate = MotionMask_Num{i-1};
        end
        temp = SaliencyM_Copy1(1:SP_all{j-1}.SuperPixelNumber,j-1);
        [index value] = find(temp<mean(temp)*0.5);
        temp(index) = 0;
        Threshold = mean(mean(abs(temp)));
        count = 0;
        
%%
        %Update the Motion/Foreground Mask
        while(count<50)
            count = count + 1;
            [index value] = find(temp>Threshold*rate);
            MotionMask1 = zeros(MaxDim,1);
            MotionMask1(index,1) = 1;
            [index value] = find(MotionMask1==1);
            MotionMaskLength1 = size(index,1);
            if(~exist('lastMotionMaskLength1'))
                lastMotionMaskLength1 = MotionMaskLength1;
            end
            if(abs(MotionMaskLength1-lastMotionMaskLength1)<lastMotionMaskLength1/4)%Minimal Number of Selected Superpixels
                break;
            end
            if(MotionMaskLength1<lastMotionMaskLength1)
                rate = rate * 0.95;
            else
                rate = rate * 1.05;
            end
        end
        MotionMaskAll(:,j-1)=MotionMask1;
        [index value] = find(MotionMaskAll(:,j-1)==1);
        MaskLength = size(index,1);
        %Update Pattern Models
        if MaskLength~=0
            result2{j-1}=FindPair(j-1,MotionMaskAll,SP_all,Cluster_Gradient);
            result{j-1}=result2{j-1};
        end
    end
    %Re-compute the Low-rank Coherency
    [A_hat E_hat A]=TenDH(LengthFiles,I_CN,I,MotionMaskAll,SaliencyM,SP_all,Mode,ImageNameContainer,oW,oH,MaxDim);
    E_Hat=sum(abs(E_hat));
    E_Hat=MatrixNormalization(E_Hat);
    E_SUM=E_Hat;
    E_Mean=mean(E_Hat);
    E_SUM1=E_Hat;
    for kkk=1:size(E_SUM,2)
        if E_SUM(kkk)>1.5*E_Mean
            E_SUM(kkk)=E_SUM(kkk)*0.1+0.2+a_Par;
        else
            E_SUM(kkk)=E_SUM(kkk)*0.2+a_Par;
        end
    end
    clear lastMotionMaskLength1;
    if(i==2)
        for j= 1:AnchorIndex(i)-1
            S_Mean=zeros(MaxDim,1);
            NewSM3=zeros(MaxDim,1);
            NewSM4=zeros(MaxDim,1);
            clear Pair;
            ColorDist=zeros(MaxDim,size(result{j},2));
            GradDist=zeros(MaxDim,size(result{j},2));
            MainColorD=zeros(MaxDim,size(result{j},2));
            Result_RGB=zeros(size(result{j},2),12);
            Result_LAB=zeros(size(result{j},2),8);
            Result_POS=zeros(size(result{j},2),2);
            tic
            for k=1:size(result{j},2)
                if  result{j}{k}{1}(2)~=0
                    Result_RGB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),3:5) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),3:5)];
                    Result_LAB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),10:11) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),10:11)];
                    Result_POS(k,:)=SP_all{j}.MiddlePoint(result{j}{k}{1}(1),1:2);
                end
            end
            for k=1:size(result{j},2)
                if result{j}{k}{1}(2)~=0
                    [Tt,Ss,Ww]=FrameTrans(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j+1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_F,Result_POS(k,1:2),Distance,ring(:,j+1),Weg1,Weg2);
                    ColorDist(:,k)=Tt;
                    GradDist(:,k)=Ss;
                    MainColorD(:,k)=Ww;
                end
            end
            GradDist=exp(0.6*GradDist);
            GradDist=MatrixNormalization(GradDist);
            GradDist=GradDist*0.3+0.7;
            Trans_W=zeros(MaxDim,1);
            for k=1:size(result{j},2)
                temp_sum=0;
                count1=0;
                if result{j}{k}{1}(2)~=0
                    for z=1:SP_all{j+1}.SuperPixelNumber
                        count1=count1+1;
                        if(ColorDist(z,k)~=1000)
                            WW=exp(-1*ColorDist(z,k)*Weight2);
                            temp_sum=temp_sum+SaliencyM(result{j}{k}{1}(1),j)*WW;
                            NewSM3(z)=NewSM3(z)+SaliencyM(result{j}{k}{1}(1),j)*WW;
                            Trans_W(z)=Trans_W(z)+exp(-1*MainColorD(z,k)*Weight2);
                        end
                    end
                    if count1~=0
                        S_Mean(result{j}{k}{1}(1))=temp_sum/count1;
                    end
                end
            end
            for k=1:SP_all{j+1}.SuperPixelNumber
                if ColorWeight2{j+1}(k)~=0
                    NewSM3(k)=NewSM3(k)/ColorWeight2{j+1}(k);
                end
            end
            
            if j-1~=0
                clear Pair;
                ColorDist=zeros(MaxDim,size(result{j-1},2));
                GradDist=zeros(MaxDim,size(result{j-1},2));
                MainColorD=zeros(MaxDim,size(result{j-1},2));
                Result_RGB=zeros(size(result{j-1},2),12);
                Result_LAB=zeros(size(result{j-1},2),8);
                Result_POS=zeros(size(result{j-1},2),2);
                for k=1:size(result{j-1},2)
                    if  result{j-1}{k}{1}(2)~=0
                        Result_RGB(k,:)=[SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(2),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{2}(2),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{3}(2),3:5)];
                        Result_LAB(k,:)=[SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(2),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{2}(2),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{3}(2),10:11)];
                        Result_POS(k,:)=SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),1:2);
                    end
                end
                for k=1:size(result{j-1},2)
                    if result{j-1}{k}{1}(2)~=0
                        [Tt,Ss,Ww]=FrameTrans(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j+1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_F,Result_POS(k,1:2),Distance,ring(:,j+1),Weg1,Weg2);
                        ColorDist(:,k)=Tt;
                        GradDist(:,k)=Ss;
                        MainColorD(:,k)=Ww;
                    end
                end
                GradDist=exp(0.6*GradDist);
                GradDist=MatrixNormalization(GradDist);
                GradDist=GradDist*0.3+0.7;
                Trans_W=zeros(MaxDim,1);
                for k=1:size(result{j-1},2)
                    if result{j-1}{k}{1}(2)~=0
                        for z=1:SP_all{j+1}.SuperPixelNumber
                            if(ColorDist(z,k)~=1000)
                                WW=exp(-1*ColorDist(z,k)*Weight2);
                                NewSM4(z)=NewSM4(z)+SaliencyM(result{j-1}{k}{1}(1),j-1)*WW;
                                Trans_W(z)=Trans_W(z)+exp(-1*MainColorD(z,k)*Weight2);
                            end
                        end
                    end
                end
                for k=1:SP_all{j+1}.SuperPixelNumber
                    if ColorWeight2{j+1}(k)~=0
                        NewSM4(k)=NewSM4(k)/ColorWeight2{j+1}(k);
                    end
                end
            end
            ModelTran=zeros(MaxDim,ModelNumber);
            Color_dist=zeros(MaxDim,ModelNumber);
            SUM=zeros(MaxDim,ModelNumber);
            S_Sum=zeros(MaxDim,ModelNumber);
            NewSM=zeros(MaxDim,1);
            MaxCD=0;
            for k=1:ModelNumber
                if Model_Sm(k)~=0
                    [Tt,Ss,Ww]=ModelTransfer(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Model(k,1:12),SP_all{j+1}.Neighbor,Model_Hsv(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_M,Model_Pos(k,1:2),DDis,ring(:,j+1),Weg1,Weg2);
                    Color_dist(:,k)=Tt;
                    SUM(:,k)=Ss;
                    MainColor(:,k)=Ww;
                end
            end
            
            SUM=exp(0.6*SUM);
            SUM=MatrixNormalization(SUM);
            SUM=SUM*0+1;
            count2=1;
            [NewSM, ModelTran1,S_Sum,ModelTran]=ModelT(double(ModelNumber),Model_Sm,double(SP_all{j+1}.SuperPixelNumber),MaxDim,Color_dist,SUM,MainColor);
            for k=1:ModelNumber
                if Model_Sm(k)~=0
                    S_Sum(:,k)=sort(S_Sum(:,k),'descend');
                end
            end
            S_SUM=sum(S_Sum(1:3,:),1);
            Sum_ModelTran=sum(ModelTran1,2);
            for k=1:SP_all{j+1}.SuperPixelNumber
                if ColorWeight{j+1}(k)~=0
                    NewSM(k)=NewSM(k)/ColorWeight{j+1}(k);
                end
            end
            S_SUM=MatrixNormalization(S_SUM);
            for k=1:ModelNumber
                for z=1:SP_all{j+1}.SuperPixelNumber
                    if ModelTran(z,k)>0
                        ModelTran(z,k)=abs(ModelTran(z,k)-SaliencyM(z,j+1));
                    else
                        ModelTran(z,k)=0;
                    end
                end
            end
            ModelTranD=sum(abs(ModelTran));
            ModelTranD=MatrixNormalization(ModelTranD);
            Pos=zeros(ModelNumber,1);
            for k=1:ModelNumber
                if S_SUM(k)~=0
                    Pos(k)=1/(1+exp(-ModelTranD(k)/S_SUM(k)))*2-1;
                end
            end
            rand('state',2);
            for k=1:ModelNumber
                if Model_Energy(k)~=0
                    P=rand(1);
                    if(P>1)
                        aaaa=1
                    end
                    if P>Pos(k)
                        Model_Energy(k)=min(Model_Energy(k)+1,Energy_Num);
                    else
                        Model_Energy(k)=Model_Energy(k)-1;
                    end
                end
            end
            for k=1:ModelNumber
                if Model_Energy(k)==0
                    Model_Sm(k)=0;
                    Model(k,1:12)=[0 0 0 0 0 0 0 0 0 0 0 0];
                    Model_res(k)=0;
                    Model_Hsv(k,:)=[0 0 0 0 0 0 0 0];
                    Model_Pos(k,:)=[0 0];
                end
            end
            NewSM3=NewSM3+NewSM4;
            NewSM3=MatrixNormalization(NewSM3);
            par2=3*CoarseRetemp{j+1}*mean(NewSM3);
            for ii=1:size(NewSM3,1)
                NewSM3(ii)=1/(1+(exp(-Weight6*(NewSM3(ii)-par2))));
            end
            NewSM3=MatrixNormalization(NewSM3);
            
%%
            %Spatial-temporal Smoothing
            NewSM3=midsmooth(mid,ring,NewSM3,AfterSaliencyM,MaxDim,j+1-1,Weight_SmoothS1,SPnum,smrg1,LengthFiles-3,FlagAfter,smrg2);
            NewSM3=MatrixNormalization(NewSM3);
            AfterSaliencyM(:,j+1)=NewSM3;
            FlagAfter(j+1)=1;
            NewSM=MatrixNormalization(NewSM);
            par2=CoarseRetemp{j+1}*mean(NewSM);
            for ii=1:size(NewSM,1)
                NewSM(ii)=1/(1+(exp(-Weight6*(NewSM(ii)-par2))));
            end
            NewSM=MatrixNormalization(NewSM);
            NewSM=midsmooth(mid,ring,NewSM,AfterSaliencyM2,MaxDim,j+1-1,Weight_SmoothS1,SPnum,smrg1,LengthFiles-3,FlagAfter2,smrg2);
            NewSM=MatrixNormalization(NewSM);
            AfterSaliencyM2(:,j+1)=NewSM;
            FlagAfter2(j+1)=1;
            Temp_SaliencyM2=zeros(MaxDim,1);
            Temp_SaliencyM1=zeros(MaxDim,1);
            %Fusion
            for k=1:SP_all{j+1}.SuperPixelNumber
                Temp_SaliencyM1(k)=(NewSM3(k)+NewSM(k))/2;
                Temp_SaliencyM2(k)=(E_SUM(j+1)*NewSM(k)+SaliencySM_MotionM(k,j+1))*(NewSM(k)+(1-E_SUM(j+1))*SaliencySM_MotionM(k,j+1));
            end
            Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
            Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
            
           %%
            Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
            AfterSaliencyM3(:,j+1)=Temp_SaliencyM1;
            FlagAfter3(j+1)=1;
            Temp_SaliencyM2=midsmooth(mid,ring,Temp_SaliencyM2,AfterSaliencyM4,MaxDim,j+1-1,Weight_SmoothS2,SPnum,smrg1,LengthFiles-3,FlagAfter4,smrg2);            %%
            Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
            AfterSaliencyM4(:,j+1)=Temp_SaliencyM2;
            FlagAfter4(j+1)=1;
            Refine_Temp_SaliencyM1=Temp_SaliencyM1*0.3+0.7;
            SaliencyM(:,j+1)= 0.5*(MatrixNormalization(E_SUM(j+1)*Temp_SaliencyM1+(1-E_SUM(j+1))*MotionM(:,j+1)).*Refine_Temp_SaliencyM1)+SaliencySM_MotionM(:,j+1)*0.5;
            SaliencyM(:,j+1) = 1./(1+exp(-SaliencyM(:,j+1)*Par_sigmoid));
            SaliencyM(:,j+1) = MatrixNormalization(SaliencyM(:,j+1));
            SaliencyM_Copy2(:,j+1)=MatrixNormalization(E_SUM(j+1)*Temp_SaliencyM1+(1-E_SUM(j+1))*MotionM(:,j+1)).*Refine_Temp_SaliencyM1;%.*Temp_Copy
            for k=1:SP_all{j+1}.SuperPixelNumber
                SaliencyM_Copy2(k,j+1)=min(SaliencyM_Copy2(k,j+1),1);
            end
            SaliencyM(:,j+1)=midsmooth(mid,ring,SaliencyM(:,j+1),AfterSaliencyM5,MaxDim,j+1-1,Weight_SmoothS,SPnum,smrg1,LengthFiles-3,FlagAfter5,smrg2);
            SaliencyM(:,j+1)=MatrixNormalization(SaliencyM(:,j+1));
            AfterSaliencyM5(:,j+1)=SaliencyM(:,j+1);
            FlagAfter5(j+1)=1;
            %%
            rate = MotionMask_Num{i-1};
            temp = SaliencyM_Copy2(1:SP_all{j+1}.SuperPixelNumber,j+1);
            [index value] = find(temp<mean(temp)*0.5);
            temp(index) = 0;
            Threshold = mean(mean(abs(temp)));
            count = 0;
            while(count<50)
                count = count + 1;
                [index value] = find(temp>Threshold*rate);
                MotionMask1 = zeros(MaxDim,1);
                MotionMask1(index,1) = 1;
                [index value] = find(MotionMask1==1);
                MotionMaskLength1 = size(index,1);
                if(~exist('lastMotionMaskLength1'))
                    lastMotionMaskLength1 = MotionMaskLength1;
                end
                if(abs(MotionMaskLength1-lastMotionMaskLength1)<lastMotionMaskLength1/4)%Minimal Number of Selected Superpixels
                    break;
                end
                if(MotionMaskLength1<lastMotionMaskLength1)
                    rate = rate * 0.95;
                else
                    rate = rate * 1.05;
                end
            end
            MotionMaskAll(:,j+1)=MotionMask1;
            [index value] = find(MotionMaskAll(:,j+1)==1);
            MaskLength = size(index,1);
            if j<LengthFiles-3&&MaskLength~=0
                result2{j+1}=FindPair(j+1,MotionMaskAll,SP_all,Cluster_Gradient);
                result{j+1}=result2{j+1};
            end
            
%%
            %Update Pattern Models
            FusionedResult=zeros(300,300);
            for jj=1:size(MotionMaskAll,1)
                if MotionMaskAll(jj,j+1)~=0
                    ClusteringPixelNumber = SP_all{j+1}.ClusteringPixelNum(1,jj);
                    for z=1:ClusteringPixelNumber
                        XIndex= SP_all{j+1}.Clustering(jj,z,1);
                        YIndex= SP_all{j+1}.Clustering(jj,z,2);
                        FusionedResult(XIndex,YIndex) = 1;
                    end
                end
            end
            se = strel('disk',30);
            FusionedResult_Copy1 = imdilate(FusionedResult,se);
            FusionedResult_Copy1=MatrixNormalization(FusionedResult_Copy1);
            if(j+2<LengthFiles-3)
                for jj=1:300
                    for z=1:300
                        if FusionedResult_Copy1(jj,z)>0
                            MotionMaskAll_Copy1(SEGMENTS{j+2}(jj,z),j+2)=MotionMaskAll_Copy1(SEGMENTS{j+2}(jj,z),j+2)+1;
                        end
                    end
                end
                for jj=1:SP_all{j+2}.SuperPixelNumber
                    if MotionMaskAll_Copy1(jj,j+2)>SP_all{j+2}.ClusteringPixelNum(1,jj)/3
                        MotionMaskAll_Copy1(jj,j+2)=1;
                    else
                        MotionMaskAll_Copy1(jj,j+2)=0;
                    end
                end
                [Dis2Um{j+2},SP_all{j+2}.Neighbor]=get_neighbor(SP_all{j+2},MotionMaskAll_Copy1(:,j+2),mean(SN),PairInitDis);
            end
            
%%
            %Update the Appearance Model
            Pairs_Store=zeros(MaxDim,12);
            Hsv_Store=zeros(MaxDim,8);
            Saliency_Store=zeros(MaxDim,1);
            Pos=zeros(MaxDim,2);
            for k=1:size(result2{j+1},2)
                if ~isempty(result2{j+1}{k})&&result2{j+1}{k}{1}(2)~=0
                    Pairs_Store(result2{j+1}{k}{1}(1),1:3)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),3:5);
                    Hsv_Store(result2{j+1}{k}{1}(1),1:2)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),10:11);
                    Pairs_Store(result2{j+1}{k}{1}(1),4:6)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(2),3:5);
                    Hsv_Store(result2{j+1}{k}{1}(1),3:4)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(2),10:11);
                    Pairs_Store(result2{j+1}{k}{1}(1),7:9)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{2}(2),3:5);
                    Hsv_Store(result2{j+1}{k}{1}(1),5:6)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{2}(2),10:11);
                    Pairs_Store(result2{j+1}{k}{1}(1),10:12)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{3}(2),3:5);
                    Hsv_Store(result2{j+1}{k}{1}(1),7:8)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{3}(2),10:11);
                    Pos(result2{j+1}{k}{1}(1),:)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),1:2);
                    Saliency_Store(result2{j+1}{k}{1}(1),1)=SaliencyM(result2{j+1}{k}{1}(1),j+1);
                end
            end
            count=1;
            kk=1;
            [val,ind]=sort(Saliency_Store,'descend');
            index=find(Model_Energy==0);
            midReplace_Num=min(size(index,1),size(result2{j+1},2));
            for k=1:ModelNumber
                if Model_Energy(k)==0&&count<midReplace_Num&&kk<size(result2{j+1},2)
                    Model(k,:)=Pairs_Store(ind(kk),:);
                    Model_Energy(k)=Energy_Num;
                    Model_Pos(k,:)=Pos(ind(kk),:);
                    Model_Hsv(k,:)=Hsv_Store(ind(kk),:);
                    Model_Sm(k)=val(kk);
                    Model_res(k)=1;
                    count=count+1;
                    kk=kk+1;
                end
            end
        end
        [A_hat E_hat A]=TenDH(LengthFiles,I_CN,I,MotionMaskAll,SaliencyM,SP_all,Mode,ImageNameContainer,oW,oH,MaxDim);
        E_Hat=sum(abs(E_hat));
        E_Hat=MatrixNormalization(E_Hat);
        E_SUM=E_Hat;
        E_Mean=mean(E_Hat);
        for kkk=1:size(E_SUM,2)
            if E_SUM(kkk)>1.5*E_Mean
                E_SUM(kkk)=E_SUM(kkk)*0.1+0.2+a_Par;
            else
                E_SUM(kkk)=E_SUM(kkk)*0.2+a_Par;
            end
        end
        clear lastMotionMaskLength1;
    end
    
    for j= AnchorIndex(i):AnchorIndex(i+1)-1
        NewSM3=zeros(MaxDim,1);
        NewSM4=zeros(MaxDim,1);
        clear Pair;
        ColorDist=zeros(MaxDim,size(result{j},2));
        GradDist=zeros(MaxDim,size(result{j},2));
        MainColorD=zeros(MaxDim,size(result{j},2));
        Result_RGB=zeros(size(result{j},2),12);
        Result_LAB=zeros(size(result{j},2),8);
        Result_POS=zeros(size(result{j},2),2);
        tic
        for k=1:size(result{j},2)
            if  result{j}{k}{1}(2)~=0
                Result_RGB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),3:5) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),3:5) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),3:5)];
                Result_LAB(k,:)=[SP_all{j}.MiddlePoint(result{j}{k}{1}(1),10:11) SP_all{j}.MiddlePoint(result{j}{k}{1}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{2}(2),10:11) SP_all{j}.MiddlePoint(result{j}{k}{3}(2),10:11)];
                Result_POS(k,:)=SP_all{j}.MiddlePoint(result{j}{k}{1}(1),1:2);
            end
        end
        for k=1:size(result{j},2)
            if result{j}{k}{1}(2)~=0
                [Tt,Ss,Ww]=FrameTrans(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j+1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_F,Result_POS(k,1:2),Distance,ring(:,j+1),Weg1,Weg2);
                ColorDist(:,k)=Tt;
                GradDist(:,k)=Ss;
                MainColorD(:,k)=Ww;
            end
        end
        GradDist=exp(0.6*GradDist);
        GradDist=MatrixNormalization(GradDist);
        GradDist=GradDist*0.3+0.7;
        for k=1:size(result{j},2)
            if result{j}{k}{1}(2)~=0
                for z=1:SP_all{j+1}.SuperPixelNumber
                    if(ColorDist(z,k)~=1000)
                        WW=exp(-1*ColorDist(z,k)*Weight2);
                        NewSM3(z)=NewSM3(z)+SaliencyM(result{j}{k}{1}(1),j)*WW;%
                    end
                end
            end
        end
        for k=1:SP_all{j+1}.SuperPixelNumber
            if ColorWeight2{j+1}(k)~=0
                NewSM3(k)=NewSM3(k)/ColorWeight2{j+1}(k);
            end
        end
        if j-1~=0&&j~=AnchorIndex(i)
            clear Pair;
            ColorDist=zeros(MaxDim,size(result{j-1},2));
            GradDist=zeros(MaxDim,size(result{j-1},2));
            MainColorD=zeros(MaxDim,size(result{j-1},2));
            Result_RGB=zeros(size(result{j-1},2),12);
            Result_LAB=zeros(size(result{j-1},2),8);
            Result_POS=zeros(size(result{j-1},2),2);
            tic
            for k=1:size(result{j-1},2)
                if  result{j-1}{k}{1}(2)~=0
                    Result_RGB(k,:)=[SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(2),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{2}(2),3:5) SP_all{j-1}.MiddlePoint(result{j-1}{k}{3}(2),3:5)];
                    Result_LAB(k,:)=[SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(2),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{2}(2),10:11) SP_all{j-1}.MiddlePoint(result{j-1}{k}{3}(2),10:11)];
                    Result_POS(k,:)=SP_all{j-1}.MiddlePoint(result{j-1}{k}{1}(1),1:2);
                end
            end
            for k=1:size(result{j-1},2)
                if result{j-1}{k}{1}(2)~=0
                    [Tt,Ss,Ww]=FrameTrans(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Result_RGB(k,1:12),SP_all{j+1}.Neighbor,Result_LAB(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_F,Result_POS(k,1:2),Distance,ring(:,j+1),Weg1,Weg2);
                    ColorDist(:,k)=Tt;
                    GradDist(:,k)=Ss;
                    MainColorD(:,k)=Ww;
                end
            end
            GradDist=exp(0.6*GradDist);
            GradDist=MatrixNormalization(GradDist);
            GradDist=GradDist*0.3+0.7;
            for k=1:size(result{j-1},2)
                if result{j-1}{k}{1}(2)~=0
                    for z=1:SP_all{j+1}.SuperPixelNumber
                        if(ColorDist(z,k)~=1000)
                            WW=exp(-1*ColorDist(z,k)*Weight2);
                            NewSM4(z)=NewSM4(z)+SaliencyM(result{j-1}{k}{1}(1),j-1)*WW;%
                        end
                    end
                end
            end
            for k=1:SP_all{j+1}.SuperPixelNumber
                if ColorWeight2{j+1}(k)~=0
                    NewSM4(k)=NewSM4(k)/ColorWeight2{j+1}(k);
                end
            end
            
        end
        Color_dist=zeros(MaxDim,ModelNumber);
        SUM=zeros(MaxDim,ModelNumber);
        MainColor=zeros(MaxDim,ModelNumber);
        for k=1:ModelNumber
            if Model_Sm(k)~=0
                [Tt,Ss,Ww]=ModelTransfer(double(SP_all{j+1}.SuperPixelNumber),SP_all{j+1}.MiddlePoint,Model(k,1:12),SP_all{j+1}.Neighbor,Model_Hsv(k,1:8),GradientMatrix{j+1},double(MaxDim),WeightD_M,Model_Pos(k,1:2),DDis,ring(:,j+1),Weg1,Weg2);
                Color_dist(:,k)=Tt;
                SUM(:,k)=Ss;
                MainColor(:,k)=Ww;
            end
        end
        SUM=exp(0.6*SUM);
        SUM=MatrixNormalization(SUM);
        SUM=SUM*0+1;
        [NewSM, ModelTran1,S_Sum,ModelTran]=ModelT(double(ModelNumber),Model_Sm,double(SP_all{j+1}.SuperPixelNumber),MaxDim,Color_dist,SUM,MainColor);
        for k=1:ModelNumber
            if Model_Sm(k)~=0
                S_Sum(:,k)=sort(S_Sum(:,k),'descend');
            end
        end
        S_SUM=sum(S_Sum(1:3,:),1);
        Sum_ModelTran=sum(ModelTran1,2);
        for k=1:SP_all{j+1}.SuperPixelNumber
            if ColorWeight{j+1}(k)~=0
                NewSM(k)=NewSM(k)/ColorWeight{j+1}(k);
            end
        end
        S_SUM=MatrixNormalization(S_SUM);
        for k=1:ModelNumber
            for z=1:SP_all{j+1}.SuperPixelNumber
                if ModelTran(z,k)>0
                    ModelTran(z,k)=ModelTran(z,k)-SaliencyM(z,j+1);
                else
                    ModelTran(z,k)=0;
                end
            end
        end
        ModelTranD=sum(abs(ModelTran));
        ModelTranD=MatrixNormalization(ModelTranD);
        Pos=zeros(ModelNumber,1);
        for k=1:ModelNumber
            if S_SUM(k)~=0
                Pos(k)=1/(1+exp(-ModelTranD(k)/S_SUM(k)))*2-1;
            end
        end
        count=1;
        while count<=ModelNumber&&Model_Energy(count)~=0
            count=count+1;
        end
        
        rand('state',2);
        for k=1:ModelNumber
            if Model_Energy(k)~=0
                P=rand(1);
                if P>Pos(k)
                    Model_Energy(k)=min(Model_Energy(k)+1,Energy_Num);
                else
                    Model_Energy(k)=Model_Energy(k)-1;
                end
            end
        end
        for k=1:ModelNumber
            if Model_Energy(k)==0
                Model_Sm(k)=0;
                Model(k,1:12)=[0 0 0 0 0 0 0 0 0 0 0 0];
                Model_Hsv(k,1:8)=[0 0 0 0 0 0 0 0];
                Model_res(k)=0;
                Model_Pos(k,:)=[0 0];
            end
        end
        
        NewSM3=NewSM3+NewSM4;
        NewSM3=MatrixNormalization(NewSM3);
        par2=3*CoarseRetemp{j+1}*mean(NewSM3);
        for ii=1:size(NewSM3,1)
            NewSM3(ii)=1/(1+(exp(-Weight6*(NewSM3(ii)-par2))));
        end
        NewSM3=MatrixNormalization(NewSM3);
        
       %%
        NewSM3=midsmooth(mid,ring,NewSM3,AfterSaliencyM,MaxDim,j+1-1,Weight_SmoothS1,SPnum,smrg1,LengthFiles-3,FlagAfter,smrg2);
        NewSM3=MatrixNormalization(NewSM3);
        AfterSaliencyM(:,j+1)=NewSM3;
        FlagAfter(j+1)=1;
        NewSM=MatrixNormalization(NewSM);
        par2=3*CoarseRetemp{j+1}*mean(NewSM);
        for ii=1:size(NewSM,1)
            NewSM(ii)=1/(1+(exp(-Weight6*(NewSM(ii)-par2))));
        end
        NewSM=MatrixNormalization(NewSM);
        NewSM=midsmooth(mid,ring,NewSM,AfterSaliencyM2,MaxDim,j+1-1,Weight_SmoothS1,SPnum,2,LengthFiles-3,FlagAfter2,smrg2);
        NewSM=MatrixNormalization(NewSM);
        AfterSaliencyM2(:,j+1)=NewSM;
        FlagAfter2(j+1)=1;
       %%
        Temp_SaliencyM2=zeros(MaxDim,1);
        Temp_SaliencyM1=zeros(MaxDim,1);
        for k=1:SP_all{j+1}.SuperPixelNumber
            Temp_SaliencyM1(k)=(NewSM3(k)+NewSM(k))/2;
            Temp_SaliencyM2(k)=(E_SUM(j+1)*NewSM(k)+SaliencySM_MotionM(k,j+1))*(NewSM(k)+(1-E_SUM(j+1))*SaliencySM_MotionM(k,j+1));
        end
        Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
        Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
       %%
        Temp_SaliencyM1=MatrixNormalization(Temp_SaliencyM1);
        AfterSaliencyM3(:,j+1)=Temp_SaliencyM1;
        FlagAfter3(j+1)=1;
        Temp_SaliencyM2=midsmooth(mid,ring,Temp_SaliencyM2,AfterSaliencyM4,MaxDim,j+1-1,Weight_SmoothS2,SPnum,smrg1,LengthFiles-3,FlagAfter4,smrg2);            %%
        Temp_SaliencyM2=MatrixNormalization(Temp_SaliencyM2);
        AfterSaliencyM4(:,j+1)=Temp_SaliencyM2;
        FlagAfter4(j+1)=1;
        Refine_Temp_SaliencyM1=Temp_SaliencyM1*0.3+0.7;
        SaliencyM(:,j+1)= 0.5*(MatrixNormalization(E_SUM(j+1)*Temp_SaliencyM1+(1-E_SUM(j+1))*MotionM(:,j+1)).*Refine_Temp_SaliencyM1)+SaliencySM_MotionM(:,j+1)*0.5;
        SaliencyM(:,j+1) = 1./(1+exp(-SaliencyM(:,j+1)*Par_sigmoid));
        SaliencyM(:,j+1) = MatrixNormalization(SaliencyM(:,j+1));
        SaliencyM_Copy2(:,j+1)=MatrixNormalization(E_SUM(j+1)*Temp_SaliencyM1+(1-E_SUM(j+1))*MotionM(:,j+1)).*Refine_Temp_SaliencyM1;
        for k=1:SP_all{j+1}.SuperPixelNumber
            SaliencyM_Copy2(k,j+1)=min(SaliencyM_Copy2(k,j+1),1);
        end
        SaliencyM(:,j+1)=midsmooth(mid,ring,SaliencyM(:,j+1),AfterSaliencyM5,MaxDim,j+1-1,Weight_SmoothS,SPnum,smrg1,LengthFiles-3,FlagAfter5,smrg2);
        SaliencyM(:,j+1)=MatrixNormalization(SaliencyM(:,j+1));
        AfterSaliencyM5(:,j+1)=SaliencyM(:,j+1);
        FlagAfter5(j+1)=1;
        %%
        rate = (MotionMask_Num{i}*0.7+MotionMask_Num{i-1}*0.3)*0.8;
        temp = SaliencyM_Copy2(1:SP_all{j+1}.SuperPixelNumber,j+1);
        [index value] = find(temp<mean(temp)*0.5);
        temp(index) = 0;
        Threshold = mean(mean(abs(temp)));
        count = 0;
        while(count<50)
            count = count + 1;
            [index value] = find(temp>Threshold*rate);
            MotionMask1 = zeros(MaxDim,1);
            MotionMask1(index,1) = 1;
            [index value] = find(MotionMask1==1);
            MotionMaskLength1 = size(index,1);
            if(~exist('lastMotionMaskLength1'))
                lastMotionMaskLength1 = MotionMaskLength1;
            end
            if(abs(MotionMaskLength1-lastMotionMaskLength1)<lastMotionMaskLength1/4)
                break;
            end
            if(MotionMaskLength1<lastMotionMaskLength1)
                rate = rate * 0.95;
            else
                rate = rate * 1.05;
            end
        end
        MotionMaskAll(:,j+1)=MotionMask1;
        [index value] = find(MotionMaskAll(:,j+1)==1);
        
        MaskLength = size(index,1);
        if j<LengthFiles-3&&MaskLength~=0
            result2{j+1}=FindPair(j+1,MotionMaskAll,SP_all,Cluster_Gradient);
            result{j+1}=result2{j+1};
        end
        %%
        if(j+2<=LengthFiles-3)
            FusionedResult=zeros(300,300);
            for jj=1:size(MotionMaskAll,1)
                if MotionMaskAll(jj,j+1)~=0
                    ClusteringPixelNumber = SP_all{j+1}.ClusteringPixelNum(1,jj);
                    for z=1:ClusteringPixelNumber
                        XIndex= SP_all{j+1}.Clustering(jj,z,1);
                        YIndex= SP_all{j+1}.Clustering(jj,z,2);
                        FusionedResult(XIndex,YIndex) = 1;
                    end
                end
            end
            se = strel('disk',30);
            FusionedResult_Copy1 = imdilate(FusionedResult,se);
            FusionedResult_Copy1=MatrixNormalization(FusionedResult_Copy1);
            if(j+2<LengthFiles-3)
                for jj=1:300
                    for z=1:300
                        if FusionedResult_Copy1(jj,z)>0
                            MotionMaskAll_Copy1(SEGMENTS{j+2}(jj,z),j+2)=MotionMaskAll_Copy1(SEGMENTS{j+2}(jj,z),j+2)+1;
                        end
                    end
                end
                for jj=1:SP_all{j+2}.SuperPixelNumber
                    if MotionMaskAll_Copy1(jj,j+2)>SP_all{j+2}.ClusteringPixelNum(1,jj)/3
                        MotionMaskAll_Copy1(jj,j+2)=1;
                    else
                        MotionMaskAll_Copy1(jj,j+2)=0;
                    end
                end
                
                [Dis2Um{j+2},SP_all{j+2}.Neighbor]=get_neighbor(SP_all{j+2},MotionMaskAll_Copy1(:,j+2),mean(SN),PairInitDis);
            end
        end
        %%
        Pairs_Store=zeros(MaxDim,12);
        Hsv_Store=zeros(MaxDim,8);
        Saliency_Store=zeros(MaxDim,1);
        Pos=zeros(MaxDim,2);
        for k=1:size(result2{j+1},2)
            if ~isempty(result2{j+1}{k})&&result2{j+1}{k}{1}(2)~=0
                Pairs_Store(result2{j+1}{k}{1}(1),1:3)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),3:5);
                Hsv_Store(result2{j+1}{k}{1}(1),1:2)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),10:11);
                Pairs_Store(result2{j+1}{k}{1}(1),4:6)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(2),3:5);
                Hsv_Store(result2{j+1}{k}{1}(1),3:4)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(2),10:11);
                Pairs_Store(result2{j+1}{k}{1}(1),7:9)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{2}(2),3:5);
                Hsv_Store(result2{j+1}{k}{1}(1),5:6)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{2}(2),10:11);
                Pairs_Store(result2{j+1}{k}{1}(1),10:12)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{3}(2),3:5);
                Hsv_Store(result2{j+1}{k}{1}(1),7:8)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{3}(2),10:11);
                Pos(result2{j+1}{k}{1}(1),:)=SP_all{j+1}.MiddlePoint(result2{j+1}{k}{1}(1),1:2);
                Saliency_Store(result2{j+1}{k}{1}(1),1)=SaliencyM(result2{j+1}{k}{1}(1),j+1);
            end
        end
        count=1;
        kk=1;
        [val,ind]=sort(Saliency_Store,'descend');
        index=find(Model_Energy==0);
        
        midReplace_Num=min(size(index,1),size(result2{j+1},2));
        for k=1:ModelNumber
            if Model_Energy(k)==0&&count<midReplace_Num&&kk<size(result2{j+1},2)
                Model(k,:)=Pairs_Store(ind(kk),:);
                Model_Energy(k)=Energy_Num;
                Model_Pos(k,:)=Pos(ind(kk),:);
                Model_Hsv(k,:)=Hsv_Store(ind(kk),:);
                Model_Sm(k)=val(kk);
                Model_res(k)=1;
                count=count+1;
                kk=kk+1;
            end
        end
    end
    [A_hat E_hat A]=TenDH(LengthFiles,I_CN,I,MotionMaskAll,SaliencyM,SP_all,Mode,ImageNameContainer,oW,oH,MaxDim);
    E_Hat=sum(abs(E_hat));
    E_Hat=MatrixNormalization(E_Hat);
    E_Mean=mean(E_Hat);
    E_SUM1=E_Hat;
    E_SUM=E_Hat;
    for kkk=1:size(E_SUM,2)
        if E_SUM(kkk)>1.5*E_Mean
            E_SUM(kkk)=E_SUM(kkk)*0.1+0.2+a_Par;
        else
            E_SUM(kkk)=E_SUM(kkk)*0.2+a_Par;
        end
    end
    clear lastMotionMaskLength1;
end
SaliencyM=(SaliencyM_Copy1+SaliencyM_Copy2)/2;
CountImg=3;
for i=1:LengthFiles-3
    %%
    SaliencyM(:,i)=MatrixNormalization(SaliencyM(:,i));
    [W H] = size(I{1});
    FusionedResult = zeros(W,H/3);
    for l=1:SP_all{i}.SuperPixelNumber
        ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,l);
        for j=1:ClusteringPixelNumber%for Test Only
            XIndex= SP_all{i}.Clustering(l,j,1);
            YIndex= SP_all{i}.Clustering(l,j,2);
            FusionedResult(XIndex,YIndex) = SaliencyM(l,i);
        end
    end
    FusionedResult = MatrixNormalization(FusionedResult);
    FusionedResult = imresize(FusionedResult,[oW oH]);
    CountImg=CountImg+1;
end
SaliencyM=Final_smooth(mid,ring,SaliencyM,MaxDim,LengthFiles-3,5.0,SPnum,5.0,20.0);

%%
%pixel-wise smoothing
CountImg=3;
for i=1:LengthFiles-3
    [W H] = size(I{1});
    FusionedResult = zeros(W,H/3);
    for l=1:SP_all{i}.SuperPixelNumber
        ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,l);
        for j=1:ClusteringPixelNumber%for Test Only
            XIndex= SP_all{i}.Clustering(l,j,1);
            YIndex= SP_all{i}.Clustering(l,j,2);
            FusionedResult(XIndex,YIndex) = SaliencyM(l,i);
        end
        %         end
    end
    FusionedResult = MatrixNormalization(FusionedResult);
    FusionedResult = imresize(FusionedResult,[oW oH]);
    CountImg=CountImg+1;
end
%%pixel_wise assaign
N=LengthFiles-3;
LastBatchSaliencyRecord=zeros(MaxDim,N);
for i=1:N
    [W H] = size(ISmoothed{1});
    Result = zeros(W,H/3);
    
    SaliencyAssignTemp=(SaliencyM(:,i));
    if i>1
        SaliencyAssignTemp0=(SaliencyM(:,i-1));
    else
        SaliencyAssignTemp0=(SaliencyM(:,i));
    end
    if i<N
        SaliencyAssignTemp2=(SaliencyM(:,i+1));
    else
        SaliencyAssignTemp2=(SaliencyM(:,i));
    end
    LastBatchSaliencyRecord(:,i)=colorsm2(mid,ring,i-1,MaxDim,SaliencyAssignTemp,SaliencyAssignTemp0,SaliencyAssignTemp2,N,SmoothRange*0.5,30,SPnum);%cuda and C program
    for l=1:SP_all{i}.SuperPixelNumber
        ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,l);
        for j=1:ClusteringPixelNumber
            XIndex= SP_all{i}.Clustering(l,j,1);
            YIndex= SP_all{i}.Clustering(l,j,2);
            Result(XIndex,YIndex) = LastBatchSaliencyRecord(l,i);
        end
    end
    %Pixel-wise Assignment
    LocalSize = 20;
    ImageSize = 300;
    ResultBox1 = imresize(Result,[ImageSize,ImageSize]);
    ResultBox2 = imresize(I{i},[ImageSize,ImageSize]);
    ResultBox3=pixelAssign(ResultBox1,ResultBox2,LocalSize,25);%cuda and C program
    ResultBox3 = MatrixNormalization(ResultBox3);
    ResultBox3 = imresize(ResultBox3,[oW oH]);
    Result = MatrixNormalization(Result);
    imwrite(ResultBox3*2,['.\result\', Mode ,'\FinalSaliency',ParR2,'\' , ImageNameContainer{i+2}]);
end




