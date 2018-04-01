function [SEGMENT,SP] = getSPInfo(rgbI,hsvI,IHsv,ILab,spnum,I_CN)
Distance=sqrt(300*300/spnum)*1.5;
[SEGMENTS, numlabels] = slicmex((rgbI),spnum,20);
% SEGMENTS = vl_slic(single(rgbI),13,0.001);%13 is the best choice
SEGMENTS = SEGMENTS + 1;
SEGMENT=SEGMENTS;
SP.SuperPixelNumber = numlabels;
% SP.Neighbor=zeros(SP.SuperPixelNumber,25);%

SP.Clustering = zeros(SP.SuperPixelNumber,int32(600*300/spnum)*1.5,11);%
SP.ClusteringPixelNum = zeros(1,SP.SuperPixelNumber);
% Temp_ICN=zeros(SP.SuperPixelNumber,int32(600*300/spnum),10);
for i=1:size(rgbI,1)
    for j=1:size(rgbI,2)
        SIndex = SEGMENTS(i,j);
        if(SP.ClusteringPixelNum(1,SIndex)+1<=1.5*int32(600*300/spnum))
            SP.ClusteringPixelNum(1,SIndex) = SP.ClusteringPixelNum(1,SIndex)+1;
            SP.Clustering(SIndex,SP.ClusteringPixelNum(1,SIndex),:) = [i j hsvI(i,j,1) hsvI(i,j,2) hsvI(i,j,3) IHsv(i,j,1) IHsv(i,j,2) IHsv(i,j,3) ILab(i,j,1) ILab(i,j,2) ILab(i,j,3)]';%记录全部像素点
%             Temp_ICN(SIndex,SP.ClusteringPixelNum(1,SIndex),:)=I_CN(i,j,:);
        end
    end
end
SP.MiddlePoint = zeros(SP.SuperPixelNumber,11);
SP.Icn = zeros(SP.SuperPixelNumber,10);
InvalidIndex = zeros(1,SP.SuperPixelNumber);
BoundaryIndex = zeros(1,SP.SuperPixelNumber);
MarginLock = 8;
ImageSize = size(rgbI,1);

for i=1:SP.SuperPixelNumber
    sum_x = 0;sum_y = 0;sum_r = 0;sum_g = 0;sum_b = 0;sum_h=0;sum_s=0;sum_v=0;sum_l=0;sum_a=0;sum_bb=0;
    sum_ICN=zeros(10,1);
    for j=1:SP.ClusteringPixelNum(1,i)
       XIndex = SP.Clustering(i,j,1);
       YIndex = SP.Clustering(i,j,2);
       sum_x = sum_x + XIndex;
       sum_y = sum_y + YIndex;
       sum_r = sum_r + hsvI(XIndex,YIndex,1);
       sum_g = sum_g + hsvI(XIndex,YIndex,2);
       sum_b = sum_b + hsvI(XIndex,YIndex,3);
       sum_h = sum_h + IHsv(XIndex,YIndex,1);
       sum_s = sum_s + IHsv(XIndex,YIndex,2);
       sum_v = sum_v + IHsv(XIndex,YIndex,3);
       sum_l = sum_l + ILab(XIndex,YIndex,1);
       sum_a = sum_a + ILab(XIndex,YIndex,2);
       sum_bb = sum_bb + ILab(XIndex,YIndex,3);
       sum_ICN=reshape(I_CN(XIndex,YIndex,1:10),10,1)+sum_ICN;
    end
    if(SP.ClusteringPixelNum(1,i)~=0)
       SP.MiddlePoint(i,:) = ([sum_x sum_y sum_r sum_g sum_b sum_h sum_s sum_v sum_l sum_a sum_bb]./SP.ClusteringPixelNum(1,i))';
       SP.Icn(i,:)=sum_ICN/SP.ClusteringPixelNum(1,i);
       XIndex = sum_x/SP.ClusteringPixelNum(1,i);
       YIndex = sum_y/SP.ClusteringPixelNum(1,i);
       if(XIndex<2*MarginLock || XIndex>ImageSize-2*MarginLock || YIndex<2*MarginLock || YIndex>ImageSize-2*MarginLock)
           SP.BoundaryIndex(1,i) = 0.5;%
           if(XIndex<MarginLock || XIndex>ImageSize-MarginLock || YIndex<MarginLock || YIndex>ImageSize-MarginLock)
               SP.BoundaryIndex(1,i) = 0.25;
           end
       else
           SP.BoundaryIndex(1,i) = 1;%
       end 
    else
        SP.MiddlePoint(i,:) = ([0 0 0 0 0 0 0 0 0 0 0 ])';
        SP.Icn(i,:)=([0 0 0 0 0 0 0 0 0 0])';
        InvalidIndex(i,1) = 1;
    end
    
end
[value index] = find(InvalidIndex==1);
if(size(index,2)==1)
SP.SuperPixelNumber = SP.SuperPixelNumber - size(index,1);
SP.MiddlePoint(index,:) = [];
SP.ClusteringPixelNum(:,index) = [];
SP.Clustering(index,:,:) = [];
SP.BoundaryIndex(:,index) = [];
end
ii=1;

