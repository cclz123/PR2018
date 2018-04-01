function [color,motion,location,vx,vy,hsv,lab,vx_copy,vy_copy] = getHistogramFeatures(I,SP)

W = size(I{1},1);
H = size(I{1},2);
parfor i=1:size(I,2)-1
   [vx{i},vy{i}] = getMotionFeature(I{i},I{i+1});% motion feature 
   vx_copy{i}=vx{i};
   vy_copy{i}=vy{i};
   MotionLength{i} = abs(vx{i}+vy{i});
   MotionAngle{i} = atan(vy{i}./vx{i});
end

maxValue = max(max(max(cell2mat(vx))),max(max(cell2mat(vy))));
minValue = min(min(min(cell2mat(vx))),min(min(cell2mat(vy))));
for i=1:size(I,2)-1
   if(maxValue-minValue~=0)
      vx{i} = (vx{i}-minValue)/(maxValue-minValue);
      vy{i} = (vy{i}-minValue)/(maxValue-minValue);
   end
end

MotionLengthMax = max(max(cell2mat(MotionLength)));% 
MotionLengthMin = min(min(cell2mat(MotionLength)));
MotionAngleMax = max(max(cell2mat(MotionAngle)));% 
MotionAngleMin = min(min(cell2mat(MotionAngle)));
for i=1:size(I,2)-1
   if(MotionLengthMax-MotionLengthMin~=0)
      MotionLength{i} = (MotionLength{i}-MotionLengthMin)/(MotionLengthMax-MotionLengthMin);
   end
   if(MotionAngleMax-MotionAngleMin~=0)
       MotionAngle{i} = (MotionAngle{i}-MotionAngleMin)/(MotionAngleMax-MotionAngleMin);
   end
end


for i=1:size(I,2)-1
    for j=1:SP{i}.SuperPixelNumber
        PixelPool = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),3:5));
        color{i}{j} = mean(abs(PixelPool));% color histogram
        PixelPool2 = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),6:8));
        hsv{i}{j} = mean(abs(PixelPool2));% color histogram
        PixelPool3 = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),9:11));
        lab{i}{j} = mean(abs(PixelPool3));% color histogram
        SPIndex = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),1:2));
        MotionPool = zeros(size(SPIndex,1),2);
        for k=1:size(SPIndex,1)
           MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2))];
        end
        motion{i}{j} = mean((MotionPool));
        location{i}{j} = [SP{i}.MiddlePoint(j,1)/W SP{i}.MiddlePoint(j,2)/H];     
        if(SP{i}.ClusteringPixelNum(1,j)==0)% prevent NaN
            motion{i}{j} = zeros(1,2);
            color{i}{j} = zeros(1,3);
        end
    end
end



