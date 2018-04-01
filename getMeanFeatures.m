%参数d为histogram的维数

function [color,motion,location,d,vx,vy] = getMeanFeatures(I,SP)


d = 1;

W = size(I{1},1);
H = size(I{1},2);

%计算各图像的motion
fprintf('compute optical flows: '); 
for i=1:size(I,2)-1
   fprintf('%d ', i); 
   [vx{i},vy{i}] = getMotionFeature(I{i},I{i+1});% motion feature 
end
fprintf('successed\n'); 
maxValue = max(max(max(cell2mat(vx))),max(max(cell2mat(vy))));% 归一化处理
minValue = min(min(min(cell2mat(vx))),min(min(cell2mat(vy))));
for i=1:size(I,2)-1
   if(maxValue-minValue~=0)
      vx{i} = (vx{i}-minValue)/(maxValue-minValue);
      vy{i} = (vy{i}-minValue)/(maxValue-minValue);
   end
end

%计算各图像中，各像素点的颜色histogram
for i=1:size(I,2)-1
    for j=1:SP{i}.SuperPixelNumber
        PixelPool = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),3:5));
        color{i}{j} = mean(PixelPool);
        
        SPIndex = shiftdim(SP{i}.Clustering(j,1:SP{i}.ClusteringPixelNum(1,j),1:2));
        MotionPool = zeros(size(SPIndex,1),2);
        for k=1:size(SPIndex,1)
           MotionPool(k,:) = [vx{i}(SPIndex(k,1),SPIndex(k,2)) vy{i}(SPIndex(k,1),SPIndex(k,2))];
        end
        motion{i}{j} = mean(MotionPool);
        
        location{i}{j} = [SP{i}.MiddlePoint(j,1)/W SP{i}.MiddlePoint(j,2)/H];
        
        if(SP{i}.ClusteringPixelNum(1,j)==0)% prevent NaN
            motion{i}{j} = [1 1];
            color{i}{j} = [1 1 1];
        end
    end
end
