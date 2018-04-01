function [result,GradientMatrix,S_num,Cluster_Gradient] = computeMotionGradientMap(I_all,vx_all,vy_all,SP_all,Mode,oW,oH,vx_copy,vy_copy)

% vx = OF.vx;
% vy = OF.vy;
D_Sum=0;
for FrameIndex=1:size(vx_all,2)
    vx=vx_all{FrameIndex};
    vy=vy_all{FrameIndex};
    I=I_all{FrameIndex};
    SP=SP_all{FrameIndex};
    [gx gy] = gradient(vx);
    vxGradientMap = sqrt(gx.^2+gy.^2);
    [gx gy] = gradient(vy);
    vyGradientMap = sqrt(gx.^2+gy.^2);
    [CSGMask, gb_thin_CS, gb_CS, orC, edgeImage, edgeComponents] = Gb_CSG(I);
%     GradientMatrix{FrameIndex}=(CSGMask-min(min(CSGMask)))/(max(max(CSGMask))-min(min(CSGMask)));
    temp1 = MatrixNormalization(CSGMask);
    temp2 = MatrixNormalization(vxGradientMap+vyGradientMap);
    FinalGradientMap = temp1.*temp2;
    temp = zeros(size(I,1),size(I,2));
    T = mean(mean(FinalGradientMap));
    [index] = find(FinalGradientMap(:)>40*T);
    temp(index) = 1;
    [index2] = find(FinalGradientMap(:)>120*T);
    temp2(index2) = 1;
    FinalGradientMap = temp;
  
    IndexPool = zeros(2,length(index));
    for i = 1 : length(index)
        [x y] = ind2sub([size(I,1), size(I,2)], index(i));
        IndexPool(:,i) = [x y]';
    end
    
    IndexPool2 = zeros(2,length(index2));
    for i = 1 : length(index2)
        [x y] = ind2sub([size(I,1), size(I,2)], index2(i));
        IndexPool2(:,i) = [x y]';
    end
    num_G=size(IndexPool,2);
    if  FrameIndex==15
        aaa=1;
    end
    
    if mean(mean(abs(vx_copy{FrameIndex})))+mean(mean(abs(vy_copy{FrameIndex})))<0.15&&FrameIndex~=1
        GradientMatrix{FrameIndex}=GradientMatrix{FrameIndex-1};
        result{FrameIndex} = result{FrameIndex-1};
        IndexPool=result{FrameIndex-1};
        IndexPool2=result_copy{FrameIndex-1};
        result_copy{FrameIndex}=result_copy{FrameIndex-1};
    else
        GradientMatrix{FrameIndex}=MatrixNormalization(FinalGradientMap);
        result{FrameIndex} = IndexPool;
        result_copy{FrameIndex}=IndexPool2;
    end
    
    [Idx,Ctrs,SumD,D] = kmeans(IndexPool',int32(num_G*0.3),'Replicates',4);
    Cluster_Gradient{FrameIndex}=Ctrs';%
    [Idx,Ctrs,SumD,D] = kmeans(IndexPool2',1,'dist','sqEuclidean','Replicates',4);
    D=sqrt(D);
    [newD,ind]=sort(D);
    MeanD1=mean(newD(int32(size(IndexPool2,2))/10:int32(4*size(IndexPool2,2)/5)));
    [Idx2,Ctrs2,SumD2,D2] = kmeans(IndexPool2',2,'Replicates',4);
    D2=sqrt(D2);
    Distance_C=sqrt(sum((Ctrs2(1,:)-Ctrs).^2))+sqrt(sum((Ctrs2(2,:)-Ctrs).^2));
    newD21=zeros(size(Idx2,1),1);
    newD22=zeros(size(Idx2,1),1);
    for k=1:size(Idx2,1)
        if Idx2(k)==1
            newD21(k)=D2(k,1);
        else
            newD22(k)=D2(k,2);
        end
    end
    
    Ind1=find(newD21>0);
    Ind2=find(newD22>0);
    [NEWD21,~]=sort(newD21,'descend');
    [NEWD22,~]=sort(newD22,'descend');
    
    MeanD21=mean(NEWD21(int32(max(size(Ind1,1)/5,1)):int32(max(9*size(Ind1,1)/10,1))));
    MeanD22=mean(NEWD22(int32(max(size(Ind2,1)/5,1)):int32(max(9*size(Ind2,1)/10,1))));
    
    
    if Distance_C>120
        S_num{FrameIndex}=double(3.14*(MeanD21^2+MeanD22^2)/(300.0*300.0)*SP_all{1}.SuperPixelNumber);
    else
        S_num{FrameIndex}=double(3.14*(MeanD1^2)/(300.0*300.0)*SP_all{1}.SuperPixelNumber);
    end
    S_num{FrameIndex}=min(S_num{FrameIndex},200);
    D_Sum=D_Sum+Distance_C;
    tempP=zeros(300,300);
    for iii=1:size(IndexPool,2)
        tempP(IndexPool(1,iii),IndexPool(2,iii))=1;
    end
%     FIndex=num2str(FrameIndex);
%     tempP=imresize(tempP,[oW oH]);
%     imwrite(tempP*1.5,['.\result\' Mode ,'\Gradient\' ,FIndex,'.jpg']);
%     
%     FIndex=num2str(FrameIndex+1000000);
%     tempP=imresize(GradientMatrix{FrameIndex},[oW oH]);
%     imwrite(tempP*10,['.\result\' Mode ,'\Gradient\' ,FIndex,'.jpg']);
end
%  