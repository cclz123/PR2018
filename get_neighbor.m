function [Dis2Um,Neighbor]=get_neighbor(SP,MotionMask,Mean_SN,PairInitDis)
Distance=zeros(SP.SuperPixelNumber,1);%40
Neighbor=zeros(SP.SuperPixelNumber,50);%邻居信息，暂定最多为25个

for i=1:SP.SuperPixelNumber
    minD=1000;
    if MotionMask(i)==1
        posi=SP.MiddlePoint(i,1:2);
        for j=1:SP.SuperPixelNumber
            if MotionMask(j)==0
                posj=SP.MiddlePoint(j,1:2);
                minD=min(sqrt(sum((posi-posj).^2)),minD);
            end
        end
        Distance(i)=minD;
    else
        Distance(i)=0;
    end
end
Max=0;
Min=100;
for i=1:SP.SuperPixelNumber
    if(Distance(i)~=0)
        Max=max(Max,Distance(i));
        Min=min(Min,Distance(i));
    end
end
for i=1:SP.SuperPixelNumber
    if(Distance(i)~=0&&(Max-Min)~=0)
        Distance(i)=(Distance(i)-Min)/(Max-Min);
    end
end
Distance=MatrixNormalization(Distance);
Distance=PairInitDis+(Distance*Mean_SN);
% Distance=100+(Distance*0);
Dis2Um=Distance;
for i=1:SP.SuperPixelNumber
    cout=1;
    posi=SP.MiddlePoint(i,1:2);
    for j=1:SP.SuperPixelNumber
        posj=SP.MiddlePoint(j,1:2);
        if(i~=j&&sqrt(sum((posi-posj).^2))<Distance(i)&&cout<=50)
            Neighbor(i,cout)=j;
            cout=cout+1;
        end
    end
end