function [result]=FindPair3(i,MotionMaskAll,SP_all,MotionGradientMatrix_all)
[index value] = find(MotionMaskAll(:,i)==1);
MaskLength = size(index,1);
PairNum=6;
for j=1:MaskLength
    if index(j)<=SP_all{i}.SuperPixelNumber
        x=SP_all{i}.MiddlePoint(index(j),1);
        y=SP_all{i}.MiddlePoint(index(j),2);
        temp_M=bsxfun(@minus,MotionGradientMatrix_all{i},[x y]');
        Dis=sqrt(sum(temp_M.^2));
        Flag=zeros(1,3);%选取三个最近的梯度点
        CCount=1;
        FFlag=zeros(SP_all{i}.SuperPixelNumber,1);
        [sorted,IND]=sort(Dis);
        for zz=1:PairNum
            Flag(zz)=IND(zz);
            weight1=200;
            MinDistance=weight1;
            pair=0;
            if Flag(zz)~=0
                for z=1:SP_all{i}.SuperPixelNumber
                    if sqrt((SP_all{i}.MiddlePoint(z,1)-MotionGradientMatrix_all{i}(1,Flag(zz)))^2+(SP_all{i}.MiddlePoint(z,2)-MotionGradientMatrix_all{i}(2,Flag(zz)))^2)<MinDistance&&~ismember(z,index)&&FFlag(z)~=1
                        if sum((SP_all{i}.MiddlePoint(z,1:2)-MotionGradientMatrix_all{i}(1:2,Flag(zz))').*([x y]-MotionGradientMatrix_all{i}(1:2,Flag(zz))'))<0
                            pair=z;
                            MinDistance=sqrt((SP_all{i}.MiddlePoint(z,1)-MotionGradientMatrix_all{i}(1,Flag(zz)))^2+(SP_all{i}.MiddlePoint(z,2)-MotionGradientMatrix_all{i}(2,Flag(zz)))^2);
                        end
                    end
                end
            end
            if pair~=0
                FFlag(pair)=1;
                result{j}{CCount}=[index(j),pair];
                CCount=CCount+1;
            end
            if CCount==4
                break;
            end
        end
        if CCount<4
            result{j}{1}=[0 0];
            result{j}{2}=[0 0];
            result{j}{3}=[0 0];
        end
        if result{j}{1}(2)==0||result{j}{2}(2)==0||result{j}{3}(2)==0
            result{j}{1}=[0 0];
            result{j}{2}=[0 0];
            result{j}{3}=[0 0];
        end
    end
end
