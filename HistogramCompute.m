function [A_hat,E_Copy,A,LocationRgb]=HistogramCompute(MotionMaskAll,SP_all,LengthFiles,SaliencyM,MaxD)
weight=1;
WWW=10;
W=2;
W_D=2;
for i=1:LengthFiles-3
    RgbHisto{i}=zeros(WWW,WWW,WWW);
    RgbHisto_Num{i}=zeros(WWW,WWW,WWW);
    [index value] = find(MotionMaskAll(:,i)==1);
    MaskLength = size(index,1);
    LocationRgb{i}=zeros(SP_all{i}.SuperPixelNumber,3);
    for j=1:SP_all{i}.SuperPixelNumber
        r=floor(SP_all{i}.MiddlePoint(j,3)*WWW)+1;
        g=floor(SP_all{i}.MiddlePoint(j,4)*WWW)+1;
        b=floor(SP_all{i}.MiddlePoint(j,5)*WWW)+1;
        LocationRgb{i}(j,:)=[r,g,b];
    end
    for j=1:MaskLength
        r=floor(SP_all{i}.MiddlePoint(index(j),3)*WWW)+1;
        g=floor(SP_all{i}.MiddlePoint(index(j),4)*WWW)+1;
        b=floor(SP_all{i}.MiddlePoint(index(j),5)*WWW)+1;
        if r>WWW
            r=WWW;
        end
        if g>WWW
            g=WWW;
        end
        if b>WWW
            b=WWW;
        end
        RgbHisto{i}(r,g,b)=RgbHisto{i}(r,g,b)+SaliencyM(index(j),i);
        RgbHisto_Num{i}(r,g,b)=RgbHisto_Num{i}(r,g,b)+1;
    end
    for j=1:WWW*WWW*WWW
        if RgbHisto_Num{i}(j)~=0
            RgbHisto{i}(j)=RgbHisto{i}(j)/RgbHisto_Num{i}(j);
        end
    end
    RgbHis_Mask{i}=zeros(WWW,WWW,WWW);
    for ia=1:MaskLength
        r=floor(SP_all{i}.MiddlePoint(index(ia),3)*WWW)+1;
        g=floor(SP_all{i}.MiddlePoint(index(ia),4)*WWW)+1;
        b=floor(SP_all{i}.MiddlePoint(index(ia),5)*WWW)+1;
        if r>WWW
            r=WWW;
        end
        if g>WWW
            g=WWW;
        end
        if b>WWW
            b=WWW;
        end
        for ib=-1*W:W
            for ic=-1*W:W
                for id=-1*W:W
                    if(r+ib>0&&r+ib<=WWW&&g+ic>0&&g+ic<=WWW&&b+id>0&&b+id<=WWW)
                        if(RgbHis_Mask{i}(r+ib,g+ic,b+id)==0)
                            RgbHis_Mask{i}(r+ib,g+ic,b+id)=RgbHisto{i}(r,g,b)*(exp(-1*W_D*sqrt((ib)^2+(ic)^2+(id)^2)));
                        else
                            RgbHis_Mask{i}(r+ib,g+ic,b+id)=max(RgbHisto{i}(r,g,b)*(exp(-1*W_D*sqrt((ib)^2+(ic)^2+(id)^2))),RgbHis_Mask{i}(r+ib,g+ic,b+id));
                        end
                    end
                end
            end
        end
    end
    RgbHis_Mask{i}=RgbHisto{i}(:);
end
W_mu=-99/80*(LengthFiles-3)+499/4;
RgbHisMatrix=cell2mat(RgbHis_Mask);
RgbHisMatrix=MatrixNormalization(RgbHisMatrix);
[A_hat E_hat] = LK(RgbHisMatrix);
E_Copy=E_hat;
% E_hat=abs(E_hat);
% E_Color=sum(E_hat,2);
% E_Color=(E_Color-min(E_Color))/(max(E_Color)-min(E_Color));
% E_Color=reshape(E_Color,WWW,WWW,WWW);
% E_SSUM=zeros(LengthFiles-3,1);
% A_Copy=zeros(size(A_hat));
A_Copy=sum(A_hat,2)/(LengthFiles-3);
for i=1:LengthFiles-3
%     if(i-50<=0)
%         Front=1;
%     else
%         Front=i-50;
%     end
%     if(i+50>LengthFiles-3)
%         End=LengthFiles-3;
%     else
%         End=i+50;
%     end
%     A_Copy(:,i)=sum(A_hat(:,Front:End),2)/(End-Front+1);

    A{i}=reshape(A_Copy(:),WWW,WWW,WWW);
    E{i}=reshape(E_hat(:,i),WWW,WWW,WWW);
%     E{i}=(E{i}-min(min(min(E{i}))))/(max(max(max(E{i})))-min(min(min(E{i}))));
    A{i}=MatrixNormalization(A{i});
%     A{i}=exp(A{i});
%     A{i}=MatrixNormalization(A{i});
     A{i}=0.3*A{i}+0.7;
    wwwww=SP_all{i}.SuperPixelNumber;
    SS=zeros(MaxD,1);
    for kk=1:wwwww
        SS(kk)=A{i}(LocationRgb{i}(kk,1),LocationRgb{i}(kk,2),LocationRgb{i}(kk,3));
    end
    A{i}=SS;
%     for kk=1:wwwww
%         E_SSUM(i)=E_SSUM(i)+abs(E{i}(LocationRgb{i}(kk,1),LocationRgb{i}(kk,2),LocationRgb{i}(kk,3)));
%     end
    
end