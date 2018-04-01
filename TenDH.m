function [A_hat E_hat B]=TenDH(LengthFiles,I_CN,I,MotionMaskAll_Copy,SaliencyM,SP_all,Mode,ImageNameContainer,oW,oH,MaxDim)
% I_copy=I_CN;
% I_COPY=I_CN;
% [W,H] = size(I{1});
parfor i=1:LengthFiles-3
    MotionContrastSmoothed=SaliencyM(:,i).*MotionMaskAll_Copy(:,i);%.
    IC_copy=zeros(SP_all{i}.SuperPixelNumber,10);
    I_COPY=zeros(SP_all{i}.SuperPixelNumber,10);
    for l=1:SP_all{i}.SuperPixelNumber
       I_COPY(l,:)=SP_all{i}.Icn(l,:)*MotionContrastSmoothed(l);
       if MotionContrastSmoothed(l)>0
            IC_copy(l,:)=SP_all{i}.Icn(l,:);
       end
    end
   
    I_SUM=sum((I_COPY),1);
    Ic_SUM=sum((IC_copy),1);
    for j=1:10
        if Ic_SUM(j)~=0
            I_Temp{i}(j)=I_SUM(j)/Ic_SUM(j);%
        else
            I_Temp{i}(j)=I_SUM(j);
        end
    end

end
% I_SUM=sum(sum(I_copy,1),2);

I_Temp=cell2mat(I_Temp');
[A_hat,E_hat] = LK(double(I_Temp'));
E_hat=abs(E_hat);
B=1;
% I_copy=SP_all{i}.Icn;
for qqq=1:LengthFiles-3
    I_copy=SP_all{qqq}.Icn;
     SSSS=zeros(SP_all{qqq}.SuperPixelNumber,1);
     for i=1:SP_all{qqq}.SuperPixelNumber
              tt=[I_copy(i,1) I_copy(i,2) I_copy(i,3) I_copy(i,4) I_copy(i,5) I_copy(i,6) I_copy(i,7) I_copy(i,8) I_copy(i,9) I_copy(i,10)];
              tt=MatrixNormalization(tt);
              [value,index]=max(tt);
%             [vaule,INDEX]=max(tt);
%             SSSS(i,j)=sum(A_hat(:,qqq).*tt');
              SSSS(i)=(E_hat(index,qqq)*value);
     end
     SSSS=MatrixNormalization(SSSS);
    
%    path=['.\result\' Mode ,'\Low_Map\' ,ImageNameContainer{qqq+2}];
%    write2file(qqq,SP_all,path,SSSS,I,oW,oH);
     SSSS=0.2*SSSS+0.8;
     A{qqq}=SSSS;
end;
% for qqq=1:LengthFiles-3
%     Temp=zeros(MaxDim,1);
%     for i=1:SP_all{qqq}.SuperPixelNumber
%         Temp(i)=A{qqq}(int32(SP_all{qqq}.MiddlePoint(i,1)),int32(SP_all{qqq}.MiddlePoint(i,2)));
%     end
%     B{qqq}=Temp;
% end

% tic
    % I_copy=I_CN;
    % parfor i=1:LengthFiles-3
    %     [W,H] = size(I{1});
    %     MotionContrastSmoothed=SaliencyM(:,i).*MotionMaskAll_Copy(:,i);%.
    %     FusionedResult = zeros(W,H/3,10);
    %     FusionedResult1= zeros(W,H/3);
    %     for l=1:SP_all{i}.SuperPixelNumber
    %         ClusteringPixelNumber = SP_all{i}.ClusteringPixelNum(1,l);
    %         for z=1:ClusteringPixelNumber
    %             XIndex= SP_all{i}.Clustering(l,z,1);
    %             YIndex= SP_all{i}.Clustering(l,z,2);
    %             ttt=MotionContrastSmoothed(l);
    %             FusionedResult(XIndex,YIndex,:) =[ttt ttt ttt ttt ttt ttt ttt ttt ttt ttt];
    %             FusionedResult1(XIndex,YIndex)=ttt;
    %         end
    %     end
    % %     for ii=1:300
    % %         for jj=1:300
    % %             I_CN{i}(ii,jj,:)= (I_CN{i}(ii,jj,:)*FusionedResult(ii,jj));
    % %         end
    % %     end
    %     FusionedResult=reshape(FusionedResult,300*300,10);
    %     I_CN{i}=reshape(I_CN{i},300*300,10);
    %     I_CN{i}=I_CN{i}.*FusionedResult;
    %     nums{i}=sum(sum(FusionedResult1));
    %     I_CN{i}=MatrixNormalization(abs(I_CN{i}));
    % %     I_CN{i} = I_CN{i}.*FusionedResult;
    % end
    % toc
    % tic
    % for ind=1:LengthFiles-3
    %     I_Temp{ind}=sum((I_CN{ind}),1);
    %     I_Temp{ind}=reshape( I_Temp{ind},10,1)/nums{ind};
    %     I_Temp{ind}=(I_Temp{ind}-min(I_Temp{ind}))/(max(I_Temp{ind})-min(I_Temp{ind}));
    % end
    % toc
    % I_Temp=cell2mat(I_Temp);
    % [A_hat E_hat] = LK(double(I_Temp));
    
    % qqq=1;
    % SSSS=zeros(300,300);
    %  A_hat=abs(A_hat);
    % for i=1:300
    %     for j=1:300
    %         tt=[I_copy{qqq}(i,j,1) I_copy{qqq}(i,j,2) I_copy{qqq}(i,j,3) I_copy{qqq}(i,j,4) I_copy{qqq}(i,j,5) I_copy{qqq}(i,j,6) I_copy{qqq}(i,j,7) I_copy{qqq}(i,j,8) I_copy{qqq}(i,j,9) I_copy{qqq}(i,j,10)];
    %         [vaule,INDEX]=max(tt);
    %         SSSS(i,j)=sum(A_hat(:,qqq).*tt');
    %     end
    % end
    % % SSSS=MatrixNormalization(SSSS);
    % figure,imshow(SSSS)
    %
    %
    %  E_hat=abs(E_hat);
    % SSSS=zeros(300,300);
    % for i=1:300
    %     for j=1:300
    %         tt=[I_copy{qqq}(i,j,1) I_copy{qqq}(i,j,2) I_copy{qqq}(i,j,3) I_copy{qqq}(i,j,4) I_copy{qqq}(i,j,5) I_copy{qqq}(i,j,6) I_copy{qqq}(i,j,7) I_copy{qqq}(i,j,8) I_copy{qqq}(i,j,9) I_copy{qqq}(i,j,10)];
    %          [vaule,INDEX]=max(tt);
    %         SSSS(i,j)=sum(E_hat(:,qqq).*tt');
    %     end
    % end
    % % SSSS=MatrixNormalization(SSSS);
    % figure,imshow(SSSS)
    %
    %  I_Temp_t=abs(I_Temp');
    % SSSS=zeros(300,300);
    % for i=1:300
    %     for j=1:300
    %         tt=[I_copy{qqq}(i,j,1) I_copy{qqq}(i,j,2) I_copy{qqq}(i,j,3) I_copy{qqq}(i,j,4) I_copy{qqq}(i,j,5) I_copy{qqq}(i,j,6) I_copy{qqq}(i,j,7) I_copy{qqq}(i,j,8) I_copy{qqq}(i,j,9) I_copy{qqq}(i,j,10)];
    %         SSSS(i,j)=sum(I_Temp_t(:,qqq).*tt');
    %     end
    % end
    % % SSSS=MatrixNormalization(SSSS);
    % figure,imshow(SSSS)