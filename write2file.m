function write2file(index,SP_all,path,NewSM,I,oW,oH)
[W,H] = size(I{1});
MotionContrastSmoothed=NewSM;
FusionedResult = zeros(W,H/3);
for l=1:SP_all{index}.SuperPixelNumber
    ClusteringPixelNumber = SP_all{index}.ClusteringPixelNum(1,l);
    for z=1:ClusteringPixelNumber
        XIndex= SP_all{index}.Clustering(l,z,1);
        YIndex= SP_all{index}.Clustering(l,z,2);
        FusionedResult(XIndex,YIndex) = MotionContrastSmoothed(l);
    end
end
FusionedResult = MatrixNormalization(FusionedResult);
FusionedResult = imresize(FusionedResult,[oW oH]);
imwrite(FusionedResult*1.5,path);