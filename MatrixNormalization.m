function result = MatrixNormalization(M)

maxValue = max(max(max(M)));
minValue = min(min(min(M)));
if(maxValue-minValue~=0)
    M = (M-minValue)/(maxValue-minValue);
end
result = M;