%返回的F的维数为d*5，
%输入为超像素点中的全部RGB像素分布, 为一个sp*3的矩阵，其中sp为当前超像素点所包含的像素个数

function [F] = getHistogram(X,d)

%初始化参数
%X = abs(X - 0.0000001);
%d = 5;
row = size(X,1);

span = 1/d;
R = floor(X./span)+1;

for i=1:size(X,2)
    H{i} = zeros(1,d);%存储histogram
end


%统计R中各个分量的次数
for i=1:row
    for j=1:size(X,2)
        H{j}(1,R(i,j)) = H{j}(1,R(i,j)) + 1;
    end
end
for j=1:size(X,2)
    temp = (H{j});
    maxValue = max((temp));
    minValue = min((temp));
    if((maxValue-minValue)~=0)
        temp = (temp-minValue)./(maxValue-minValue);
    end
    H{j} = temp;
end


F=[];
for i=1:size(X,2)%　归一化处理
    F=[F H{i}];%返回结果
end

%归一化处理
%{
temp = cell2mat(H);
maxValue = max(max(temp));
minValue = min(min(temp));
F=[];
for i=1:size(X,2)%　归一化处理
    if(maxValue-minValue~=0)
        H{i} = (H{i}-minValue)/(maxValue-minValue);
    end
    F=[F H{i}];%返回结果
end
%}
