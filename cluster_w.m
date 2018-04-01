%随机获取150个点
X = [randn(50,2)+ones(50,2);randn(50,2)-ones(50,2);randn(50,2)+[ones(50,1),-ones(50,1)]];
 
opts = statset('Display','final');
 
%调用Kmeans函数
%X N*P的数据矩阵
%Idx N*1的向量,存储的是每个点的聚类标号
%Ctrs K*P的矩阵,存储的是K个聚类质心位置
%SumD 1*K的和向量,存储的是类间所有点与该类质心点距离之和
%D N*K的矩阵，存储的是每个点与所有质心的距离;
 
[Idx,Ctrs,SumD,D] = kmeans(X,4,'Replicates',4);
 
Point1=X(Idx==1,:);
Point2=X(Idx==2,:);
Point3=X(Idx==3,:);
Point4=X(Idx==4,:);
Dis1=sqrt(sum((bsxfun(@minus,Point1,Ctrs(1,:))).^2,2));
Dis2=sqrt(sum((bsxfun(@minus,Point2,Ctrs(2,:))).^2,2));
Dis3=sqrt(sum((bsxfun(@minus,Point3,Ctrs(3,:))).^2,2));
Dis4=sqrt(sum((bsxfun(@minus,Point4,Ctrs(4,:))).^2,2));
%画出聚类为1的点。X(Idx==1,1),为第一类的样本的第一个坐标；X(Idx==1,2)为第二类的样本的第二个坐标
plot(X(Idx==1,1),X(Idx==1,2),'r.','MarkerSize',14)
hold on
plot(X(Idx==2,1),X(Idx==2,2),'b.','MarkerSize',14)
hold on
plot(X(Idx==3,1),X(Idx==3,2),'g.','MarkerSize',14)
 hold on
plot(X(Idx==4,1),X(Idx==4,2),'g.','MarkerSize',14)
%绘出聚类中心点,kx表示是圆形
plot(Ctrs(1,1),Ctrs(1,2),'kx','MarkerSize',14,'LineWidth',4)
plot(Ctrs(2,1),Ctrs(2,2),'kx','MarkerSize',14,'LineWidth',4)
plot(Ctrs(3,1),Ctrs(3,2),'kx','MarkerSize',14,'LineWidth',4)
 plot(Ctrs(4,1),Ctrs(4,2),'kx','MarkerSize',14,'LineWidth',4)
legend('Cluster 1','Cluster 2','Cluster 3','Centroids','Location','NW')
%  
% Ctrs
% SumD