function idx=KMeansClassify(centers,x)
[m,n]=size(centers);
d=[];
for i=1:m
    %d=[d pdist2(centers(i,:),x,'correlation')];
    d=[d norm(centers(i,:)-x)];
end
[mini,idx]=min(d);
end