function v=fcmeansClassify(centers,x)
expo=2;
[m,n]=size(centers);
d=[];
for i=1:m
    d=[d;norm(centers(i,:)-x)];
end
tmp=d.^(-2/(expo-1));
v=tmp./repmat(sum(tmp),m,1);
end