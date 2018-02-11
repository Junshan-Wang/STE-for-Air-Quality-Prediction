clear all;
clc;

load air_bj.mat;

fStatistic=zeros(1,35);

center=air_bj{1}(:,6);

for i=1:35
    sta=air_bj{i}(:,6);
    [F,c]=granger_cause(center,sta,0.01,1);
    fStatistic(i)=F;
end
[a,b]=sort(fStatistic);
select=b(30:35);

plot(air_bj{1}(1:100,6));
for i=1:6
    plot(air_bj{select(i)}(1:100,6));
    hold on;
end


    