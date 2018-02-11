%%对比版本：regression tree
clear all;
clc;

load air_bj.mat;
%%%%%%%%%%%%%%%%%%%%%%
adj=5;
adjMatrix=zeros(35,adj); %% include it self
for i=1:35
    [a,b]=sort(pdist2(stationLocation,stationLocation(i,:)));
    adjMatrix(i,:)=b(1:adj);
end
%%%%%%%%%%%%%%%%%%%%
numTimeDelay=[5];
inputPolluents=[1 2 3 4 5];     %输入过去污染物数据
outputPolluents=6;                  %输出PM2.5

weatherFactors=[7 8 9 10];      %输入未来天气预测

count=1;

%%%%%%%%%%%%%%%%%预测多少天以后%%%%%%%%%%%%%%%%%%%%
for iTimeDelay=1:length(numTimeDelay)
    
for iStation=1:35
outputStation=iStation 

timeDelay=numTimeDelay(iTimeDelay);    %预测几小时之后的值，0为一天
range=timeDelay+1;

inputSize=range*length(inputPolluents)+3*length(weatherFactors)*adj+range*length(outputPolluents)*adj;    
outputSize=1;

trainingLength=2400;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%构造输入输出矩阵%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStation}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStation,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(outputStation,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStation}(i+range+timeDelay,outputPolluents);    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testingLength=240;
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStation}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStation,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(outputStation,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    test_x(i-trainingLength,:)=t;  
    test_y(i-trainingLength,1)=air_bj{outputStation}(i+range+timeDelay,outputPolluents);   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
regtree=fitrtree(train_x,train_y,'MinLeaf',200);
predict_y=predict(regtree,test_x);

e1=predict_y-test_y;
mae1=mean(abs(e1))*500
acc1=1-sum(abs(e1))/sum(abs(test_y));
rmse1=sqrt(mean((e1*500).^2));

result(count).station=outputStation;
result(count).timeDelay=timeDelay;
result(count).mae1=mae1;
result(count).acc1=acc1;
result(count).rmse1=rmse1;
count=count+1;

end
end
save result.mat result;