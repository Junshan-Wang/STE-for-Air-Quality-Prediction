%%对比版本：考虑周围站点，使用NN

clear all;
clc;

load air_bj.mat;

adj=5;
adjMatrix=zeros(35,adj); %% include it self
for i=1:35
    [a,b]=sort(pdist2(stationLocation,stationLocation(i,:)));
    adjMatrix(i,:)=b(1:adj);
end

numTimeDelay=[59];
numLearningRate=[1];
numScaling_learningRate=[0.99];
numEpoches=[100];
numBatchSize=[40];
ifPenalty=[0];
numMomentum=[0.5];

inputPolluents=[1 2 3 4 5];     %输入过去污染物数据
outputPolluents=6;                  %输出PM2.5
weatherFactors=[7 8 9 10];      %输入未来天气预测


count=1;

%%%%%%%%%%%%%%%%%预测多少天以后%%%%%%%%%%%%%%%%%%%%
for iTimeDelay=1:length(numTimeDelay)

for i=1:35
outputStation=i   

timeDelay=numTimeDelay(iTimeDelay);    %预测几小时之后的值，0为一天
range=timeDelay+1;

inputSize=range*length(inputPolluents)+adj*range*length(outputPolluents)+adj*3*length(weatherFactors);    
outputSize=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%构造输入输出矩阵%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
trainingLength=2400;
train_x=rand(trainingLength,inputSize);
train_y=rand(trainingLength,outputSize);

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
hiddenLayers=2;     %隐藏层数

hiddenNodes1=50;    %第一层结点数量
hiddenNodes2=20;    %第二层结点数量
hiddenStruct=[hiddenNodes1 hiddenNodes2];

nn=nnsetup([inputSize hiddenStruct outputSize]);

opts.sae=0;
opts.numepochs = 100;
opts.batchsize = 40;
opts.silent = 1;
nn.learningRate = 1;
nn.scaling_learningRate = 0.99;
nn.weightPenaltyL2=0;

nn = nntrain(nn, train_x, train_y, opts); 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testingLength=240;
testBegin=2400;

test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);

for i=testBegin+1:testBegin+testingLength
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
    test_x(i-testBegin,:)=t;
    
    test_y(i-testBegin,1)=air_bj{outputStation}(i+range+timeDelay,outputPolluents);   
end

nn=nnff(nn,test_x,test_y);
predict_y=nn.a{end};
e=predict_y-test_y;


result(count).station=outputStation;
result(count).timeDelay=numTimeDelay(iTimeDelay);
result(count).MAE=mean(abs(e))*500;
result(count).RMSE=sqrt(mean(e.^2))*500;
result(count).accuracy=1-sum(abs(e))/sum(abs(test_y));
count=count+1;

end
end

mae=0;
acc=0;
for i=1:35
    mae=mae+result(i).MAE;
    acc=acc+result(i).accuracy;
end
mae/35
acc/35

save myContrast1.mat result;
%%%%%%%%%%%%%%%%%%%%%%End iTimeDelay%%%%%%%%%%%%%%%%%%%%%%%


