clear all;
clc;

load my_air_bj.mat;
%%%%%%%%%%%%%%%%%%%%%%
adj=5;
adjMatrix=zeros(36,adj); %% include it self
for i=1:36
    [a,b]=sort(pdist2(stationLocation,stationLocation(i,:)));
    adjMatrix(i,:)=b(1:adj);
end
%%%%%%%%%%%%%%%%%%%%
numTimeDelay=23;
inputPolluents=[1 2 3 4 5 6];     %输入过去污染物数据
outputPolluents=1;                  %输出PM2.5

weatherFactors=[7 8 9 10 11 12];      %输入未来天气预测

count=1;

%%%%%%%%%%%%%%%%%预测多少天以后%%%%%%%%%%%%%%%%%%%%
for iTimeDelay=1:length(numTimeDelay)
    
for iStation=1:36
outputStation=iStation

timeDelay=numTimeDelay(iTimeDelay);    %预测几小时之后的值，0为一天
range=timeDelay+1;

%inputSize=range*length(inputPolluents)+3*length(weatherFactors)*adj+range*length(outputPolluents)*adj;    
inputSize=3*length(inputPolluents)+3*length(weatherFactors)*adj+3*length(outputPolluents)*adj;    
outputSize=1;

trainingLength=7000;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%构造输入输出矩阵%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStation}(i+range-3:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStation,k)}(i+range-3:i+range-1,outputPolluents)'];
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
testingLength=1000;
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStation}(i+range-3:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStation,k)}(i+range-3:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(outputStation,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(outputStation,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    test_x(i-trainingLength,:)=t;  
    test_y(i-trainingLength,1)=air_bj{outputStation}(i+range+timeDelay,outputPolluents);   
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%神经网络%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
size=[inputSize 30 10 80 outputSize];
opt.active_funcs={'sigm','tanh_opt'};
opt.learningRate=1;
opt.weightPenaltyL1=0;
opt.weightPenaltyL2=0;
opt.momentum=0.5;
opt.scaling_learningRate=0.99;
opt.numepochs=100;
opt.batchsize=40;
%opt.batches=batches;
lstm=lstmsetup3(size,opt);

%lstm.nn.weightPenaltyL1=0.001;

lstm=lstmtrain3(lstm,train_x,train_y,opt);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%test%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lstm=lstmff3(lstm,test_x,test_y);
predict_y=lstm.nn.a{end};

e1=predict_y-test_y;
mae1=mean(abs(e1))*500
rmse1=sqrt((mean(e1.^2)))*500;
acc1=1-sum(abs(e1))/sum(abs(test_y));

result(count).station=outputStation;
result(count).timeDelay=timeDelay;
result(count).mae1=mae1;
result(count).rmse1=rmse1;
result(count).acc1=acc1;
count=count+1;

end

mae=0;
acc=0;
for i=1:36
    mae=mae+result(i).mae1;
    acc=acc+result(i).acc1;
end
mae/36
acc/36

end
save result.mat result;