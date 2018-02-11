clear all;
clc;

load air_bj.mat;
%%%%%%%%%%%%%%%%%%%%%%
numTimeDelay=[5];
inputPolluents=[1 2 3 4 5];     %输入过去污染物数据
outputPolluents=6;                  %输出PM2.5
weatherFactors=[7 8 9 10];      %输入未来天气预测

count=1;
for iTimeDelay=1:length(numTimeDelay)
for iStation=1:35
outputStations=iStation
timeDelay=numTimeDelay(iTimeDelay);  
range=timeDelay+1;
%%%%%%%%%%%%%%%%% 获得相关站点 %%%%%%%%%%%%%
Kernel=zeros(35,1);
for i=1:35
	%Kernel(i)=abs(pdist2(air_bj{outputStations}(:,6)',air_bj{i}(:,outputPolluents)','correlation'));
    [F,c]=granger_cause(air_bj{outputStations}(:,outputPolluents),air_bj{i}(:,outputPolluents),0.01,range);
    Kernel(i)=F;
end
adj=5;
adjMatrix=zeros(35,adj);
[a,b]=sort(Kernel);
adjMatrix(outputStations,:)=b(1:adj);

%%%%%%%%%%%%%%%%%% 相关区域 %%%%%%%%%%%%%%%%%%
numDir=4;
fStatistic=zeros(35,length(weatherFactors)+1);
for k=1:35
    for i=1:length(weatherFactors)
        [F,c]=granger_cause(air_bj{outputStations}(1:2400,outputPolluents),air_bj{k}(1:2400,weatherFactors(i)),0.01,1);
        fStatistic(k,i)=F;
    end
    [F,c]=granger_cause(air_bj{outputStations}(1:2400,outputPolluents),air_bj{k}(1:2400,outputPolluents),0.01,1);
    fStatistic(k,length(weatherFactors)+1)=F;
end

stationsU=fcmeans(fStatistic,numDir);
for i=1:numDir
    dir{i}=[];
    for j=1:35
        if j~=outputStations && stationsU(i,j)>0.3 %index(j)==i %u(i,j)>0.25
            dir{i}=[dir{i} j];
        end
    end
end

%%%%%%%%%%%%%%%%% 预测多少天以后 %%%%%%%%%%%%%%%%%%%%

inputSize=range*length(inputPolluents)+range*length(outputPolluents)*adj+3*adj*length(weatherFactors)+numDir*(range*length(outputPolluents)+length(weatherFactors));    
outputSize=1;

trainingLength=2400;

%%%%%%%%%%%%%%%%%%% 构造输入输出矩阵 %%%%%%%%%%%%%%%%%%%%%%
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStations,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(outputStations,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(outputStations,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(outputStations,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    for iDir=1:numDir
        for j=1:length(outputPolluents)
           w=[];
           for k=1:length(dir{iDir})
               w=[w;air_bj{dir{iDir}(k)}((i):(i+range-1),outputPolluents)'];
           end
           t=[t mean(w,1)];
        end    
        
        for j=1:length(weatherFactors)
            w=[];
            for k=1:length(dir{iDir})
                w=[w mean(air_bj{dir{iDir}(k)}((i+range):(i+range+timeDelay),weatherFactors(j)))];
            end
            t=[t mean(w)];
        end
    end
      
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end

%%%%%%%%%%%%%%%%%%%%%% 所有训练集训练神经网络 %%%%%%%%%%%%%%%%%%%%%%%%%%
Size=[inputSize 30 10 10 outputSize];

opt.active_funcs={'sigm','tanh_opt'};
opt.learningRate=1;
opt.weightPenaltyL1=0;
opt.weightPenaltyL2=0;
opt.momentum=0.5;
opt.scaling_learningRate=0.99;
opt.numepochs=100;
opt.batchsize=40;
%opt.batches=batches;
lstm=lstmsetup3(Size,opt);

%lstm.lstmcell{1}.weightPenaltyL2=0.001;
%lstm.lstmcell{2}.weightPenaltyL2=0.001;
%lstm.lstmcell{1}.weightPenaltyGroup=0.001;
%lstm.lstmcell{1}.group=3;

lstm=lstmtrain3(lstm,train_x,train_y,opt);

lstm=lstmff3(lstm,train_x,train_y);

%%%%%%%%%%%%%%%%%%%%%% 构造测试集 %%%%%%%%%%%%%%%%%%%%%%%%
testingLength=240;
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(outputStations,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(outputStations,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(outputStations,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(outputStations,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    for iDir=1:numDir
        for j=1:length(outputPolluents)
           w=[];
           for k=1:length(dir{iDir})
               w=[w;air_bj{dir{iDir}(k)}((i):(i+range-1),outputPolluents)'];
           end
           t=[t mean(w,1)];
        end    
        
        for j=1:length(weatherFactors)
            w=[];
            for k=1:length(dir{iDir})
                w=[w mean(air_bj{dir{iDir}(k)}((i+range):(i+range+timeDelay),weatherFactors(j)))];
            end
            t=[t mean(w)];
        end
    end
    test_x(i-trainingLength,:)=t;  
    test_y(i-trainingLength,1)=air_bj{outputStations}(i+range+timeDelay,outputPolluents);   
end

%%%%%%%%%%%%%%%%%% 计算测试集结果 %%%%%%%%%%%%%%%%%%%

lstm=lstmff3(lstm,test_x,test_y);
py1=lstm.nn.a{end};

%%%%%%%%%%%%%%% 误差计算 %%%%%%%%%%%%%%%%
e1=py1-test_y;
mae1=mean(abs(e1))*500;
acc1=1-sum(abs(e1))/sum(abs(test_y));
mae1
acc1

%%%%%%%%%%%%%%%% 保存结果 %%%%%%%%%%%%%%%%%%%
result(count).station=outputStations;
result(count).timeDelay=timeDelay;
result(count).mae1=mae1;
result(count).acc1=acc1;
count=count+1;
end
end
save result.mat result;
%}
