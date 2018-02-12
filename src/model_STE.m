% 四个LSTM，根据时间序列划分天气模式 by fcmean，空间选择也不同，回归层线性回归／SVM，空间关联为gg

clear all;
clc;

load air_bj.mat;
%%%%%%%%%%%%%%%%%%%%%%
numTimeDelay=[5];
inputPolluents=[1 2 3 4 5];     %输入过去污染物数据
outputPolluents=6;                  %输出PM2.5
weatherFactors=[7 8 9 10];      %输入未来天气预测

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 预测站点 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count=1;
for iTimeDelay=1:length(numTimeDelay)
for iStation=1
outputStations=iStation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 预测多少天以后 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timeDelay=numTimeDelay(iTimeDelay);  
range=timeDelay+1;
trainingLength=2400;
%%%%%%%%%%%%%%%%%%%%%%%%% 参数 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=4;                            % 根据天气因素分类的类别数
lag=40;                         % 样本序列长度
M=floor(trainingLength/lag);    % 样本序列数
adj=5;                          % 相关站点数量
adjMatrix=zeros(K,adj);         % 四个训练子集的相关站点
numDir=4;                       % 空间相关区域数量

inputSize=range*length(inputPolluents)+range*length(outputPolluents)*adj+3*adj*length(weatherFactors)+numDir*(range*length(outputPolluents)+length(weatherFactors));    
outputSize=1;

%%%%%%%%%%%%%%%%% 根据天气因素将样本序列分类 %%%%%%%%%%%%%%%%

wx=air_bj{outputStations}(1+timeDelay+range:trainingLength+timeDelay+range,weatherFactors);
wy=air_bj{outputStations}(1+timeDelay+range:trainingLength+timeDelay+range,outputPolluents);
fStatistic=zeros(M,length(weatherFactors));
for iLag=1:M
    for i=1:length(weatherFactors)
        [F,c]=granger_cause(wy(lag*(iLag-1)+1:lag*iLag,1),wx(lag*(iLag-1)+1:lag*iLag,i),0.01,1);
        fStatistic(iLag,i)=F;
    end
end
[u,centers]=fcmeans2(fStatistic,K);     % u:每个样本序列输入每个类的概率矩阵
second_x=[];    % 回归层的输入

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子模型 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for kSub=1:K
    
index=[];
for i=1:M
    if u(kSub,i)>0.1
        index=[index lag*(i-1)+1:lag*i];
    end
end
subTrainingLength=length(index);

%%%%%%%%%%%%%%%%% 空间相关站点和相关区域获取 %%%%%%%%%%%%%%%%
Kernel=zeros(35,1);
for i=1:35
	%Kernel(i)=abs(pdist2(air_bj{outputStations}(:,6)',air_bj{i}(:,outputPolluents)','correlation'));
    [F,c]=granger_cause(air_bj{outputStations}(:,outputPolluents),air_bj{i}(:,outputPolluents),0.01,range);
    Kernel(i)=F;
end
[a,b]=sort(Kernel);
adjMatrix(kSub,:)=b(1:adj);

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
    dir{kSub}{i}=[];
    for j=1:35
        if j~=outputStations && stationsU(i,j)>0.3 %index(j)==i %u(i,j)>0.25
            dir{kSub}{i}=[dir{kSub}{i} j];
        end
    end
end
%%%%%%%%%%%%%%%%%%% 构造训练输入输出矩阵 %%%%%%%%%%%%%%%%%%%
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(kSub,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(kSub,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(kSub,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(kSub,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    for iDir=1:numDir
        for j=1:length(outputPolluents)
           w=[];
           for k=1:length(dir{kSub}{iDir})
               w=[w;air_bj{dir{kSub}{iDir}(k)}((i):(i+range-1),outputPolluents)'];
           end
           t=[t mean(w,1)];
        end    
        
        for j=1:length(weatherFactors)
            w=[];
            for k=1:length(dir{kSub}{iDir})
                w=[w mean(air_bj{dir{kSub}{iDir}(k)}((i+range):(i+range+timeDelay),weatherFactors(j)))];
            end
            t=[t mean(w)];
        end
    end    
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end

subtrain_x=train_x(index,:);
subtrain_y=train_y(index,:);
%%%%%%%%%%%%%%%%%%%%%% 训练Deep LSTM %%%%%%%%%%%%%%%%%%%%%%
Size=[inputSize 30 10 50 outputSize];
opt.active_funcs={'sigm','tanh_opt'};
opt.learningRate=1;
opt.weightPenaltyL1=0;
opt.weightPenaltyL2=0;
opt.momentum=0.5;
opt.scaling_learningRate=0.99;
opt.numepochs=100;
opt.batchsize=40;
lstm{kSub}=lstmsetup3(Size,opt);
%lstm.lstmcell{1}.weightPenaltyL2=0.001;
%lstm.lstmcell{2}.weightPenaltyL2=0.001;
lstm{kSub}.lstmcell{1}.weightPenaltyGroup=0;
lstm{kSub}.lstmcell{1}.group=3;
lstm{kSub}=lstmtrain3(lstm{kSub},subtrain_x,subtrain_y,opt);

%%%%%%%%%%%%%%%% 获取子模型对于全体输入数据的输出%%%%%%%%%%%%
lstm{kSub}=lstmff3(lstm{kSub},train_x,train_y);
weight=reshape(repmat(u(kSub,:),lag,1),trainingLength,1);
second_x=[second_x lstm{kSub}.nn.a{end}];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 子模型训练结束，训练回归层 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%second_x=[second_x wx];
theta=regress(train_y,second_x);
svr=svmtrain(train_y,second_x,'-s 3 -p 0.01 -q');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 测试 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testingBegin=1;
testingLength=1200;
wx=air_bj{outputStations}(testingBegin+1+timeDelay+range:testingBegin+testingLength+timeDelay+range,weatherFactors);
wy=air_bj{outputStations}(testingBegin+1+timeDelay+range:testingBegin+testingLength+timeDelay+range,outputPolluents);
m=floor(testingLength/lag);
v=zeros(K,m);
for iLag=1:m
    f=[];
    for i=1:length(weatherFactors)
        [F,c]=granger_cause(wy(lag*(iLag-1)+1:lag*iLag,1),wx(lag*(iLag-1)+1:lag*iLag,i),0.01,1);
        f=[f F];
    end
    v(:,iLag)=fcmeansClassify(centers,f);
end
second_x=[];

%%%%%%%%%%%%%%%%%%%%%%%% 子模型 %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for kSub=1:K

%%%%%%%%%%%%%%%%%%%% 构造测试输入输出矩阵 %%%%%%%%%%%%%%%%%%%
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=testingBegin+1:testingBegin+testingLength
    t=[];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for k=1:adj
        t=[t air_bj{adjMatrix(kSub,k)}(i:i+range-1,outputPolluents)'];
    end
    for j=1:length(weatherFactors)
        for k=1:adj
            t=[t mean(air_bj{adjMatrix(kSub,k)}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{adjMatrix(kSub,k)}(i+range:i+range+timeDelay,weatherFactors(j))) air_bj{adjMatrix(kSub,k)}(i+range+timeDelay,weatherFactors(j))];
        end
    end
    for iDir=1:numDir
        for j=1:length(outputPolluents)
           w=[];
           for k=1:length(dir{kSub}{iDir})
               w=[w;air_bj{dir{kSub}{iDir}(k)}((i):(i+range-1),outputPolluents)'];
           end
           t=[t mean(w,1)];
        end    
        
        for j=1:length(weatherFactors)
            w=[];
            for k=1:length(dir{kSub}{iDir})
                w=[w mean(air_bj{dir{kSub}{iDir}(k)}((i+range):(i+range+timeDelay),weatherFactors(j)))];
            end
            t=[t mean(w)];
        end
    end
    test_x(i-testingBegin,:)=t;  
    test_y(i-testingBegin,1)=air_bj{outputStations}(i+range+timeDelay,outputPolluents);   
end

%%%%%%%%%%%%%%%%%%%%%% 计算测试集结果 %%%%%%%%%%%%%%%%%%%%%%%
lstm{kSub}=lstmff3(lstm{kSub},test_x,test_y);
weight=reshape(repmat(v(kSub,:),lag,1),testingLength,1);
second_x=[second_x lstm{kSub}.nn.a{end}];

end
%%%%%%%%%%%%%%%%%%%%%%% 测试集回归 %%%%%%%%%%%%%%%%%%%%%%%%%%
%second_x=[second_x wx];
py1=second_x*theta;
[py2,mse,dec]=svmpredict(test_y,second_x,svr);
%%%%%%%%%%%%%%%%%%%%%%%% 误差计算 %%%%%%%%%%%%%%%%%%%%%%%%%%%
e1=py1-test_y;
mae1=mean(abs(e1))*500;
acc1=1-sum(abs(e1))/sum(abs(test_y));
mae1
acc1

e2=py2-test_y;
mae2=mean(abs(e2))*500;
acc2=1-sum(abs(e2))/sum(abs(test_y));
mae2
acc2
%{
e2=py2-test_y;
mae2=mean(abs(e2))*500;
acc2=1-sum(abs(e2))/sum(abs(test_y));
mae2
acc2
%}
%%%%%%%%%%%%%%%% 保存结果 %%%%%%%%%%%%%%%%%%%
result(count).station=outputStations;
result(count).timeDelay=timeDelay;
result(count).mae1=mae1;
result(count).acc1=acc1;
result(count).mae2=mae2;
result(count).acc2=acc2;
%result(count).mae2=mae2;
%result(count).acc2=acc2;
count=count+1;
end
end
save result.mat result;
%}
