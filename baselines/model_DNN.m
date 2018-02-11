%%�ԱȰ汾2��������Χվ�㣬ʹ��DNN:SAE+NN

clear all;
clc;

load air_bj.mat;

adj=5;
adjMatrix=zeros(35,adj); %% include it self
for i=1:35
    [a,b]=sort(pdist2(stationLocation,stationLocation(i,:)));
    adjMatrix(i,:)=b(1:adj);
end

numTimeDelay=[47];
numLearningRate=[1];
numScaling_learningRate=[0.99];
numEpoches=[100];
numBatchSize=[40];
ifPenalty=[0];
numMomentum=[0.5];

inputPolluents=[1 2 3 4 5];     %�����ȥ��Ⱦ������
outputPolluents=6;                  %���PM2.5
weatherFactors=[7 8 9 10];      %����δ������Ԥ��

count=1;
stations=1:35;

%%%%%%%%%%%%%%%%%Ԥ��������Ժ�%%%%%%%%%%%%%%%%%%%%
for iTimeDelay=1:length(numTimeDelay)

for iStation=1:length(stations)

outputStation=stations(iStation)  

timeDelay=numTimeDelay(iTimeDelay);    %Ԥ�⼸Сʱ֮���ֵ��0Ϊһ��
range=timeDelay+1;

inputSize=range*length(inputPolluents)+adj*range*length(outputPolluents)+adj*3*length(weatherFactors);    
outputSize=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%���������������%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
hiddenLayers=2;     %���ز���
hiddenNodes1=100;    %��һ��������
hiddenNodes2=80;    %�ڶ���������
hiddenStruct=[hiddenNodes1 hiddenNodes2];
Epochs=100;
Batchsize=40;

%%%%%%%%%sae��ȡ����%%%%%%%%%%%
sae=saesetup([inputSize hiddenStruct]);
for iSae=1:numel(hiddenStruct)
    sae.ae{iSae}.learningRate=1;
    sae.ae{iSae}.scaling_learningRate=0.99;
    sae.ae{iSae}.sae=1;
end
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
sae=saetrain(sae,train_x,opts);

%%%%ÿһ��������ȡ���ƫ�ú�Ȩ�ؽ��fnn,�õ�train_feature%%%%
fnn=nnsetup([inputSize hiddenStruct]);
for iFnn=1:numel(hiddenStruct)
    fnn.W{iFnn}=sae.ae{iFnn}.W{1};
    fnn.b{iFnn}=sae.ae{iFnn}.b{1};
end
fnn=nnff(fnn,train_x,zeros(length(train_x),hiddenStruct(end)));
train_feature=fnn.a{end};

%%%%%%%��ͨ������nn�ع�Ԥ��%%%%%%%%%
nn=nnsetup([hiddenStruct(end) outputSize]);
opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
nn.learningRate = 1;
nn.scaling_learningRate = 0.99;
nn = nntrain(nn, train_feature, train_y, opts); 

%%%%ջʽ�Ա������ͨ������%%%%%%
dnn=nnsetup([inputSize hiddenStruct outputSize]);
for iDnn=1:numel(hiddenStruct)
    dnn.W{iDnn}=fnn.W{iDnn};
    dnn.b{iDnn}=fnn.b{iDnn};
end
dnn.W{numel(hiddenStruct)+1}=nn.W{1};
dnn.b{numel(hiddenStruct)+1}=nn.b{1};

opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
dnn.learningRate = 1;
dnn.scaling_learningRate = 0.99;
dnn.weightPenaltyL2=0;

dnn=nntrain(dnn,train_x,train_y,opts);

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

dnn=nnff(dnn,test_x,test_y);
predict_y=dnn.a{end};
e=predict_y-test_y;


result(count).station=outputStation;
result(count).timeDelay=timeDelay;
result(count).MAE=mean(abs(e))*500;
result(count).RMSE=sqrt(mean(e.^2))*500;
result(count).accuracy=1-sum(abs(e))/sum(abs(test_y));
count=count+1;
end
end

mae=0;
for i=1:10
    mae=mae+result(i).MAE;
end
mae/10

save myContrast2.mat result;
%%%%%%%%%%%%%%%%%%%%%%End iTimeDelay%%%%%%%%%%%%%%%%%%%%%%%


