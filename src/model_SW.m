% �ĸ�DNN������ʱ�����л�������ģʽ by fcmean���ռ�ѡ��Ҳ��ͬ���ع�����Իع飯SVM���ռ����Ϊgg

clear all;
clc;

load air_bj.mat;
%%%%%%%%%%%%%%%%%%%%%%
numTimeDelay=[35];
inputPolluents=[1 2 3 4 5];     %�����ȥ��Ⱦ������
outputPolluents=6;                  %���PM2.5
weatherFactors=[7 8 9 10];      %����δ������Ԥ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ԥ��վ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
count=1;
for iTimeDelay=1:length(numTimeDelay)
for iStation=1:35
outputStations=iStation

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Ԥ��������Ժ� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

timeDelay=numTimeDelay(iTimeDelay);  
range=timeDelay+1;
trainingLength=2400;
%%%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
K=4;                            % �����������ط���������
lag=40;                         % �������г���
M=floor(trainingLength/lag);    % ����������
adj=5;                          % ���վ������
adjMatrix=zeros(K,adj);         % �ĸ�ѵ���Ӽ������վ��
numDir=4;                       % �ռ������������

inputSize=range*length(inputPolluents)+range*length(outputPolluents)*adj+3*adj*length(weatherFactors)+numDir*(range*length(outputPolluents)+length(weatherFactors));    
%inputSize=3*length(inputPolluents)+3*length(outputPolluents)*adj+3*adj*length(weatherFactors)+numDir*(3*length(outputPolluents)+length(weatherFactors));    
outputSize=1;

%%%%%%%%%%%%%%%%% �����������ؽ��������з��� %%%%%%%%%%%%%%%%

wx=air_bj{outputStations}(1+timeDelay+range:trainingLength+timeDelay+range,weatherFactors);
wy=air_bj{outputStations}(1+timeDelay+range:trainingLength+timeDelay+range,outputPolluents);
fStatistic=zeros(M,length(weatherFactors));
for iLag=1:M
    for i=1:length(weatherFactors)
        [F,c]=granger_cause(wy(lag*(iLag-1)+1:lag*iLag,1),wx(lag*(iLag-1)+1:lag*iLag,i),0.01,1);
        fStatistic(iLag,i)=F;
    end
end
[u,centers]=fcmeans2(fStatistic,K);     % u:ÿ��������������ÿ����ĸ��ʾ���
second_x=[];    % �ع�������

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ��ģ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for kSub=1:K
    
index=[];
for i=1:M
    if u(kSub,i)>0.1
        index=[index lag*(i-1)+1:lag*i];
    end
end
subTrainingLength=length(index);

%%%%%%%%%%%%%%%%% �ռ����վ�����������ȡ %%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%% ����ѵ������������� %%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%% ѵ��Deep NN %%%%%%%%%%%%%%%%%%%%%%
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
dnn{kSub}=nnsetup([inputSize hiddenStruct outputSize]);
for iDnn=1:numel(hiddenStruct)
    dnn{kSub}.W{iDnn}=fnn.W{iDnn};
    dnn{kSub}.b{iDnn}=fnn.b{iDnn};
end
dnn{kSub}.W{numel(hiddenStruct)+1}=nn.W{1};
dnn{kSub}.b{numel(hiddenStruct)+1}=nn.b{1};

opts.sae=0;
opts.numepochs = Epochs;
opts.batchsize = Batchsize;
opts.silent = 1;
dnn{kSub}.learningRate = 1;
dnn{kSub}.scaling_learningRate = 0.99;
dnn{kSub}.weightPenaltyL2=0;

dnn{kSub}=nntrain(dnn{kSub},train_x,train_y,opts);

%%%%%%%%%%%%%%%% ��ȡ��ģ�Ͷ���ȫ���������ݵ����%%%%%%%%%%%%
dnn{kSub}=nnff(dnn{kSub},train_x,train_y);
weight=reshape(repmat(u(kSub,:),lag,1),trainingLength,1);
second_x=[second_x dnn{kSub}.a{end}];

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ��ģ��ѵ��������ѵ���ع�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%second_x=[second_x wx];
%theta=regress(train_y,second_x);
svr=svmtrain(train_y,second_x,'-s 3 -p 0.01 -q');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ���� %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
testingBegin=2400;
testingLength=240;
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

%%%%%%%%%%%%%%%%%%%%%%%% ��ģ�� %%%%%%%%%%%%%%%%%%%%%%%%%%%%
for kSub=1:K

%%%%%%%%%%%%%%%%%%%% ������������������ %%%%%%%%%%%%%%%%%%%
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

%%%%%%%%%%%%%%%%%%%%%% ������Լ���� %%%%%%%%%%%%%%%%%%%%%%%
dnn{kSub}=nnff(dnn{kSub},test_x,test_y);
weight=reshape(repmat(v(kSub,:),lag,1),testingLength,1);
second_x=[second_x dnn{kSub}.a{end}];

end
%%%%%%%%%%%%%%%%%%%%%%% ���Լ��ع� %%%%%%%%%%%%%%%%%%%%%%%%%%
%second_x=[second_x wx];
%py1=second_x*theta;
[py2,mse,dec]=svmpredict(test_y,second_x,svr);
%%%%%%%%%%%%%%%%%%%%%%%% ������ %%%%%%%%%%%%%%%%%%%%%%%%%%%
%e1=py1-test_y;
%mae1=mean(abs(e1))*500;
%acc1=1-sum(abs(e1))/sum(abs(test_y));
%mae1
%acc1

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
%%%%%%%%%%%%%%%% ������ %%%%%%%%%%%%%%%%%%%
result(count).station=outputStations;
result(count).timeDelay=timeDelay;
%result(count).mae1=mae1;
%result(count).acc1=acc1;
result(count).mae2=mae2;
result(count).acc2=acc2;
%result(count).mae2=mae2;
%result(count).acc2=acc2;
count=count+1;
end

mae=0;
acc=0;
for i=1:35
    mae=mae+result(i).mae2;
    acc=acc+result(i).acc2;
end
mae/35
acc/35

end
save result.mat result;
%}
