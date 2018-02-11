%%对比版本：FFA模型
clear all;
clc;
load air_bj.mat;

timeDelay=47;
inputPolluents=[1 2 3 4 5];     %输入过去污染物数据
outputPolluents=6;                  %输出PM2.5
weatherFactors=[7 8 9 10];      %输入未来天气预测
trainingLength=2400;
testingLength=240;
count=1;

for outputStations=1:35
outputStations

range=timeDelay+1;


%%%%%%%%%% temproal predictor %%%%%%%%%%
inputSize=range*(length(outputPolluents)+length(inputPolluents))+(range+1)*length(weatherFactors)+2;    
outputSize=1;
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=[air_bj{outputStations}(i:i+range-1,outputPolluents)'];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for j=1:length(weatherFactors)
        t=[t air_bj{outputStations}(i+range-1:i+range+timeDelay,weatherFactors(j))'];
    end
    t=[t mod(i,24) mod(i,24*7)];
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=[air_bj{outputStations}(i:i+range-1,outputPolluents)'];
    for j=1:length(inputPolluents)
        t=[t air_bj{outputStations}(i:i+range-1,inputPolluents(j))'];
    end
    for j=1:length(weatherFactors)
        t=[t air_bj{outputStations}(i+range-1:i+range+timeDelay,weatherFactors(j))'];
    end
    t=[t mod(i,24) mod(i,24*7)];
    test_x(i-trainingLength,:)=t;    
    test_y(i-trainingLength,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end

theta=regress(train_y,train_x);

train_temporal=train_x*theta;
test_temporal=test_x*theta;

mae_temporal=mean(abs(test_temporal-test_y))*500

%%%%%%%%%%% spatial predictor %%%%%%%%%%
dist=0.3;
numRegions=16;
regions{1}=[];regions{2}=[];regions{3}=[];regions{4}=[];
regions{5}=[];regions{6}=[];regions{7}=[];regions{8}=[];
regions{9}=[];regions{10}=[];regions{11}=[];regions{12}=[];
regions{13}=[];regions{14}=[];regions{15}=[];regions{16}=[];
regions{17}=[];
center=stationLocation(outputStations,:);

for i=1:35
    t=stationLocation(i,:);
    if center(1)==t(1) && center(2)==t(2)
        j=1;
    elseif center(1)>t(1) && center(2)>t(2)
        j=1;
    elseif center(1)>t(1) && center(2)<=t(2)
        j=5;
    elseif center(1)<=t(1) && center(2)>t(2)
        j=9;
    else 
        j=13;  
    end
    if abs((center(1)-t(1))/(center(2)-t(2)))>1
        j=j+2;
    end
    if pdist2(center,t)>dist
        j=j+1;
    end
    regions{j}=[regions{j} i];
end
%{
for i=1:35
    t=stationLocation(i,:);
    if center(1)>t(1) && center(2)>t(2)
        j=1;
    elseif center(1)>t(1) && center(2)<t(2)
        j=2;
    elseif center(1)<t(1) && center(2)>t(2)
        j=3;
    else 
        j=4;  
    end
    regions{j}=[regions{j} i];
end
%}
inputSize=3*(length(inputPolluents)+length(outputPolluents))+length(weatherFactors);
totalInputSize=inputSize*numRegions;
outputSize=1;
train_x=zeros(trainingLength,totalInputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength   
    t=[];
    for r=1:numRegions
        if isempty(regions{r}) 
            t=[t zeros(1,inputSize)]; 
            continue;
        end
        t2=zeros(length(regions{r}),inputSize);
        for k=1:length(regions{r})
            t3=[air_bj{regions{r}(k)}(i+range-3:i+range-1,outputPolluents)'];
            
            for j=1:length(inputPolluents)
                t3=[t3 air_bj{regions{r}(k)}(i+range-3:i+range-1,inputPolluents(j))'];
            end
            for j=1:length(weatherFactors)
                t3=[t3 air_bj{regions{r}(k)}(i+range-1,weatherFactors(j))'];
            end
            t2(k,:)=t3;
        end
        t2=mean(t2,1);
        t=[t t2];  
    end
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);
end
test_x=zeros(testingLength,totalInputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=[];
    for r=1:numRegions
        if isempty(regions{r}) 
            t=[t zeros(1,inputSize)]; 
            continue;
        end
        t2=zeros(length(regions{r}),inputSize);
        for k=1:length(regions{r})
            t3=[air_bj{regions{r}(k)}(i+range-3:i+range-1,outputPolluents)'];
            for j=1:length(inputPolluents)
                t3=[t3 air_bj{regions{r}(k)}(i+range-3:i+range-1,inputPolluents(j))'];
            end
            for j=1:length(weatherFactors)
                t3=[t3 air_bj{regions{r}(k)}(i+range-1,weatherFactors(j))'];
            end
            t2(k,:)=t3;
        end
        t2=mean(t2,1);
        t=[t t2];  
    end
    test_x(i-trainingLength,:)=t;    
    test_y(i-trainingLength,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);
end

hiddenLayers=2;     %隐藏层数
hiddenNodes1=100;    %第一层结点数量
hiddenNodes2=80;    %第二层结点数量
hiddenStruct=[hiddenNodes1 hiddenNodes2];
nn=nnsetup([totalInputSize hiddenStruct outputSize]);

opts.numepochs = 100;
opts.batchsize = 40;
nn = nntrain(nn, train_x, train_y, opts); 

nn=nnff(nn,train_x,train_y);
train_spatial=nn.a{end};
nn=nnff(nn,test_x,test_y);
test_spatial=nn.a{end};

mae_spatial=mean(abs(test_spatial-test_y))*500

%%%%%%%%% predictor aggregator %%%%%%%%%
inputSize=(length(weatherFactors))+1;    
outputSize=1;
train_x=zeros(trainingLength,inputSize);
train_y=zeros(trainingLength,outputSize);
for i=1:trainingLength
    t=air_bj{outputStations}(i+range-1,outputPolluents);
    %t=[];
    for j=1:length(weatherFactors)
        t=[t air_bj{outputStations}(i+range+timeDelay,weatherFactors(j))];
        %t=[t mean(air_bj{outputStations}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{outputStations}(i+range:i+range+timeDelay,weatherFactors(j))) (air_bj{outputStations}(i+range+timeDelay,weatherFactors(j)))];
    end
    train_x(i,:)=t;    
    train_y(i,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end
train_x=[train_x train_temporal train_spatial];
test_x=zeros(testingLength,inputSize);
test_y=zeros(testingLength,outputSize);
for i=trainingLength+1:trainingLength+testingLength
    t=air_bj{outputStations}(i+range-1,outputPolluents);
    %t=[];
    for j=1:length(weatherFactors)
        t=[t air_bj{outputStations}(i+range+timeDelay,weatherFactors(j))];
        %t=[t mean(air_bj{outputStations}(i+range:i+range+timeDelay,weatherFactors(j))) max(air_bj{outputStations}(i+range:i+range+timeDelay,weatherFactors(j))) (air_bj{outputStations}(i+range+timeDelay,weatherFactors(j)))];
    end
    test_x(i-trainingLength,:)=t;    
    test_y(i-trainingLength,1)=air_bj{outputStations}(i+range-1,outputPolluents)-air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
    test(i-trainingLength,1)=air_bj{outputStations}(i+range+timeDelay,outputPolluents);    
end
test_x=[test_x test_temporal test_spatial];

%regtree=fitrtree([train_x train_y],'y~temporal+spatial','MinLeafSize',100, ...
%    'PredictorNames',{'temMean','temMax','temLast','humMean','humMax','humLast',...
%    'wdMean','wdMax','wdLast','wsMean','wsMax','wsLast','temporal','spatial','y'});%,'CategoricalPredictors','all');
%predict_y=predict(regtree,test_x);

svr=svmtrain(train_y,train_x,'-s 3 -p 0.01 -q');
[predict_y,mse,dec]=svmpredict(test_y,test_x,svr);

%regTree=createTree([train_x,train_y],1e-9,100);
%predict_y=predictTree([test_x,test_y],regTree);

%theta=regress(train_y,train_x);
%predict_y=test_x*theta;

%{
regtree=fitrtree(train_x,train_y,'MinLeaf',2000);
predict_y=predict(regtree,train_x);
for j=1:regtree.NumNodes
    x_index{j}=find(predict_y==regtree.NodeMean(j));
    regress_b{j}=regress(train_y(x_index{j},:),[train_x(x_index{j},:) ones(length(x_index{j}),1)]);
end

predict_y=predict(regtree,test_x);
predict_y2=predict_y;
for j=1:regtree.NumNodes
    i=find(predict_y2==regtree.NodeMean(j));
    predict_y2(i)=[test_x(i,:) ones(length(i),1)]*regress_b{j};
end
%}
e=predict_y-test_y;
mae=mean(abs(e))*500
rmse=sqrt((mean(e.^2)))*500;
acc=1-sum(abs(e))/sum(abs(test));
%{
e2=predict_y2-test_y;
mae2=mean(abs(e2))*500
rmse2=sqrt((mean(e2.^2)))*500;
acc2=1-sum(abs(e2))/sum(abs(test_y));
%}

result(count).station=outputStations;
result(count).timeDelay=timeDelay;
result(count).mae=mae;
result(count).rmse=rmse;
result(count).acc=acc;
%{
result(count).mae2=mae2;
result(count).rmse2=rmse2;
result(count).acc2=acc2;
%}
result(count).mae_temporal=mae_temporal;
result(count).mae_spatial=mae_spatial;

count=count+1;

end

mae=0;
acc=0;
for i=1:35
    mae=mae+result(i).mae;
    acc=acc+result(i).acc;
end
mae/35
acc/35

save result.mat result;