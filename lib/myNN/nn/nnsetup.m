function nn=nnsetup(size)

    nn.size=size;
    nn.n=numel(nn.size);
    
    nn.learningRate=0.99;
    nn.scaling_learningRate=1;
    nn.sae=0;
    nn.weightPenaltyL2=0;
    nn.weightPenaltyL1=0;
    
    for i=2:nn.n
        nn.b{i-1}=zeros(nn.size(i),1);
        nn.W{i-1}=(rand(nn.size(i),nn.size(i-1))-0.5)*2*4*sqrt(6/(nn.size(i)+nn.size(i-1)));
        nn.p{i}=zeros(1,nn.size(i));
    end
    
end
