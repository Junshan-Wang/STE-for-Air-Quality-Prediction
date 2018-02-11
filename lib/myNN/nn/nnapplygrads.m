function nn=nnapplygrads(nn)
    
    for i=1:(nn.n-1)
        nn.W{i}=nn.W{i}-nn.learningRate*(nn.dW{i}+nn.weightPenaltyL2*nn.W{i}+nn.weightPenaltyL1*sign(nn.W{i}));
        nn.b{i}=nn.b{i}-nn.learningRate*nn.db{i};
    end
    
end