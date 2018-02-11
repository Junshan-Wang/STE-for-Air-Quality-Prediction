function lstm = lstmsetup(size,opt)
    lstm.lstmcell=lstmcellsetup(size(1),size(2),opt);
    
    lstm.nn=nnsetup(size(1,2:end));
    lstm.nn.learningRate = opt.learningRate;
    lstm.nn.scaling_learningRate = opt.scaling_learningRate;
    %lstm.nn.weightPenaltyL2=opt.weightPenaltyL2;
end