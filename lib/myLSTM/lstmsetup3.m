function lstm = lstmsetup3(size,opt)
    
    lstm.lstmcell{1}=lstmcellsetup(size(1),size(2),opt);
    lstm.lstmcell{2}=lstmcellsetup(size(2),size(3),opt);
    lstm.nn=nnsetup(size(1,3:end));
    lstm.nn.learningRate = opt.learningRate;
    lstm.nn.scaling_learningRate = opt.scaling_learningRate;
end