function lstm = lstmsetup2(size,opt)
    layer=floor((length(size)-1)/2);
    lstm.layer=layer;
    
    for k=1:layer
        lstm.lstmcell{k}=lstmcellsetup(size((k-1)*2+1),size(k*2),opt);
        if k<layer
            lstm.nn{k}=nnsetup(size(1,k*2:k*2+1));
        else
            lstm.nn{k}=nnsetup(size(1,k*2:end));
        end
        lstm.nn{k}.learningRate = opt.learningRate;
        lstm.nn{k}.scaling_learningRate = opt.scaling_learningRate;
        %lstm.nn{k}.weightPenaltyL2=opt.weightPenaltyL2;
    end
end