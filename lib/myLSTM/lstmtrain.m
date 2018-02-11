function lstm = lstmtrain(lstm,x,y,opts)
    numepochs=opts.numepochs;
    batchsize=opts.batchsize;
    numbatches=size(x,1)/batchsize;
    
    
    for i=1:numepochs
        lstm.loss(i)=0;
        lstm.lstmcell.learningRate=lstm.lstmcell.learningRate*lstm.lstmcell.scaling_learningRate;
        lstm.nn.learningRate=lstm.nn.learningRate*lstm.nn.scaling_learningRate;
        kk=randperm(numbatches);
        for j=1:numbatches
            batch_x=x((kk(j)-1)*batchsize+1:kk(j)*batchsize,:);
            batch_y=y((kk(j)-1)*batchsize+1:kk(j)*batchsize,:);
            %batch_x=x((j-1)*batchsize+1:j*batchsize,:);
            %batch_y=y((j-1)*batchsize+1:j*batchsize,:);
            %batch_x=x;
            %batch_y=y;
            
            lstm.lstmcell=lstmcellff(lstm.lstmcell,batch_x,batch_y);

            lstm.nn=nnff(lstm.nn,lstm.lstmcell.mh,batch_y);
            lstm.nn=nnbp(lstm.nn);
            lstm.nn=nnapplygrads(lstm.nn);

            lstm.lstmcell=lstmcellbp(lstm.lstmcell,lstm.nn.d);
            lstm.lstmcell=lstmcellupdate(lstm.lstmcell);   
            
            lstm.loss(i)=lstm.loss(i)+mean(abs(lstm.nn.e));
        end
    lstm.loss(i)=lstm.loss(i)/numbatches;
       
    end
    
    
end