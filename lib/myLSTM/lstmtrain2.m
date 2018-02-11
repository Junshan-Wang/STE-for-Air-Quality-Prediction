function lstm = lstmtrain2(lstm,x,y,opts)
    numepochs=opts.numepochs;
    batchsize=opts.batchsize;
    numbatches=size(x,1)/batchsize;
    
    layer=lstm.layer;
    
    for i=1:numepochs
        lstm.loss(i)=0;
        
        for k=1:layer
            lstm.lstmcell{k}.learningRate=lstm.lstmcell{k}.learningRate*lstm.lstmcell{k}.scaling_learningRate;
            lstm.nn{k}.learningRate=lstm.nn{k}.learningRate*lstm.nn{k}.scaling_learningRate;
        end
        
        kk=randperm(numbatches);
        for j=1:numbatches
            batch_x=x((kk(j)-1)*batchsize+1:kk(j)*batchsize,:);
            batch_y=y((kk(j)-1)*batchsize+1:kk(j)*batchsize,:);

            for k=1:layer
                lstm.lstmcell{k}=lstmcellff(lstm.lstmcell{k},batch_x,batch_y);                
                lstm.nn{k}=nnff(lstm.nn{k},lstm.lstmcell{k}.mh,zeros(size(batch_x,1),lstm.nn{k}.size(end)));
                batch_x=lstm.nn{k}.a{end};
            end
            
            for k=layer:-1:1     
                if k<layer
                    lstm.nn{k}.e=-lstm.lstmcell{k+1}.dx;
                else
                    lstm.nn{k}.e=batch_y-lstm.nn{k}.a{end};
                end
                lstm.nn{k}=nnbp(lstm.nn{k});
                lstm.nn{k}=nnapplygrads(lstm.nn{k});            
                lstm.lstmcell{k}=lstmcellbp(lstm.lstmcell{k},lstm.nn{k}.d);
                lstm.lstmcell{k}=lstmcellupdate(lstm.lstmcell{k});              
            end
            
            lstm.loss(i)=lstm.loss(i)+mean(abs(lstm.nn{layer}.e));
        end
    lstm.loss(i)=lstm.loss(i)/numbatches;
       
    end
    
    
end