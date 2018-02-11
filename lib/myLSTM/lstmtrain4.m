function lstm = lstmtrain4(lstm,x,y,opts)
    numepochs=opts.numepochs;
    batchsize=opts.batchsize;
    batches=opts.batches;
    numbatches=length(batches);
    batches=[batches length(y)];
    
    for i=1:numepochs
        lstm.loss(i)=0;
        
        lstm.lstmcell{1}.learningRate=lstm.lstmcell{1}.learningRate*lstm.lstmcell{1}.scaling_learningRate;    
        lstm.lstmcell{2}.learningRate=lstm.lstmcell{2}.learningRate*lstm.lstmcell{2}.scaling_learningRate;
        lstm.nn.learningRate=lstm.nn.learningRate*lstm.nn.scaling_learningRate;
      
        kk=randperm(numbatches);
       
        for j=1:numbatches
            if batches(kk(j))+1==batches(kk(j)+1)
                continue;
            end
            batch_x=x(batches(kk(j))+1:batches(kk(j)+1),:);
            batch_y=y(batches(kk(j))+1:batches(kk(j)+1),:);
            
            lstm.lstmcell{1}=lstmcellff(lstm.lstmcell{1},batch_x,batch_y);                
            lstm.lstmcell{2}=lstmcellff(lstm.lstmcell{2},lstm.lstmcell{1}.mh,batch_y);                
            lstm.nn=nnff(lstm.nn,lstm.lstmcell{2}.mh,batch_y);

            lstm.nn=nnbp(lstm.nn);
            lstm.nn=nnapplygrads(lstm.nn);            
            lstm.lstmcell{2}=lstmcellbp(lstm.lstmcell{2},lstm.nn.d);
            lstm.lstmcell{2}=lstmcellupdate(lstm.lstmcell{2});              
            lstm.lstmcell{1}=lstmcellbp(lstm.lstmcell{1},lstm.lstmcell{2}.dx);
            lstm.lstmcell{1}=lstmcellupdate(lstm.lstmcell{1});              

            lstm.loss(i)=lstm.loss(i)+mean(abs(lstm.nn.e));
            
            %disp(kk(j));
            %disp(mean(abs(lstm.nn.e)));
        end
        %disp(lstm.loss(i));
    lstm.loss(i)=lstm.loss(i)/numbatches;
       
    end
    
    
end