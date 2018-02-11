function  nn=nntrain(nn, x, y, opts)
    m=size(x,1);
    
    batchsize=opts.batchsize;
    numepochs=opts.numepochs;
    
    numbatches=floor(m/batchsize);
    %numbatches=1;
    
    for i=1:numepochs
        
        kk=randperm(m);
        nn.learningRate=nn.learningRate*nn.scaling_learningRate;
        
        for l=1:numbatches
            if (l==numbatches)
                batch_x=x(kk((numbatches-1)*batchsize+1:end), :);
                batch_y=y(kk((numbatches-1)*batchsize+1:end), :);
            else
                batch_x=x(kk((l-1)*batchsize+1:l*batchsize), :);
                batch_y=y(kk((l-1)*batchsize+1:l*batchsize), :);
            end
            
            %f=clock;
            nn=nnff(nn,batch_x,batch_y);
            %ff=etime(clock,f);
            %disp(ff);
            
            nn=nnbp(nn);
            nn=nnapplygrads(nn);
            
            if (nn.sae==1)
                tempW=(nn.W{1,1}+nn.W{1,2}')/2;
                nn.W{1,1}=tempW;
                nn.W{1,2}=tempW';
            end
        end
        
    end
end
    
            
            
            
                
        
        