function nn=nnbp(nn)
    
    n=nn.n;
    
    d{n}=-nn.e.*(nn.a{n}.*(1-nn.a{n}));
    
    for i=(n-1):-1:2
        d{i}=(d{i+1}*nn.W{i}).*(nn.a{i}.*(1-nn.a{i}));
    end
    
    for i=1:(n-1)
        nn.dW{i}=(d{i+1}'*nn.a{i})/size(d{i+1},1);
        nn.db{i}=sum(d{i+1},1)'/size(d{i+1},1);
    end
    
    nn.d=(d{2}*nn.W{1});
end