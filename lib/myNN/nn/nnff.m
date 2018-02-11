function nn=nnff(nn,x,y)

    n=nn.n;
    m=size(x,1);
    
    nn.a{1}=x;
    
    %ttt=clock;

    for i=2:n
        %ttt=clock;
        nn.a{i}=sigm(repmat(nn.b{i-1}',m,1)+nn.a{i-1}*nn.W{i-1}');
        %disp(etime(clock,ttt));
    end
    
    %disp(etime(clock,ttt));
    nn.e=y-nn.a{n};
    
end

        