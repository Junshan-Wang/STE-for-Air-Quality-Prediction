function lstmcell = lstmcellff(lstmcell,x,y)

    [m,n]=size(x);
    
    mi=zeros(m,lstmcell.outputlen);
    mai=zeros(m,lstmcell.outputlen);
    mf=zeros(m,lstmcell.outputlen);
    mai=zeros(m,lstmcell.outputlen);
    mc=zeros(m,lstmcell.outputlen);
    mac=zeros(m,lstmcell.outputlen);
    mgac=zeros(m,lstmcell.outputlen);
    mo=zeros(m,lstmcell.outputlen);
    mao=zeros(m,lstmcell.outputlen);
    mgc=zeros(m,lstmcell.outputlen);
    mh=zeros(m,lstmcell.outputlen);
    
    mai(1,:) =  x(1,:) * lstmcell.W_ix'+lstmcell.b_i;
    mi(1,:) = active_func(mai(1,:), lstmcell.delta);
    maf(1,:) = x(1,:) * lstmcell.W_fx'+lstmcell.b_f;
    mf(1,:) = active_func(maf(1,:), lstmcell.delta);
    mac(1,:) = x(1,:) * lstmcell.W_cx'+lstmcell.b_c;
    mgac(1,:) = active_func(mac(1,:), lstmcell.g);
    mc(1,:) = mi(1,:) .*  mgac(1,:);
    mao(1,:) = x(1,:) * lstmcell.W_ox' + mc(1,:) .* lstmcell.W_oc'+lstmcell.b_o;
    mo(1,:) = active_func(mao(1,:), lstmcell.delta);
    mgc(1, :) = active_func(mc(1,:), lstmcell.g);
    mh(1,:) = mo(1,:) .* mgc(1, :);

    for t = 2 : m
        
        mai(t,:) =  x(t,:) * lstmcell.W_ix' + mh(t-1, :) * lstmcell.W_ih' + mc(t-1, :) .* lstmcell.W_ic' + lstmcell.b_i;
        mi(t,:) = active_func(mai(t,:), lstmcell.delta);

        maf(t,:) = x(t,:) * lstmcell.W_fx' + mh(t-1, :) * lstmcell.W_fh' + mc(t-1, :) .* lstmcell.W_fc' + lstmcell.b_f;
        mf(t,:) = active_func(maf(t,:), lstmcell.delta); 

        mac(t,:) = x(t,:) * lstmcell.W_cx' +  mh(t-1, :) * lstmcell.W_ch' + lstmcell.b_c;
        mgac(t,:) = active_func(mac(t,:), lstmcell.g);

        mc(t,:) = mf(t,:) .* mc(t-1, :) + mi(t,:) .*  mgac(t,:);

        mao(t,:) = x(t,:) * lstmcell.W_ox' + mh(t-1, :) * lstmcell.W_oh' + mc(t,:) .* lstmcell.W_oc' + lstmcell.b_o;
        mo(t,:) = active_func(mao(t,:), lstmcell.delta); 
        
        mgc(t, :) = active_func(mc(t,:), lstmcell.g);
        mh(t,:) = mo(t,:) .* mgc(t, :);
    end
    
    lstmcell.x = x;
    lstmcell.mi = mi;
    lstmcell.mai = mai;
    lstmcell.mf = mf;
    lstmcell.maf = maf;
    lstmcell.mc = mc;
    lstmcell.mac = mac;
    lstmcell.mgac = mgac;
    lstmcell.mo = mo;
    lstmcell.mao = mao;
    lstmcell.mgc = mgc;
    lstmcell.mh = mh;
    
end