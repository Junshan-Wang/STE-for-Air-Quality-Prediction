function lstmcell = lstmcellbp(lstmcell,e)

    [m,n]=size(e);
    
    dmi = zeros(m + 1, lstmcell.outputlen);
    dmai = zeros(m + 1, lstmcell.outputlen);
    dmf = zeros(m + 1, lstmcell.outputlen);
    dmaf = zeros(m + 1, lstmcell.outputlen);
    dmc = zeros(m + 1, lstmcell.outputlen);
    dmac = zeros(m + 1, lstmcell.outputlen);
    dmgac = zeros(m + 1, lstmcell.outputlen);
    dmo = zeros(m + 1, lstmcell.outputlen);
    dmao = zeros(m + 1, lstmcell.outputlen);
    dmh = zeros(m + 1,lstmcell.outputlen);
    dx = zeros(m + 1,lstmcell.inputlen);

    for t = m : -1 :1
        dmh(t, :) = e(t, :) + dmai(t + 1, : ) * lstmcell.W_ih + dmaf(t + 1, :) * lstmcell.W_fh + dmao(t + 1, :) * lstmcell.W_oh + dmac(t + 1, :) * lstmcell.W_ch;
        
        dmo(t, :) = dmh(t, :) .* lstmcell.mgc(t, :);
        dmao(t, :) = dmo(t, :) .* active_func_d(lstmcell.mo(t, :), lstmcell.delta);
        
        if t == m
            dmc(t, :) = dmh(t, :) .* lstmcell.mo(t, :) .* active_func_d(lstmcell.mgc(t, :), lstmcell.g) + ...
                + dmai(t + 1, :) .* lstmcell.W_ic' + dmaf(t + 1, :) .* lstmcell.W_fc' + dmao(t, :) .* lstmcell.W_oc'; % diagonal matrix represent by vertor
        else
            dmc(t, :) = dmh(t, :) .* lstmcell.mo(t, :) .* active_func_d(lstmcell.mgc(t, :), lstmcell.g) + dmc(t + 1, :) .* lstmcell.mf(t + 1, :) ...
                + dmai(t + 1, :) .* lstmcell.W_ic' + dmaf(t + 1, :) .* lstmcell.W_fc' + dmao(t, :) .* lstmcell.W_oc'; % diagonal matrix represent by vertor
        end        
        dmac(t, :) = dmc(t, :) .* active_func_d(lstmcell.mgac(t, :), lstmcell.g) .* lstmcell.mi(t, :);
       
        if t > 1
            dmf(t, :) = dmc(t, :) .* lstmcell.mc(t - 1, :);
            dmaf(t, :) = dmf(t, :) .* active_func_d(lstmcell.mf(t, :), lstmcell.delta);
        end

        dmi(t, :) = dmc(t, :) .* lstmcell.mgac(t, :);
        dmai(t, :) = dmi(t, :) .* active_func_d(lstmcell.mi(t, :), lstmcell.delta);

        dx(t, :) = dmai(t, :) * lstmcell.W_ix + dmaf(t, :) * lstmcell.W_fx + dmao(t, :) * lstmcell.W_ox + dmac(t, :) * lstmcell.W_cx;       
    end
    
    lstmcell.e = e(1:m, :);
    lstmcell.dmi = dmi(1:m, :);
    lstmcell.dmai = dmai(1:m, :);
    lstmcell.dmf = dmf(1:m, :);
    lstmcell.dmaf = dmaf(1:m, :);
    lstmcell.dmc = dmc(1:m, :);
    lstmcell.dmac = dmac(1:m, :);
    lstmcell.dmgac = dmgac(1:m, :);
    lstmcell.dmo = dmo(1:m, :);
    lstmcell.dmao = dmao(1:m, :);
    lstmcell.dmh = dmh(1:m, :);
    lstmcell.dx = dx(1:m, :);
    
    lstmcell.dW_ix = lstmcell.dmai' * lstmcell.x / m;
    lstmcell.dW_ih = lstmcell.dmai(2 : end, :)' * lstmcell.mh(1 : end-1, :) / (m - 1);
    lstmcell.dW_ic = (sum(lstmcell.dmai(2 : end, :) .* lstmcell.mc(1 : end-1, :)) / (m - 1))';
    lstmcell.db_i = sum(lstmcell.dmai) / m;
    
    lstmcell.dW_fx = lstmcell.dmaf' * lstmcell.x / m;
    lstmcell.dW_fh = lstmcell.dmaf(2 : end, :)' * lstmcell.mh(1 : end-1, :) / (m - 1);
    lstmcell.dW_fc = (sum(lstmcell.dmaf(2 : end, :) .* lstmcell.mc(1 : end-1, :)) / (m - 1))';
    lstmcell.db_f = sum(lstmcell.dmaf) / m;
    
    lstmcell.dW_cx = lstmcell.dmac' * lstmcell.x / m;
    lstmcell.dW_ch = lstmcell.dmac(2 : end, :)' * lstmcell.mh(1:end - 1, :) / (m - 1);
    lstmcell.db_c = sum(lstmcell.dmac) / m;
    
    lstmcell.dW_ox = lstmcell.dmao' * lstmcell.x / m;
    lstmcell.dW_oh = lstmcell.dmao(2 : end, :)' * lstmcell.mh(1 : end-1, :) / (m - 1);
    lstmcell.dW_oc = (sum(lstmcell.dmao .* lstmcell.mc) / (m - 1))';
    lstmcell.db_o = sum(lstmcell.dmao) / m;

end
