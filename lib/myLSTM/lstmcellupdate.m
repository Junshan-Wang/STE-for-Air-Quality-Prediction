function lstmcell = lstmcellupdate(lstmcell)

if lstmcell.weightPenaltyGroup>0

    groupSize=lstmcell.outputlen/lstmcell.group;
    for i=1:lstmcell.group   
        for j=1:groupSize

            dW_ix((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_ix((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * [zeros(1, 1) lstmcell.W_ix((i-1)*groupSize+j,2:end)] / sqrt(sum(lstmcell.W_ix((i-1)*groupSize+j,2:end).^2)));
            dW_ih((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_ih((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_ih((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_ih((i-1)*groupSize+j,:).^2))) ;
            dW_ic((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_ic((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_ic((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_ic((i-1)*groupSize+j,:).^2)));
            db_i(1,(i-1)*groupSize+j) = lstmcell.learningRate * lstmcell.db_i(1,(i-1)*groupSize+j);

            dW_fx((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_fx((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * [zeros(1, 1) lstmcell.W_fx((i-1)*groupSize+j,2:end)] / sqrt(sum(lstmcell.W_fx((i-1)*groupSize+j,2:end).^2)));
            dW_fh((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_fh((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_fh((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_fh((i-1)*groupSize+j,:).^2)));
            dW_fc((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_fc((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_fc((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_fc((i-1)*groupSize+j,:).^2)));
            db_f(1,(i-1)*groupSize+j) = lstmcell.learningRate * lstmcell.db_f(1,(i-1)*groupSize+j);

            dW_cx((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_cx((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * [zeros(1, 1) lstmcell.W_cx((i-1)*groupSize+j,2:end)] / sqrt(sum(lstmcell.W_cx((i-1)*groupSize+j,2:end).^2)));
            dW_ch((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_ch((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_ch((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_ch((i-1)*groupSize+j,:).^2)));
            db_c(1,(i-1)*groupSize+j) = lstmcell.learningRate * lstmcell.db_c(1,(i-1)*groupSize+j);

            dW_ox((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_ox((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * [zeros(1, 1) lstmcell.W_ox((i-1)*groupSize+j,2:end)] / sqrt(sum(lstmcell.W_ox((i-1)*groupSize+j,2:end).^2)));
            dW_oh((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_oh((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_oh((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_oh((i-1)*groupSize+j,:).^2)));
            dW_oc((i-1)*groupSize+j,:) = lstmcell.learningRate * (lstmcell.dW_oc((i-1)*groupSize+j,:) + lstmcell.weightPenaltyGroup * lstmcell.W_oc((i-1)*groupSize+j,:) / sqrt(sum(lstmcell.W_oc((i-1)*groupSize+j,:).^2)));
            db_o(1,(i-1)*groupSize+j) = lstmcell.learningRate * lstmcell.db_o(1,(i-1)*groupSize+j);
        end
    end

    
elseif lstmcell.weightPenaltyL1>0
        
    dW_ix = lstmcell.learningRate * (lstmcell.dW_ix + lstmcell.weightPenaltyL1 * sign([zeros(size(lstmcell.W_ix,1), 1) lstmcell.W_ix(:,2:end)]));
    dW_ih = lstmcell.learningRate * (lstmcell.dW_ih + lstmcell.weightPenaltyL1 * sign(lstmcell.W_ih));
    dW_ic = lstmcell.learningRate * (lstmcell.dW_ic + lstmcell.weightPenaltyL1 * sign(lstmcell.W_ic));
    db_i = lstmcell.learningRate * lstmcell.db_i;
    
    dW_fx = lstmcell.learningRate * (lstmcell.dW_fx + lstmcell.weightPenaltyL1 * sign([zeros(size(lstmcell.W_fx,1), 1) lstmcell.W_ix(:,2:end)]));
    dW_fh = lstmcell.learningRate * (lstmcell.dW_fh + lstmcell.weightPenaltyL1 * sign(lstmcell.W_fh));
    dW_fc = lstmcell.learningRate * (lstmcell.dW_fc + lstmcell.weightPenaltyL1 * sign(lstmcell.W_fc));
    db_f = lstmcell.learningRate * lstmcell.db_f;

    dW_cx = lstmcell.learningRate * (lstmcell.dW_cx + lstmcell.weightPenaltyL1 * sign([zeros(size(lstmcell.W_cx,1), 1) lstmcell.W_ix(:,2:end)]));
    dW_ch = lstmcell.learningRate * (lstmcell.dW_ch + lstmcell.weightPenaltyL1 * sign(lstmcell.W_ch));
    db_c = lstmcell.learningRate * lstmcell.db_c;

    dW_ox = lstmcell.learningRate * (lstmcell.dW_ox + lstmcell.weightPenaltyL1 * sign([zeros(size(lstmcell.W_ox,1), 1) lstmcell.W_ix(:,2:end)]));
    dW_oh = lstmcell.learningRate * (lstmcell.dW_oh + lstmcell.weightPenaltyL1 * sign(lstmcell.W_oh));
    dW_oc = lstmcell.learningRate * (lstmcell.dW_oc + lstmcell.weightPenaltyL1 * sign(lstmcell.W_oc));
    db_o = lstmcell.learningRate * lstmcell.db_o;

elseif lstmcell.weightPenaltyL2>0
        
    dW_ix = lstmcell.learningRate * (lstmcell.dW_ix + lstmcell.weightPenaltyL2 * [zeros(size(lstmcell.W_ix,1), 1) lstmcell.W_ix(:,2:end)]);
    dW_ih = lstmcell.learningRate * (lstmcell.dW_ih + lstmcell.weightPenaltyL2 * lstmcell.W_ih);
    dW_ic = lstmcell.learningRate * (lstmcell.dW_ic + lstmcell.weightPenaltyL2 * lstmcell.W_ic);
    db_i = lstmcell.learningRate * lstmcell.db_i;
    
    dW_fx = lstmcell.learningRate * (lstmcell.dW_fx + lstmcell.weightPenaltyL2 * [zeros(size(lstmcell.W_fx,1), 1) lstmcell.W_ix(:,2:end)]);
    dW_fh = lstmcell.learningRate * (lstmcell.dW_fh + lstmcell.weightPenaltyL2 * lstmcell.W_fh);
    dW_fc = lstmcell.learningRate * (lstmcell.dW_fc + lstmcell.weightPenaltyL2 * lstmcell.W_fc);
    db_f = lstmcell.learningRate * lstmcell.db_f;

    dW_cx = lstmcell.learningRate * (lstmcell.dW_cx + lstmcell.weightPenaltyL2 * [zeros(size(lstmcell.W_cx,1), 1) lstmcell.W_ix(:,2:end)]);
    dW_ch = lstmcell.learningRate * (lstmcell.dW_ch + lstmcell.weightPenaltyL2 * lstmcell.W_ch);
    db_c = lstmcell.learningRate * lstmcell.db_c;

    dW_ox = lstmcell.learningRate * (lstmcell.dW_ox + lstmcell.weightPenaltyL2 * [zeros(size(lstmcell.W_ox,1), 1) lstmcell.W_ix(:,2:end)]);
    dW_oh = lstmcell.learningRate * (lstmcell.dW_oh + lstmcell.weightPenaltyL2 * lstmcell.W_oh);
    dW_oc = lstmcell.learningRate * (lstmcell.dW_oc + lstmcell.weightPenaltyL2 * lstmcell.W_oc);
    db_o = lstmcell.learningRate * lstmcell.db_o;
        
else
    dW_ix = lstmcell.learningRate * lstmcell.dW_ix;
    dW_ih = lstmcell.learningRate * lstmcell.dW_ih;
    dW_ic = lstmcell.learningRate * lstmcell.dW_ic;
    db_i = lstmcell.learningRate * lstmcell.db_i;

    dW_fx = lstmcell.learningRate * lstmcell.dW_fx;
    dW_fh = lstmcell.learningRate * lstmcell.dW_fh;
    dW_fc = lstmcell.learningRate * lstmcell.dW_fc;
    db_f = lstmcell.learningRate * lstmcell.db_f;

    dW_cx = lstmcell.learningRate * lstmcell.dW_cx;
    dW_ch = lstmcell.learningRate * lstmcell.dW_ch;
    db_c = lstmcell.learningRate * lstmcell.db_c;

    dW_ox = lstmcell.learningRate * lstmcell.dW_ox;
    dW_oh = lstmcell.learningRate * lstmcell.dW_oh;
    dW_oc = lstmcell.learningRate * lstmcell.dW_oc;
    db_o = lstmcell.learningRate * lstmcell.db_o;
end


if(lstmcell.momentum>0)
    lstmcell.vW_ix = lstmcell.momentum * lstmcell.vW_ix - dW_ix;
    dW_ix = lstmcell.vW_ix;
    lstmcell.vW_ih = lstmcell.momentum * lstmcell.vW_ih - dW_ih;
    dW_ih = lstmcell.vW_ih;
    lstmcell.vW_ic = lstmcell.momentum * lstmcell.vW_ic - dW_ic;
    dW_ic = lstmcell.vW_ic;
    lstmcell.vb_i = lstmcell.momentum * lstmcell.vb_i - db_i;
    db_i = lstmcell.vb_i;

    lstmcell.vW_fx = lstmcell.momentum * lstmcell.vW_fx - dW_fx;
    dW_fx = lstmcell.vW_fx;
    lstmcell.vW_fh = lstmcell.momentum * lstmcell.vW_fh - dW_fh;
    dW_fh = lstmcell.vW_fh;
    lstmcell.vW_fc = lstmcell.momentum * lstmcell.vW_fc - dW_fc;
    dW_fc = lstmcell.vW_fc;
    lstmcell.vb_f = lstmcell.momentum * lstmcell.vb_f - db_f;
    db_f = lstmcell.vb_f;

    lstmcell.vW_cx =  lstmcell.momentum * lstmcell.vW_cx - dW_cx;
    dW_cx = lstmcell.vW_cx;
    lstmcell.vW_ch = lstmcell.momentum * lstmcell.vW_ch - dW_ch;
    dW_ch = lstmcell.vW_ch;
    lstmcell.vb_c = lstmcell.momentum * lstmcell.vb_c - db_c;
    db_c = lstmcell.vb_c;

    lstmcell.vW_ox = lstmcell.momentum * lstmcell.vW_ox - dW_ox;
    dW_ox = lstmcell.vW_ox;
    lstmcell.vW_oh = lstmcell.momentum * lstmcell.vW_oh - dW_oh;
    dW_oh = lstmcell.vW_oh;
    lstmcell.vW_oc = lstmcell.momentum * lstmcell.vW_oc - dW_oc;
    dW_oc = lstmcell.vW_oc;
    lstmcell.vb_o = lstmcell.momentum * lstmcell.vb_o - db_o;
	db_o = lstmcell.vb_o;
end


lstmcell.W_ix = lstmcell.W_ix + dW_ix;
lstmcell.W_ih = lstmcell.W_ih + dW_ih;
lstmcell.W_ic = lstmcell.W_ic + dW_ic;
lstmcell.b_i = lstmcell.b_i + db_i;

lstmcell.W_fx = lstmcell.W_fx + dW_fx;
lstmcell.W_fh = lstmcell.W_fh + dW_fh;
lstmcell.W_fc = lstmcell.W_fc + dW_fc;
lstmcell.b_f = lstmcell.b_f + db_f;

lstmcell.W_cx = lstmcell.W_cx + dW_cx;
lstmcell.W_ch = lstmcell.W_ch + dW_ch;
lstmcell.b_c = lstmcell.b_c + db_c;

lstmcell.W_ox = lstmcell.W_ox + dW_ox;
lstmcell.W_oh = lstmcell.W_oh + dW_oh;
lstmcell.W_oc = lstmcell.W_oc + dW_oc;
lstmcell.b_o = lstmcell.b_o + db_o;
   
end