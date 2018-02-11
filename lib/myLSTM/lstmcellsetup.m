function lstmcell = lstmcellsetup(inputlen, outputlen, opt)

    lstmcell.inputlen = inputlen;
    lstmcell.outputlen = outputlen;
    lstmcell.delta = opt.active_funcs{1}; 
    lstmcell.g = opt.active_funcs{2}; 
    lstmcell.learningRate = opt.learningRate;
    lstmcell.momentum = opt.momentum;
    lstmcell.weightPenaltyL1 = 0;
    lstmcell.weightPenaltyL2 = 0;
    lstmcell.weightPenaltyGroup = 0;
    lstmcell.scaling_learningRate = opt.scaling_learningRate;
  
%%initializaton of weights and bias
    %i_t
    lstmcell.W_ix=(rand(outputlen,inputlen)-0.5)/inputlen;
    lstmcell.W_ih=(rand(outputlen,outputlen)-0.5)/outputlen;
    lstmcell.W_ic=(rand(outputlen,1)-0.5)/outputlen;
    lstmcell.b_i=zeros(1,outputlen);
    
    %f_t
    lstmcell.W_fx=(rand(outputlen, inputlen) - 0.5) / inputlen;
    lstmcell.W_fh=(rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.W_fc=(rand(outputlen, 1) - 0.5) / outputlen; 
    lstmcell.b_f=ones(1,outputlen);
    
    %c_t
    lstmcell.W_cx=(rand(outputlen, inputlen) - 0.5) / inputlen;
    lstmcell.W_ch=(rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.b_c=zeros(1,outputlen);
    
    %o_t
    lstmcell.W_ox=(rand(outputlen, inputlen) - 0.5) / inputlen;
    lstmcell.W_oh=(rand(outputlen, outputlen) - 0.5) / outputlen;
    lstmcell.W_oc=(rand(outputlen, 1) - 0.5) / outputlen;
    lstmcell.b_o=zeros(1,outputlen);
         
    if lstmcell.momentum  > 0
        lstmcell.vW_ix = zeros(size(lstmcell.W_ix));
        lstmcell.vW_ih = zeros(size(lstmcell.W_ih));
        lstmcell.vW_ic = zeros(size(lstmcell.W_ic));
        lstmcell.vb_i = zeros(size(lstmcell.b_i));

        lstmcell.vW_fx = zeros(size(lstmcell.W_fx));
        lstmcell.vW_fh = zeros(size(lstmcell.W_fh));
        lstmcell.vW_fc = zeros(size(lstmcell.W_fc));
        lstmcell.vb_f = zeros(size(lstmcell.b_f));

        lstmcell.vW_cx = zeros(size(lstmcell.W_cx));
        lstmcell.vW_ch = zeros(size(lstmcell.W_ch));
        lstmcell.vb_c = zeros(size(lstmcell.b_c));

        lstmcell.vW_ox = zeros(size(lstmcell.W_ox));
        lstmcell.vW_oh = zeros(size(lstmcell.W_oh));
        lstmcell.vW_oc = zeros(size(lstmcell.W_oc ));
        lstmcell.vb_o = zeros(size(lstmcell.b_o));
        
    end
    
end