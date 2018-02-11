clear all;
clc;

seq_len = 240;
len_in = 10;
len_out = 30;
opt.active_funcs = {'sigm', 'sigm'};
opt.learningRate = 0.1;
opt.weightPenaltyL2 = 0.001;
opt.scaling_learningRate = 0.5;
opt.momentum=0.5;
lstmcell = lstmcellsetup(len_in, len_out, opt);

x = rand(seq_len, len_in);
y = rand(seq_len, len_out);

lstmcell = lstmcellff(lstmcell, x, y);
e = y - lstmcell.mh;
loss_1 = sum(sum(e .* e)) / 2 / seq_len;
lstmcell = lstmcellbp(lstmcell, e);

for i = 1:100
    lstmcell = lstmcellff(lstmcell, x, y);
    e = y - lstmcell.mh;
    loss(i) = sum(sum(e .* e)) / 2 / seq_len;
    lstmcell = lstmcellbp(lstmcell, -e);
    lstmcell = lstmcellupdate(lstmcell);
end
plot(loss);