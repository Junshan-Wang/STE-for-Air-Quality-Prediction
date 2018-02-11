function lstm = lstmff3(lstm,x,y)
    lstm.lstmcell{1}=lstmcellff(lstm.lstmcell{1},x,y);                
    lstm.lstmcell{2}=lstmcellff(lstm.lstmcell{2},lstm.lstmcell{1}.mh,y);                
    lstm.nn=nnff(lstm.nn,lstm.lstmcell{2}.mh,zeros(size(x,1),lstm.nn.size(end)));


    lstm.error=y-lstm.nn.a{end};
end