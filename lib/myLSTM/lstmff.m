function lstm = lstmff(lstm,x,y)
    lstm.lstmcell=lstmcellff(lstm.lstmcell,x,y);
    lstm.nn=nnff(lstm.nn,lstm.lstmcell.mh,y);
    lstm.error=lstm.nn.e;
end