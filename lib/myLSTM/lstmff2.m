function lstm = lstmff2(lstm,x,y)
    layer=lstm.layer;
    for k=1:layer
        lstm.lstmcell{k}=lstmcellff(lstm.lstmcell{k},x,y);                
        lstm.nn{k}=nnff(lstm.nn{k},lstm.lstmcell{k}.mh,zeros(size(x,1),lstm.nn{k}.size(end)));
        x=lstm.nn{k}.a{end};
    end

    lstm.error=y-lstm.nn{layer}.a{end};
end