function sae=saetrain(sae,x,opts)
    for i=1:numel(sae.ae)
        sae.ae{i}=nntrain(sae.ae{i},x,x,opts);
        t=nnff(sae.ae{i},x,x);
        x=t.a{2};
    end
end 