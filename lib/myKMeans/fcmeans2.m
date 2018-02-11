function [u,center] = fcmeans2(data,k)
    
    [m,n]=size(data);
    
    max_iter=1000;
    min_impro=0.0000001;
    expo=2;
    obj_new=0;
    
    u=rand(k,m);
    u=u./repmat(sum(u),k,1);
    
    for i=1:max_iter       
        mf=u.^expo;
        center=mf*data./repmat(sum(mf,2),1,n);
        
        dist=zeros(k,m);
        for j=1:k
            for l=1:m 
                %disp(pdist2(center(j,1:500),data(l,1:500),'correlation'));
                %disp(pdist2(center(j,501:502),data(l,501:502)));
                %dist(j,l)=pdist2(center(j,:),data(l,:),'correlation');%*pdist2(center(j,501:502),data(l,501:502));
                dist(j,l)=norm(center(j,:)-data(l,:));
                %dist(j,l)=dtw(center(j,:),data(l,:));              
            end
            %dist(j,:)=sqrt(sum(((data-repmat(center(j,:),m,1)).^2)'));
        end
        
        tmp=dist.^(-2/(expo-1));
        u=tmp./repmat(sum(tmp),k,1);
        
        obj_old=obj_new;
        obj_new=sum(sum((dist.^2).*mf));
        if abs(obj_new-obj_old)<min_impro
            break;
        end
        
    end
end