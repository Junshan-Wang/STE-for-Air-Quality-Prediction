function [index,c]=KMeans(data,k)
    [m, n]=size(data);
    c=zeros(k,n);
    for i=1:n
        maxi=max(data(:,i));
        mini=min(data(:,i));
        for j=1:k
            c(j,i)=mini+(maxi-mini)*rand();
        end
    end
    %disp(c);
    
    times=0;
    while 1
        index=zeros(m,1);
        dist=zeros(m,1);
        cc=zeros(k,n);
        numcc=zeros(k,1);
        for i=1:m
            d=[];
            for j=1:k                
                %ÂíÊÏ¾àÀë?
                %x1=c(j,:); x2=data(i,:);  disp(x1); disp(x2); d=[d pdist([x1;x2],'mahalanobis')];  %sqrt((x1-x2)*inv(cov([x1;x2]))*(x1-x2)')];
                %Ïà¹ØÏµÊı
                %corr=corrcoef(c(j,:),data(i,:)); d=[d (1-corr(1,2))];  
                %d=[d pdist2(c(j,:),data(i,:),'correlation')];
                %Å·Ê½¾àÀë
                %disp(norm(c(j,:)-data(i,:)));
                d=[d norm(c(j,:)-data(i,:))];
                %Âü¹ş¶Ù¾àÀë
                %d=[d mean(c(j,:)-data(i,:))];
                %ÓàÏÒ¾àÀë
                %d=[d pdist2(c(j,:),data(i,:),'cosine')];
                %dtw¶¯Ì¬¹æ»®
                %d=[d dtw(c(j,:),data(i,:))];
                
                %corr=corrcoef(c(j,:),data(i,:)); 
                %d=[d (1-corr(1,2))+norm(c(j,:)-data(i,:))];  
            end
            [dist(i,1),index(i,1)]=min(d);
            cc(index(i,1),:)=cc(index(i,1),:)+data(i,:);
            numcc(index(i,1),1)=numcc(index(i,1),1)+1;
        end
        
        for i=1:k
            if numcc(i,1)>0 
                cc(i,:)=cc(i,:)/numcc(i,1);
            else                
                [t1,t2]=max(dist);
                %disp(t2);
                cc(i,:)=data(t2,:)+rand(1,n);
            end                
        end

        %disp(c); disp(cc);
        %disp(c-cc);
        if sum(sum((cc-c).^2))<0.00001 || times>100
            %disp(sum(sum((cc-c).^2)));
            %disp(times);
            break;
        else
            c=cc;
            times=times+1;
        end 
        
    end
end