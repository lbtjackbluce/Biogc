function [smallgraphlabelandsequence,maxleng,averleng] = shendupre(NCItraingraphlabelandsequence,M)

smallgraphlabel=NCItraingraphlabelandsequence{1,1};
[nrows1,ncols1]=size(NCItraingraphlabelandsequence{1,1});
smallgraphlabel=cell2mat(smallgraphlabel);


smallgraphsequence=NCItraingraphlabelandsequence{1,2};
[nrows2,ncols2]=size(smallgraphsequence);

allchangdu=[];
for i=1:nrows2
    changdu=length(smallgraphsequence{i,1});
    allchangdu=[allchangdu,changdu];
end
maxleng=max(allchangdu);
averleng=mean(allchangdu,2); 

allsmallgraphafterbuqi=zeros(nrows2,M);
for i=1:nrows2
    a0=smallgraphsequence{i,1};
    if M>length(a0)
        smallgraphafterbuqi=[a0, zeros(1,M-length(a0))];
    end
    if M==length(a0)
        smallgraphafterbuqi=a0;
    end
    if M<length(a0)
        smallgraphafterbuqi=a0(1:M);
    end
    allsmallgraphafterbuqi(i,:)=smallgraphafterbuqi;
end


smallgraphlabelandsequence=[smallgraphlabel,allsmallgraphafterbuqi];
end

