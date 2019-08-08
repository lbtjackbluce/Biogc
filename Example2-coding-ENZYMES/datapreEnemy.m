function [traingraphlabeandsequence] = datapre(sourceegdgestr1,nodeinsmallgraphindexstr2,shiyanfengeshujistr3,graphlabelstr4)


NCI1egde=importdata(sourceegdgestr1);
NCI1adj=FormNet(NCI1egde);
% subNCI1adj=NCI1adj(1:30,1:30);
% NCI1adjfull=full(subNCI1adj);

nodeandgraphindex=importdata(nodeinsmallgraphindexstr2);
totalgraph=unique(nodeandgraphindex);

%% given smallgraphnumber then find corresponding smallgraph-small adj
%cross-fold 10 
traingraph=importdata(shiyanfengeshujistr3);
traingraph=traingraph+1; 

smallgraph={}; 
nodenuminsmallgraph={};
for i=1:length(traingraph)
    [row, col] = find(nodeandgraphindex==traingraph(i));
    nodenuminsmallgraph{i}=row;
    smallgraph{i}=NCI1adj(row,row);
end



graphtosequence={};
for j=1:length(smallgraph)

    
    smallgraphadj=cell2mat(smallgraph(j));
    smallgraphnodexuhao=cell2mat(nodenuminsmallgraph(j));
   
    InputMatrix=smallgraphadj;
    Num = size(InputMatrix,1);
    EdgeMatrix = [];
    for i = 1:Num
        Temp0 = find(InputMatrix(i,i+1:end));
        TempEdge = Temp0 + i;    
        LengthTemp = length(TempEdge);
        TempPoint = i*ones(LengthTemp,1);
        TempFinal = [TempPoint,TempEdge'];
        EdgeMatrix = [EdgeMatrix;TempFinal];
    end 
    s = EdgeMatrix(:,1);
    t = EdgeMatrix(:,2);
    G = graph(s,t);
    %plot(G)
    v0 = dfsearch(G,1);
    maxnodenum=max(size(G.Nodes));
    if length(v0)~=length(smallgraphadj)
        if length(v0)~=maxnodenum %length(smallgraphadj)
            
            while 1
            smallgraphnode=1:maxnodenum;
            shengyunode=setdiff(smallgraphnode',v0);
    %         tiaonode=randperm(length(smallgraphadj));
            tiaonode= shengyunode(1);
           
            v1 = dfsearch(G,tiaonode);
            newv=[v0;v1];
            v0=newv;
                if length(newv)==maxnodenum
                    
                     break  
                end
            end  
        end
    smallgraphnode=1:length(smallgraphadj);  
    shengyunode2=setdiff(smallgraphnode',v0);
    v02=[v0;shengyunode2];
    v0=v02;
    if length(v02)==length(smallgraphadj)
        
    end
    end
    
    
    if length(v0)==length(smallgraphadj)
        
    end 
    graphtosequence{j}=v0; 
end

allbiggraphsequence={};
for n=1:length(smallgraph)
    totalnodeinsmallgraph=length(cell2mat(smallgraph(n)));
    bianhao=1:totalnodeinsmallgraph;
    biggraphnodenum=cell2mat(nodenuminsmallgraph(n));
    smallgraphnumandbiggraphnum=[bianhao',biggraphnodenum];
    smallgraphsequencematrix=cell2mat(graphtosequence(n));
    numbermap=smallgraphnumandbiggraphnum;
    
    newsequence=[];
    for m=1:length(smallgraphsequencematrix)
        weizhi=smallgraphsequencematrix(m);
        dingyingbiggraphnum=numbermap(weizhi,2);
        newsequence=[newsequence,dingyingbiggraphnum];   
    end
   allbiggraphsequence{n}=newsequence;
end
traingraphandduiyingsequence={traingraph,allbiggraphsequence'};

%%
graphlabel=importdata(graphlabelstr4);
traingraphlabel=[];
for t=1:length(traingraph)
    eachonegraph=traingraph(t);
    smallgraphlabel=graphlabel(eachonegraph,1);
    traingraphlabel=[traingraphlabel,smallgraphlabel];
end
traingraphlabel=num2cell(traingraphlabel);
traingraphlabeandsequence={traingraphlabel',allbiggraphsequence'};

end

