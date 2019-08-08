clear

str1='data2/NCI1/NCI1_A.txt';
str2='data2/NCI1/NCI1_graph_indicator.txt';
str4='data2/NCI1/NCI1_graph_labels.txt';


for timesfold=1:10  % 10 times 10fold cv
    


    for fold=1:10  

    str3=sprintf('data2/NCI1/10fold_idx/train_idx-%dtimes%d.txt', fold,timesfold);

    str5=sprintf('data2/NCI1/10fold_idx/test_idx-%dtimes%d.txt', fold,timesfold);
    [NCItraingraphlabelandsequence2] = datapre(str1,str2,str3,str4);
    [NCItestgraphlabelandsequence2] = datapre(str1,str2,str5,str4);
    M=111;
    [trainlabelandsequence,maxleng1,averleng1] = shendupre(NCItraingraphlabelandsequence2,M);
    [testlabelandsequence,maxleng2,averleng2] = shendupre(NCItestgraphlabelandsequence2,M);

    save(sprintf('matforpy10times/NCI1/NCI1cv%dtraintimes%d.mat', fold,timesfold),'trainlabelandsequence');
    save(sprintf('matforpy10times/NCI1/NCI1cv%dtesttimes%d.mat', fold,timesfold),'testlabelandsequence');
    end

end



