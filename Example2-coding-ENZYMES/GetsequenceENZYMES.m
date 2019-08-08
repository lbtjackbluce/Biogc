clear

str1='data2/ENZYMES/ENZYMES_A.txt';
str2='data2/ENZYMES/ENZYMES_graph_indicator.txt';
str4='data2/ENZYMES/ENZYMES_graph_labels.txt';

for timesfold=1:10  % 10 times 10-fold cv

for fold=1:10
% str3=sprintf('data/ENZYMES/10fold_idx/train_idx-%d.txt', fold);
str3=sprintf('data2/ENZYMES/10fold_idx/train_idx-%dtimes%d.txt', fold,timesfold);
str5=sprintf('data2/ENZYMES/10fold_idx/test_idx-%dtimes%d.txt', fold,timesfold);

[NCItraingraphlabelandsequence1] = datapreEnemy(str1,str2,str3,str4);

[NCItestgraphlabelandsequence1] = datapreEnemy(str1,str2,str5,str4);
M=126;
[trainlabelandsequence,maxleng1,averleng1] = shendupre(NCItraingraphlabelandsequence1,M);
[testlabelandsequence,maxleng2,averleng2] = shendupre(NCItestgraphlabelandsequence1,M);
save(sprintf('matforpy10times/ENZYMES/ENZYMEScv%dtraintimes%d.mat', fold,timesfold),'trainlabelandsequence');
save(sprintf('matforpy10times/ENZYMES/ENZYMEScv%dtesttimes%d.mat', fold,timesfold),'testlabelandsequence');
end
end


