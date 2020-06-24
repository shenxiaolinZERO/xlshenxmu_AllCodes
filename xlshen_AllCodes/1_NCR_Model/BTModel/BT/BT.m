function out = BT(seed,session,varargin)
rng(seed) 
%% Parse parameters
%设置默认参数以及更新参数
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x)); 
params.addParameter('maxIter',100,@(x) isnumeric(x)); 
params.addParameter('topN',5,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x)); 
params.addParameter('earlyStop',false,@(x) islogical(x));
params.parse(varargin{:}); 
par = params.Results;
par.m = session{end}.allUser; 
par.n = session{end}.allItem;
session(end) = []; 
%% Run BTL_baseline and use k-folds cross validation
methodSolver = str2func([par.method,'_solver']); 
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);  
out = zeros(par.K,8);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    [U,V] = feval(methodSolver,session,trainIdx,testIdx,par); 
    filename = sprintf('BT_FilmTrust_fold_%i_para.mat',i);
    
    save (filename,'testIdx','U','V','par','-mat');
    out(i,:) = BT_prediction(session,testIdx,U,V,par);
    fprintf('BT : fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end

resname = sprintf('BT_FilmTrust_5fold_result.mat');
meanValue=mean(out);
allOut = out;
allOut(end+1,:)=meanValue;  
result = sprintf('Final Results:AUC = %f, Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',mean(out));
save (resname,'out','meanValue','allOut','result','-mat');


fprintf('Final Results:AUC = %f, Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',mean(out));
end

function [U,V] = graded_solver(session,trainIdx,testIdx,par)
%模型函数
%初始化参数，注意归一化
temp = rand(par.m,par.F);  
U = temp./sum(temp,2);   
temp = rand(par.n,par.F); 
V = temp./sum(temp); 

temp = zeros(length(trainIdx),1);  
cellD = cell(length(trainIdx),1);   
for i = 1:length(trainIdx) 
    sample = session{trainIdx(i)};
    temp(i) = sample.user; 
    buyItem = sample.buy(1,:);  
    noBuyItem = sample.noBuy(1,:);  
       
    comparePair = combvec(buyItem,noBuyItem);  
 
    cellD{i} = [repmat(temp(i),[size(comparePair,2),1]),comparePair'];  
end


dLen = arrayfun(@(x) size(cellD{x},1),1:length(cellD));

[userSet,userP] = numunique(temp); 

userLen = arrayfun(@(x) length(userP{x}),1:length(userSet));
matD = cell2mat(cellD); 
[wItemSet,wP] = numunique(matD(:,2)); 
[lItemSet,lP] = numunique(matD(:,3));
AllItemSet = [matD(:,2),matD(:,3)];
AllItemSet =unique(AllItemSet);
idx = ismember(lItemSet,wItemSet); 

lItemSet(~idx) = [];  
lP(~idx) = []; 
fprintf('init completed\n')
bestAUC = 0;
loseNum = 0;
lastLoss = 0;
%1110
tempUU=rand(par.m,par.F);
tempVV=rand(par.n,par.F);

for i = 1:par.maxIter
    tic;
 
    [cT,fT] = cellfun(@(x) calCTFT(x,U,V,par),cellD,'UniformOutput',false); 

    cT = cell2mat(cT);
    fT = cell2mat(fT); 

    temp = arrayfun(@(x) sum(fT(userP{x},:))./sum(cT(userP{x},:)),(1:length(userSet))','UniformOutput',false);
    
    U(userSet,:) = cell2mat(temp);
    U = U./sum(U,2); 
    fprintf('update U completed\n')
   

     tempNumerator =  zeros(par.n,par.F);
     temp1 = arrayfun(@(x)  calVk(matD(wP{x},:),U,V,par,1),(1:length(wItemSet))','UniformOutput',false);
     tempNumerator(wItemSet,:) =  tempNumerator (wItemSet,:)+cell2mat(temp1);

     tempDenominator =  zeros(par.n,par.F);
     temp21 = arrayfun(@(x)  calVk(matD(wP{x},:),U,V,par,0),(1:length(wItemSet))','UniformOutput',false);
     tempDenominator(wItemSet,:) =  tempDenominator (wItemSet,:)+cell2mat(temp21); 
     temp22 = arrayfun(@(x)  calVk(matD(lP{x},:),U,V,par,0),(1:length(lItemSet))','UniformOutput',false);
     tempDenominator(lItemSet,:) =  tempDenominator (lItemSet,:)+cell2mat(temp22);
     V = tempNumerator./tempDenominator;

    V(isnan(V)) = min(V(:));
    V = V./sum(V);
    fprintf('update V completed\n')
  
    lh = cellfun(@(x) lhFunc(x,U,V,par),cellD,'UniformOutput',false); 
    lh = cell2mat(lh);
    lh = sum(log(lh));
    t=toc;
    deltaLoss = lastLoss-lh;
    lastLoss = lh;

    cU = (tempUU-U).^2;
    deltaU = sqrt(sum(cU(:)));
    tempUU = U;
    
    cV = (tempVV-V).^2;
    deltaV = sqrt(sum(cV(:)));
    tempVV=V;
    
    if abs(deltaU)<1e-4 ||  abs(deltaV)<1e-4
        break;
    end


    out = BT_prediction(session,testIdx,U,V,par); 

       fprintf('BT: iter [%d/%d] completed, time = %f,deltaU = %f,deltaV = %f\n',i,par.maxIter,t,abs(deltaU),abs(deltaV));
       fprintf('AUC = %f, Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',out);
end
end

function [cT,fT] = calCTFT(sess,U,V,par)   
u = U(sess(1,1),:);

w = V(sess(:,2),:);
v = V(sess(:,3),:);

 
 
cTBase =sum(u.*w,2) + sum(u.*v,2);
fTBase =sum(w,2);

 fT = zeros(size(v,1),par.F);
 cT =sum(( (w+v)./cTBase),1); 
 fT = sum(w ./fTBase,1); 

end

function r = calVk(sess,U,V,par,isWin)
u = U(sess(1,1),:); 
w = V(sess(:,2),:);
v = V(sess(:,3),:);
r = zeros(1,par.F); 
[cT,~] = calCTFT(sess,U,V,par);
eT = sum(u,2);

if isWin  
    eT = u./eT;
    r =  sum(eT,1);
else    
    r = sum(u./cT,1);
end
end

function result = lhFunc(sess,U,V,par)

u = U(sess(:,1),:);

w = V(sess(:,2),:);
v = V(sess(:,3),:);
temp = zeros(size(w,1),par.F);  
temp = temp+ sum(u.*w)./(sum(u.*w)+sum(u.*v));

p = ones(1,par.F);
for i = 1:size(temp,1)  
    tp = p.*temp(i,:);
    p = tp/sum(tp);  
end
result = p;
end