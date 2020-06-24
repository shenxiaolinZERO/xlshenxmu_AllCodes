function [U,V,yItem] = FSBPR(seed,session,varargin)
rng(seed)
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('lr',5,@(x) isnumeric(x));
params.addParameter('regU',0.01,@(x) isnumeric(x));
params.addParameter('regV',0.01,@(x) isnumeric(x));
params.addParameter('momentum',0.8,@(x) isnumeric(x));
params.addParameter('batchNum',10,@(x) isnumeric(x));
params.addParameter('maxIter',100,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));
params.addParameter('earlyStop',true,@(x) islogical(x));
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;
par.m = session{end}.allUser;
par.n = session{end}.allItem;
session(end) = [];
%% Run FSBPR_baseline and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);
temp = arrayfun(@(x) session{x}.user,(1:length(session))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,8);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    [U,V,yItem] = feval(methodSolver,session,trainIdx,testIdx,par);
    filename = sprintf('FSBPR_Yoochoose_fold_%i.mat',i);
    save (filename,'testIdx','U','V','yItem','par','-mat');
    out(i,:) = FSBPR_prediction(session,testIdx,U,V,yItem,par);
 
    fprintf('FSBPR : fold [%d/%d] completed\n',i,cvObj.NumTestSets);
end

resname = sprintf('FSBPR_Yoochoose_5fold_result.mat');
meanValue=mean(out);
allOut = out;
allOut(end+1,:)=meanValue;  
result = sprintf('FSBPR Final Results: AUC=%f,Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',mean(out));
save (resname,'out','meanValue','allOut','result','-mat');

fprintf('FSBPR Final Results: AUC=%f,Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',mean(out));
end

function [U,V,yItem] = graded_solver(session,trainIdx,testIdx,par)
D = cell(length(trainIdx),1);
for i = 1:length(trainIdx)
    sample = session{trainIdx(i)};
    buyItem = sample.buy(1,:);
    noBuyItem = sample.noBuy(1,:);
    comparePair = combvec(buyItem,noBuyItem);
    D{i} = [repmat(sample.user,[size(comparePair,2),1]),comparePair'];
end
D = cell2mat(D);
batchIdx = discretize(1:size(D,1),par.batchNum); 
[~,p] = numunique(batchIdx);
fprintf('generate data completed\n');
U = rand(par.m,par.F);
V = rand(par.n,par.F);
yItem =  rand(par.n,par.F); 

incU = zeros(par.m,par.F);
incW = zeros(par.n,par.F);
incV = zeros(par.n,par.F);
incyItem = zeros(par.n,par.F);  
incyItemW = zeros(par.n,par.F);
incyItemV = zeros(par.n,par.F);
bestAUC = 0;
bestPr = 0;
loseNum = 0;
lastLoss = 0;
%1110
tempUU=rand(par.m,par.F);
tempVV=rand(par.n,par.F);
tempyItem=rand(par.n,par.F);

[userSet,userP] = numunique(D(:,1));  
userLen = arrayfun(@(x)  calyULen(D(userP{x},:)),(1:length(userSet))','UniformOutput',false);
userLen = cell2mat(userLen);
sqrt_U = sqrt(userLen);

for i = 1:length(D)
     D(i,4) =sqrt_U(D(i,1));
end

uOnly = arrayfun(@(x)  calyU(D(userP{x},:),U,yItem),(1:length(userSet))','UniformOutput',false);     
uOnly = cell2mat(uOnly)./sqrt_U; 
U = uOnly;

for i = 1:par.maxIter
    loss = 0;
    for j = 1:par.batchNum
        winPred = sum(U(D(p{j},1),:).*V(D(p{j},2),:),2); 
        losePred = sum(U(D(p{j},1),:).*V(D(p{j},3),:),2);
        compareDiff = logsig(losePred-winPred);
        loss = loss+sum(-log(logsig(winPred-losePred)));

        ixyItemW = -compareDiff.*(V(D(p{j},2),:)-V(D(p{j},3),:)).*U(D(p{j},1),:)+par.regU*yItem(D(p{j},2),:);
        ixyItemV = -compareDiff.*(V(D(p{j},2),:)-V(D(p{j},3),:)).*U(D(p{j},1),:)+par.regU*yItem(D(p{j},3),:);
         
        ixW = -compareDiff.*U(D(p{j},1),:)+par.regV*V(D(p{j},2),:);
        ixV = -compareDiff.*(-U(D(p{j},1),:))+par.regV*V(D(p{j},3),:);
        
        gW = zeros(par.n,par.F);
        gV = zeros(par.n,par.F);
        gyItemW = zeros(par.n,par.F);
        gyItemV = zeros(par.n,par.F);
        
        for z = 1:length(p{j})
            gW(D(p{j}(z),2),:) = gW(D(p{j}(z),2),:)+ixW(z,:);
            gV(D(p{j}(z),3),:) = gV(D(p{j}(z),3),:)+ixV(z,:);
            gyItemW(D(p{j}(z),2),:) = gyItemW(D(p{j}(z),2),:)+ixyItemW(z,:);
            gyItemV(D(p{j}(z),3),:) = gyItemV(D(p{j}(z),3),:)+ixyItemV(z,:);           
        end

        incW = par.momentum*incW+par.lr*gW/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incyItemW = par.momentum*incyItemW+par.lr*gyItemW/length(p{j});
        incyItemV = par.momentum*incyItemV+par.lr*gyItemV/length(p{j});
    
        incW = par.momentum*incW+par.lr*gW/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incyItemW = par.momentum*incyItemW+par.lr*gyItemW/length(p{j});
        incyItemV = par.momentum*incyItemV+par.lr*gyItemV/length(p{j});

        V = V - incW;
        V = V - incV;
        yItem = yItem - incyItemW;
        yItem = yItem - incyItemV;
  
        loss = loss+par.regU*sum(sum(U(D(p{j},1),:).^2))+par.regV*sum(sum(V(D(p{j},2),:).^2))+...
            par.regV*sum(sum(V(D(p{j},3),:).^2));

    end  
    deltaLoss = lastLoss-0.5*loss;

    cU = (tempUU-U).^2;
    deltaU = sqrt(sum(cU(:)));
    tempUU = U;
    
    cyItem= (tempyItem-yItem).^2;
    deltayItem = sqrt(sum(cyItem(:)));
    tempyItem = yItem;
    
    cV = (tempVV-V).^2;
    deltaV = sqrt(sum(cV(:)));
    tempVV=V;
    
    if abs(deltayItem)<1e-4 ||  abs(deltaV)<1e-4
        break;
    end
    out = FSBPR_prediction(session,testIdx,U,V,yItem,par);
  
    if par.adaptive && i > 2
        if lastLoss > 0.5*loss
            par.lr = 1.05*par.lr;
        else
            par.lr = 0.5*par.lr;
        end
        lastLoss = 0.5*loss;
    else
        lastLoss = 0.5*loss;
    end
    
    fprintf('FSBPR : iter [%d/%d] completed, loss: %f, delta_loss = %f, lr: %f\n',i,par.maxIter,0.5*loss,deltaLoss,par.lr);
    fprintf('FSBPR : iter [%d/%d] completed,deltayItem = %f,deltaV = %f\n',i,par.maxIter,abs(deltayItem),abs(deltaV));
    fprintf('AUC=%f,Pr = %f, Re = %f, MAP = %f, Ndcg = %f, MRR = %f, oPr = %f, oMRR = %f\n',out);
  
end
end

function r = calyULen(sess)
u = sess(:,1);
item1 = sess(:,2);
item2 = sess(:,3);

allItem = [item1', item2']; 
unique_allItem = unique(allItem); 

r = length(unique_allItem);
end

function r = calyU(sess,U,yItem)
u = U(sess(1,1),:);
wItem = sess(:,2);
vItem = sess(:,3);

allItem = [wItem', vItem']; 
unique_allItem = unique(allItem); 

v = yItem(unique_allItem,:);
r = sum(v,1);
end