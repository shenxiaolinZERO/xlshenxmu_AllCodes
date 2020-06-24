function out = AMF_NCR(seed,rawData,varargin)
rng(seed);
%% Parse parameters
params = inputParser;
params.addParameter('method','graded',@(x) ischar(x));
params.addParameter('F',5,@(x) isnumeric(x));
params.addParameter('lr',1,@(x) isnumeric(x)); 
params.addParameter('momentum',0.8,@(x) isnumeric(x)); 
params.addParameter('batchNum',10,@(x) isnumeric(x));   
params.addParameter('maxIter',100,@(x) isnumeric(x));
params.addParameter('K',5,@(x) isnumeric(x));
params.addParameter('adaptive',true,@(x) islogical(x));  
params.addParameter('topN',5,@(x) isnumeric(x));
params.parse(varargin{:});
par = params.Results;

%% Run AMF and use K-folds cross validation
methodSolver = str2func([par.method,'_solver']);

[userSet,p] = numunique(rawData(:,1));
for i = 1:length(userSet)
    rawData(p{i},1) = i;
end
par.m = max(rawData(:,1));
par.n = max(max(rawData(:,2)));
temp = arrayfun(@(x) rawData(x,1),(1:length(rawData))');
cvObj = cvpartition(temp,'KFold',par.K);
out = zeros(par.K,5);
for i = 1:cvObj.NumTestSets
    trainIdx = find(cvObj.training(i));
    testIdx = find(cvObj.test(i));
    %模型训练得到参数
    [U,V,yItem,theta] = feval(methodSolver,rawData,trainIdx,testIdx,par);
    filename = sprintf('AMF_NCR_FilmTrust_fold_%i_para_lr1.mat',i);
%     filename = sprintf('AMF_NCR_CiaoDVD_fold_%i_para_lr1.mat',i);
%     filename = sprintf('AMF_NCR_MovieLens_fold_%i_para_lr1.mat',i);
    save (filename,'testIdx','U','V','yItem','theta','par','-mat');%保存结果
    %模型测试得到指标结果
    out(i,:) = AMF_NCR_prediction(rawData,testIdx,U,V,yItem,theta,par);
    fprintf('%d/%d fold completed\n',i,cvObj.NumTestSets);
end
%---------- 保存结果---start-------
resname = sprintf('AMF_NCR_FilmTrust_5fold_result_lr1.mat');
% resname = sprintf('AMF_NCR_CiaoDVD_5fold_result_lr1.mat');
% resname = sprintf('AMF_NCR_Movielens_5fold_result_lr1.mat');

meanValue=mean(out);
allOut = out;
allOut(end+1,:)=meanValue;  
result = sprintf('Final Results : AUC = %f, NDCG = %f, MRR = %f, RMSE = %f, MAE = %f \n',mean(out));
save (resname,'out','meanValue','allOut','result','-mat');% 保存结果
%---------- 保存结果---end-------
fprintf('Final Results : AUC = %f, NDCG = %f, MRR = %f, RMSE = %f, MAE = %f \n',mean(out));
end

function [U,V,yItem,theta] = graded_solver(rawData,trainIdx,testIdx,par)
trainData = rawData(trainIdx,:);
trainData = trainData(randperm(size(trainData,1)),:);
batchIdx = discretize(1:size(trainData,1),par.batchNum);  

[~,p] = numunique(batchIdx); 
fprintf('generate data completed\n');

U = normrnd(0,0.1,par.m,par.F); 
V = normrnd(0,0.1,par.n,par.F); 
yItem = normrnd(0,0.1,par.n,par.F);  
theta = 0.5;

incU = zeros(par.m,par.F); % 
incV = zeros(par.n,par.F); 

incyItem = zeros(par.n,par.F);  
incTheta = 0;


lastLoss = 0;

tempUU=rand(par.m,par.F);
tempVV=rand(par.n,par.F);
tempyItem=rand(par.n,par.F);

[userSet,userP] = numunique(trainData(:,1));  
userLen = arrayfun(@(x) length(userP{x}),1:length(userSet));
sqrt_U = sqrt(userLen);

for i = 1:length(trainData)
     trainData(i,4) =sqrt_U(trainData(i,1));
end

yU = arrayfun(@(x)  calyU(trainData(userP{x},:),U,yItem),(1:length(userSet))','UniformOutput',false);

yU = cell2mat(yU)./sqrt_U'; 
U = yU;

for i = 1:par.maxIter
    loss = 0;
    for j = 1:par.batchNum 

        % pred 预测评分
        ixTheta = zeros(length(trainData(p{j})),1);
        x1 = zeros(length(trainData(p{j})),par.F);
        x2 = zeros(length(trainData(p{j})),par.F);
        u = U(trainData(p{j},1),:);
        v = V(trainData(p{j},2),:);
        
        ixV = zeros(length(trainData(p{j})),par.F);
        for m = 1:par.F
            x1(:,m) = exp(u(:,m))./sum(exp(u),2);
            x2(:,m) = exp(theta).*(v(:,m))+sum(v,2)-v(:,m);                
            ixV(:,m) = (exp(u(:,m)).*exp(theta)+sum(exp(u),2)-exp(u(:,m)))./sum(exp(u),2);                     
            ixTheta = ixTheta + (x1(:,m).*(v(:,m)).*exp(theta));    
        end

         ixyItem = x2.*(x1-x1.^2).*yItem(trainData(p{j},2),:)./trainData(p{j},4); 

        pred = sum(x1.*x2,2);


  
        error = pred-trainData(p{j},3);
        loss = loss+sum(error.^2);

        ixV = error.*ixV;
        y_i = arrayfun(@(x)  sgdY(trainData(userP{x},:),U),(1:length(userSet))','UniformOutput',false);
        y_i = cell2mat(y_i);

        ixyItem = error.*ixyItem;
        ixTheta = error.*ixTheta;
        
        gU = zeros(par.m,par.F);
        gV = zeros(par.n,par.F);

        gyItem = zeros(par.n,par.F);
        gTheta = sum(ixTheta);

        for z = 1:length(p{j})
            gV(trainData(p{j}(z),2),:) = gV(trainData(p{j}(z),2),:)+ixV(z,:);
            gyItem(trainData(p{j}(z),2),:) = gyItem(trainData(p{j}(z),2),:)+ixyItem(z,:);         
        end
    
        incU = par.momentum*incU+par.lr*gU/length(p{j});
        incV = par.momentum*incV+par.lr*gV/length(p{j});
        incyItem = par.momentum*incyItem+par.lr*gyItem/length(p{j});
        incTheta = par.momentum*incTheta+par.lr*gTheta/length(p{j});
        % -----
        U = U - incU;
        V = V - incV;
        yItem = yItem - incyItem;
        theta = theta - incTheta;

        if theta <0
            theta =0.1;
        end

        loss = loss;
        
    end
    deltaLoss = lastLoss-0.5*loss;
    
     %----------------------收敛条件:参数--------------------
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
    %----------------------收敛条件:参数
    
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
    fprintf('AMF_NCR: iter [%d/%d] completed, theta = %f,loss = %f, delta_loss: %f, lr: %f\n',i,par.maxIter,theta,0.5*loss,deltaLoss,par.lr);
    out = AMF_NCR_prediction(rawData,testIdx,U,V,yItem,theta,par);
    fprintf('AMF_NCR : iter [%d/%d] completed,deltayItem = %f,deltaV = %f\n',i,par.maxIter,abs(deltayItem),abs(deltaV));
    fprintf('AUC = %f, NDCG = %f, MRR = %f, RMSE = %f, MAE = %f \n',out);
end
end

function r = calyU(sess,U,yItem)
u = U(sess(1,1),:); 
v= yItem(sess(:,2),:);
r = sum(v,1);
end

function r = sgdY(sess,U)
u = U(sess(1,1),:);
v = sess(:,2);
r = v;
end