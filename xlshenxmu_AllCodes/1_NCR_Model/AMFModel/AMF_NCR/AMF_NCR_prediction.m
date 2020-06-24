function out = AMF_NCR_prediction(rawData,testIdx,U,V,yItem,theta,par)
out = nan*ones(1,5); %AUC/NDCG/MRR/RMSE/MAE
EvalMetric=EvaluationMetric;
rawData = rawData(testIdx,:);
rawData = [rawData,zeros(size(rawData,1),1)];

[userSet,userP] = numunique(rawData(:,1));  
for i = 1:length(userSet)
    rawData(userP{i},1) = i;
end

userLen = arrayfun(@(x) length(userP{x}),1:length(userSet)); 
sqrt_U = sqrt(userLen);
for j = 1:length(rawData)
     rawData(j,4) =sqrt_U(rawData(j,1));
end

yU = arrayfun(@(x)  calyU(rawData(userP{x},:),U,yItem),(1:length(userSet))','UniformOutput',false);

yU = cell2mat(yU)./sqrt_U'; 
u=yU(rawData(:,1),:);
v=V(rawData(:,2),:);
predRating =zeros(length(testIdx),1);
for m =1:par.F
    x1 =exp(u(:,m))./sum(exp(u),2);
    x2 = exp(theta).*v(:,m)+sum(v,2)-v(:,m);
    predRating = predRating+x1.*x2;
end

rawData(:,5) = predRating; 

% 计算 AUC/NDCG/MRR三个指标 :
temp_out = nan*ones(length(userSet),5);
for i=1:length(userSet)
    sample = rawData(userP{i},:);
    item = sample(:,2);
    rating = sample(:,3);
    pred = sample(:,5); 
    %auc
    target = zeros(length(rating),1);
    target(rating>3) = 1;
    temp_out(i,1) = EvalMetric.aucEval(target,pred);
    %ndcg
    [~,pred_idx] = sort(pred,'descend');
    temp_out(i,2) = EvalMetric.ndcgEval(item(pred_idx),item(rating>3),par.topN);
    %mrr
    temp_out(i,3) = EvalMetric.mrrEval(item(pred_idx),item(rating>3),par.topN);
    
    temp_out(i,4) = EvalMetric.rmseEval(pred,rating);
    temp_out(i,5) = EvalMetric.maeEval(pred,rating);
    
    if isnan(temp_out(i,4))
        a=1;
    end
    
end

for i=1:5
    del_idx = isnan(temp_out(:,i));
    temp_out(del_idx,:) = [];
end

out(1) = mean(temp_out(:,1));
out(2) = mean(temp_out(:,2));
out(3) = mean(temp_out(:,3));
out(4) = mean(temp_out(:,4));
out(5) = mean(temp_out(:,5));
end

function s  = predFunc(u,v,theta,F)
s = zeros(1,size(v,1));
    for z=1:F
        x=exp(u(:,z))./sum(exp(u),2);  
        y=exp(theta).*v(:,z)+sum(v,2)-v(:,z);
        s= s + (x.*y);
    end
end

function r = calyU(sess,U,yItem)
u = U(sess(1,1),:); 
v= yItem(sess(:,2),:);
r = sum(v,1);
end
