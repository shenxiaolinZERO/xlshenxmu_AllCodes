function out = FSBPR_NCR_prediction(session,testIdx,U,V,yItem,theta,par)
out = nan*ones(length(testIdx),8);
EvalMetric=EvaluationMetric;
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];
    userLen = length(candItems); 
    sqrt_U = sqrt(userLen);
  
    yv= yItem(candItems,:);
    yU = sum(yv,1);  
    yU = yU/sqrt_U; 
    
    itemNum=length(candItems);

    x=repmat(yU,itemNum,1);
    y=V(candItems,:);  

    s = pred(candItems,U(u,:),V,theta,par);

    [~,idx] = sort(s,'descend');
    rankedItems = candItems(idx);
    
    label_0 = zeros(1,length(sample.noBuy(1,:)));
    label_1 = ones(1,length(sample.buy(1,:)));

    label = [label_0,label_1];
    
    out(i,1) = EvalMetric.aucEval(label,s);
    out(i,2) = EvalMetric.prEval(rankedItems,correctItems,par.topN);
    out(i,3) = EvalMetric.reEval(rankedItems,correctItems,par.topN);
    out(i,4) = EvalMetric.mapEval(rankedItems,correctItems,par.topN);
    out(i,5) = EvalMetric.ndcgEval(rankedItems,correctItems,par.topN);
    out(i,6) = EvalMetric.mrrEval(rankedItems,correctItems,par.topN);
    out(i,7) = EvalMetric.oPrEval(rankedItems,correctItems);
    out(i,8) = EvalMetric.oMrrEval(rankedItems,correctItems);
end

for i=1:8
    idx = isnan(out(:,i));
    out(idx,:) = [];
end

out = mean(out);
end

function s  = pred1(u,v,theta,F)
s = zeros(1,size(v,1));
for i=1:F
    s= s + (exp(u(i))./sum(exp(u)).*(exp(theta)*v(:,i)+sum(v,2)-v(:,i)));
end
end

function r = pred(items,u,V,theta,par)
r = zeros(1,length(items));
w = V(items,:);
for i = 1:par.F
    w(:,[1,i]) = w(:,[i,1]);
    r1 = exp(u(:,i))./sum(exp(u),2);
    r2 = exp(theta).*w(:,1)+sum(w(:,2:end),2);
    r = r + r1.*(r2');
end
end