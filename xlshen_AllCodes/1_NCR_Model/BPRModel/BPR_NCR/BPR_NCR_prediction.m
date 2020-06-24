function out = BPR_NCR_prediction(session,testIdx,U,V,theta,par)
out = nan*ones(length(testIdx),5);
EvalMetric=EvaluationMetric;
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;    
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];
    s = pred(candItems,U(u,:),V,theta,par);
    [~,idx] = sort(s,'descend');
    rankedItems = candItems(idx);
    target = zeros(length(candItems),1)';
    target(length(sample.noBuy(1,:))+1:end) = 1;
    out(i,1) = EvalMetric.aucEval(target,s);
    out(i,2) = EvalMetric.ndcgEval(rankedItems,correctItems,par.topN);
    out(i,3) = EvalMetric.mrrEval(rankedItems,correctItems,par.topN);
    out(i,4) = EvalMetric.mapEval(rankedItems,correctItems,par.topN);
    out(i,5) = EvalMetric.prEval(rankedItems,correctItems,par.topN);
end
for i=1:5
    idx = isnan(out(:,i));
    out(idx,:) = [];
end
out = mean(out);
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
