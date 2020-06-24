function out = BPR_prediction(session,testIdx,U,V,par)
out = nan*ones(length(testIdx),5);
EvalMetric=EvaluationMetric;
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];
    s = U(u,:)*V(candItems,:)';
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