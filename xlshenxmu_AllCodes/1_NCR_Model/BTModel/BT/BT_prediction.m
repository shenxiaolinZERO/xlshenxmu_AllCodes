function out =BT_prediction(session,testIdx,U,V,par)
out = nan*ones(length(testIdx),8);  
EvalMetric=EvaluationMetric;
for i = 1:length(testIdx)
    sample = session{testIdx(i)};
    u = sample.user;
    u_k =U(u,:);
    correctItems = sort(sample.buy(1,:));
    candItems = [sample.noBuy(1,:),sample.buy(1,:)];

    s = predItemP(candItems,u_k,V);
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
idx = isnan(out(:,1));
out(idx,:) = [];
out = mean(out);
end

function r = predItemP(items,u,V)
r = zeros(1,length(items));
w = V(items,:);

comparePair = ones(length(items)-1,2);
comparePair(:,2) = 2:length(items);

for i = 1:length(r)
    w([1,i],:) = w([i,1],:);
    r(i) = prod( sum( u.*w(comparePair(:,1),:) )./sum( u.*w(comparePair(:,1),:) + u.*w(comparePair(:,2),:)) );
end
end

function r = pred(items,g,V,theta)
r = zeros(1,length(items));
w = V(items,:);
w(:,[1,g]) = w(:,[g,1]);
comparePair = ones(length(items)-1,2);
comparePair(:,2) = 2:length(items);

for i = 1:length(r)
    w([1,i],:) = w([i,1],:);
     r_log = sum(log(w(comparePair(:,1),1))-log((w(comparePair(:,1),1)+theta*w(comparePair(:,2),1)))+...
        sum(log(theta*w(comparePair(:,1),2:end))-log((w(comparePair(:,2),2:end)+theta*...
        w(comparePair(:,1),2:end))),2));
    r(i) = exp(r_log);

end
end
