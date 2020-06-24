function out = MF_prediction(rawData,testIdx,U,V,par)
out = nan*ones(1,5);
EvalMetric=EvaluationMetric;
rawData = rawData(testIdx,:);
rawData = [rawData,zeros(size(rawData,1),1)];
rawData(:,4) = sum(U(rawData(:,1),:).*V(rawData(:,2),:),2);
[userSet,p] = numunique(rawData(:,1));
temp_out = nan*ones(length(userSet),3);
for i=1:length(userSet)
    sample = rawData(p{i},:);
    item = sample(:,2);
    rating = sample(:,3);
    pred = sample(:,4);  
    [~,pred_idx] = sort(pred,'descend');
    target = zeros(length(rating),1);
    target(rating>3) = 1;
    if sum(target~=zeros(length(rating),1))==0 || sum(target~=ones(length(rating),1))==0
        target = zeros(length(rating),1);
        for j=1:floor(length(rating)/2)
            target(pred_idx==j)=1;
        end
    end
    temp_out(i,1) = EvalMetric.aucEval(target,pred);
    temp_out(i,2) = EvalMetric.ndcgEval(item(pred_idx),item(rating>3),par.topN);
    temp_out(i,3) = EvalMetric.mrrEval(item(pred_idx),item(rating>3),par.topN);
    temp_out(i,4) = EvalMetric.rmseEval(pred,rating);
    temp_out(i,5) = EvalMetric.maeEval(pred,rating);

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
