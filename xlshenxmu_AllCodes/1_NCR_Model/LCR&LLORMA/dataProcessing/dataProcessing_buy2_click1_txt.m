%% ����Ԥ�����Ϊ���²��衣
% ��ȡ���ݡ�
% ��ͬ�н�����һ�С�
% ȥ�����й���û�е�����С�
% ȥ������С��2������100��û�й����session��
% ����ѭ��ֱ��ÿ��userӵ�еķ�������session>5��ÿ��itemӵ�е�user>5��
% �����ļ���ͳ��user��session��item����
%% 1 ��ȡ���ݡ�
tic;
fileID = fopen('user_log_format1.csv');
fmt = repmat('%f',1,7); 
rawData = textscan(fileID,fmt,'Delimiter',',','HeaderLines',1); 
fclose(fileID);
rawData = cell2mat(rawData); 
rawData = rawData(:,[1,2,3,6,7]); 
[~,p] = numunique(rawData(:,end));  
rawData(p{1},end) = 1; %1: click
rawData(p{4},end) = 2; %2: add to favorite
rawData(p{2},end) = 3; %3: add to cart
rawData(p{3},end) = 4; %4: buy
clearvars -except rawData 
t = toc;
fprintf('step1 - load raw data and convert completed, row cmount: %d, time: %f\n',size(rawData,1),t); 
tic;
rawData = sortrows(rawData,5);  
[~,ia,ic] = unique(rawData(:,1:end-1),'rows','legacy'); 
[~,p] = numunique(ic); 
actionNum = arrayfun(@(x) length(p{x}),(1:length(p))'); 
rawData = rawData(ia,:); 
idx = rawData(:,end)>1; 
actionNum(idx) = actionNum(idx)-1; 

rawData(:,end+1) = actionNum; 
rawData(rawData(:,end)==0,:) = [];


[~,p]=numunique(rawData(:,end-1)); 
rawData(p{4},end-1) = 2;
rawData([p{2},p{3}],:) = [ ]; 

rawData = sortrows(rawData,[1,4]);  
clearvars -except rawData
t = toc;
fprintf('step2 - merge repeat row and count completed, row cmount: %d, time: %f\n',size(rawData,1),t); 
tic;
mask = nan*ones(size(rawData,1),1); 
sessIdx = 1;

[C,~,ic] = unique(rawData(:,[1,4]),'rows'); 
[~,p] = numunique(ic); 
sessLen = arrayfun(@(x) length(p{x}),1:length(p));
idx = find(sessLen>=2|sessLen<=100);
for i = 1:length(idx)
   
    isBuy = find(rawData(p{idx(i)},5)==2,1); 
    if ~isempty(isBuy)
        mask(p{idx(i)}) = sessIdx;
        sessIdx = sessIdx+1;
    end
end
sessIdx = sessIdx-1;
clearvars -except rawData mask sessIdx
t = toc;
fprintf('step3 - filter session completed, session amount: %d, time: %f\n',sessIdx,t);

tic;
rawData(isnan(mask),:) = [];
mask(isnan(mask)) = [];
temp1 = [rawData(:,1),mask]; 
temp2 = [rawData(:,2),mask]; 
k = 5;
while 1
    userMask = zeros(size(temp1,1),1);
    itemMask = zeros(size(temp2,1),1);
    sessMask = zeros(length(mask),1);
   
    [userSet,p] = numunique(temp1(:,1));
    userLen = arrayfun(@(x) length(unique(temp1(p{x},2))),1:length(userSet));
    removeUser = find(userLen<k); %
    for i = 1:length(removeUser) 
        userMask(p{removeUser(i)}) = 1;
    end
    
    
    [itemSet,p] = numunique(temp2(:,1));
    itemLen = arrayfun(@(x) length(p{x}),1:length(itemSet));
    removeItem = find(itemLen<k); 
    for i = 1:length(removeItem)
        itemMask(p{removeItem(i)}) = 1;
    end
    
    temp = find((userMask+itemMask)==0);
    [sessSet,p] = numunique(mask(temp));
    
    sessLen = arrayfun(@(x) length(p{x}),1:length(sessSet));
   
    buyLen = arrayfun(@(x) length(find(rawData(temp(p{x}),5)==2)),1:length(sessSet));
    removeSess = find(sessLen<2|buyLen==0|(sessLen-buyLen)==0);  
    for i = 1:length(removeSess)
        sessMask(temp(p{removeSess(i)})) = 1;
    end
    removeIdx = find((userMask+itemMask+sessMask)~=0);
    if isempty(removeIdx)
        break;
    else
        temp1(removeIdx,:) = [];
        temp2(removeIdx,:) = [];
        mask(removeIdx) = [];
        rawData(removeIdx,:) = [];
    end
end
t = toc;
fprintf('step4 - extract %d-core completed, session amount: %d, time: %f\n',k,length(unique(mask)),t);
clearvars -except rawData mask

tic;

[userSet,p] = numunique(rawData(:,1));
for i = 1:length(userSet)
    rawData(p{i},1) = i;
end
userNum = length(userSet); 


[itemSet,p] = numunique(rawData(:,2));
for i = 1:length(itemSet)
    rawData(p{i},2) = i;
end
itemNum = length(itemSet);

t=toc;
fprintf('step5 - renumbering and generating data completed, user count:%d,time: %f\n', userNum,t);
rawData = rawData(:,[1,2,5]); 
fid=fopen('Tmall_single.txt','w');   %�½�һ��txt�ļ���Ž��
fprintf(fid,'%d,%d,%d\n',rawData');
fclose(fid);
