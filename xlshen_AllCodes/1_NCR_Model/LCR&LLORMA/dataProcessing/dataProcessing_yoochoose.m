%%
tic;
fileID = fopen('Yoochoose[session_item_type].txt');
fmt = repmat('%f',1,3);
rawData = textscan(fileID,fmt,'Delimiter',',','HeaderLines',1);
fclose(fileID);
rawData = cell2mat(rawData);
t = toc;
fprintf('step1 - load raw data and convert completed, row cmount: %d, time: %f\n',size(rawData,1),t);

%%
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
rawData = sortrows(rawData,[1,2]);
t=toc;
fprintf('step2 - renumbering and generating data completed, userNum: %d,itemNum: %d, time: %f\n',userNum,itemNum,t);
fid=fopen('Yoochoose.txt','w');   %新建一个txt文件存放结果
fprintf(fid,'%d,%d,%d\n',rawData');
fclose(fid);
