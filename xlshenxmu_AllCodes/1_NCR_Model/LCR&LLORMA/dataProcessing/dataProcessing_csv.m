%%
tic;
fileID = fopen('CiaoDVD_result02_5session.csv');
fmt = repmat('%f',1,7);
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
t=toc;
fprintf('step2 - renumbering and generating data completed,  time: %f\n',t);
fid=fopen('CiaoDVD.csv','w');   %新建一个txt文件存放结果
fprintf(fid,'%d,%d,%d,%d,%d,%d,%d\n',rawData');
fclose(fid);
