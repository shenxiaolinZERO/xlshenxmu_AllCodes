tic;

load('D:\CB0\1110Matlab\data\session_CiaoDVD_new.mat')
fprintf('Start session_CiaoDVD_new ����\n');

seed = 1;

fprintf('Starting run FSBPR_NCR ����\n ');
out = FSBPR_NCR(seed,session);

t=toc;
fprintf('total_time: %f\n',t);