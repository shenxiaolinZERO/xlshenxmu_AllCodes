tic;

 load('D:\111-Matlab-Code\0630NCR\NCR\session.mat')
 fprintf('Start session_buy_nobuy ����\n');


seed = 1;

fprintf('Starting run FSBPR_NCR ����\n ');
out = FSBPR_NCR(seed,session);

t=toc;
fprintf('total_time: %f\n',t);