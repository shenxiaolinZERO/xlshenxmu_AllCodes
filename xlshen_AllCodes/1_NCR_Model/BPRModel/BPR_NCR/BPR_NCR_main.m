tic;

load('D:\111-Matlab-Code\0630NCR\NCR\NCR_V1_ratingDataset\data\session_MovieLens_new_small.mat')
fprintf('Start session_MovieLens_new_small бнбн\n'); 
seed = 1;


fprintf('Starting run  BPR бнбн\n ');
out =BPR_NCR(seed,session);


t=toc;
fprintf('total_time: %f\n',t);
