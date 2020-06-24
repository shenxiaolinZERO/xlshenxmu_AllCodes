tic;

load('D:\111-Matlab-Code\0630NCR\NCR\NCR_V1_ratingDataset\data\session_FilmTrust_new.mat')
fprintf('Start session_FilmTrust_new бнбн\n');

seed = 1;

fprintf('Starting run BT_NCR бнбн\n ');
out =BT_NCR(seed,session);


t=toc;
fprintf('total_time: %f\n',t);
