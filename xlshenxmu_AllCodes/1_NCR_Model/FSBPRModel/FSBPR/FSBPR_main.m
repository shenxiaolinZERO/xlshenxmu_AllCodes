tic;

load('D:\111-Matlab-Code\0630NCR\NCR\NCR_V1_ratingDataset\data\session_FilmTrust_new.mat')  
fprintf('Start FilmTrust бнбн\n');

seed = 1;

fprintf('Starting run FSBPR_baseline бнбн\n ');
out = FSBPR(seed,session);

t=toc;
fprintf('total_time: %f\n',t);