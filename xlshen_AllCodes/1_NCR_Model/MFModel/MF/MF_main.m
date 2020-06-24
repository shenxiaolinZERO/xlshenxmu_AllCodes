tic;

seed = 1;

load('..\FilmTrust_u_i_r_3.mat')
fprintf('This is filmTrust 3......')
out =MF(seed,D3);

fprintf('Starting run MF бнбн\n ');

t=toc;
fprintf('total_time: %f\n',t);
