tic;

seed = 1;

% load('filmTrust_u_i_r.mat')
% load('CiaoDVD_u_i_r.mat')
% load('MovieLens_u_i_r.mat')

%---1--
load('..\filmTrust_u_i_r.mat')
fprintf('This is filmTrust......')
%---2--
% load('..\CiaoDVD_u_i_r.mat')
% fprintf('This is CiaoDVD......')
%---3--
% load('..\MovieLens_u_i_r.mat')
% fprintf('This is MovieLens......')

fprintf('Starting run AMF бнбн\n ');
out =AMF(seed,D);


t=toc;
fprintf('total_time: %f\n',t);
