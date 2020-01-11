%% Residential
clear; clc;
rng(0);
warning('off','all');
dataset = 'Residential';
sigmamin = 0.01; smin = 0.14; sigmamax = 8; smax = 0.99;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run RegressionComparison_cv_dim_mle.m

%% Music
clear; clc;
rng(0);
warning('off','all');
dataset = 'music';
sigmamin = 0.07; smin = 0.03; sigmamax = 6; smax = 0.95;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run RegressionComparison_cv_dim_mle.m

%% BarshA
clear; clc;
rng(0);
warning('off','all');
dataset = 'BarshanRegressionData_A';
sigmamin = 0.01; smin = 0.05; sigmamax = 3; smax = 0.90;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run RegressionComparison_cv_dim_mle.m

%% DLBCL
clear; clc;
rng(0);
warning('off','all');
dataset = 'DLBCL';
sigmamin = 18; smin = 0.95; sigmamax = 200; smax = 0.67;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run RegressionComparison_cv_dim_mle.m

%% Bair Hard
clear; clc;
rng(0);
warning('off','all');
dataset = 'BairRegressionData_hard';
sigmamin = 10; sigmamax = 400;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run RegressionComparison_cv_dim_mle.m

% %% BarshB
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'BarshanRegressionData_B';
% sigmamin = 0.01; smin = 0.05; sigmamax = 3; smax = 0.77;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m
% 
% %% BarshC
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'BarshanRegressionData_C';
% sigmamin = 0.01; smin = 0.03; sigmamax = 3; smax = 0.64;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m