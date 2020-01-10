% %% Residential
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'Residential';
% sigmamin = 0.01; smin = 0.14; sigmamax = 8; smax = 0.99;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m
% 
% %% Music
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'music';
% sigmamin = 0.07; smin = 0.03; sigmamax = 4; smax = 0.95;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m
% 
% %% BarshA
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'BarshanRegressionData_A';
% sigmamin = 0.01; smin = 0.05; sigmamax = 3; smax = 0.90;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m
% 
% %% DLBCL
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'DLBCL';
% sigmamin = 18; smin = 0.95; sigmamax = 200; smax = 0.67;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run RegressionComparison_cv_dim.m
% 
% %% Ionosphere
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'Ionosphere';
% sigmamin = 0.12; smin = 0.02; sigmamax = 6; smax = 0.87;
% numsigmas = 10;
% %generate sigmas for kernel methods
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run ClassificationComparison_cv_dim.m

%% Sonar
clear; clc;
rng(0);
warning('off','all');
dataset = 'Sonar';
sigmamin = 1; smin = 0.015; sigmamax = 20; smax = 0.835;
numsigmas = 10;
% generate sigmas for kernel methods
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m

%% Colon
clear; clc;
rng(0);
warning('off','all');
dataset = 'colon';
sigmamin = 14; smin = 0.05; sigmamax = 130; smax = 0.90;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m

%% Arcene
clear; clc;
rng(0);
warning('off','all');
dataset = 'Arcene';
sigmamin = 26; smin = 0.015; sigmamax = 300; smax = 0.81;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m
