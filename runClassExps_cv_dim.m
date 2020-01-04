%% Ionosphere
clear; clc;
rng(0);
warning('off','all');
dataset = 'Ionosphere';
sigmamin = 0.12; smin = 0.02; sigmamax = 6; smax = 0.87;
numsigmas = 10;
%generate sigmas for kernel methods
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m

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

% %% Lung
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'lung';
% sigmamin = 23.1; smin = 0.05; sigmamax = 110; smax = 0.77;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run ClassificationComparison_cv_dim.m

% %% Lymphoma
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'lymphoma';
% sigmamin = 20; smin = 0.03; sigmamax = 153; smax = 0.64;
% generate sigmas for kernel methods
%numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run ClassificationComparison_cv_dim.m
% 
% %% Glioma
% clear; clc;
% rng(0);
% warning('off','all');
% dataset = 'GLIOMA';
% sigmamin = 15; smin = 0.05; sigmamax = 200; smax = 0.87;
% % generate sigmas for kernel methods
% numsigmas = 10;
% sigmas = linspace(sigmamin, sigmamax, numsigmas);
% run ClassificationComparison_cv_dim.m

%% Basehock
clear; clc;
rng(0);
warning('off','all');
dataset = 'BASEHOCK';
sigmamin = 8; smin = 0.01; sigmamax = 80; smax = 0.98;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m

%% PCMAC
clear; clc;
rng(0);
warning('off','all');
dataset = 'PCMAC';
sigmamin = 8; smin = 0.014; sigmamax = 70; smax = 0.98;
% generate sigmas for kernel methods
numsigmas = 10;
sigmas = linspace(sigmamin, sigmamax, numsigmas);
run ClassificationComparison_cv_dim.m
clear; clc;

