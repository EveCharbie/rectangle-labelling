% Addapté du code de Fabien pour différence en SPM2D des données de EEG
% Permet de déterminer les différences entre des groupes de matrices (dans
% mon cas appliqué aux fixation de la toile)

clear all, close all ; clc

addpath(genpath(['/usr/local/MATLAB/R2019b/toolbox/fieldtrip-20220314']))



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Documentation sur la fonction ft_freqstatistics
% http://www.fieldtriptoolbox.org/reference/ft_freqstatistics
% http://www.fieldtriptoolbox.org/tutorial/cluster_permutation_freq
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



statcfg.channel = 'all' ;
statcfg.latency = 'all' ;
statcfg.frequency = 'all' ;
statcfg.avgovertime = 'no' ;
statcfg.avgchan = 'no' ;
statcfg.avgoverfreq = 'no' ;
statcfg.method = 'montecarlo' ;
statcfg.numrandomization = 1000 ; % devrait etre 1000
statcfg.statistic = 'depsamplesT' ;
statcfg.tail = 0 ;
statcfg.correcttail = 'prob' ;
statcfg.parameter = 'powspctrm' ;
statcfg.ivar = 2 ;
statcfg.uvar = 1 ;
statcfg.correctm = 'cluster' ;
statcfg.clusteralpha = 0.05 ;
statcfg.clustertail = 0 ;
statcfg.minnbchan = 0 ;
statcfg.clusterstatistic = 'maxsum' ;
statcfg.neighbours = [] ;



Subjects = {...
'P1','Sujet1';'P2','Sujet2';'P3','Sujet3';'P4','Sujet4' ;'P5','Sujet5';'P6','Sujet6';'P7','Sujet7';'P8','Sujet8';'P9','Sujet9';...
'P10','Sujet10';'P11','Sujet11';'P12','Sujet12';'P13','Sujet13';'P14','Sujet14';'P15','Sujet15';'P17','Sujet17';...
'P18','Sujet18';'P19','Sujet19';'P20','Sujet20';'P21','Sujet21';'P22','Sujet22';'P23','Sujet23';'P24','Sujet24';} ;



Xsens_name = {'Head','Stern','Pelvis','RShoulder','RUArm','RFArm','RHand'} ;
Degre={'rawAcc','rawGyr'};



j=1 ;



i=1 ;
for iSubjects = 1:length(Subjects) % Novices
    load(['blabla.mat'])
    % Novices
    ftAllFiles{1,i}.powspctrm(1,:,:) = toto ; % données 2D
    ftAllFiles{1,i}.dimord = 'chan_time_freq' ;
    ftAllFiles{1,i}.time = 1:100 ; % combien de points sur x
    ftAllFiles{1,i}.freq = 0.05:0.05:15; % combien de points sur y
    ftAllFiles{1,i}.label = 'Trampo' ;
    design(1,i) = iSubjects;
    design(2,i) = 1;
    i=i+1 ;
end

for iSubjects = 1:length(Subjects) % Experts
    % Task termination
    ftAllFiles{1,i}.powspctrm(1,:,:) = toto ; % données 2D
    ftAllFiles{1,i}.dimord = 'chan_time_freq' ;
    ftAllFiles{1,i}.time = 1:100 ;
    ftAllFiles{1,i}.freq = 0.05:0.05:15;
    ftAllFiles{1,i}.label = 'Trampo' ;
    design(1,i) = iSubjects +i;
    design(2,i) = 2;
    i=i+1 ;

end

statcfg.design = design;

ftStat = ft_freqstatistics(statcfg, ftAllFiles{:}) ;
ftStat.prob(ftStat.prob>0.05) = 1 ;

Stat_TF.ftStat=ftStat ;
Stat_TF.Init=Compil_init ;
Stat_TF.Termi=Compil_termi ;

imagesc(squeeze(ftStat.prob)') ; j=j+1 ;

clear ftAllFiles ftStat Stat_TF design Compil_init Compil_termi



