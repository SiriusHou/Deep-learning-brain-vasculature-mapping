clc;
clear all;
close all;

%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('L:\CVR_backup\smooth8\Park\script_CO2');
wkdir = 'L:\CVR_backup\smooth8\Park';

cd(wkdir);
subSeries = dir();
subSeries(ismember({subSeries.name}, {'.', '..', 'script_CO2'})) =[];

for ii=1:length(subSeries)
    subName{ii} = subSeries(ii).name;
end

for ii=1:length(subName)
    disp(ii)
    hc_CVR_pipeline_corrMap(wkdir, subName{ii});
end
