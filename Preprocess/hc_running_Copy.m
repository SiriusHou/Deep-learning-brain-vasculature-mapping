clc;
clear all;
close all;

%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('D:\xhou4\Park\script_CO2');
wkdir = 'D:\xhou4\Park';

cd(wkdir);
subSeries = dir();
subSeries(ismember({subSeries.name}, {'.', '..', 'script_CO2'})) =[];

for ii=1:length(subSeries)
    subName{ii} = subSeries(ii).name;
end

for ii=86:length(subName)
    hc_CVR_pipeline(wkdir, subName{ii});
end
