clc;
clear all;
close all;

%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('/data1/xhou/ML_CVR/smooth8/preprocessed_RS/Park/script_CO2')
wkdir = '/data1/xhou/ML_CVR/smooth8/preprocessed_RS/Park';

cd(wkdir);
subSeries = dir();
subSeries(ismember({subSeries.name}, {'.', '..', 'script_CO2'})) =[];

for ii=1:length(subSeries)
    subName{ii} = subSeries(ii).name;
end

% p = parpool(12);
for ii=1:1%length(subName)
    resting_state_pipeline(wkdir, subName{ii});
end
% delete(p);
