clc;
clear all;
close all;

%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('K:\CVR_backup\smooth8\Park\script_CO2');
wkdir = 'K:\CVR_backup\smooth8\Park';

cd(wkdir);
subSeries = dir();
subSeries(ismember({subSeries.name}, {'.', '..', 'script_CO2'})) =[];

for ii=1:length(subSeries)
    subName{ii} = subSeries(ii).name;
end

for ii=1:length(subName)
    disp(subName{ii});
    mask = spm_read_vols(spm_vol([wkdir, filesep, subName{ii}, filesep, 'mask', filesep, 'brainMask_HC.nii']));
    CVR_mask = spm_read_vols(spm_vol([wkdir, filesep, subName{ii}, filesep, 'CVR_globalshift', filesep, 'mask.nii']));
    mask_HC = mask.*CVR_mask;
    write_hdrimg(mask_HC, [wkdir, filesep, subName{ii}, filesep, 'mask', filesep, 'brainMask_HC.nii'], [2,2,2], 16);
end