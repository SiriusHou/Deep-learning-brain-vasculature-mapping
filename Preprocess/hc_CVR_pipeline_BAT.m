%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hc_CVR_pipeline_BAT(wkdir, folder_name)

% wkdir = 'L:\CVR_backup\smooth8\Park';
% addpath('L:\CVR_backup\smooth8\Park\script_CO2');
% folder_name = '3T0807';

cd([wkdir, filesep, folder_name]);
para_file_ID = fopen('parameter_HC.txt', 'r');

while ~feof(para_file_ID)
    
    tline = fgetl(para_file_ID);
    if regexp(tline, 'SmoothFWHM')  %second line indicates the SmoothFWHmm        
        colon_loc =regexp(tline, ':');
        SmoothFWHMmm = str2num(tline(colon_loc+1:end));
    end
    
    if regexp(tline, 'TR')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, ':');
        TR = str2num(tline(colon_loc+1:end));
    end
    
    if regexp(tline, 'mprageFile')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        [foo, mpr_file_name, mpr_file_ext] = fileparts(tline(colon_loc(3)+1:colon_loc(4)-1));
    end
    
    if regexp(tline, 'boldFile')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        [foo, bold_file_name, bold_file_ext] = fileparts(tline(colon_loc(3)+1:colon_loc(4)-1));
    end
    
    if regexp(tline, 'sliceOrderFile')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, '"');
        slice_order_file = tline(colon_loc(3)+1:colon_loc(4)-1);
    end
    
    if regexp(tline, 'refSlice')  %second line indicates the SmoothFWHmm
        colon_loc =regexp(tline, ':');
        ref_slice = str2num(tline(colon_loc+1:end));
    end
end


%% Global Parameter Setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
addpath('/data1/xhou/toolbox');
addpath('/data1/xhou/lib');
addpath('/data1/xhou/spm12');

spm_get_defaults;
global defaults;
defaults.mask.thresh = 0;
mni_resolution = [2, 2, 2];
mni_type = 16;
envelope_interp_rate = 10; 
sample_rate_co2 = 48;
select = 0.1;
tpm_loc = 'D:\Lulab\resting_state_pipleline\spm12\tpm'; %brain normlization template location

%% get in-brain mask
if exist([wkdir, filesep, folder_name, filesep, 'CVR_voxelshift_etco2_v3'])
    rmdir([wkdir, filesep, folder_name, filesep, 'CVR_voxelshift_etco2_v3'], 's');
end

mask = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC.nii']));
% %mask_cerebellum = mask;
% mask_cerebellum = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC_cerebellum.nii']));
% mask_size = size(mask);
% avgBold = zeros(rs_dynnum, 1);

%%write 3D w file into 4D nii file %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rs_dir = [wkdir, filesep, folder_name, filesep, bold_file_name];
cd(rs_dir);

P = cell(1,1);
P{1}   = spm_select('List', rs_dir, 'img', ['^warro', bold_file_name, '*']);

% get the scan's data
V       = spm_vol(P);
V       = cat(1,V{:});

for ii = 1:length(V)
    tmparray = spm_read_vols(spm_vol(V(ii).fname));
    if ii == 1
        tmparray_4D = zeros(cat(2, size(tmparray), length(V)));
    end
    tmparray_4D(:, :, :, ii) = tmparray;
end

ftempname = [rs_dir filesep 'warro' bold_file_name '.nii'];
write_hdrimg(tmparray_4D, ftempname, mni_resolution, mni_type);

%% Voxel-shift Beta Maps %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path_temp = [wkdir, filesep, folder_name, filesep, bold_file_name];
fn = spm_select('List',path_temp,'Sync_EtCO2_timecourse.txt'); 
[r1] = textread([path_temp filesep fn],'%f');
thre = 4;
nave=floor(length(r1)/thre);
[Y,I]=sort(r1,'descend');
EtCO2_min =mean(Y(end-nave:end));  % lowest 1/4 as baseline
EtCO2_mean=mean(Y);
EtCO2_max =mean(Y(1:nave));        % highest 1/4 for output
    
% find beta maps, EtCO2_mean, and EtCO2_min (used to find CVR map)and BAT
% map using the voxel-shift method
cvrdir_v = [wkdir, filesep, folder_name, filesep, 'CVR_voxelshift_etco2_v3'];
mkdir(cvrdir_v);
fixedDelayRange(1) = -9;
fixedDelayRange(2) = 30;
etco2_file_sync = [wkdir, filesep, folder_name, filesep, 'IP_etco2_time_', folder_name, '_boldSynced.txt'];
voxelwiseResult = CVR_mapping_voxelwise_GLM_CO2_only([wkdir, filesep, folder_name, filesep, bold_file_name], bold_file_name, TR, SmoothFWHMmm, fixedDelayRange, mask, ...
    etco2_file_sync, EtCO2_mean, EtCO2_min, cvrdir_v);
fclose all;
close all;
return
