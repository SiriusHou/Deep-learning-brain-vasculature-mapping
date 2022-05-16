%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hc_CVR_pipeline_BAT(wkdir, folder_name)

% wkdir = 'L:\CVR_backup\smooth8\Park';
% addpath('L:\CVR_backup\smooth8\Park\script_CO2');
% folder_name = '3T0807';

cd([wkdir, filesep, folder_name]);
para_file_ID = fopen('parameter_RS.txt', 'r');

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
return
