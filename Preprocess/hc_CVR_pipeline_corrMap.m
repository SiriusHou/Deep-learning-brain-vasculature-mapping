%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hc_CVR_pipeline_corrMap(wkdir, folder_name)

% wkdir = 'D:\Lulab\rsCVR_PCA\Moyamoya';
% addpath('D:\Lulab\rsCVR_PCA\Moyamoya\script');
% folder_name = 'HLu_CS_MR1_04152016';
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
addpath('D:\Lulab\resting_state_pipleline\toolbox');
addpath('D:\Lulab\resting_state_pipleline\lib');
addpath('D:\Lulab\resting_state_pipleline\spm12');

spm_get_defaults;
global defaults;
defaults.mask.thresh = 0;
mni_resolution = [2, 2, 2];
mni_type = 16;

%%filter and detrend %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rs_dir = [wkdir, filesep, folder_name, filesep, bold_file_name];
cd(rs_dir)

% concatenate BOLD images
all_norm_epis = spm_select('List', rs_dir, 'img', ['^s' int2str(SmoothFWHMmm) 'warro*']);
norm_epifiles = spm_vol(all_norm_epis);
norm_epifiles_4D = zeros(91, 109, 91, length(norm_epifiles));

% get in-brain mask
mask = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC.nii']));
mask_size = size(mask);

for ii = 1 : length(norm_epifiles)
    imgStruct = loadimage(norm_epifiles(ii).fname,1);
    norm_epifiles_4D(:,:,:,ii) = imgStruct;
end

% get in-brain mask
brainVox = find(mask == 1);
regressedImg_4D = zeros(size(norm_epifiles_4D));

for vox = 1:length(brainVox)
    [row,col,sl] = ind2sub(mask_size, brainVox(vox));
    TS1 = squeeze(norm_epifiles_4D(row,col,sl,:));
    
    sig = detrend(TS1);
    meansig = mean(TS1-sig);
    TS2 =  sig + meansig;     % detrended signal
%     TS3 = filtfilt(b,a,TS2);  % Filtering
    regressedImg_4D(row,col,sl,:) = TS2;
end

% write filtered and detrended images
ftempname = [rs_dir filesep 'drs' int2str(SmoothFWHMmm) 'warro' bold_file_name '.nii'];
write_hdrimg(regressedImg_4D, ftempname, mni_resolution, mni_type);

return