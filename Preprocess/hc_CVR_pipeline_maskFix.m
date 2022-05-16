%%User Defined Parameter Setup%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hc_CVR_pipeline_maskFix(wkdir, folder_name)

% wkdir = 'K:\CVR_backup\smooth8\Park';
% addpath('K:\Lulab\rsCVR_PCA\Park\script_CO2');
% folder_name = '3T3352';

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
envelope_interp_rate = 10;
sample_rate_co2 = 48;
select = 0.1;
tpm_loc = 'D:\Lulab\resting_state_pipleline\spm12\tpm'; %brain normlization template location

% %% Reorient BOLD Image%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [rs_dir, rs_dynnum, rs_vox, rs_dim, rs_ss, rs_offset, rs_origin, rs_type, rs_precision] = PARRECtoANALYZE_new([wkdir, filesep, folder_name, filesep, bold_file_name, bold_file_ext], '3D');
% boldfiles = dir([rs_dir, filesep, bold_file_name, '*.img']);
%
% cd(rs_dir);
% for ii = 1:length(boldfiles)
%     tmparray = spm_read_vols(spm_vol( boldfiles(ii).name ));
%     tmparray = RealignImageArray(tmparray,'+x-y+z',-1);%reorient the image
%     write_hdrimg(tmparray, ['ro', boldfiles(ii).name], rs_vox, rs_type);
% end
%
%
% %% Realign BOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % get original bold scan
% P      = cell(1,1);
% P{1}   = spm_select('FPList', rs_dir, 'img', ['^ro', bold_file_name, '*']);
%
% % get the bold scan's data
% V       = spm_vol(P);
% V       = cat(1,V{:});
%
% % realign bold scan
% disp(sprintf(['realigning ' bold_file_name]));
% FlagsC = struct('quality',defaults.realign.estimate.quality,...
%     'fwhm',5,'rtm',0);
% spm_realign(V, FlagsC);
%
% % reslice bold scan
% which_writerealign = 2;
% mean_writerealign = 1;
% FlagsR = struct('interp',defaults.realign.write.interp,...
%     'wrap',defaults.realign.write.wrap,...
%     'mask',defaults.realign.write.mask,...
%     'which',which_writerealign,'mean',mean_writerealign);
% spm_reslice(P,FlagsR);
%
% %% Slice Timing Correction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % skip this step if sliceOrderFile is omitted
% sliceOrderFile = [wkdir, filesep, folder_name, filesep, slice_order_file];
% P = spm_select('FPList', rs_dir, 'img', ['^rro', bold_file_name, '*']);
%
% if ~isempty(sliceOrderFile)
%
%     % get original bold scan
%     % set TA and refSlice
%     scanInfo = spm_vol(P(1,:));  %use first dynamic to get number of slices
%     nslices = scanInfo(1).dim(3);
%     TA = TR-(TR/nslices);
%
%     % get order of slices
%     fid = fopen([sliceOrderFile]);
%     sliceOrder = textscan(fid,'%f');
%     sliceOrder = sliceOrder{1};
%     fclose(fid);
%
%     % get timing for correction
%     timing(2)=TR-TA;
%     timing(1)=TA/(nslices-1);
%
%     % correct slice timing (produces a new bold scan with the filename prefixed
%     % with 'a')
%     spm_slice_timing(P, sliceOrder', ref_slice, timing);
% end
%
% %% Reorient MPRAGE File %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% [mpr_dir, mpr_dyn, mpr_vox, mpr_dim, mpr_ss, mpr_offset, mpr_origin, mpr_type, mpr_precision] = PARRECtoANALYZE_new([wkdir, filesep, folder_name, filesep, mpr_file_name, mpr_file_ext]); %convert par/rec file to analyze format
% copyfile(mpr_dir, [mpr_dir, '_HC']);
% mpr_dir = [mpr_dir, '_HC'];
% cd(mpr_dir)
%
% mpr_brain_vol = spm_read_vols(spm_vol([mpr_file_name, '.img']));
% [outVol,varargout] = reorientVol(mpr_brain_vol, '-y-z-x');
% write_hdrimg(outVol, [mpr_file_name, '.img'], [mpr_vox(3), mpr_vox(1), mpr_vox(2)], mpr_type);
% write_hdrimg(outVol, [mpr_file_name, '.nii'], [mpr_vox(3), mpr_vox(1), mpr_vox(2)], mpr_type);
%
% %% Coregistration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% % coregister MPRAGE to BOLD image
% coreg_source = [mpr_dir, filesep, mpr_file_name, '.nii'];
%
% coreg_target = spm_select('FPList', rs_dir, ['^meanro', bold_file_name, '-001-001.img']);
% coreg_job(coreg_target, coreg_source);
%
% %% Segment MPRAGE and Create Mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cd(mpr_dir)
% segment_job([mpr_file_name, '.nii'], tpm_loc); %segment mprage image
%
% %% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% deform_field = spm_select('FPList', mpr_dir, ['^y_', mpr_file_name, '.nii']);
% norm_job(rs_dir, bold_file_name, rs_dynnum, deform_field) %resoultion = 2*2*2mm
%
% anaFile = spm_select('FPList', mpr_dir, ['^m', mpr_file_name, '.nii']);
% norm_ana_job(anaFile, deform_field); %resoultion = 2*2*2mm
%
% %% Skull-stripped Brain Mask%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% suit_isolate_seg({['wm', mpr_file_name, '.nii']}); %segement cerebellum
% segment_job(['wm', mpr_file_name, '.nii'], tpm_loc); %segment mprage image
%
% % segmented probability map
% norm_segmentedList = dir(['c*wm' mpr_file_name, '.nii']);
%
% for ii = 1:3 %GM, WM and CSF
%     if ii == 1
%         brainTissue = spm_read_vols(spm_vol(norm_segmentedList(ii).name));
%     else
%         brainTissue = brainTissue + spm_read_vols(spm_vol(norm_segmentedList(ii).name));
%     end
% end
%
% levelBW = 0.8; %set up the threshold for brain tissue mask
%
% %write the mpr_brain
% mpr_brain = spm_read_vols(spm_vol(['wm', mpr_file_name, '.nii']));
% mpr_brain(brainTissue < levelBW) = 0;
%
% %write the brain mask
% brain_mask = zeros(size(brainTissue));
% brain_mask(brainTissue >= levelBW) = 1;
%
% %padding the hole in the brain mask
% DD = bwconncomp(brain_mask);
%
% numPixels = cellfun(@numel,DD.PixelIdxList);
% [biggest,idx] = max(numPixels);
% c1 = zeros(size(brain_mask));
% c1(DD.PixelIdxList{idx}) = 1;
%
% c2 = ones(size(brain_mask));
% c2(DD.PixelIdxList{idx}) = 0;
%
% c2DD = bwconncomp(c2, 6);
% numPixels2 = cellfun(@numel,c2DD.PixelIdxList);
% [biggest2,idx2] = max(numPixels2);
% c2(c2DD.PixelIdxList{idx2}) = 0;
%
% c3 = zeros(size(brain_mask));
% c3(find(c1>0))=1;
% c3(find(c2>0))=1;
%
% mkdir([wkdir, filesep, folder_name, filesep, 'mask']);
% write_hdrimg(c3, [wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC.nii'], mni_resolution, rs_type);
%
%
% %cerebellum mask
% c_cerebellum = spm_read_vols(spm_vol([mpr_dir, filesep, 'wm', mpr_file_name, '_seg1.nii']));
% cerebellum_thresh = 0.9;
% c_cerebellum(c_cerebellum>cerebellum_thresh) = 1;
% c_cerebellum(c_cerebellum<=cerebellum_thresh) = 0;
% write_hdrimg(c_cerebellum, [wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC_cerebellum.nii'], mni_resolution, rs_type);
%
% %% Smooth BOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% cd(rs_dir)
%
% % get realigned and resliced bold scan (prefixed with 'r')
% P = cell(1,1);
% P{1}   = spm_select('List', rs_dir, 'img', ['^warro', bold_file_name, '*']);
%
% % get the scan's data
% V       = spm_vol(P);
% V       = cat(1,V{:});
%
% % smooth scan (creates a 4D smoothed scan prefixed with 's' and the FWHM of
% % the gaussian kernel) (also creates a 3D mean scan prefixed with 'mean')
% disp(sprintf(['smoothing ' bold_file_name]));
%
% for ii = 1:length(V)
%     [pth,nam,ext] = fileparts(V(ii).fname);
%     fnameIn       = fullfile(pth, [nam ext]);
%     fname         = fullfile(pth, ['s' int2str(SmoothFWHMmm) nam ext]);
%     spm_smooth(fnameIn, fname, SmoothFWHMmm);
% end
%
% % clear V for further use
% clear V;

%% shift EtCO2/O2, filter and detrend %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rs_dir = [wkdir, filesep, folder_name, filesep, bold_file_name];
cd(rs_dir)

% concatenate BOLD images
all_norm_epis = spm_select('List', rs_dir, 'img', ['^s' int2str(SmoothFWHMmm) 'warro*']);
rs_dynnum = length(all_norm_epis);
norm_epifiles = spm_vol(all_norm_epis);

% get in-brain mask
mask = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC.nii']));
mask_1 = find(mask==1);
mask_0 = find(mask==0);
norm_epifiles_4D = zeros(cat(2, size(mask), rs_dynnum));

%check if mask size equals to CVR map
cvrdir_g = [wkdir, filesep, folder_name, filesep, 'CVR_globalshift'];

CVR_map = spm_read_vols(spm_vol([cvrdir_g, filesep, 'HC_CVRmap_s8.img']));
CVR_map(mask_0) = 0;
CVR_resid = zeros(size(CVR_map));
rs_type = 16;
if length(find(isnan(CVR_map(mask_1))))>0
    CVR_resid_loc = find(isnan(CVR_map));
    
    CVR_resid(CVR_resid_loc) = 1;
    disp(length(find(isnan(CVR_map(mask_1)))))
    disp('rewrite CVR map');
    write_hdrimg(CVR_resid, [cvrdir_g, filesep, 'resid.nii'], mni_resolution, rs_type);
    %mask_cerebellum = mask;
    mask_cerebellum = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_HC_cerebellum.nii']));
    mask_size = size(mask);
    avgBold = zeros(rs_dynnum, 1);
    
    for ii = 1 : rs_dynnum
        imgStruct = loadimage(norm_epifiles(ii).fname,1);
        avgBold(ii) = mean(imgStruct(mask_cerebellum==1));
        norm_epifiles_4D(:,:,:,ii) = imgStruct;
    end
    
    rmdir(cvrdir_g, 's');
    % save the whole-brain bold signal
    name_avgboldPath = strcat(rs_dir, filesep, 'AvgBOLD_WB_ns.txt');
    dlmwrite(name_avgboldPath, avgBold);
    
    % get the whole-brain bold signal
    avgBold = [TR*(0:length(avgBold)-1)',avgBold];
    
    % get the co2 envelope signal (timestamps in 1st column, mmHgCo2 in 2nd)
    etco2_file_name = [wkdir, filesep, folder_name, filesep, 'IP_etco2_time_', folder_name, '.txt'];
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % get the co2 envelope signal (timestamps in 1st column, mmHgCo2 in 2nd)
    dd = dlmread(etco2_file_name, '\t', 0, 0);
    etco2timecourse = dd(:, 1:2);
    
    etco2timestep = linspace(min(etco2timecourse(:, 1)), max(etco2timecourse(:, 1)), envelope_interp_rate*(max(etco2timecourse(:, 1))-min(etco2timecourse(:, 1)))+1)';
    etco2_inter = interp1(etco2timecourse(:, 1), etco2timecourse(:, 2), etco2timestep);
    etco2course = cat(2, etco2timestep, etco2_inter); %interpolate
    
    % find the optimal delay between the whole-brain (cerebellum) bold signal and the co2
    % envelope (by finding the lowest residual values between the two shifted
    % curves after linear fitting)
    
    % delay range in seconds (it is assumed that the EtCO2 curve will cover a
    % much larger range than the BOLD; therefore, the shift of the EtCO2 curve
    % will only be negative (to the left))
    minRange = -abs((length(etco2timecourse)./envelope_interp_rate) -...
        (size(avgBold,1).*TR));
    % minRange = -200;
    maxRange = 100;
    delayrange = [minRange maxRange];
    outDir = [wkdir, filesep, folder_name, filesep, bold_file_name];
    
    % finds the optimal delay in the delay range between the two curves
    [optDelay, ~] = cvr_func_findCO2delay(avgBold, TR,...
        etco2course, delayrange,1,1,1,outDir,1);
    
    % repeat process once with +/- 10s delays with 0.1s iteration bewteen
    % delays to approach (zoom in on) the optimal delay
    delayrange = [optDelay-5, optDelay+5];
    [optDelay, optEtCO2] = cvr_func_findCO2delay(avgBold, TR,...
        etco2course, delayrange,1,0,0.1,outDir);
    
    co2delay=optDelay;
    
    % save delays
    filename = fullfile([wkdir, filesep, folder_name], 'Sync_EtCO2_timecourse.txt');
    save(filename,'optEtCO2','-ascii');
    filename = fullfile([wkdir, filesep, folder_name], 'EtCO2_BOLD_delay.txt');
    save(filename,'co2delay','-ascii');
    
    close all force
    % use delay to generate an EtCO2 and O2 file with modified timestamps,
    % where t=0 is at the beginning of the curve synced with the global BOLD
    % signal
    boldSyncedEtco2Path = zach_syncCurveWithBold(etco2_file_name, co2delay);
    
    %% Global-shift CVR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % find top 25% and get delta (top-bttm) %%%%%%%%%%%%%%%%
    % use mean/min etco2 in future %%%%%%%%%%%%%%%%%%%%%%%%
    
    % find and save CVR map using global-shift method
    cvrdir_g = [wkdir, filesep, folder_name, filesep, 'CVR_globalshift'];
    mkdir(cvrdir_g);
    copyfile([wkdir, filesep, folder_name, filesep, 'Sync_EtCO2_timecourse.txt'], [wkdir, filesep, folder_name, filesep, bold_file_name]);
    [EtCO2_mean,EtCO2_min,EtCO2_max] =CVR_mapping_spm_GLM_CO2_only([wkdir, filesep, folder_name, filesep, bold_file_name], bold_file_name, TR, SmoothFWHMmm, mask, 1, cvrdir_g);
    
    %% Voxel-shift Beta Maps %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % find beta maps, EtCO2_mean, and EtCO2_min (used to find CVR map)and BAT
    % map using the voxel-shift method
    cvrdir_v = [wkdir, filesep, folder_name, filesep, 'CVR_voxelshift_etco2'];
    mkdir(cvrdir_v);
    fixedDelayRange(1) = -15;
    fixedDelayRange(2) = 40;
    etco2_file_sync = [wkdir, filesep, folder_name, filesep, 'IP_etco2_time_', folder_name, '_boldSynced.txt'];
    voxelwiseResult = CVR_mapping_voxelwise_GLM_CO2_only([wkdir, filesep, folder_name, filesep, bold_file_name], bold_file_name, TR, SmoothFWHMmm, fixedDelayRange, mask, ...
        etco2_file_sync, EtCO2_mean, EtCO2_min, cvrdir_v);
end
fclose all;
return