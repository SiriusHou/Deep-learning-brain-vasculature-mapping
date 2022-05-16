function resting_state_pipeline(wkdir, folder_name)

% wkdir = 'D:\Lulab\rsCVR_PCA\Park';
% folder_name = '3T0807';

cd([wkdir, filesep, folder_name]);
para_file_ID = fopen('parameter_RS.txt', 'r');

% addpath('D:\Lulab\rsCVR_PCA\Moyamoya\script');

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
tpm_loc = '/data1/xhou/spm12/tpm'; %brain normlization template location

%% Reorient BOLD Image%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist([wkdir, filesep, folder_name, filesep, bold_file_name])
    rmdir([wkdir, filesep, folder_name, filesep, bold_file_name], 's');
end

[rs_dir, rs_dynnum, rs_vox, rs_dim, rs_ss, rs_offset, rs_origin, rs_type, rs_precision] = PARRECtoANALYZE_new([wkdir, filesep, folder_name, filesep, bold_file_name, bold_file_ext], '3D');
boldfiles = dir([rs_dir, filesep, bold_file_name, '*.img']);

cd(rs_dir);
for ii = 1:length(boldfiles)
    tmparray = spm_read_vols(spm_vol( boldfiles(ii).name ));
    tmparray = RealignImageArray(tmparray,'+x-y+z',-1);%reorient the image
    write_hdrimg(tmparray, ['ro', boldfiles(ii).name], rs_vox, rs_type);
end


%% Realign BOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% get original bold scan
P      = cell(1,1);
P{1}   = spm_select('FPList', rs_dir, 'img', ['^ro', bold_file_name, '*']);

% get the bold scan's data
V       = spm_vol(P);
V       = cat(1,V{:});

% realign bold scan
disp(sprintf(['realigning ' bold_file_name]));
FlagsC = struct('quality',defaults.realign.estimate.quality,...
    'fwhm',5,'rtm',0);
spm_realign(V, FlagsC);

% reslice bold scan
which_writerealign = 2;
mean_writerealign = 1;
FlagsR = struct('interp',defaults.realign.write.interp,...
    'wrap',defaults.realign.write.wrap,...
    'mask',defaults.realign.write.mask,...
    'which',which_writerealign,'mean',mean_writerealign);
spm_reslice(P,FlagsR);

%% Slice Timing Correction %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% skip this step if sliceOrderFile is omitted
sliceOrderFile = [wkdir, filesep, folder_name, filesep, slice_order_file];
P = spm_select('FPList', rs_dir, 'img', ['^rro', bold_file_name, '*']);

if ~isempty(sliceOrderFile)
    
    % get original bold scan
    % set TA and refSlice
    scanInfo = spm_vol(P(1,:));  %use first dynamic to get number of slices
    nslices = scanInfo(1).dim(3);
    TA = TR-(TR/nslices);
    
    % get order of slices
    fid = fopen([sliceOrderFile]);
    sliceOrder = textscan(fid,'%f');
    sliceOrder = sliceOrder{1};
    fclose(fid);
    
    % get timing for correction
    timing(2)=TR-TA;
    timing(1)=TA/(nslices-1);
    
    % correct slice timing (produces a new bold scan with the filename prefixed
    % with 'a')
    spm_slice_timing(P, sliceOrder', ref_slice, timing);
end

%% Reorient MPRAGE File %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if exist([wkdir, filesep, folder_name, filesep, mpr_file_name])
    rmdir([wkdir, filesep, folder_name, filesep, mpr_file_name], 's');
end

if exist([wkdir, filesep, folder_name, filesep, mpr_file_name, '_RS'])
    rmdir([wkdir, filesep, folder_name, filesep, mpr_file_name, '_RS'], 's');
end

[mpr_dir, mpr_dyn, mpr_vox, mpr_dim, mpr_ss, mpr_offset, mpr_origin, mpr_type, mpr_precision] = PARRECtoANALYZE_new([wkdir, filesep, folder_name, filesep, mpr_file_name, mpr_file_ext]); %convert par/rec file to analyze format
copyfile(mpr_dir, [mpr_dir, '_RS']);
mpr_dir = [mpr_dir, '_RS'];
cd(mpr_dir)

mpr_brain_vol = spm_read_vols(spm_vol([mpr_file_name, '.img']));
[outVol,varargout] = reorientVol(mpr_brain_vol, '-y-z-x');
write_hdrimg(outVol, [mpr_file_name, '.img'], [mpr_vox(3), mpr_vox(1), mpr_vox(2)], mpr_type);
write_hdrimg(outVol, [mpr_file_name, '.nii'], [mpr_vox(3), mpr_vox(1), mpr_vox(2)], mpr_type);

%% Coregistration %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Segment MPRAGE and Create preMask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(mpr_dir)
segment_job([mpr_file_name, '.nii'], tpm_loc); %segment mprage image
for ii = 1:3
    if ii == 1
        mask_p = spm_read_vols(spm_vol([mpr_dir, filesep, 'c', num2str(ii), mpr_file_name, '.nii']));
    else
        mask_p = mask_p + spm_read_vols(spm_vol([mpr_dir, filesep, 'c', num2str(ii), mpr_file_name, '.nii']));
    end
end

new_info = spm_vol([mpr_file_name, '.nii']);
mpr_vox = mri_resolution(new_info.mat);
mask_p(mask_p>0.05) = 1;
write_hdrimg(mask_p, 'mask_p.nii', mpr_vox, mni_type);
delete(['y_', mpr_file_name, '.nii']);
delete([mpr_file_name, '_seg8.mat']);

mp_data = spm_read_vols(spm_vol([mpr_dir, filesep, mpr_file_name, '.nii']));
mp_data = mp_data.*mask_p;
mpr_mask_name = [mpr_dir, filesep, mpr_file_name, '_mask.nii'];
write_hdrimg(mp_data, mpr_mask_name, mpr_vox, mni_type);

%coregister MPRAGE to BOLD image
coreg_target = spm_select('FPList', rs_dir, ['^meanro', bold_file_name, '-001-001.img']);
coreg_other = {[mpr_dir, filesep, mpr_file_name, '.nii']};
coreg_job(coreg_target, mpr_mask_name, coreg_other);

%% Segment MPRAGE and Create Mask %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(mpr_dir)
segment_job([mpr_file_name, '.nii'], tpm_loc); %segment mprage image

%% Normalization %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
deform_field = spm_select('FPList', mpr_dir, ['^y_', mpr_file_name, '.nii']);
norm_job(rs_dir, bold_file_name, rs_dynnum, deform_field) %resoultion = 2*2*2mm

anaFile = spm_select('FPList', mpr_dir, ['^m', mpr_file_name, '.nii']);
norm_ana_job(anaFile, deform_field); %resoultion = 2*2*2mm

%% Skull-stripped Brain Mask%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
suit_isolate_seg({['wm', mpr_file_name, '.nii']}); %segement cerebellum

segment_job(['wm', mpr_file_name, '.nii'], tpm_loc); %segment mprage image

% segmented probability map
norm_segmentedList = dir(['c*wm' mpr_file_name, '.nii']);

for ii = 1:3 %GM, WM and CSF
    if ii == 1
        brainTissue = spm_read_vols(spm_vol(norm_segmentedList(ii).name));
    else
        brainTissue = brainTissue + spm_read_vols(spm_vol(norm_segmentedList(ii).name));
    end
end

levelBW = 0.8; %set up the threshold for brain tissue mask

%write the mpr_brain
mpr_brain = spm_read_vols(spm_vol(['wm', mpr_file_name, '.nii']));
mpr_brain(brainTissue < levelBW) = 0;

%write the brain mask
brain_mask = zeros(size(brainTissue));
brain_mask(brainTissue >= levelBW) = 1;

%padding the hole in the brain mask
DD = bwconncomp(brain_mask);

numPixels = cellfun(@numel,DD.PixelIdxList);
[biggest,idx] = max(numPixels);
c1 = zeros(size(brain_mask));
c1(DD.PixelIdxList{idx}) = 1;

c2 = ones(size(brain_mask));
c2(DD.PixelIdxList{idx}) = 0;

c2DD = bwconncomp(c2, 6);
numPixels2 = cellfun(@numel,c2DD.PixelIdxList);
[biggest2,idx2] = max(numPixels2);
c2(c2DD.PixelIdxList{idx2}) = 0;

c3 = zeros(size(brain_mask));
c3(find(c1>0))=1;
c3(find(c2>0))=1;

mkdir([wkdir, filesep, folder_name, filesep, 'mask']);
write_hdrimg(c3, [wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_RS.nii'], mni_resolution, rs_type);

%cerebellum mask
c_cerebellum = spm_read_vols(spm_vol([mpr_dir, filesep, 'wm', mpr_file_name, '_seg1.nii']));
% c_cerebellum_c1 = spm_read_vols(spm_vol([mpr_dir, filesep, 'wm', mpr_file_name, '_seg1.nii']));
% c_cerebellum_c2 = spm_read_vols(spm_vol([mpr_dir, filesep, 'wm', mpr_file_name, '_seg2.nii']));
% 
% c_cerebellum = c_cerebellum_c1 + c_cerebellum_c2;
cerebellum_thresh = 0.9;
c_cerebellum(c_cerebellum>cerebellum_thresh) = 1;
c_cerebellum(c_cerebellum<=cerebellum_thresh) = 0;
write_hdrimg(c_cerebellum, [wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_RS_cerebellum.nii'], mni_resolution, rs_type);

%% Smooth BOLD %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(rs_dir)

% get realigned and resliced bold scan (prefixed with 'r')
P = cell(1,1);
P{1}   = spm_select('List', rs_dir, 'img', ['^warro', bold_file_name, '*']);

% get the scan's data
V       = spm_vol(P);
V       = cat(1,V{:});

% smooth scan (creates a 4D smoothed scan prefixed with 's' and the FWHM of
% the gaussian kernel) (also creates a 3D mean scan prefixed with 'mean')
disp(sprintf(['smoothing ' bold_file_name]));

for ii = 1:length(V)
    [pth,nam,ext] = fileparts(V(ii).fname);
    fnameIn       = fullfile(pth, [nam ext]);
    fname         = fullfile(pth, ['s' int2str(SmoothFWHMmm) nam ext]);
    spm_smooth(fnameIn, fname, SmoothFWHMmm);
end

% clear V for further use
clear V;

%% shift EtCO2/O2, filter and detrend %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd(rs_dir)

% concatenate BOLD images
all_norm_epis = spm_select('List', rs_dir, 'img', ['^s' int2str(SmoothFWHMmm) 'warro*']);
norm_epifiles = spm_vol(all_norm_epis);
norm_epifiles_4D = zeros(cat(2, size(brain_mask), rs_dynnum));

% get in-brain mask
mask = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_RS.nii']));
%mask_cerebellum = mask;
mask_cerebellum = spm_read_vols(spm_vol([wkdir, filesep, folder_name, filesep, 'mask', filesep, 'brainMask_RS_cerebellum.nii']));
mask_size = size(mask);
avgBold = zeros(rs_dynnum,1);

for ii = 1 : rs_dynnum
    imgStruct = loadimage(norm_epifiles(ii).fname,1);
    avgBold(ii) = mean(imgStruct(mask_cerebellum==1));
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

% save the whole-brain bold signal
name_avgboldPath = strcat(rs_dir, filesep, 'AvgBOLD_WB_ns.txt');
save(name_avgboldPath, 'avgBold','-ascii');

% get the whole-brain bold signal
avgBold = textread(name_avgboldPath,'%f');
return
