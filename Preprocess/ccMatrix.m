function [cc, folder_name] = ccMatrix(processed_folder, folder_name, temp_folder, rs_dynnum)

%GENERAL
roinum = 114;
linethick1 = 1.2;
linethick2 = 1.2;
colorrgb1 = [150,150,150]./256;
colorrgb2 = [250,250,250]./256;

%import the brain parcellation template
cd(temp_folder);
label_brain = spm_read_vols(spm_vol('Yeo2011_17Networks_N1000.split_components.FSL_MNI152_2mm.nii'));
[network_17_num, foo, network_17_raw] = xlsread('17Networks.xlsx');
[network_7_num, foo, network_7_raw] = xlsread('7Networks.xlsx');
% module = {'ContA','ContB','ContC','DefaultA','DefaultB','DefaultC','DefaultD','DorsAttnA',...
%     'DorsAttnB','LimbicA','LimbicB','SalVentAttnA','SalVentAttnB','SomMotorA','SomMotorB','VisCent','VisPeri'};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subject_folder = [processed_folder, filesep, folder_name];
cd([subject_folder]);

% import censor file
censor_data = csvread('censor.csv');
censor_loc=find(censor_data==1);

% import global WM and CSF time course
filegl = fopen('gl.txt','r');
gl_data = fscanf(filegl,'%f',[1,inf])';

filecsf = fopen('csf.txt','r');
csf_data = fscanf(filecsf,'%f',[1,inf])';

filewm = fopen('wm.txt','r');
wm_data = fscanf(filewm,'%f',[1,inf])';

% import 6-D rigid translation
filerp = fopen('rp.txt','r');
rp_data = fscanf(filerp,'%f %f %f %f %f %f',[6,inf])';
fclose('all');

% import brain mask
mymask = spm_read_vols(spm_vol('brainMask_RS.nii')); % adjust for more accurate mask
mymask_2D = reshape(mymask,91*109*91,1);
mymask_loc = find(mymask_2D==0);

bold_data_mean = zeros(roinum,length(censor_loc));
bold_data = spm_read_vols(spm_vol('rs_f.nii'));
bold_data_2D = reshape(bold_data,91*109*91,rs_dynnum);

%compile the data
for kk = 1 : rs_dynnum
    bold_data_2D(mymask_loc,kk) = nan;
end

bold_x = [gl_data(:,1), csf_data(:,1), wm_data(:,1), rp_data(:,1:6), ones(length(gl_data), 1)];
bold_x1 = bold_x(censor_loc,:); %motion scrubbing

%regress out the Global, WM, CSF and rp effect
for jj = 1:roinum
    bold_ROI = squeeze(bold_data_2D(find(label_brain==network_17_num(jj,1)),:));
    bold_y = nanmean(bold_ROI,1)';
    bold_y1 = bold_y(censor_loc,:);%censor location
    
    bold_para1 = regress(bold_y1, bold_x1);
    
    for kk=1:size(bold_x1,2)-1
        if kk==1
            bold_inter1 = bold_para1(kk)*bold_x1(:,kk);
        else
            bold_inter1 = bold_inter1+bold_para1(kk)*bold_x1(:,kk);
        end
    end
    bold_data_y2 = bold_y1-bold_inter1;
    bold_data_mean(jj,:) = bold_data_y2';
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%generate correlation matrix
cc = zeros(size(bold_data_mean,1), size(bold_data_mean,1));
cc_disp = zeros(size(bold_data_mean,1), size(bold_data_mean,1));

for kk = 1:size(bold_data_mean,1)
    bold_data1 = bold_data_mean(kk,:);
    for jj = kk:size(bold_data_mean,1)
        if jj == kk
            cc(kk,jj) = 2;%%diagonal term is set as 2 
        else
            bold_data2 = bold_data_mean(jj,:);
            cc_value = corrcoef(bold_data1,bold_data2);
            if isnan(cc_value(1,2))%prevent some mni node was not covered in individual map
                cc_value = zeros(2,2);
            end
            cc_fisher = 0.5*log((cc_value(1,2)+1)/(1-cc_value(1,2))); %fisher r to z transform
            cc(kk,jj) = cc_fisher;
            cc(jj,kk) = cc_fisher;
            
            cc_disp(jj,kk) = cc_fisher;
            cc_disp(kk,jj) = nan;
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%draw the cross-correlation matrix
oneMatrix = ones(roinum,roinum)-diag(ones(roinum,1));%diag=0,off-diag=1;

f = figure('units', 'normalized', 'outerposition', [0 0 1 1], 'visible', 'off');
cc_disp = cc_disp.*oneMatrix;
imagesc(cc_disp);%color the diagnoal term black

%%upper triangle as zero
[nr,nc] = size(cc_disp);
pcolor([cc_disp nan(nr,1); nan(1,nc+1)]);
shading flat
set(gca, 'ydir', 'reverse');

colormap fireice
axis square
colorbar
caxis([-0.75 0.75]);
hold on
title('Resting-State FC', 'FontSize', 18)

label_len = size(bold_data_mean,1)+1;
for kk = 1:17
    seg_num_loc = find(network_17_num(:,2) == kk);
    x_label_loc(kk) = seg_num_loc(ceil(end/2)+1);
    p = plot([seg_num_loc(1) seg_num_loc(1)], [seg_num_loc(1) label_len],'Color',colorrgb1);
    p(1).LineWidth = linethick1;
    p = plot([0 seg_num_loc(1)],[seg_num_loc(1) seg_num_loc(1)],'Color',colorrgb1);
    p(1).LineWidth = linethick1;
    module{kk, 1} = network_17_raw{seg_num_loc(1), 3};
end

for kk = 1:7
    seg_num_loc = find(network_7_num(:,2) == kk);
    p = plot([seg_num_loc(1) seg_num_loc(1)], [seg_num_loc(1) label_len],'Color',colorrgb2);
    p(1).LineWidth = linethick2;
    p = plot([0 seg_num_loc(1)],[seg_num_loc(1) seg_num_loc(1)],'Color',colorrgb2);
    p(1).LineWidth = linethick2;
end

box off
set(gca,'TickDir','out')
set(gca,'xtick',x_label_loc,'xticklabel',module,'FontSize', 18)
xtickangle(90)
set(gca,'ytick',x_label_loc,'yticklabel',module,'FontSize', 18)

saveas(f, ['cc_' folder_name '.png']);

cc(find(cc(:) == 2)) = nan;%set diagonal as nan
save('ccMatrix.mat', 'cc', 'folder_name');

parcellation_name = network_17_raw(:, 4);
cc_summary = cat(2, parcellation_name, num2cell(cc));
parcellation_name_title = cat(1, folder_name, parcellation_name);
cc_summary = cat(1, parcellation_name_title', cc_summary);
%writecell(cc_summary, 'ccMatrix.txt', 'Delimiter', ' '); 
xlswrite('ccMatrix.xlsx', cc_summary);
