clear all;
clc;
warning off;

%GENERAL
addpath('/data1/xhou/toolbox');
addpath('/data1/xhou/lib');
addpath('/data1/xhou/spm12');
spm_defaults;
global defaults;

folderMother = fullfile('/data1/xhou/ML_CVR/smooth8/preprocessed_RS/Park/script_CO2');
addpath(folderMother);

folder = fullfile('/data1/xhou/ML_CVR/smooth8/preprocessed_RS/Park');

cd(folder);
subSeries = dir();
subSeries(ismember({subSeries.name}, {'.', '..', 'script_CO2'})) =[];

for ii=1:length(subSeries)
    subName{ii} = subSeries(ii).name;
end

for jj = 1:length(subSeries) 

    subname = char(subName{jj}); %foldername for each subject CASE SENSITIVE
    subfolder = [folder, filesep, subname];

    cd(subfolder);%project main path
    folder_list = dir();
    folder_list(ismember({folder_list.name}, {'.', '..', 'script_CO2'})) =[];
    folder_list = folder_list([folder_list(:).isdir] == 1);
%     
%     for ii = 1: length(folder_list)
%         rmdir([subfolder, filesep, folder_list(ii).name],'s');
%     end
    
    fileID_w = fopen('parameter_RS.txt','w');
    fprintf(fileID_w,'"UserInput" : { \n');
    fprintf(fileID_w,'\t "SmoothFWHMmm" : 8\n');

    %mprage image counts
    listFile = cat(1, dir('*.par'), dir('*.PAR')); %for linux system
%     listFile = dir('*.par');

    count_MPR = 0;
    ww = 0;
    for ii = 1:length(listFile)
        fileID = fopen(listFile(ii).name, 'r');
        while ~feof(fileID)
            tline = fgetl(fileID);
            if ~isempty(regexp(tline, 'MPRAGE')) 
                ww = ww+1;
                MPRFile{ww} = listFile(ii).name;
                count_MPR = count_MPR+1;
            end
            
        end
        fclose(fileID);
    end
    
    count_HC = 0;
    kk = 0;
    for ii = 1:length(listFile)
        fileID = fopen(listFile(ii).name, 'r');
        index_hc = 0;
        while ~feof(fileID)
            tline = fgetl(fileID);
            if ~isempty(regexp(tline, 'Resting State SENSE')) 
                index_hc = 1;
                kk = kk+1;
                HCFile{kk} = listFile(ii).name;
                count_HC = count_HC+1;
            end
                        
            if index_hc == 1
                if ~isempty(regexp(tline, 'Repetition time'))  
                    TR_string = tline;
                end
            end
           
        end
        fclose(fileID);
    end
    TR_split = regexp(TR_string, '\:', 'split');
    TR = str2num(TR_split{end})/1000; %string to number.
    fprintf(fileID_w,'\t "TRs" : %d\n', TR);
    fprintf(fileID_w,'\t \"mprageFileName" : "%s"\n', MPRFile{1});
    fprintf(fileID_w,'\t \"boldFileName" : "%s"\n', HCFile{1});
    fprintf(fileID_w,'\t \"sliceOrderFile" : "slice_order_RS.txt"\n');
    fprintf(fileID_w,'\t \"refSlice" : 1\n}');
    
    if count_HC > 1 || count_MPR > 1
        disp('More than one BOLD/MPRAGE data');
    end
    
    [rs_dir, rs_dynnum, rs_vox, rs_dim, rs_ss, rs_offset, rs_origin, rs_type, rs_precision] = PARRECtoANALYZE_new([subfolder, filesep, HCFile{1}], '3D');

    fileID_s = fopen('slice_order_RS.txt','w');  
    for kk = 1: rs_dim(3)
        fprintf(fileID_s,'%d\n', kk);
    end
    
    fclose('all');
    
end


