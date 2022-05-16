clear all;
clc;
warning off;

folderMother = fullfile('D:\Lulab\project1');
addpath(folderMother);

folder = fullfile('O:\BrainTumor');

cd(folder);
subSeries = dir('hlu*');

jj=1;
for ii=1:length(subSeries)
    subName{jj} = subSeries(ii).name;
    jj = jj+1;
end

tadir = 'D:\Lulab\rsCVR_PCA\BrainTumor';

for jj = 1:length(subSeries)
    
    disp(jj)
    subname = char(subName{jj}); %foldername for each subject CASE SENSITIVE
    subfolder = [folder, '\', subname];
    
    sub_tadir = [tadir, filesep, subname];
    cd(subfolder);%project main path
    listFile = cat(1, dir('*.par'), dir('*.rec'), dir('*O2.txt'), dir(['physio', filesep, '*_etco2_timecourse.txt']), ...
        dir(['physio', filesep, '*_eto2_timecourse.txt']),...
        dir(['physio', filesep, 'timecourse*O2.txt']), ...
        dir(['physio', filesep, 'co2*.txt']), dir(['physio', filesep, 'o2*.txt']), ...
        dir(['anato', filesep, subname(end-7:end), '*pre.*']));
    
    if length(dir(['anato', filesep, subname(end-7:end), '*t1_pre.*']))~=2
        disp(subname);
    end
    
    mkdir(sub_tadir);
    for ii = 1:length(listFile)
        copyfile([listFile(ii).folder, filesep, listFile(ii).name], sub_tadir);   
    end
    
end


