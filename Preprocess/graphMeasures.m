function [E_RS, cluster_RS, Q_RS, S_RS]= graphMeasures(processed_folder, folder_name, temp_folder)

cd(temp_folder);
[num, txt, raw] = xlsread('7Networks.xlsx');
M=num(:,2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subject_folder = [processed_folder, filesep, folder_name];
cd([subject_folder]);

load('ccMatrix.mat');
ccMatrix=cc;

for j=1:size(ccMatrix,3)
    disp(j)
    A=squeeze(ccMatrix(:,:,j));
    A_norm = weight_conversion(A, 'normalize');
    A_norm(A_norm<0)=0;
    
    for i=1:length(A)
        A(i,i)=0;
        A_norm(i,i)=0;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [Q1,cpos,cneg,M1]=module_q(A,M);
     Q_RS(j,1)=Q1;
     
     A(A<0)=0;
     E_RS(j,1)=efficiency_wei(A,0);  
     [S_RS(j,1), ~, ~]=segregation(A, M');
     
     cluster_RS(j,1)=mean(clustering_coef_wu(A_norm)); 
end

graphMeasures = cat(2, E_RS, cluster_RS, Q_RS, S_RS);
graphMeasuresName = {'', 'Global_Efficiency', 'Cluster_Coefficient', 'Modularity', 'Segregation'};
finalList = cat(2, cellstr(folder_name), num2cell(graphMeasures));
finalList = cat(1, graphMeasuresName, finalList);
%writecell(finalList, 'graphMeasures.txt', 'Delimiter', ' ');
xlswrite('graphMeasures.xlsx', finalList);

return

