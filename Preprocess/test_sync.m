dd = dlmread('D:\Lulab\rsCVR_PCA\BrainTumor\hlu_mr1_ba122018\hlu_mr1_ba122018_2_1\Sync_EtCO2_timecourse.txt', ',', 0, 0);
etco2timecourse = dd(:, 1:2);

dd = dlmread('D:\Lulab\rsCVR_PCA\BrainTumor\hlu_mr1_ba122018\Sync_EtO2_timecourse.txt', ',', 0, 0);
eto2timecourse = dd(:, 1:2);

figure(1024)
plot(etco2timecourse(:, 2))
hold on
plot(etco2yang(:, 2))
hold off

figure(2048)
plot(etco2timecourse(:, 1)-etco2yang(:, 1))
hold off

minDelay = -20;
maxDelay = 30;

P = spm_select('FPList','D:\Lulab\rsCVR_PCA\BrainTumor\hlu_mr1_ba122018\CVR_voxelshift_etco2',['^hlu_mr1', '.*\.img']); 
V = spm_read_vols(spm_vol(P));
nVol = size(V,1);

for i=1:nVol
    img_temp=spm_read_vols(V(i));
    img1=reshape(img_temp,[size(mask,1)*size(mask,2)*size(mask,3),1]);
    img(:,i)=img1;
end
sig = img(511415, :)';
[optDelay, optEtCO2, optEtO2, minres, coefs] = cvr_func_findDelayFixed(sig, TR, etco2timecourse, eto2timecourse, [minDelay maxDelay],0,0,1);