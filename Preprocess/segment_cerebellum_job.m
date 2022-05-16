function segment_cerebellum_job(mprimage)
                                         
segment_cerebellum_job(mprimage)

spm_jobman('defaults', 'FMRI')
spm_jobman('run',matlabbatch)
return