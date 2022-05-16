function segment_cerebellum(mprimage)

matlabbatch{1}.spm.tools.suit.isolate_seg.source = {[mprimage ',1']};
matlabbatch{1}.spm.tools.suit.isolate_seg.bb = [-90 90
                                                -126 90
                                                -72 108];
matlabbatch{1}.spm.tools.suit.isolate_seg.maskp = 0.2;                                            
matlabbatch{1}.spm.tools.suit.isolate_seg.keeptempfiles = 0;                                            

return