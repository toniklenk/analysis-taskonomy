clear
clc
warning off

MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};

DATASET_NAMES = {'places1', 'places2', 'oasis'};


IMPORT_PATH = './data mat/integration';
EXPORT_PATH = './data mat/integration blocked';

%%

blocks = [1 repelem(2:17, 3)];

m_blocked = nan()
%%

[nRows, nCols] = size(tmp)
for i=1:nRows

    x=tmp(i,:)

end

%%
for m = 1:length(MODEL_NAMES)
    for d =1:length(DATASET_NAMES)
        load(fullfile(IMPORT_PATH, MODEL_NAMES{1}, ['cnn_' MODEL_NAMES{m} '_res_' DATASET_NAMES{d} '.mat']), "cnn")


        dat.c{study}{analysis}{scale} = splitapply(@mean, dat.c{study}{analysis}{scale}, blocks);
        dat.p{study}{analysis}{scale} = splitapply(@min, dat.p{study}{analysis}{scale}, blocks);

    end
    save(fullfile(SAVE_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")
end



clear