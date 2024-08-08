clear
clc
warning off


DATA_PATH = './results taskonomy';

MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};

SAVE_PATH = './results taskonomy blocked';



blocks = [1 repelem(2:17, 3)]';

for model = 1:length(MODEL_NAMES)
    load(fullfile(DATA_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")

    for study=1:4
        for analysis=1:5 
            for scale=1:5
                dat.c{study}{analysis}{scale} = splitapply(@mean, dat.c{study}{analysis}{scale}, blocks);
                dat.p{study}{analysis}{scale} = splitapply(@min, dat.p{study}{analysis}{scale}, blocks);

            end
        end
    end
    save(fullfile(SAVE_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")
end



clear