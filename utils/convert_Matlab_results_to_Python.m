clear

MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};

STUDY_NAMES = {'short presentation','long presentation','complexity order','oasis'};
SCALE_NAMES = {'scale2','scale4','scale8','scale16','scale32'};

RESULTS_PATH = './results taskonomy blocked';
EXPORT_PATH = '../Taskonomy Integration/analysis results taskonomy blocked_all_layers';

%%
load(fullfile(RESULTS_PATH,['cnn_prediction_' MODEL_NAMES{1} '.mat']), "dat")

%%


%% correlations
for model = 1:length(MODEL_NAMES)
    for study = 1:length(STUDY_NAMES)
        for scale = 1:length(SCALE_NAMES)
            clear dat
            load(fullfile(RESULTS_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")

            writematrix(dat.c{study}{1}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'ib_correlations.csv'));
            writematrix(dat.c{study}{2}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'self_similarity.csv'));
            writematrix(dat.c{study}{3}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'ib_correlation_ss_partialed.csv'));
        end
    end
end

%% p-values
for model = 1:length(MODEL_NAMES)
    for study = 1:length(STUDY_NAMES)
        for scale = 1:length(SCALE_NAMES)
            clear dat
            load(fullfile(RESULTS_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")

            writematrix(dat.p{study}{1}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'ib_correlations_pvalues.csv'));
            writematrix(dat.p{study}{2}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'self_similarity_pvalues.csv'));
            writematrix(dat.p{study}{3}{scale}, fullfile(EXPORT_PATH, MODEL_NAMES{model}, STUDY_NAMES{study}, SCALE_NAMES{scale}, 'ib_correlation_ss_partialed_pvalues.csv'));
        end
    end
end