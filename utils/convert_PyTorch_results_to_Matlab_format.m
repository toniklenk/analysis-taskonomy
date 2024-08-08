%% vgg16
clear

DATASET_NAMES = {'places1', 'places2', 'oasis'};
SCALE_NAMES = {'scale2','scale4','scale8','scale16','scale32'};

SAVE_PATH = './data vgg16';
 
task = 'vgg16'; % placeholder for different taskonomy models
%task = 'autoencoding';
PYTORCH_RESULTS_PATH = '../Taskonomy Integration/results_vgg16';


for dataset = 1:3
    for scale = 1:5
        corr{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'correlations.csv')));
        sim{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'selfsimilarity.csv')));
        l2{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'l2norm.csv')));
    end

    % combine into single struct
    cnn.corr = corr;
    cnn.sim = sim;
    cnn.l2 = l2;

    % export csv for one study
    save(fullfile(SAVE_PATH, task, ['cnn_' task '_res_' DATASET_NAMES{dataset} '.mat']), "cnn");
end

clear


%% autoencoding
clear


DATASET_NAMES = {'places1', 'places2', 'oasis'};
SCALE_NAMES = {'scale2','scale4','scale8','scale16','scale32'};

SAVE_PATH = './data taskonomy';
task = 'autoencoding';
PYTORCH_RESULTS_PATH = '../Taskonomy Integration/results_taskonomy';

for dataset = 1:1
    for scale = 1:5
        corr{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'correlations.csv')));
        sim{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'selfsimilarity.csv')));
        l2{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, task, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'l2norm.csv')));
    end

    % combine into single struct
    cnn.corr = corr;
    cnn.sim = sim;
    cnn.l2 = l2;

    % export csv for one study
    save(fullfile(SAVE_PATH, task, ['cnn_' task '_res_' DATASET_NAMES{dataset} '.mat']), "cnn");
end

clear


%% all Taskonomy models
clear

MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};

DATASET_NAMES = {'places1', 'places2', 'oasis'};
SCALE_NAMES = {'scale2','scale4','scale8','scale16','scale32'};

SAVE_PATH = './data taskonomy';
PYTORCH_RESULTS_PATH = '../Taskonomy Integration/results_taskonomy';

for model = 1:length(MODEL_NAMES)
    for dataset = 1:length(DATASET_NAMES)
        for scale = 1:length(SCALE_NAMES)
            corr{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, MODEL_NAMES{model}, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'correlations.csv')));
            sim{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, MODEL_NAMES{model}, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'selfsimilarity.csv')));
            l2{scale} = single(readmatrix(fullfile(PYTORCH_RESULTS_PATH, MODEL_NAMES{model}, DATASET_NAMES{dataset}, SCALE_NAMES{scale}, 'l2norm.csv')));
        end
    
        % combine into single struct
        cnn.corr = corr;
        cnn.sim = sim;
        cnn.l2 = l2;
    
        % export csv for one study
        save(fullfile(SAVE_PATH, MODEL_NAMES{model}, ['cnn_' MODEL_NAMES{model} '_res_' DATASET_NAMES{dataset} '.mat']), "cnn");
    end
end

clear
