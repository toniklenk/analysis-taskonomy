clear
clc
warning off


% import correlation, self-similarity and l2norm (computed in PyTorch)
DATA_PATH = './data taskonomy';

MODEL_NAMES = {'autoencoding' 'depth_euclidean' 'jigsaw' 'reshading' ...
               'edge_occlusion' 'keypoints2d' 'room_layout' ...  %'colorization' currently not working
               'curvature' 'edge_texture' 'keypoints3d' 'segment_unsup2d' ...
               'class_object' 'egomotion' 'nonfixated_pose' 'segment_unsup25d' ...
               'class_scene' 'fixated_pose' 'normal' 'segment_semantic' ...
               'denoising' 'inpainting' 'point_matching' 'vanishing_point'};


BEHAVIOR_PATH = './behavior';
BEHAVIOR_NAMES = {'study1_places1_short.mat', 'study2_places1.mat', 'study3_places2.mat', 'study4_oasis.mat'};

DATASET_NAMES = {'places1', 'places1', 'places2', 'oasis'}; % study 1&2 use only different behavioral data

SAVE_PATH = './results taskonomy';




%%


for model = 1:length(MODEL_NAMES)

    for study=1:4
        
        load(fullfile(DATA_PATH, MODEL_NAMES{model}, ['cnn_' MODEL_NAMES{model} '_res_' DATASET_NAMES{study} '.mat']));

        % behaviour
        load(fullfile(BEHAVIOR_PATH, BEHAVIOR_NAMES{study}));
        beh=mean(res.beauty,2); 

        for scale=1:5
    
            % images
            for img=1:size(cnn.corr{1},2)
                
                % COMPUTE CORRELATIONS

                %choose dnn variables (integration, self-similarity, competition)
                int_=-cnn.corr{scale}(:,img);
                sim_=cnn.sim{scale}(:,img);
                l2_=cnn.l2{scale}(:,img);
    
                %integration effect
                [r,p]=corr(beh,int_,'type','Spearman');
                dat.c{study}{1}{scale}(img,1)=r;
                dat.p{study}{1}{scale}(img,1)=p;
    
                %part-similarity effect
                [r,p]=corr(beh,sim_,'type','Spearman');
                dat.c{study}{2}{scale}(img,1)=r;
                dat.p{study}{2}{scale}(img,1)=p;
    
                %integration effect (unique, wrt part-similarity)
                [r,p]=partialcorr(beh,int_,sim_,'type','Spearman');
                dat.c{study}{3}{scale}(img,1)=r;
                dat.p{study}{3}{scale}(img,1)=p;
    
                %L2 for whole image
                [r,p]=corr(beh,l2_,'type','Spearman');
                dat.c{study}{4}{scale}(img,1)=r;
                dat.p{study}{4}{scale}(img,1)=p;
    
                %integration effect (unique, wrt L2)
                [r,p]=partialcorr(beh,int_,l2_,'type','Spearman');
                dat.c{study}{5}{scale}(img,1)=r;
                dat.p{study}{5}{scale}(img,1)=p;
    
                %store all variables
                dat.data{study}{scale}{img}=[beh,int_,sim_,l2_];
    
                %partial out complexity and order
                if study==3
                    dat.corr_co(:,:,img)=corr([beh,int_,mean(res.complexity,2),mean(res.order,2)],'type','Spearman');
                    [r,p]=partialcorr(beh,int_,mean(res.complexity,2),'type','Spearman');
                    dat.c_co{1}{1}{scale}(img,1)=r;
                    dat.p_co{1}{1}{scale}(img,1)=p;
                    [r,p]=partialcorr(beh,int_,mean(res.order,2),'type','Spearman');
                    dat.c_co{2}{1}{scale}(img,1)=r;
                    dat.p_co{2}{1}{scale}(img,1)=p;
                end                
                
            end
        end
    
        % COMPUTE GLMs
        
        %choose behaviour
        beh=mean(res.beauty,2);
    
        %compute regression for each spatial scale
        for scale=1:5
    
            X=-cnn.corr{scale};
    
            %zscore
            X=zscore(X,0,1);
            Y=beh;
            Y=zscore(Y,0,1);
    
            %perform cross-validated glm
            Z=zeros(size(X,1),1);
    
            % Leave-One-Out cross-validation
            for i=1:size(X,1)
    
                %split into train and test set
                all_i=1:size(X,1);
                train_i=all_i(not(all_i==i));
                test_i=i;
                
                %get regression coefficients
                B=regress(Y(train_i),[X(train_i,:),ones(size(Y(train_i)))]);
    
                %get prediction for left-out image
                Z(i,1)=sum(B(1:end-1)'.*X(test_i,:));
    
            end
    
            %evaluate predictions
            [r,p]=corr(Z,Y,'type','Spearman');
            dat.r_crossval{study}{1}(1,scale)=r;
            dat.p_crossval{study}{1}(1,scale)=p;
    
        end
    
    end
    save(fullfile(SAVE_PATH,['cnn_prediction_' MODEL_NAMES{model} '.mat']), "dat")
end


clear