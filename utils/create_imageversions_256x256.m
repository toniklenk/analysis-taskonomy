%% Taskonomy models
% resize to 256x256
% rescale to [-1, 1]

%% Places 1
clear
DIR_IMAGES = dir('./image datasets/stimuli_places1/*.jpg');
DIR_SAVE = '../Taskonomy Integration/data_256x256/places1';
%TRANSFORMS = {'untransformed'};

for scale = [2 4 8 16 32]
    for i=1:length(DIR_IMAGES)
        imgname = DIR_IMAGES(i).name;
        imgpath = [DIR_IMAGES(i).folder,'/',imgname];
        imgname = imgname(1:end-4); %need this later
    
        im = imread(imgpath);
        im = imresize(im,[640,640]);
        imv1 = im;
        imv2 = im;
        

        scale_factor = 1/scale;
        for i1=1:scale
            for i2=1:scale
                if mod(i1+i2,2)==0
                    imv1([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                else
                    imv2([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                end
            end
        end
    
        im = single(im); % note: 255 range
        im = im / 127.5 - 1;
        im = imresize(im, [256,256]);
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'full', imgname), "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [256,256]);
        imv1 = imv1 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version1', imgname), "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [256,256]);
        imv2 = imv2 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version2', imgname), "imv2")
    end
end

%% Places 2
clear
DIR_IMAGES = dir('./image datasets/stimuli_places2/*.jpg');
DIR_SAVE = '../Taskonomy Integration/data_256x256/places2';
%TRANSFORMS = {'untransformed'};    

for scale = [2 4 8 16 32]
    for i=1:length(DIR_IMAGES)
        imgname = DIR_IMAGES(i).name;
        imgpath = [DIR_IMAGES(i).folder,'/',imgname];
        imgname = imgname(1:end-4); %need this later
    
        im = imread(imgpath);
        im = imresize(im,[640,640]);
        imv1 = im;
        imv2 = im;
        

        scale_factor = 1/scale;
        for i1=1:scale
            for i2=1:scale
                if mod(i1+i2,2)==0
                    imv1([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                else
                    imv2([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                end
            end
        end
    
        im = single(im); % note: 255 range
        im = im / 127.5 - 1;
        im = imresize(im, [256,256]);
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'full', imgname), "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [256,256]);
        imv1 = imv1 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version1', imgname), "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [256,256]);
        imv2 = imv2 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version2', imgname), "imv2")
    end
end
clear
%% oasis
clear
DIR_IMAGES = dir('./image datasets/stimuli_oasis/*.jpg');
DIR_SAVE = '../Taskonomy Integration/data_256x256/oasis';
%TRANSFORMS = {'untransformed'};

for scale = [2 4 8 16 32]
    for i=1:length(DIR_IMAGES)
        imgname = DIR_IMAGES(i).name;
        imgpath = [DIR_IMAGES(i).folder,'/',imgname];
        imgname = imgname(1:end-4); %need this later
    
        im = imread(imgpath);
        im = imresize(im,[640,640]);
        imv1 = im;
        imv2 = im;
        

        scale_factor = 1/scale;
        for i1=1:scale
            for i2=1:scale
                if mod(i1+i2,2)==0
                    imv1([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                else
                    imv2([1:640*scale_factor]+640*scale_factor*(i1-1),[1:640*scale_factor]+640*scale_factor*(i2-1),:)=127;
                end
            end
        end
    
        im = single(im); % note: 255 range
        im = im / 127.5 - 1;
        im = imresize(im, [256,256]);
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'full', imgname), "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [256,256]);
        imv1 = imv1 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version1', imgname), "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [256,256]);
        imv2 = imv2 / 127.5 - 1;
        save(fullfile(DIR_SAVE, ['scale' num2str(scale)], 'version2', imgname), "imv2")
    end
end

clear


