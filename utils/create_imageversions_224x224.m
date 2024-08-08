%% Places 1
dir_images = dir('./image datasets/stimuli_places1/*.jpg');
dir_save = '../Taskonomy Integration/data_224x224/places1';

for scale = [2 4 8 16 32]
    for i=1:length(dir_images)
        imgname = dir_images(i).name;
        imgpath = [dir_images(i).folder,'/',imgname];
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
        im = imresize(im, [224,224]);
        save([dir_save '/scale' num2str(scale) '/full/',imgname], "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version1/',imgname], "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version2/',imgname], "imv2")
    end
end


%% Places 2
clear
dir_images = dir('./image datasets/stimuli_places2/*.jpg');
dir_save = '../Taskonomy Integration/data_224x224/places2';

for scale = [2 4 8 16 32]
    for i=1:length(dir_images)
        imgname = dir_images(i).name;
        imgpath = [dir_images(i).folder,'/',imgname];
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
        im = imresize(im, [224,224]);
        save([dir_save '/scale' num2str(scale) '/full/',imgname], "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version1/',imgname], "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version2/',imgname], "imv2")
    end
end

%% OASIS
dir_images = dir('./image datasets/stimuli_oasis/*.jpg');
dir_save = '../Taskonomy Integration/data_224x224/oasis';

for scale = [2 4 8 16 32]
    for i=1:length(dir_images)
        imgname = dir_images(i).name;
        imgpath = [dir_images(i).folder,'/',imgname];
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
        im = imresize(im, [224,224]);
        save([dir_save '/scale' num2str(scale) '/full/',imgname], "im")
    
        imv1 = single(imv1); % note: 255 range
        imv1 = imresize(imv1, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version1/',imgname], "imv1")
    
        imv2 = single(imv2); % note: 255 range
        imv2 = imresize(imv2, [224,224]);
        save([dir_save '/scale' num2str(scale) '/version2/',imgname], "imv2")
    end
end