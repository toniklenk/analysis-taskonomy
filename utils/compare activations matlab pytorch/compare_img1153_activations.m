%% This script: Compare one-tensor activations in Matlab and Pytorch network
% potentially with permutation from weights and biases analysis applied
load('./vgg16_places365/vggnet16_places365.mat');
%%
load('py_activations1153.mat')
im = imread('stimuli_places1/Places365_val_00001153.jpg');
im = imresize(im,[640,640]);
im = single(im);
im = imresize(im, [224,224]) ;
%% quick-check: predictions
[B, I] = sort(predict(net, im));


%% I. Layer 1
% activations
pyt = conv1;
pyt = permute(pyt, [2 3 1]); % fit dimensions
pyt1 = pyt(:,:,1);

mat = activations(net, im, 'conv1_1');
mat1 = mat(:,:,1);

size(mat)
size(pyt)
%% compute L1 difference
sum( abs( pyt - mat), 'all')
%% exclude borders
a = mat(2:223,2:223,:);
b = pyt(2:223,2:223,:);
sum(abs(b - a), 'all')
%% compare whole first layer
for i = 1:64
    p = squeeze(pyt(:,:,i));
    m = squeeze(mat(:,:,i));

    disp(sum(abs(p - m),'all')) 
end


%% II. Individual single activations
%% first convolution filter is quite similar, but still has some difference
% despite model weights being exactly the same

% compare: datatype (float/single)
% to some degree yes, try that, but since we have activation differences
% in orders of integers, this is unlikely


% manually calculate activation I think is made
% plain convolution: 27 weights * imageRGb + bias

% manually calculate activation of matlab
% how does matlab get to the retrievied number

% manually calculate activation of pytorch
% manually implement activation extraction in pytorch




%% single cell 
% matlab
% input 224 224 3
% conv1 3 3 3 64
X = im(1:3, 1:3, :);
X2 = im(1:3,222:224,:);
%% matlab conv1_1: first conv cell
W = net.Layers(2).Weights(:,:,:,1);
B = net.Layers(2).Bias(:,:,1);

%% value from activation
% upper/lower left/right corner
ul = mat1(2,2)
ll = mat1(223,2)
ur = mat1(2,223)
lr = mat1(223,223)



% now: compute corresponding cell in PyTorch






























%% II. use permutation from Layer one on all Layers
net.Layers
%%
mat_layers = ["conv1_1", "conv1_2", "conv2_1", "conv2_2", "conv3_1","conv3_2",...
    "conv3_3","conv4_1","conv4_2","conv4_3","conv5_1","conv5_2","conv5_3"];
%%
py_layers = {conv1, conv1_1, conv2, conv2_1, conv3, conv3_1, conv3_2,...
    conv4, conv4_1, conv4_2, conv5, conv5_1, conv5_2};
%%
for i = 1:length(mat_layers)
    mat = activations(net, im, mat_layers(i));
    pyt = py_layers{i};

    pyt = permute(pyt, [2 3 1]);

    sum( abs( pyt - mat), 'all')
end
%doesnt work

%% III Layer 2
%% I. Layer 1
% activations
pyt = conv1_1;
py_c2_d1 = squeeze(pyt(3,:,:));

mat = activations(net, ones(224,224,3), 'conv1_2');
mat_c2_d1 = mat(:,:,3);
%%
size(mat)
size(pyt)
%% find necessary dimension permutations
pypyt_c2 = permute(pyt, [2 3 1]);

%%
a = mat(2:223,2:223,:);
b = py_c1(:,2:223,2:223);

%% compute L1 difference
sum( abs( pyt - mat), 'all')
%% exclude borders
sum(abs(b - a), 'all')
% no significant difference

%% compare whole first layer
for i = 1:64
    p = squeeze(py_c1(:,:,i));
    m = squeeze(mat(:,:,i));

    disp(sum(abs(p - m),'all')) 
end

