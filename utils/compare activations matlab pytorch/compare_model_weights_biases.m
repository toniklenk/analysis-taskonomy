load('./vgg16_places365/vggnet16_places365.mat');
load('pynet_parameters.mat');
%% Matlab net overview
net.Layers

%% I. Comparison of layer 1
%% I.1 Weights
% fetch Matlab net Layer 1 weights
mat = net.Layers(2).Weights;
% fetch PyTorch net Layer 1 weights
pyt = conv1_1_weight;


%% I.1.1 Comparison of layer 1 convolution 1
% Which dimensions correspond to each other
%% weights
a = mat(:,:,:,1);
b = squeeze(pyt(1,:,:,:)); % squeeze when not indexing the last dim

% necessary transformation steps
%c = flip(flip(b,1), 3);
%d = permute(c ,[3 2 1]);
%e = permute(d, [2 1 3]);
%f = flip(e, 2);
% 1->3->3 2->2->1 3->1->2
%direct = flip(permute(b, [2 3 1]),3);

direct = permute(flip(b, 1), [2 3 1]);
%%
sum(abs(a - direct), 'all')




%% I.1.2 Comparison of whole layer 1
size(mat)
size(pyt)

%%
size(mat)
size(permute(flip(pyt, 2), [3 4 2 1]))
%%
A = permute(flip(pyt, 2), [3 4 2 1]); % also align additional dimensions for comparison
B = mat;
sum(abs(A - B), 'all')


%% I.2 Biases
% fetch Matlab net Layer 1 biases
mat = net.Layers(2).Bias;
% fetch PyTorch net Layer 1 biases
pyt = conv1_1_bias;
%%
size(mat)
size(pyt)
%%
mat = squeeze(mat);
pyt = squeeze(pyt);
%% 
% need a transpose
pyt = pyt';
%%
sum(abs(mat - pyt))



%% EDITORIAL: compare to input image dimensionality
matim1153 = imread('stimuli_places1/Places365_val_00001153.jpg');
size(matim1153) % 683 512 3

load('pyimg1153.mat')
size(pyimg1153) % 3 683 512

% weirdly, for input images we need only this transform to match
% maybe this is a reason for the activation differences
matim1153 - permute(pyimg1153, [2 3 1]);


%% EDIT: already applied transformation from step 3, now perfect alignment
%% II. Comparison with layer 1 permutation now for all network layers
%% Compute L1 distances between weights and biases for all network layers
% iterate of all layers
PyWeights = {conv1_1_weight; conv1_2_weight; conv2_1_weight; conv2_2_weight;
    conv3_1_weight; conv3_2_weight; conv3_3_weight;
    conv4_1_weight; conv4_2_weight; conv4_3_weight;
    conv5_1_weight; conv5_2_weight; conv5_3_weight};

MatWeights = {net.Layers(2).Weights;net.Layers(4).Weights; net.Layers(7).Weights;
    net.Layers(9).Weights; net.Layers(12).Weights; net.Layers(14).Weights;
    net.Layers(16).Weights; net.Layers(19).Weights; net.Layers(21).Weights;
    net.Layers(23).Weights; net.Layers(26).Weights; net.Layers(28).Weights;
    net.Layers(30).Weights};
disp("Differences in Weights")
for i = 1:length(PyWeights)
    A = MatWeights{i};
    if i == 1
        B = flip(PyWeights{i}, 2);
    else
        B = PyWeights{i};
    end
    B = permute(B, [3 4 2 1]);
    disp("Layer "+i+ "    "+ sum(abs(A - B),"all"))
end

PyBias = {conv1_1_bias; conv1_2_bias; conv2_1_bias; conv2_2_bias;
    conv3_1_bias; conv3_2_bias; conv3_3_bias;
    conv4_1_bias; conv4_2_bias; conv4_3_bias;
    conv5_1_bias; conv5_2_bias; conv5_3_bias};

MatBias = {net.Layers(2).Bias;net.Layers(4).Bias; net.Layers(7).Bias;
    net.Layers(9).Bias; net.Layers(12).Bias; net.Layers(14).Bias;
    net.Layers(16).Bias; net.Layers(19).Bias; net.Layers(21).Bias;
    net.Layers(23).Bias; net.Layers(26).Bias; net.Layers(28).Bias;
    net.Layers(30).Bias};

disp("Differences in Biases")
for i = 1:length(PyBias)
    A = squeeze(MatBias{i});
    B = squeeze(PyBias{i})';
    disp("Layer "+i+ "    "+ sum(abs(A - B),"all"))
end

% this transformation seems to apply only to the first layer
% investigate transformations in other layers

%% III. Comparison of layer 2
net.Layers
%% III.1 Weights
% fetch Matlab net Layer 1 weights
mat = net.Layers(4).Weights;
% fetch PyTorch net Layer 1 weights
pyt = conv1_2_weight;
%%
size(mat)
size(pyt)

%% III.1.1 Comparison of layer 1 convolution 1
% Which dimensions correspond to each other
%% III.1.1.1 try for 3x3x3 part first
m = mat(:,:,1:3,1);
p = squeeze(pyt(1,1:3,:,:)); % squeeze when not indexing the last dim
%%
direct = permute(p, [2 3 1]); %direct permutation for p to align b
%
sum(abs(m - direct), 'all')
%% III.1.1.2 try for 3x3x64
m = mat(:,:,:,1);
p = squeeze(pyt(1,:,:,:)); % squeeze when not indexing the last dim
%%
direct = permute(p, [2 3 1]);
sum(abs(m - direct), 'all')
% fits. Nice!


%% III.1.2 Comparison of whole layer 1
size(mat)
size(pyt)

%%
size(mat)
size(permute(pyt, [3 4 2 1]))

%%
A = permute(pyt, [3 4 2 1]); % also align additional dimensions for comparison
B = mat;
sum(A - B, 'all')















