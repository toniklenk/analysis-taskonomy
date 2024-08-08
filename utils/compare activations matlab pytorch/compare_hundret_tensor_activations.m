%% compate one tensor activations in Matlab and Pytorch network
%% potentially with permutation from weights and biases analysis applied


load('./vgg16_places365/vggnet16_places365.mat');

%% I. Layer 1
% activations
py_c1 = conv1;
py_c1_d1 = squeeze(py_c1(3,:,:));

mat_c1 = activations(net, 100*ones(224,224,3), 'conv1_1');
mat_c1_d1 = mat_c1(:,:,3);
%%
size(mat_c1)
size(py_c1)
%% find necessary dimension permutations
py_c1 = permute(py_c1, [2 3 1]);

%%
a = mat_c1(2:223,2:223,:);
b = py_c1(2:223,2:223, :);

%% compute L1 difference
sum( abs( py_c1 - mat_c1), 'all')
%% exclude borders
sum(abs(b - a), 'all')
% no significant difference

%% compare whole first layer
for i = 1:64
    p = squeeze(py_c1(:,:,i));
    m = squeeze(mat_c1(:,:,i));

    disp(sum(abs(p - m),'all')) 
end


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
    mat = activations(net, 100*ones(224,224,3), mat_layers(i));
    pyt = py_layers{i};

    pyt = permute(pyt, [2 3 1]);

    sum( abs( pyt - mat), 'all')
end
%doesn't work

%% III Layer 2
%% I. Layer 1
% activations
py_c2 = conv1_1;
py_c2_d1 = squeeze(py_c2(3,:,:));

mat_c2 = activations(net, ones(224,224,3), 'conv1_2');
mat_c2_d1 = mat_c2(:,:,3);
%%
size(mat_c2)
size(py_c2)
%% find necessary dimension permutations
py_c2 = permute(py_c2, [2 3 1]);

%%
a = mat_c1(2:223,2:223,:);
b = py_c1(:,2:223,2:223);

%% compute L1 difference
sum( abs( py_c2 - mat_c2), 'all')
%% exclude borders
sum(abs(b - a), 'all')
% no significant difference

%% compare whole first layer
for i = 1:64
    p = squeeze(py_c1(:,:,i));
    m = squeeze(mat_c1(:,:,i));

    disp(sum(abs(p - m),'all')) 
end
