function [hn_layerActivations, sub_layeredLabels] = Autoencoder_function(label1, label2, percentage)
% autoencoder_function: create the subset with the two desired labels to 
%                       compare of the desired percentage.
%                       train the autoencoder and encode the subset.
%           INPUT: two desired labels to compare.
%                  percentage of data that the subsets to use.
%           OUTPUT: activation values of the two hidden units of a trained 
%                   autoencoder.
%                   labels of the encoded subset.


    % subsets of the training set corresponding to each label value
    sub_trainLabels = cell(1, 2);
    
    % percentage desired of subsets of the training set with the two classes
    sub_trainLabels_perc = cell(1, 2);
    
    % subsets of labels column corresponding to each label value
    labels_subsets = cell(1, 2);
    
    % array that unifies the two sub_trainLabels_perc
    sub_layeredTrain = [];
    
    % array that unifies the labels_subsets to get a layered subset of the labels column
    sub_layeredLabels = [];
    
    % desired labels to compare
    labels = [label1, label2];
    
    % retrive the subset of the training set (0) with only label1 and label2 classes
    for i = 1:2
        sub_trainLabels{1,i} = loadMNIST(0,labels(1,i));
    end
    
    % for every possible labels: permute the indexes of the i-th subset of
    % the traing set, compute the percentage of observation desired, apply
    % the percentage to the subset with random permutation of indexes,
    % create the labels related.
    for i = 1:2
        rand_indexes = randperm(size(sub_trainLabels{1,i}, 1));
        percent_obs = floor(percentage/100*size(sub_trainLabels{1,i},1)); 
        sub_trainLabels_perc{1,i} = sub_trainLabels{1,i}(rand_indexes(1,1:percent_obs),:);
        labels_subsets{1,i}=labels(1,i)*ones(percent_obs,1);
    end
    
    % unify the two subsets
    sub_layeredTrain = [sub_layeredTrain; sub_trainLabels_perc{1,1}; sub_trainLabels_perc{1,2}];
    
    % unify the two labels column
    sub_layeredLabels = [sub_layeredLabels; labels_subsets{1,1}; labels_subsets{1,2}];
    
    % number of units of the autoencoder hidden layer
    nh = 2;

    % Train the autoencoder with the obtained layered subsets
    autoencoder = trainAutoencoder(transpose(sub_layeredTrain), nh);

    % Test the autoencoder with the considered training set
    encodedData = encode(autoencoder, transpose(sub_layeredTrain));
    
    % Transpose the encoded data for the plotcl
    hn_layerActivations = transpose(encodedData);

end