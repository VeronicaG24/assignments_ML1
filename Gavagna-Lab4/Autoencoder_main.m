%% ------------------------------------------------------------------------
% Neural Networks - Task 2: Autoencoder
% -------------------------------------------------------------------------
%
% Veronica Gavagna
%
% -------------------------------------------------------------------------
%
% Machine Learning for Robotics 1 - 2022/2023
% Assignment 4:
% Task 2: The workflow of the experiment is as follows, illustrated for the
% 10-class MNIST digits problem:
%       2.1: Split the data into subsets of different classes x1, .. , x10. 
%       2.2: Create a training set with only 2 classes.
%            Experiment with different combinations.
%       2.3: Train an autoencoder on the new, reduced training set, call
%            "trainAutoencoder".
%       2.4: Encode the different classes using the encoder obtained.
%       2.5: Plot the data using the "plotcl".
%
% -------------------------------------------------------------------------
%

%% Basic startup functions
clear
close all
clc

%% +++++++++++++++++++++++++++
%  DATA PROCESSING
%  +++++++++++++++++++++++++++

% initialize max and min values for labels
max_label = 10;
min_label = 0;

% percentage of observations stored in the sub_trainLabels
percentage = 50;

% labels of the classes to compare: from 1 to 10 where 10 is 0-class
label1 = 1;
label2 = 8;

% check if the labels insert are correct, if so call the
% autoencoder_function
if (label1 <= max_label && label1 >= min_label) && (label2 <= max_label && label2 >= min_label)
    fprintf('Labels valid\n');
    [hnActivations, labels] = Autoencoder_function(label1, label2, percentage);
else
    fprintf('Labels not valid\n');
    return
end

% initialize the vector that will contain the lables for the plot 
plotLabels = zeros(size(labels,1),1);
% convert the labels returned by the autoencoder_function into "1" for the
% label1 and "2" for label2
for i = 1:size(labels,1)
    if labels(i,1) == label1
        plotLabels(i,1) = 1;
    else
        plotLabels(i,1) = 2;
    end
end

% plot the hidden units activation values using plotcl
figure
plotcl(hnActivations, plotLabels)
title(['Compare: ', num2str(label1),'vs. ', num2str(label2)])
xlabel('Activation value (hidden neuron 1)')
ylabel('Activation value (hidden neuron 2)')

