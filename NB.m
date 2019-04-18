%% Naive Bayes
clear ; close all; clc; 

%% Load data sample and prepare
load X1
attributes = {'a','b','c','d'};
description = 'Botnet Dataset';
[ds, uc, nf] = build_dataset(meas,species,attributes,description);

%% Shuffle the dataset
ds = shuffle_dataset(ds);

%% Prepare test and training sets. 
[train_dataset, test_dataset] = splitting_dataset(ds,0.7);

%% Run Naive Bayes
[train_targets_i, train_targets_l]=grp2idx(train_dataset.(5)); % Change class name into ordinal index
[test_targets_i, test_targets_l]=grp2idx(test_dataset.(5)); % Change class name into ordinal index

predicted_features = naive_bayes(double(train_dataset(:,1:4)), train_targets_i, double(test_dataset(:,1:4)));
etime = toc();

%% Checking error rate

%predicted_features_labels = train_targets_l(predicted_features); %Convert to class name

%bad_predicted = strcmp(cellstr(predicted_features_labels),cellstr(test_dataset.(5)))

bad_predicted = find(test_targets_i~=predicted_features);

error_rate = length(bad_predicted) /size(test_dataset,1);

