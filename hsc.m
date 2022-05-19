% definitely make sure we've got the validation data set
if ~exist('validation.zip')%#ok
 % you need the validation data only for running the prototype app
 try
 validation_url = 'https://archive.physionet.org/pn3/challenge/2016/
validation.zip';
 websave('validation.zip', validation_url);
 catch
 warning("Failed to access heart sound validation data on physionet.org
 - check whether path %s needs updating", validation_url)
 end
 unzip('validation.zip', 'Data');
end
% by default, skip downloading training data (may take long time, 185 MB)
% (though you won't be able to execute the feature extraction yourself below)
getTrainingData = false;
if ~exist('HeartSoundClassificationNew-FX-Oct19.zip') && getTrainingData%#ok
 % fetch training data from physionet site.
 % NOTE: unless you plan to execute the feature extraction, don't worry if
 there is an error here,
 % we only need access to the training set to run the feature
 extraction
 try
 training_url = 'https://archive.physionet.org/pn3/challenge/2016/
training.zip';
 websave('training.zip', training_url);
 catch
 warning("Failed to access heart sound training data on
 physionet.org - check your internet connection or whether path %s needs
 updating",training_url)
 end
 unzip('training.zip', 'Data/training')
end
% make sure we have copies of the two example files in the main directory
if exist('Data/validation')%#ok
 copyfile 'Data/validation'/a0002.wav;
 copyfile 'Data/validation/a0011.wav';
end
addpath(genpath(pwd));
addpath('./HelperFunctions');
warning off; % suprress warning messages
[PCG_abnormal, fs] = audioread('a0002.wav');
% Plot the sound waveform
plot(PCG_abnormal(1:fs*3))
[PCG_normal, fs] = audioread('a0011.wav');
1
% Plot the sound waveform
plot(PCG_normal(1:fs*3))
signalAnalyzer(PCG_normal, PCG_abnormal)
% expecting the training data in subfolders of 'Data\training\*': "traininga", etc
training_fds = fileDatastore(fullfile(pwd, 'Data', 'training'), 'ReadFcn',
 @importAudioFile, 'FileExtensions', '.wav','IncludeSubfolders',true);
data_dir = fullfile(pwd, 'Data', 'training');
folder_list = dir([data_dir filesep 'training*']);
reference_table = table();
for ifolder = 1:length(folder_list)
 disp(['Processing files from folder: ' folder_list(ifolder).name])
 current_folder = [data_dir filesep folder_list(ifolder).name];
 % Import ground truth labels (1, -1) from reference. 1 = Normal, -1 =
 Abnormal
 reference_table = [reference_table; importReferencefile([current_folder
 filesep 'REFERENCE.csv'])];
end
runExtraction = false; % control whether to run feature extraction (will take
 several minutes)
 % Note: be sure to have the training data downloaded before executing
 % this section!
if runExtraction | ~exist('FeatureTable.mat')%#ok
 % Window length for feature extraction in seconds
 win_len = 5;
 % Specify the overlap between adjacent windows for feature extraction in
 percentage
 win_overlap = 0;
 % Initialize feature table to accumulate observations
 feature_table = table();
 % Use Parallel Computing Toobox to speed up feature extraction by
 distributing computation across available processors
 % Create partitions of the fileDatastore object based on the number of
 processors
 n_parts = numpartitions(training_fds, gcp);
 % Note: You could distribute computation across available processors by
 using
 % parfor instead of "for" below, but you'll need to omit keeping track
 % of signal lengths
 parfor ipart = 1:n_parts
 % Get partition ipart of the datastore.
 subds = partition(training_fds, n_parts, ipart);
 % Extract features for the sub datastore
 [feature_win,sampleN] = extractFeatures(subds, win_len, win_overlap,
 reference_table);
2
 % and append that to the overall feature table we're building up
 feature_table = [feature_table; feature_win];
 % Display progress
 disp(['Part ' num2str(ipart) ' done.'])
 end
 save('FeatureTable', 'feature_table');
else % simply load the precomputed features
 load('FeatureTable.mat');
end
% Take a look at the feature table
disp(feature_table(1:5,:))
classificationLearner
% tabulate classes in training data
grpstats_all = grpstats(feature_table, 'class', 'mean');
disp(grpstats_all(:,'GroupCount'))
% using split function defined at end of script to divide feature table
% into training and test set, holding out 30%
[training_set, test_set] = splitDataSets(feature_table,0.3);
% Assign higher cost for misclassification of abnormal heart sounds
C = [0, 10; 1, 0];
% Create a random sub sample (to speed up training) of 1/4 of the training set
%subsample = randi([1 height(training_set)], round(height(training_set)/4),
 1);
% OR train on the whole training set
subsample = 1:height(training_set);
rng(1);
% Create a 5-fold cross-validation set from training data
cvp = cvpartition(length(subsample),'KFold',5);
% Step 2: train the model with hyperparameter tuning (unless you simply
% load an existing pre-trained model)
% train ensemble of decision trees (random forest)
disp("Training Ensemble classifier...")
% bayesian optimization parameters (stop after 15 iterations)
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',cvp,...
 'AcquisitionFunctionName','expected-improvementplus','MaxObjectiveEvaluations',15);
trained_model = fitcensemble(training_set(subsample,:),'class','Cost',C,...
 'OptimizeHyperparameters',
{'Method','NumLearningCycles','LearnRate'},...
 'HyperparameterOptimizationOptions',opts)
% Step 3: evaluate accuracy on held-out test set
3
% Predict class labels for the validation set using trained model
% NOTE: if training ensemble without optimization, need to use
 trained_model.Trained{idx} to predict
predicted_class = predict(trained_model, test_set);
conf_mat = confusionmat(test_set.class, predicted_class);
conf_mat_per = conf_mat*100./sum(conf_mat, 2);
% Visualize model performance in heatmap
labels = {'Abnormal', 'Normal'};
heatmap(labels, labels, conf_mat_per, 'Colormap',
 winter, 'ColorbarVisible','off');
runNCA = true; % control whether to see NCA running or just load the
 selected features
if ~runNCA && exist('SelectedFeatures.mat')%#ok
 % Load saved array of selected feature indexes
 load('SelectedFeatures.mat')
else % Perform feature selection with neighborhood component analysis
 rng(1);
 % first, let's make sure we've split the data (in case we skipped the
 % programmatic model training sections above)
 if ~exist('training_set')
 [training_set, test_set] = splitDataSets(feature_table,0.3);
 end
 mdl = fscnca(table2array(training_set(:,1:27)), ...
 table2array(training_set(:,28)), 'Lambda', 0.005, 'Verbose', 0);
 % Select features with weight above 1
 selected_feature_indx = find(mdl.FeatureWeights > 0.1);
 % Plot feature weights
 stem(mdl.FeatureWeights,'bo');
 % save for future reference
 save('SelectedFeatures', 'selected_feature_indx');
end
% Display list of selected features
disp(feature_table.Properties.VariableNames(selected_feature_indx))
trainReducedModel = false; % control whether to re-train this model or load
 it from a previous run
if trainReducedModel | ~exist('TrainedEnsembleModel_FeatSel.mat')%#ok
 % configure key parameters: cross validation, cost, and hyperparameter
 % tuning
 rng(1)
 cvp = cvpartition(length(subsample),'KFold',5);
 C = [0, 10; 1, 0]; % Assign higher cost for misclassification
 of abnormal heart sounds
4
 opts =
 struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',cvp,...
 'AcquisitionFunctionName','expected-improvementplus','MaxObjectiveEvaluations',10);
 % now we are ready to train...
 trained_model_featsel =
 fitcensemble(training_set(subsample,selected_feature_indx),training_set.class(subsample),'Cost',C,...
 'OptimizeHyperparameters',
{'Method','NumLearningCycles','LearnRate'}, 'HyperparameterOptimizationOptions',opts)
 % save the model for later reference
 save('TrainedWaveletModel_FeatSel', 'trained_model_featsel');
else
 load('TrainedEnsembleModel_FeatSel.mat')
end
