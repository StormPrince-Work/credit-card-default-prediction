#### Imports
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import csv
import sys
import os
from imblearn.over_sampling import SMOTE
from keras import Sequential, layers, models
from keras.layers import Dense,  Dropout
from keras.models import load_model
from keras.initializers import Constant
from keras.metrics import Recall, Metric
from keras.callbacks import EarlyStopping
from keras_tuner import HyperModel, Hyperband, Objective
from sklearn.utils.class_weight import compute_class_weight
from keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import f1_score

#Mormalises the train, val and test sets for better machine learning performance.
def array_normalise(train, val, test):
  # Converts train, validation and test data into numpy arrays.
  train_features = np.array(train)
  val_features = np.array(val)
  test_features = np.array(test)
  
  # Normalises test, validation and test sets for impoves ML performance.
  scaler = StandardScaler()
  train_features = scaler.fit_transform(train_features)
  val_features = scaler.transform(val_features)
  test_features = scaler.transform(test_features)
  # Returns train validation and test features repsectively.
  return train_features, val_features, test_features

# Outputs labels for train/val/test).
def label_pop_ccd(train, val, test):
  # Sets 'default payment next month' to be our class labels.
  train_labels = np.array(train.pop('default payment next month'))
  val_labels = np.array(val.pop('default payment next month'))
  test_labels = np.array(test.pop('default payment next month'))
  return train_labels, val_labels, test_labels

def create_bar_chart(optimizing_data, title_suffix):
    # Sets up figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Number of groups
    num_groups = len(optimizing_data)

    # Bar width
    bar_width = 0.25

    # Set position of bar on X axis
    indices = list(range(num_groups))
    r1 = indices
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Defines colours for our bars
    colours = ['#4682b4', '#ff7f50', '#9dc183']

    # Make the plot
    ax.bar(r1, optimizing_data['Val Precision'], width=bar_width,color=colours[0], label='Precision')
    ax.bar(r2, optimizing_data['Val Recall'], width=bar_width, color=colours[1],  label='Recall')
    ax.bar(r3, optimizing_data['Val F1 Score'], width=bar_width, color=colours[2], label='F1 Score')


    # Add xticks on the middle of the group bars
    ax.set_xlabel('Category', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(num_groups)])
    ax.set_ylim(0, 0.8)

    # Set the xtick labels to be a combination of Optimiser and Activation Function and objectives
    categories = [
       f"{row['Optimiser']}, {row['Activation Function']}, {row['Objective']}"
       for _, row in optimizing_data.iterrows()
    ]
    split_categories = [
    '\n'.join(category.split(', ')) for category in categories
    ]
    ax.set_xticklabels(split_categories, rotation=0, ha='center', fontsize=9)

    # Create legend and title
    ax.legend()
    plt.title(f'Validaton Performance Metrics across Activation Functions, Optimsers and Metric Search - for {title_suffix} Dataset')
    plt.tight_layout() 
    return fig, ax

# Creates train/val/test features.
def original_data(data):
  # Shuffles and splits data set into train, validation and test labels. With a 60%-20%-20% split respectively.
  # Ensures no leakage of test/validation data into train data, allows for evaluation of models performance.
  train_set, test_set = train_test_split(data, test_size =0.2, random_state = 42)
  train_set, val_set = train_test_split(train_set, test_size=0.25, random_state = 42)

  # Forms np arrays of train, validation and test labels. 
  train_labels, val_labels, test_labels = label_pop_ccd(train_set, val_set, test_set)

  # Forms np arrays of our test data
  train_features, val_features, test_features = array_normalise(train_set, val_set, test_set)

  # Returns data/labels for train, validation and test represectively.
  return train_features, train_labels, val_features, val_labels, test_features, test_labels

#Undersamples majority class for train data/
def undersampling(data):
  # 60%-20%-20% split between train, validation and test set respectively.
  train_set, test_set = train_test_split(data, test_size =0.2, random_state = 42)
  train_set, val_set = train_test_split(train_set, test_size=0.25, random_state = 42)
  
	# Gets size of class
  count_class_0, count_class_1 = train_set['default payment next month'].value_counts()

  # Seperates defaults and non-defaults
  CCD_class_0 = train_set[train_set['default payment next month'] == 0]
  CCD_class_1 = train_set[train_set['default payment next month'] == 1]
  
  # Randomly samples the majority class to have identical number of samples.
  CCD_class_0_under = CCD_class_0.sample(count_class_1, random_state=42)

  # Joins now under sampled majority class with untouched minoritiy class.
  train_under = pd.concat([CCD_class_0_under, CCD_class_1], axis=0)

  # Shuffles the train undersampled dataset.
  train_under = train_under.sample(frac=1, random_state=42).reset_index(drop=True)
  
  # Forms np arrays of train undersampled, validation and test labels.
  train_labels, val_labels, test_labels = label_pop_ccd(train_under, val_set, test_set)
  
  # Forms np arrays containing for train, validation and test samples.
  train_features, val_features, test_features = array_normalise(train_under, val_set, test_set)
  
  return train_features, train_labels, val_features, val_labels, test_features, test_labels

#Apply SMOTE to create synthetic train data for minority class.
def apply_smote(data):
  # Shuffles and splits data set into train, validation and test labels. With a 60%-20%-20% split respectively.
  # Ensures no leakage of test/validation data into train data, allows for evaluation of models performance.
  train_set, test_set = train_test_split(data, test_size =0.2, random_state = 42)
  train_set, val_set = train_test_split(train_set, test_size=0.25, random_state = 42)

  # Forms np arrays of train, validation and test labels. Removed them from the sets.
  train_labels, val_labels, test_labels = label_pop_ccd(train_set, val_set, test_set)

  # Creates synthetic data for the minority class.
  smote = SMOTE(random_state=42)
  train_set, train_labels = smote.fit_resample(train_set, train_labels)

  # Forms np arrays of our test data
  train_features, val_features, test_features = array_normalise(train_set, val_set, test_set)

  # Returns train data with synthetic samples. As well as the train/validation/test data and class labels.
  return train_features, train_labels, val_features, val_labels, test_features, test_labels

#Limitation of keras tuner requires model_builder to have one parameter, so use of nested functions neccisary.
def make_model_builder(ACTIVATION, OPTIM_NAME, input_shape):
  # Builds model and automatically searches for best hyperparameter combinations.
  def model_builder(hp):
    model = Sequential()
    # Searches for best number of layers, e.g. complexity of model.
    for i in range(hp.Int('num_layers', 1, 5)):
      if i == 0:  
        # First input layer definition needs to define shape, Dense layer neurons can vary to increase complexity.
        model.add(Dense(units=hp.Int('units_' + str(i), 
                                     min_value=10, max_value=150, step=10),
                                     activation=ACTIVATION, 
                                     input_shape = input_shape))

        # Creates Dropout layer and searces for optimal dropout value. Reduces overfitting.
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), 
                                        min_value=0.0, max_value=0.5, step=0.1)))
      else:
        # Adds hidden Dense and Dropout layers.
        model.add(Dense(units=hp.Int('units_' + str(i), 
                                     min_value=10, max_value=150, step=10), 
                                     activation=ACTIVATION))
        
        # Creates Dropout layer and searces for optimal dropout value. Reduces overfitting.
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), 
                                        min_value=0.0, max_value=0.5, step=0.1)))
    
    # Add output layer with sigmoid activation for binary classification, outputs prediction 1 or 0.
    model.add(Dense(1, activation = 'sigmoid'))

    # Experiments to determine optimal value for learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5, 1e-6])

    # Varies input parameters for different optimisers
    if OPTIM_NAME == 'Adam':
      #Compiles model or Adam.
      model.compile(optimizer=Adam(learning_rate=hp_learning_rate),
                    loss = 'binary_crossentropy',
                    metrics=['accuracy', 'Precision', Recall(name = 'recall')])
    elif OPTIM_NAME == 'RMSprop':
      # Determines best momenum value across possible choices.
      hp_momentum = hp.Choice('momentum', values=[0.0,0.1,0.3,0.5,0.7,0.8,0.9])

      #Compiles model using for RMSprop with specified learning rate and momentum values.
      model.compile(optimizer=RMSprop(learning_rate=hp_learning_rate, 
                                      momentum = hp_momentum),
                                      loss = 'binary_crossentropy',
                                      metrics=['accuracy', 'Precision', Recall(name = 'recall')])
    else:
      # Exits code exucution if unexpected optimiser is called.
      sys.exit("Stopping the script due to unexpected optimizer.")
    return model
  return model_builder

def build_hypermodel(train_features, train_labels ,val_features, val_labels, ACTIVATION_NAME, OPTIMISER_NAME, METRIC, BATCH_SIZE, EPOCHES, preprocessing_name):
  # Computed class weights if unbalanced data is inputted.
  class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
  class_weights = {class_label: weight for class_label, weight in zip(np.unique(train_labels), class_weights)}

  # Input shape definition
  input_shape = (train_features.shape[1],)

  #Creates model with specified Activation function and Optimiser.
  model_builder_with_params = make_model_builder(ACTIVATION_NAME, OPTIMISER_NAME, input_shape)

  # Ensure keras rebuilds tuner, by using time as a unique identifier.
  unique_id = int(time.time())

  #Intitalies instance of tuner, with metric, add metrics t
  if METRIC == 'val_loss': 
    # Builds hyperparameter tuner if metric is minimised.
    tuner = Hyperband(model_builder_with_params,
                      objective=Objective(METRIC, direction='min'),
                      max_epochs=EPOCHES,
                      hyperband_iterations=4,
                      factor=3,
                      directory='my_dir',
                      project_name=f'{preprocessing_name}_{BATCH_SIZE}_{METRIC}_{OPTIMISER_NAME}_{ACTIVATION_NAME}_{unique_id}')
    
    # Haults training if no improvement after 3 epoches. If no improvement restores previous best weights.
    stop_early = EarlyStopping(monitor=METRIC, patience=1,mode="min", restore_best_weights=True)
  else:
    # Builds hyperparameter tuner if metric needs to be maximised.
    tuner = Hyperband(model_builder_with_params,
                      objective=Objective(METRIC, direction='max'),
                      max_epochs=EPOCHES,
                      hyperband_iterations=4,
                      factor=3,
                      directory='my_dir',
                      project_name=f'{preprocessing_name}_{BATCH_SIZE}_{METRIC}_{OPTIMISER_NAME}_{ACTIVATION_NAME}_{unique_id}')
    # Haults training if no improvement after 3 epoches. If no improvement restores previous best weights.
    stop_early = EarlyStopping(monitor=METRIC, patience=2,mode="max",restore_best_weights=True)
  
	# Searches for best hyperparameter combinations.
  tuner.search(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHES, class_weight=class_weights,  validation_data=(val_features, val_labels), callbacks=[stop_early], verbose=2)

  # Retreves best hyperparameters corresponding to metric being maximised.
  best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

  # Builds model with best parameters obtained by tuner.
  hypermodel = tuner.hypermodel.build(best_hps)

  # Stores history of model during training process
  history = hypermodel.fit(train_features, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHES, class_weight=class_weights, validation_data=(val_features, val_labels))

  # Gets the best epoch (prevents overfitting).
  val_metric_per_epoch = history.history[METRIC]

  if METRIC == 'val_loss': 
    # Retreives epoch corresponding to minimum loss.
    best_epoch = val_metric_per_epoch.index(min(val_metric_per_epoch)) + 1
  else:
    #Retreives epochh corresponing to maximum precision/recall.
    best_epoch = val_metric_per_epoch.index(max(val_metric_per_epoch)) + 1

  # Retrains model best epoch number, reduces overfitting.
  hypermodel.fit(train_features, train_labels,batch_size=BATCH_SIZE,epochs=best_epoch, class_weight=class_weights, validation_data=(val_features, val_labels))

  # Returns hypermodel, the history and the best hyperparameters used.
  return hypermodel, history, best_hps, best_epoch

# Creates folders to store models, validation performance and model histories.
os.makedirs("Final_Models", exist_ok=True)
os.makedirs("Validation_Performance", exist_ok=True)
os.makedirs("Test_Performance", exist_ok=True)
os.makedirs("Model_History", exist_ok=True)
os.makedirs("Best_Hyperparameters", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

# Imports 'CCD.xls'.
CCD_import = pd.read_excel('CCD.xls', header=1)

# Preverves original dataset and removes duplicates
CCD_raw = (CCD_import.copy()).drop_duplicates()

# One hot encoding catagorical data to improve deep learning performance.
CCD = pd.get_dummies(CCD_raw, columns=["SEX", "EDUCATION", "MARRIAGE"]).astype(int)

# Defines Optimisers, Activation Functions, Metric goals and datapreprocessing methods to iterate over.
optims = ['Adam','RMSprop']
activations = ['relu','tanh','swish']
metrics = ['val_precision', 'val_loss']
data_preprocessing_methods = [("Original", original_data), ("Smote", apply_smote), ("Undersampling",undersampling)]


#Loops over Optimisers, Activation Functions, Metrics and Datapreprocessing methods.
run_tuner = False
if run_tuner == True:
    for preprocessing_name, preprocess_method in data_preprocessing_methods:
        train_features, train_labels, val_features, val_labels, test_features, test_labels = preprocess_method(CCD)
        for OPTIM in optims:
            for ACTIVATION in activations:
                for METRIC in metrics:
                    # Builds hypermodel, fetches history, best hyperparameters and best epoch.
                    hypermodel, history, best_hps, best_epoch = build_hypermodel(train_features, train_labels ,val_features, val_labels, ACTIVATION, OPTIM, METRIC, 128, 20, preprocessing_name)
       
					          #Saves model history.
                    np.save(f'Model_History/{preprocessing_name}_{OPTIM}_{ACTIVATION}_{METRIC}.npy',history.history)

                    # Creates name and saves model.
                    model_name = f'Final_Models/{preprocessing_name}_{OPTIM}_{ACTIVATION}_{METRIC}.keras'
                    hypermodel.save(model_name)

                    # Returns best hyperparameters, saves in file with correponding preprocessing/optimiser/activation function/maximised metric name
                    best_hyperparameters = best_hps.values
                    # Defines fine name and writes best hyperparameters to file.
                    hyperparamter_name = f'Best_Hyperparameters/{preprocessing_name}_{OPTIM}_{ACTIVATION}_{METRIC}.csv'
                    with open(hyperparamter_name, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([f'Epoch:, {best_epoch}'])
                        for hp_name, hp_value in best_hyperparameters.items():
                            writer.writerow([hp_name, hp_value])


# Evalutates model on validation and test set.
evaluation =  False
if evaluation == True:
    for preprocessing_name, preprocess_method in data_preprocessing_methods:
        # Defines and resets column labels.
        val_list = [['Preprocessing Method', 'Optimiser', 'Activation Function', 'Objective', 'Val Loss','Val Accuracy','Val Precision', 'Val Recall', 'Val F1 Score']]
        test_list = [['Preprocessing Method', 'Optimiser', 'Activation Function', 'Objective', 'Test Loss','Test Accuracy','Test Precision', 'Test Recall', 'Test F1 Score']]
        # Returns train/validaton/test feature and labels from preprocessing method specified.
        train_features, train_labels, val_features, val_labels, test_features, test_labels = preprocess_method(CCD)
        for OPTIM in optims:
            for ACTIVATION in activations:
                for METRIC in metrics:
                   # Sets model name and loads corresponding model.
                   model_name = f'Final_Models/{preprocessing_name}_{OPTIM}_{ACTIVATION}_{METRIC}.keras'
                   best_model = load_model(model_name)
                   
                   # Evaluates model performance on validation set.
                   validation_loss, validation_accuracy, validation_precision, validation_recall = best_model.evaluate(val_features, val_labels, verbose = 1)

                   # Returns predictions of the model for validation set.
                   val_pred = (best_model.predict(val_features)>0.5).astype(int)

                   # Calculates F1 score for validation set.
                   f1_val = f1_score(val_labels, val_pred)

                   # Appends results for each model into a list.
                   val_list.append([preprocessing_name, OPTIM, ACTIVATION, METRIC, round(validation_loss,2), round(validation_accuracy,2), round(validation_precision,2), round(validation_recall,2), round(f1_val,2)])
                   
                   # Evaluates model on test data
                   test_loss, test_accuracy, test_precision, test_recall = best_model.evaluate(test_features, test_labels, verbose = 1)

                   # Returns predictions of the model for test set.
                   test_pred = (best_model.predict(test_features)>0.5).astype(int)

                   # Calculates F1 score for test set.
                   f1_test = f1_score(test_labels, test_pred)

                   #Stores test results in list.
                   test_list.append([preprocessing_name, OPTIM, ACTIVATION, METRIC ,round(test_loss,2), round(test_accuracy,2), round(test_precision,2), round(test_recall,2), round(f1_test,2)])
        # Writes validation performance to file for each processing type.
        file_name = f'Validation_Performance/{preprocessing_name}_validation.csv'
        with open(file_name, 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerows(val_list)
        
        # Writes test performance to file for each processing type:
        file_name = f'Test_Performance/{preprocessing_name}_test.csv'
        with open(file_name, 'w', newline='') as file:
          writer = csv.writer(file)
          writer.writerows(test_list)


# Generates graphs 
graphing = False
if graphing == True:
   for preprocessing_name, preprocess_method in data_preprocessing_methods:
      # Sets file name and accesses validation data
      data_path = f'Validation_Performance/{preprocessing_name}_validation.csv' 
      data_precision_optimizing_data = pd.read_csv(data_path)
      
      # Create and save the bar chart of peformance for each combination of optimiser/activation/metric/preprocessing.
      fig1, ax1 = create_bar_chart(data_precision_optimizing_data, f'{preprocessing_name}')
      fig1.savefig(f"Figures/{preprocessing_name}")



# Plots training data against validation for best model.
best_model = True
if best_model:
  optims = 'RMSprop'
  activations = 'swish'
  metrics = 'val_precision'
  data_preprocessing_methods = 'Smote'

  # Load the model history
  history = np.load(f'Model_History/{data_preprocessing_methods}_{optims}_{activations}_{metrics}.npy', allow_pickle='TRUE').item()
  
  # Grabs epoch range
  epochs = range(1, len(history['loss']) + 1)

  # Set up the plotting figure and axes
  plt.figure(figsize=(18, 6))  # Increase figure size for better readability

  # Plot Training and Validation Loss
  plt.subplot(1, 3, 1)  
  plt.plot(history['loss'], label='Training Loss')
  plt.plot(history['val_loss'], label='Validation Loss')
  plt.title('Model Loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.xticks(epochs)
  plt.ylim(0, 1)
  plt.legend(loc='upper right')

  # Plot Training and Validation Accuracy
  plt.subplot(1, 3, 2) 
  plt.plot(history['accuracy'], label='Training Accuracy')
  plt.plot(history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.xticks(epochs)
  plt.ylim(0, 1)
  plt.legend(loc='lower right')

  # Plot Training and Validation Precision (if available)
  plt.subplot(1, 3, 3)
  plt.plot(history.get('precision', []), label='Training Precision') 
  plt.plot(history.get('val_precision', []), label='Validation Precision')
  plt.title('Model Precision')
  plt.ylabel('Precision')
  plt.xlabel('Epoch')
  plt.xticks(epochs)
  plt.ylim(0, 1)
  plt.legend(loc='lower right')

  plt.tight_layout()
  plt.show()