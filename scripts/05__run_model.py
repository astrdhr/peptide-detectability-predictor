# import libraries
import numpy as np
import pandas as pd
import time
import math
import re

import sklearn.model_selection
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

from transformers import TokenAndPositionEmbedding, TransformerBlock

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from textwrap import wrap
import seaborn as sns

# import detected and undetected datasets
detected_peptides = pd.read_table('../data/detected_peptides_all_quant_aaindex1.tsv')
undetected_peptides = pd.read_table('../data/undetected_peptides_all_quant_aaindex1.tsv')

# keep detected and undetected peptides less than or equal to 40 aa AND greater than or equal to 7 aa in length
detected_peptides = detected_peptides.loc[(detected_peptides["Peptide"].str.len()>=7) &
                                          (detected_peptides["Peptide"].str.len()<=40)].reset_index(drop=True)

undetected_peptides = undetected_peptides.loc[(undetected_peptides["Peptide"].str.len()>=7) &
                                              (undetected_peptides["Peptide"].str.len()<=40)].reset_index(drop=True)

# double check 0 peptides in undetected peptides are present in detected peptides
len(undetected_peptides[undetected_peptides["Peptide"].isin(detected_peptides["Peptide"])])

# add detectability column
detected_peptides.insert(loc=1, column='Detectability', value=1)
undetected_peptides.insert(loc=1, column='Detectability', value=0)

# remove nan values
all_peptides = pd.concat([detected_peptides, undetected_peptides])
all_peptides = all_peptides.dropna(axis=1, how='any')
detected_peptides = all_peptides[all_peptides['Detectability'] == 1]
undetected_peptides = all_peptides[all_peptides['Detectability'] == 0]

# take random sample of undetected_peptides, with equal number of rows to detected_peptides
# random_state is used for reproducibility
undetected_peptides_balanced = undetected_peptides.sample(n=detected_peptides.shape[0],
                                                         random_state=42).reset_index(drop=True)

# store unused undetected peptides
unused_undetected = undetected_peptides[~undetected_peptides["Peptide"].isin
                                        (undetected_peptides_balanced["Peptide"])]

"""
Create training, validation and test sets
"""
# split into train and test sets
# detected peptides
X_trainP, X_testP, y_trainP, y_testP = sklearn.model_selection.train_test_split(
    detected_peptides, detected_peptides['Detectability'], test_size=0.3, random_state=1)

# undetected peptides
X_trainN, X_testN, y_trainN, y_testN = sklearn.model_selection.train_test_split(
    undetected_peptides_balanced, undetected_peptides_balanced['Detectability'], test_size=0.3, random_state=1)

# split training set further into train and validation sets
# detected peptides
X_trainP, X_valP, y_trainP, y_valP = sklearn.model_selection.train_test_split(
    X_trainP, y_trainP, test_size=0.25, random_state=1)

# undetected peptides
X_trainN, X_valN, y_trainN, y_valN = sklearn.model_selection.train_test_split(
    X_trainN, y_trainN, test_size=0.25, random_state=1)

# create final training and validation sets
X_train = pd.concat([X_trainP, X_trainN])
X_val = pd.concat([X_valP] + [X_valN])
y_train = pd.concat([pd.Series(y_trainP)] + [pd.Series(y_trainN)])
y_val = pd.concat([pd.Series(y_valP)] + [pd.Series(y_valN)])

# check validation set is not in train
print(len(X_val[X_val["Peptide"].isin(X_train["Peptide"])]))

# create final test set
X_test = pd.concat([X_testP, X_testN])
y_test = pd.concat([pd.Series(y_testP)] + [pd.Series(y_testN)])

# check test is not in train or validation
print(len(X_test[X_test["Peptide"].isin(X_val["Peptide"])]))
print(len(X_test[X_test["Peptide"].isin(X_train["Peptide"])]))


"""
integer-encode peptides
"""
# this function was kindly provided by Esteban Gea and is based on prior work by Atekah Hafeez)
maxLength = 40

aaDict = {"-": 0, "A": 1, "R": 2, "N": 3, "D": 4, "C": 5, "Q": 6, "E": 7, "G": 8, "H": 9, "I": 10, "L": 11,
          "K": 12, "M": 13, "F": 14, "P": 15, "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20, "U": 21}

def convertPeptide(peptide, maxLength):
    j = 0
    hotPeptide = []
    for aa in peptide:
        hotPeptide.append(aaDict[aa])
        j = j + 1
    for k in range(maxLength - j):
        hotPeptide.append(0)

    return np.array(hotPeptide)


"""
Separate out each feature
"""
# training set
X_train = shuffle(X_train, random_state=1).reset_index(drop=True)
y_train = shuffle(y_train, random_state=1).reset_index(drop=True)
X_train_peptide = X_train['Peptide'].apply(convertPeptide, args=(maxLength,))
X_train_quant = X_train.iloc[:, 2:7]
X_train_aaindex1 = X_train.iloc[:, 7:]

# validation set
X_val = shuffle(X_val, random_state=1).reset_index(drop=True)
y_val = shuffle(y_val, random_state=1).reset_index(drop=True)
X_val_peptide = X_val['Peptide'].apply(convertPeptide, args=(maxLength,))
X_val_quant = X_val.iloc[:, 2:7]
X_val_aaindex1 = X_val.iloc[:, 7:]

# test set
X_test = shuffle(X_test, random_state=1).reset_index(drop=True)
y_test = shuffle(y_test, random_state=1).reset_index(drop=True)
X_test_peptide = X_test['Peptide'].apply(convertPeptide, args=(maxLength,))
X_test_quant = X_test.iloc[:, 2:7]
X_test_aaindex1 = X_test.iloc[:, 7:]

# convert to arrays
X_train_peptide = np.array(X_train_peptide.to_list())
X_val_peptide = np.array(X_val_peptide.to_list())
X_test_peptide = np.array(X_test_peptide.to_list())

"""
Further pre-processing
"""
# apply variance threshold
from sklearn.feature_selection import VarianceThreshold
selector = VarianceThreshold(threshold=0.005).fit(X_train_aaindex1)
mask = selector.get_support()
X_train_aaindex1 = X_train_aaindex1.loc[:, mask]
X_val_aaindex1 = X_val_aaindex1.loc[:, mask]
X_test_aaindex1 = X_test_aaindex1.loc[:, mask]

# apply scaling
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train_aaindex1)
X_train_aaindex1_scaled = scaler.transform(X_train_aaindex1)
X_val_aaindex1_scaled = scaler.transform(X_val_aaindex1)
X_test_aaindex1_scaled = scaler.transform(X_test_aaindex1)

# Apply scaled PCA
scaler = preprocessing.StandardScaler().fit(X_train_aaindex1)
X_train_aaindex1_scaled = scaler.transform(X_train_aaindex1)
X_val_aaindex1_scaled = scaler.transform(X_val_aaindex1)
X_test_aaindex1_scaled = scaler.transform(X_test_aaindex1)
from sklearn.decomposition import PCA
pca = PCA(n_components=10)
pca.fit(np.asanyarray(X_train_aaindex1_scaled))
print(pca.explained_variance_ratio_)
X_train_aaindex1_pca = pca.transform(np.asanyarray(X_train_aaindex1_scaled))
X_val_aaindex1_pca = pca.transform(np.asanyarray(X_val_aaindex1_scaled))
X_test_aaindex1_pca = pca.transform(np.asanyarray(X_test_aaindex1_scaled))
print(pca.singular_values_)

# PCA on original raw data (without scaling)
pca = PCA(n_components=3)
pca.fit(np.asanyarray(X_train_aaindex1))
print(pca.explained_variance_ratio_)
X_train_aaindex1_pca = pca.transform(np.asanyarray(X_train_aaindex1))
X_val_aaindex1_pca = pca.transform(np.asanyarray(X_val_aaindex1))
X_test_aaindex1_pca = pca.transform(np.asanyarray(X_test_aaindex1))

# Remove highly correlated (redundant) features
X_train_aaindex1_corr = X_train_aaindex1.corr()

correlated_features = set()
for i in range(len(X_train_aaindex1_corr.columns)):
    for j in range(i):
        if abs(X_train_aaindex1_corr.iloc[i, j]) > 0.5:
            colname = X_train_aaindex1_corr.columns[i]
            correlated_features.add(colname)
len(correlated_features)

X_train_aaindex1_corr = X_train_aaindex1
X_val_aaindex1_corr = X_val_aaindex1
X_test_aaindex1_corr = X_test_aaindex1

X_train_aaindex1_corr.drop(labels=correlated_features, axis=1, inplace=True)
X_val_aaindex1_corr.drop(labels=correlated_features, axis=1, inplace=True)
X_test_aaindex1_corr.drop(labels=correlated_features, axis=1, inplace=True)

X_train_aaindex1_corr.shape
X_train_aaindex1_corr.columns


"""
Build model
This model was built following prior work by Atekah Hafeez
"""
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

main_input = tf.keras.layers.Input(shape=(40,))
# embed each peptide into a 40-dimensional vector
embedding_layer = TokenAndPositionEmbedding(40, 21, embed_dim)
x = embedding_layer(main_input)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
lstm_out = tf.keras.layers.GlobalAveragePooling1D()(x)
auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

auxiliary_input = tf.keras.layers.Input(shape=(5,), name='aux_input')
aaindex_input = tf.keras.layers.Input(shape=(4,), name='aaindex_input')

x = tf.keras.layers.concatenate([auxiliary_output, auxiliary_input, aaindex_input])

x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)

model = tf.keras.Model(inputs=[main_input, auxiliary_input, aaindex_input], outputs=[main_output, auxiliary_output])

optimiser = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'], loss_weights=[1., 0.2])

# Inspect model
print(model.summary())
keras.utils.plot_model(model, "final_figures/keras_model.png", show_shapes=True)


"""
Train model
"""
start_time = time.time()

history = model.fit([np.asanyarray(X_train_peptide), np.asanyarray(X_train_quant), np.asanyarray(X_train_aaindex1_corr)],
                    [np.asanyarray(y_train), np.asanyarray(y_train)],
                    validation_data = ([np.asarray(X_val_peptide), np.asanyarray(X_val_quant), np.asarray(X_val_aaindex1_corr)],
                                       [np.asarray(y_val), np.asarray(y_val)]),
                    epochs=250, batch_size=134, verbose=2)

print("")
print("Time taken for model to run: ", time.time() - start_time)


"""
Analyse scores from training
"""
history_dict = history.history
history_dict.keys()

# save to df
df_acc = pd.DataFrame({'Loss': history_dict['loss'],
                       'main_output_loss': history_dict['main_output_loss'],
                       'aux_output_loss': history_dict['aux_output_loss'],
                      'main_output_accuracy': history_dict['main_output_accuracy'],
                       'aux_output_accuracy': history_dict['aux_output_accuracy'],
                      'val_loss': history_dict['val_loss'],
                       'val_main_output_loss': history_dict['val_main_output_loss'],
                       'val_aux_output_loss': history_dict['val_aux_output_loss'],
                      'val_main_output_accuracy': history_dict['val_main_output_accuracy'],
                       'val_aux_output_accuracy': history_dict['val_aux_output_accuracy']})
df_acc.to_csv('final_data/M8_seq_nsaf_aaindex1_var_corr05_REDO_train_134batch_250epoch.tsv', sep='\t', index=False)


# example of saving training scores from running multiple models (using different features)
df_acc_aaindex1 = pd.DataFrame({'raw_acc': history_dict['val_main_output_accuracy']})
df_acc_aaindex1['var_scaled'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['scaled'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['scaled_pca'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['raw_pca'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['corr_0.8'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['corr_0.5'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['corr_0.7'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['var_corr_0.6'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1['var_corr_0.5'] = history_dict['val_main_output_accuracy']
df_acc_aaindex1.to_csv('final_data/M7_seq_aaindex1_all-val-scores_134batch_250epoch.tsv', sep='\t', index=False)


# plot training and validation loss
train_loss = history_dict['main_output_loss']
val_loss = history_dict['val_main_output_loss']
acc = history_dict['main_output_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, train_loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss (Sequence + Combined SpC + AAIndex1)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot training and validation accuracy
plt.clf()
train_acc = history_dict['main_output_accuracy']
val_acc = history_dict['val_main_output_accuracy']

plt.plot(epochs, train_acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy (Sequence + Combined SpC + AAIndex1)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
Evaluate model
"""
# train set
train_results = model.evaluate([np.asarray(X_train_peptide), np.asanyarray(X_train_quant), np.asanyarray(X_train_aaindex1_corr)],
                               [np.asarray(y_train), np.asarray(y_train)])
print("%s: %.2f%%" % (model.metrics_names[1], train_results[1]*100))
print("%s: %.2f%%" % (model.metrics_names[3], train_results[3]*100))

# validation set
val_results = model.evaluate([np.asarray(X_val_peptide), np.asanyarray(X_val_quant), np.asanyarray(X_val_aaindex1_corr)],
                             [np.asarray(y_val), np.asarray(y_val)])
print("%s: %.2f%%" % (model.metrics_names[1], val_results[1]*100))
print("%s: %.2f%%" % (model.metrics_names[3], val_results[3]*100))

# test set
test_results = model.evaluate([np.asarray(X_test_peptide), np.asanyarray(X_test_quant), np.asanyarray(X_test_aaindex1_corr)],
                              [np.asarray(y_test), np.asarray(y_test)])
print("%s: %.2f%%" % (model.metrics_names[1], test_results[1]*100))
print("%s: %.2f%%" % (model.metrics_names[3], test_results[3]*100))


"""
Predict on test set
"""
df = pd.DataFrame({'Peptide': X_test['Peptide'], 'Detectability': y_test})
test_predictions = model.predict([np.array([
    convertPeptide(pep, maxLength) for pep in df['Peptide']]),
                                  np.array(X_test_quant), np.array(X_test_aaindex1_corr)], verbose=1)
df["Predictions"] = test_predictions[0].flatten()


"""
Get prediction metrics
"""
# confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, np.rint(df['Predictions']))

# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, np.rint(df['Predictions']))

# Recall
from sklearn.metrics import recall_score
recall_score(y_test, round(df['Predictions']), average='binary')

# Precision
from sklearn.metrics import precision_score
precision_score(y_test, np.rint(df['Predictions']), average='binary')

# F1 score
from sklearn.metrics import f1_score
f1_score(y_test, np.rint(df['Predictions']), average='binary')

# MCC (Matthew's correlation coefficient)
from sklearn.metrics import matthews_corrcoef
matthews_corrcoef(y_test, np.rint(df['Predictions']))
