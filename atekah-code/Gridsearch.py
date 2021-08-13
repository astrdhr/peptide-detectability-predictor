#!pip install q keras==2.3.1
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics as stats
import sklearn.model_selection
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import tensorboard
from tensorboard.plugins.hparams import api as hp
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.layers import LSTM
from keras.layers import Embedding, Activation
import tensorflow as tf
import datetime
import math
from keras.models import load_model
from keras.models import model_from_json
import csv as csv
from keras.backend import sigmoid
import re
from tensorflow.keras import layers
import timeit
from timeit import default_timer as timer


!rm -rf ./logs/

#Activation function swish for LSTM
def swish(x, beta = 1):
    return (x * sigmoid(beta * x))
#Assign swish activation function to 'swish'
from keras.utils.generic_utils import get_custom_objects
get_custom_objects().update({'swish': Activation(swish)})

def hasMissedCleavage(peptide):
    try:
        index = peptide.index("K")
        return index< len(peptide)-1 and peptide[index+1]!="P"

    except ValueError:
        try:
            index = peptide.index("R")<len(peptide)-1
            return index < len(peptide) - 1 and peptide[index + 1] != "P"
        except ValueError:
            return False


def reversePeptide(peptide):
    return peptide[::-1]


def filterNegativePeptides(peptides):
    f1 = open(
        "144T_mIgG1.pepXML").read()
    #f2 = open(
    #     "mzml/200415_DAS_OT_JAS_R08_230115_FWD_Fxn_1/200415_DAS_OT_JAS_R08_230115_FWD_Fxn.pepXML").read()
    # f3 = open(
    #     "mzml/200415_DAS_OT_JAS_R08_230115_REV_Fxn_0/200415_DAS_OT_JAS_R08_230115_REV_Fxn.pepXML").read()
    # f4 = open(
    #     "mzml/200415_DAS_OT_JAS_R08_230115_REV_Fxn_1/200415_DAS_OT_JAS_R08_230115_REV_Fxn.pepXML").read()


    allPeptidesMissedCleavage = set(re.findall(r'peptide="([KR]?[^P].*?[KR](?!P))[A-Z]*[KR]"\sm', f1))
                #                     +re.findall(r'peptide="([KR]?[^P].*?[KR](?!P))[A-Z]*[KR]"\sm', f2)+ \
                # re.findall(r'peptide="([KR]?[^P].*?[KR](?!P))[A-Z]*[KR]"\sm', f3)+re.findall(r'peptide="([KR]?[^P].*?[KR](?!P))[A-Z]*[KR]"\sm', f4))

    foundInPepXml = set(re.findall(r'peptide="([A-Z]*)"\sm', f1))
        #                 + re.findall(
        # r'peptide="([A-Z]*)"\sm', f2) + \
        #                             re.findall(r'peptide="([A-Z]*)"\sm', f3) + re.findall(
        # r'peptide="([A-Z]*)"\sm', f4))



    return list(set(peptides).difference(allPeptidesMissedCleavage).difference(foundInPepXml))
    print(len(allPeptidesMissedCleavage))


aaDict = {
    "-": 0,
    "A": 1,
    "R": 2,
    "N": 3,
    "D": 4,
    "C": 5,
    "Q": 6,
    "E": 7,
    "G": 8,
    "H": 9,
    "I": 10,
    "L": 11,
    "K": 12,
    "M": 13,
    "F": 14,
    "P": 15,
    "S": 16,
    "T": 17,
    "W": 18,
    "Y": 19,
    "V": 20,
    "U": 21,

}


def reversePeptide(peptide):
    return peptide[::-1]


def convertPeptide(peptide, maxLength):
    j = 0
    hotPeptide = []
    for aa in peptide:
        hotPeptide.append(aaDict[aa])
        j = j + 1
    for k in range(maxLength - j):
        hotPeptide.append(0)
    return np.array(hotPeptide)

##########Data Preprocessing steps

allPositivesQuant = pd.read_csv("allPositivesQuantification.csv")

#allPositivesQuantReverse = allPositivesQuant.copy()
#allPositivesQuantReverse["peptide"] = list(map(reversePeptide, allPositivesQuantReverse["peptide"]))


#allPositivesQuant = pd.concat([allPositivesQuant, allPositivesQuantReverse])

# reversePositivePeptides = list(map(reversePeptide, positives["peptide"]))
# reversePositive = pd.DataFrame(
#     {"peptide": reversePositivePeptides, "found": np.repeat(1, len(reversePositivePeptides)), "probability": allPositives["probability"]})

allPositivesQuant = allPositivesQuant[~allPositivesQuant["peptide"].str.contains("X")]#Remove peptides with undetected aa
allPositivesQuant = allPositivesQuant[allPositivesQuant["peptide"].str.len()<=40]#Select only peptides less than 40 aa long
print(allPositivesQuant.shape[0])


negativesQuant = pd.read_csv("negativesQuantNoMC.csv")#Use fully cleaved negative peptides


negativesQuant = negativesQuant[~negativesQuant["peptide"].isin(allPositivesQuant["peptide"])]
negativesQuant = negativesQuant.drop_duplicates("peptide")

negativesQuant = negativesQuant[~negativesQuant["peptide"].str.contains("X")]
negativesQuant = negativesQuant[negativesQuant["peptide"].str.len()<=40]
print(negativesQuant.shape[0])

#negativesSampled = negativesQuant.sample(n=allPositivesQuant.shape[0]) #use same number of negative and positive peptides
negativesSampled=negativesQuant

negativesSampled["probability"] = [allPositivesQuant["probability"].mean()]*negativesSampled.shape[0]


# negatives = negatives.sample((positives.shape[0] + reversePositive.shape[0])*5)

#positives = pd.concat([positives, reversePositive])
print("positives: "+str(allPositivesQuant.shape[0]))
print("negatives: "+str(negativesSampled.shape[0]))

#print('Positives:',allPositivesQuant)
#print('Negatives:',negativesSampled)

nbRepeats=1
#Splitting detected/positive peptides data into training and test set where X denotes both sequence and Quant data, Y= only Quantification data
X_trainP, X_testP, y_trainP, y_testP = sklearn.model_selection.train_test_split(
    allPositivesQuant, pd.Series([1]*allPositivesQuant.shape[0]), test_size=0.2)


#X_testP = X_testP[~((X_testP["peptide"].str.startswith("R")) | (X_testP["peptide"].str.startswith("K")))]
#y_testP = y_testP[X_testP.index]


#X_testP = X_testP[X_testP["probability"]>0.6]#Taking only peptides with high probability of being detected as testing set
y_testP = pd.Series([1]*X_testP.shape[0])

#X_trainP = pd.concat([X_trainP, reversePositive["peptide"]])
#y_trainP = pd.concat([y_trainP, reversePositive["found"]])

X_trainN, X_testN, y_trainN, y_testN = sklearn.model_selection.train_test_split(
    negativesSampled, np.repeat(0, negativesSampled.shape[0]), test_size=0.1)

######
#Works out balanced class weight for imbalanced data set which has more Negatives than positives. However its better to optimize these weights as these balanced weights are multiplied by the loss function
#and the performance of the model at each iteration is based on the class weight * loss function so we wont know which weights will work better as they will need to be optimised alongside the loss function
#weights=[]
#for i in range(round(nbRepeats)):
#    weights = weights + allPositivesQuant.loc[X_trainP.index,"probability"].tolist()
#weights = weights+[X_trainP.shape[0]/X_trainN.shape[0]]*X_trainN.shape[0]#To balance the class weight= Total Positive+Negative/Samples in 1 class *Total no. of classes
#print(weights)

#Data used for model prediction
X_testFinal = pd.concat([pd.DataFrame(X_testP), negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])]])["peptide"]#Combine positive and negative
X_testFinal_quant  = pd.concat([pd.DataFrame(X_testP), negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])]])["quantification"]
y_testFinal = [1]*pd.DataFrame(X_testP).shape[0] + [0]*negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])].shape[0]
X_testFinal = X_testFinal.reset_index(drop=True)
X_testFinal_quant = X_testFinal_quant.reset_index(drop=True)
y_testFinal = pd.Series(y_testFinal).reset_index(drop=True)



X_train = pd.concat([X_trainP, X_trainN])#Combining the training positive and negative peptides for sequence data into training set, which goes into the auxiliary input
X_test = pd.concat([X_testP] + [X_testN])#Auxiliary input test set labels
y_train = list(y_trainP)+list(y_trainN)#Labels for data(1= detected 0=undetected)
y_test = pd.concat([pd.Series(y_testP)] + [pd.Series(y_testN)])#Test set labels, combining labels from positive and negative peptides



train = X_train
train["y"]=y_train
train = train.sample(frac=1).reset_index(drop=True)


X_train = train
X_test = X_test.reset_index(drop=True)
y_train =train["y"]
y_test = y_test.reset_index(drop=True)
y_all = pd.concat([pd.Series(y_train)] + [pd.Series(y_test)])#Combine the test and training sets labels for bootstrapping later. y corresponds to train and test labels for aux and main input data
y_all=y_all.reset_index(drop=True)
#print('y_all: ',len(y_all))
#print('y_all: ',y_all[900:])


###################################

#y_testPos = y_test[y_test==1]
#y_testNeg = y_test[y_test==0].sample(n=y_testPos.shape[0])
#y_testNeg = y_test[y_test==0]
#y_test = pd.concat([y_testPos, y_testNeg]).sample(frac=1)
#X_test = X_test[y_test.index]

###################################
maxLength = 40
X_train1 = X_train.copy()

X_train1_quant = X_train1["quantification"]

print(len(set(X_testFinal).intersection(set(X_train1["peptide"]))))

X_train1 = X_train1["peptide"].apply(convertPeptide, args=(maxLength,))
X_test1 = X_test.copy()

X_test1_quant = X_test1["quantification"]
X_test1 = X_test1["peptide"].apply(convertPeptide, args=(maxLength,))


X_seq = pd.concat([pd.Series(X_train1)] + [pd.Series(X_test1)])#Combine the main input test and training sets for bootstrapping later
X_seq=X_seq.reset_index(drop=True)
#print('X_seq: ',X_seq[900:])
X_seq_quant= pd.concat([pd.Series(X_train1_quant)] + [pd.Series(X_test1_quant)])#Combine the aux input test and training sets for bootstrapping
X_seq_quant=X_seq_quant.reset_index(drop=True)
#X_train1 = np.array(X_train1.tolist())
#X_test1 = np.array(X_test1.tolist())

X_seq= np.array(X_seq.tolist())


#print('seq:',len(X_seq))
#print('seq+quant:',len(X_seq))
embedding_vecor_length = 128
#sample_weight=np.array(weights)

#To calculate model training time
class TimingCallback(tf.keras.callbacks.Callback):
  def __init__(self, logs={}):
    self.logs=[]
  def on_epoch_begin(self, epoch, logs={}):
    self.starttime = timer()
  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(timer()-self.starttime)

cb = TimingCallback()


#Hyperparameters that need to be optimized
HP_neurons=hp.HParam('neurons', hp.Discrete([32,64,128])) #
HP_dense_layers=hp.HParam('dense_layers', hp.Discrete([2,3,4])) #
HP_0CW= hp.HParam('cw0', hp.Discrete([0.1, 0.2, 0.3, 0.4, 0.5]))
#Adam parameters
HP_lr=hp.HParam('lr', hp.Discrete([0.001, 0.0001, 0.00001]))#Learning rate controls how much to update the weight at the end of each batch
METRIC_ACCURACY= 'accuracy'

print('The gridsearch will run for 3x3x3x5= 135 combinations of parameters')

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
    hparams=[HP_dense_layers, HP_neurons, HP_lr, HP_0CW],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='accuracy')],
  )


def train_test_model(hparams, X_train1, X_train1_quant, y_train1, X_test1, X_test1_quant, y_test1):

    x = tf.keras.Sequential()
    # create model
    main_input = tf.keras.layers.Input(shape=(40,), dtype='int32', name='main_input')#Sequence data input= main_input
    x = tf.keras.layers.Embedding(21, embedding_vecor_length, input_length=maxLength)(main_input)
    x = tf.keras.layers.LSTM(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(300, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, go_backwards=True)(x)
    x = tf.keras.layers.LSTM(200, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    lstm_out = tf.keras.layers.LSTM(50, dropout=0.3, recurrent_dropout=0.3)(x)
    auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)#Prediction of model based on sequence data only

    auxiliary_input = tf.keras.layers.Input(shape=(1,), name='aux_input')#Sequence plus quantification data
    x = tf.keras.layers.concatenate([lstm_out, auxiliary_input])#merge the LSTM trained with sequence data from main input with the aux_input(sequence and quant data)
    for l in range(dense_layers):
        x = tf.keras.layers.Dense(hparams[HP_neurons], activation='relu')(x)

    main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)
    model = tf.keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    optimiser = tf.keras.optimizers.Adam(lr=hparams[HP_lr], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(loss={'main_output': 'binary_crossentropy', 'aux_output':'binary_crossentropy'}, optimizer=optimiser, metrics=['accuracy'], loss_weights={'main_output': 1., 'aux_output': 0.2})#weight of 1 to sequence and 0.2 to quantification
    model.fit([np.asanyarray(X_train1),np.asanyarray(X_train1_quant)],[np.asanyarray(y_train1),np.asanyarray(y_train1)], epochs=30,batch_size=64, class_weights={'aux_output':{0:(hparams[HP_0CW]),1:(1-float(hparams[HP_0CW]))},'main_output':{0:(hparams[HP_0CW]),1:1-float((hparams[HP_0CW]))}},callbacks=[cb])
    #X_train1= sequence train data, X_train1_quant= Sequence+Quantification train data, y_train= labels for sequence training data input, y_test=labels for testing data
    loss,main_output_loss, aux_output_loss,main_output_accuracy, aux_output_accuracy= model.evaluate([np.asarray(X_test1), np.asarray(X_test1_quant)],[np.asarray(y_test1),np.asarray(y_test1)])
    return loss,main_output_loss, aux_output_loss,main_output_accuracy, aux_output_accuracy

    #print(model.summary())



#history = model.fit(np.asanyarray(X_train1), np.asanyarray(y_train), validation_data=(np.asarray(X_test1), np.asarray(y_test)), epochs=150, batch_size=128,
#sample_weight=np.array(weights)
#print(history)



#predictions to file
#df = pd.DataFrame({"peptide": X_testFinal, "quantification": X_testFinal_quant, "found": y_testFinal})
#predictions = model.predict([np.array([convertPeptide(pep, maxLength) for pep in df["peptide"]]), np.array(X_testFinal_quant)])
#df["predictions"] = predictions[0].flatten()
#df.to_csv("output/test1.csv", index=False)
#model.save("output/test1Model.h5")

#For each run, log an hparams summary with the hyperparameters and final accuracy:
def run(run_dir, hparams, avg_outputacc):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        avg_outputacc=avg_outputacc
        tf.summary.scalar(METRIC_ACCURACY,avg_outputacc,step=1)




#########
session_num = 0
y_all=y_all
X_seq=X_seq
X_seq_quant=X_seq_quant

#print('y_all',y_all[0:100])
#print('y_all: ',y_all.isnull().values.sum())
#print('X_seq: ',X_seq.shape[0])


typetableFile = open('/gsearch_results.csv','w')#open and name the file using the outputfile paramater
typetableFile.write('dense_layer'+','+'neurons'+','+'learning_rate'+','+'class_weight(0/negative_peptides)'+'\n')
for dense_layers in HP_dense_layers.domain.values:
    for neurons in HP_neurons.domain.values:
        for lr in HP_lr.domain.values:
            for cw0 in HP_0CW.domain.values:
                hparams = {
                    HP_dense_layers:dense_layers,
                    HP_neurons:neurons,
                    HP_lr:lr,
                    HP_0CW:cw0,
                }

                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print('Combination',(session_num+1),{h.name: hparams[h] for h in hparams})
                bootstrap_samples=5 #no of bootstrap iterations
                scores= []
                for _ in range(bootstrap_samples):
                    #select indices of data samples in bootstrap samples to match the labels
                    ix= [i for i in range(len(X_seq))]#Main input data
                    train_ix= resample(ix, replace=True, n_samples=len(X_seq))
                    #print('ix',len(ix),ix)
                    #print('train_ix',len(train_ix),train_ix)

                    #OOB data points not selected for bootstrap samples
                    test_ix=[x for x in ix if x not in train_ix]

                    #Select data main_input = seq data
                    X_train1,y_train1=X_seq[train_ix],y_all[train_ix]
                    X_test1,y_test1=X_seq[test_ix],y_all[test_ix]
                    y_train1,y_test1=y_train1.reset_index(drop=True),y_test1.reset_index(drop=True)

                    #Select data aux_input = seq+quant data
                    X_train1_quant=X_seq_quant[train_ix].reset_index(drop=True)
                    X_test1_quant=X_seq_quant[test_ix].reset_index(drop=True)

                    #Select testing set has quantif probability>0.6
                    X_prob=X_test1_quant>0.6
                    h=X_prob.index[X_prob]
                    X_test1_quant= X_test1_quant[h].reset_index(drop=True)#Update X_test1_quant data for aux input with prob>0.6
                    y_test1=y_test1[h].reset_index(drop=True)#Update y labels for only peptides with prob>0.6
                    X_test1=X_test1[h]#update sequence data for main input for only peptides with prob>0.6

                    ###Balanced classed dataset for testing the model###
                    #Find the lower represented class in the testing set
                    minimum=min(y_test1.value_counts())
                    y0,y1=y_test1.loc[y_test1 == 0], y_test1.loc[y_test1 == 1]
                    y0,y1=y0[:minimum],y1[:minimum]#Balance the dataset to have the same number of samples that are positive and negative
                    y01= pd.concat([y0,y1])
                    y01=y01.sort_index()
                    l=list(y01.index.values)#Extract indices of the positive and negative samples
                    X_test1= X_test1[l]#Update the testing set using the indices stored in l to balance the classes
                    y_test1=y01.reset_index(drop=True)#Reset the indices of the labels
                    X_test1_quant=X_test1_quant[l]#Balance the dataset for the aux input

                    #Evaluate the model
                    loss,main_output_loss, aux_output_loss,main_output_accuracy, aux_output_accuracy= train_test_model(hparams,X_train1, X_train1_quant, y_train1, X_test1, X_test1_quant,y_test1)
                    scores.append(main_output_accuracy)
                avg_outputacc=stats.mean(scores)
                print('Average accuracy:',avg_outputacc,'for',bootstrap_samples,'bootstrap samples')
                run('logs/hparam_tuning/' + run_name, hparams, avg_outputacc)
                typetableFile.write({hparams[h] for h in hparams}+str(avg_outputacc)+'\n')

                print(cb.logs)
                print(sum(cb.logs))
                session_num += 1
typetableFile.close() 
