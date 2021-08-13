

import numpy as np
import pandas as pd
import sklearn.model_selection
# from keras.models import Sequential
# from keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from keras.layers import Embedding
import tensorflow as tf
import math
# from keras.models import load_model
# from keras.models import model_from_json



import re
#from tensorflow.keras import layers
from transformers import TokenAndPositionEmbedding, TransformerBlock


def hasMissedCleavage(peptide):
    try:
        index = peptide.index("K")
        if index< len(peptide)-1 and peptide[index+1]!="P":
            return True
        index = peptide.index("R")
        return index < len(peptide) - 1 and peptide[index + 1] != "P"



    except ValueError:
        try:
            index = peptide.index("R")
            return index < len(peptide) - 1 and peptide[index + 1] != "P"
        except ValueError:
            return False


def reversePeptide(peptide):
    return peptide[::-1]


def filterNegativePeptides(peptides):
    f1 = open(
        "144T_mIgG1.pepXML").read()
    # f2 = open(
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







allPositivesQuant = pd.read_csv("allPositivesQuantification.csv")

allPositivesQuantReverse = allPositivesQuant.copy()
allPositivesQuantReverse["peptide"] = list(map(reversePeptide, allPositivesQuantReverse["peptide"]))


allPositivesQuant = pd.concat([allPositivesQuant, allPositivesQuantReverse])

# reversePositivePeptides = list(map(reversePeptide, positives["peptide"]))
# reversePositive = pd.DataFrame(
#     {"peptide": reversePositivePeptides, "found": np.repeat(1, len(reversePositivePeptides)), "probability": allPositives["probability"]})

allPositivesQuant = allPositivesQuant[~allPositivesQuant["peptide"].str.contains("X")]
allPositivesQuant = allPositivesQuant[allPositivesQuant["peptide"].str.len()<=40]




negativesQuant = pd.read_csv("negativesQuantification.csv")


negativesQuant = negativesQuant[~negativesQuant["peptide"].isin(allPositivesQuant["peptide"])]
negativesQuant = negativesQuant.drop_duplicates("peptide")

negativesQuant = negativesQuant[~negativesQuant["peptide"].str.contains("X")]
negativesQuant = negativesQuant[negativesQuant["peptide"].str.len()<=40]


negativesSampled = negativesQuant
#negativesSampled=negatives

negativesSampled["probability"] = [allPositivesQuant["probability"].mean()]*negativesSampled.shape[0]


# negatives = negatives.sample((positives.shape[0] + reversePositive.shape[0])*5)

#positives = pd.concat([positives, reversePositive])
print("positives: "+str(allPositivesQuant.shape[0]))
print("negatives: "+str(negativesSampled.shape[0]))

nbRepeats=1


allPositivesQuant["mc"] = allPositivesQuant["peptide"].apply(hasMissedCleavage)
allPositivesQuant=allPositivesQuant[allPositivesQuant["mc"]==False]



negativesSampled["mc"] = negativesSampled["peptide"].apply(hasMissedCleavage)
negativesSampled=negativesSampled[negativesSampled["mc"]==False]

negativesSampled=negativesSampled.sample(n=allPositivesQuant.shape[0])


X_trainP, X_testP, y_trainP, y_testP = sklearn.model_selection.train_test_split(
    allPositivesQuant, pd.Series([1]*allPositivesQuant.shape[0]), test_size=0.2)


X_testP = X_testP[~((X_testP["peptide"].str.startswith("R")) | (X_testP["peptide"].str.startswith("K")))]
#y_testP = y_testP.loc[X_testP.index,:]

X_testP = X_testP[X_testP["probability"]>0.6]
y_testP = pd.Series([1]*X_testP.shape[0])

#X_trainP = pd.concat([X_trainP, reversePositive["peptide"]])
#y_trainP = pd.concat([y_trainP, reversePositive["found"]])

X_trainN, X_testN, y_trainN, y_testN = sklearn.model_selection.train_test_split(
    negativesSampled, np.repeat(0, allPositivesQuant.shape[0]), test_size=0.1)




weights=[]

for i in range(round(nbRepeats)):
    weights = weights + allPositivesQuant.loc[X_trainP.index,"probability"].tolist()

weights = weights+[X_trainP.shape[0]/X_trainN.shape[0]]*X_trainN.shape[0]


X_testFinal = pd.concat([pd.DataFrame(X_testP), negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])]])["peptide"]




X_testFinal_quant  = pd.concat([pd.DataFrame(X_testP), negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])]])["quantification"]
y_testFinal = [1]*pd.DataFrame(X_testP).shape[0] + [0]*negativesQuant[~negativesQuant["peptide"].isin(X_trainN["peptide"])].shape[0]
X_testFinal = X_testFinal.reset_index(drop=True)
X_testFinal_quant = X_testFinal_quant.reset_index(drop=True)
y_testFinal = pd.Series(y_testFinal).reset_index(drop=True)



X_train = pd.concat([X_trainP, X_trainN])
X_test = pd.concat([X_testP] + [X_testN])
y_train = list(y_trainP)+list(y_trainN)
y_test = pd.concat([pd.Series(y_testP)] + [pd.Series(y_testN)])





train = X_train
train["y"]=y_train
train = train.sample(frac=1).reset_index(drop=True)




X_train = train
X_test = X_test.reset_index(drop=True)
y_train =train["y"]
y_test = y_test.reset_index(drop=True)

# X_test, uniqueIndexes = np.unique(X_test, return_index=True)
# y_test = y_test[uniqueIndexes]
# y_test = y_test.reset_index(drop=True)

###################################

#y_testPos = y_test[y_test==1]
#y_testNeg = y_test[y_test==0].sample(n=y_testPos.shape[0])
#y_testNeg = y_test[y_test==0]
#y_test = pd.concat([y_testPos, y_testNeg]).sample(frac=1)
#X_test = X_test[y_test.index]

###################################



###########################################################################
maxLength = 40
X_train1 = X_train.copy()

X_train1_quant = X_train1["quantification"]

print(len(set(X_testFinal).intersection(set(X_train1["peptide"]))))

X_train1 = X_train1["peptide"].apply(convertPeptide, args=(maxLength,))
X_test1 = X_test.copy()

X_test1_quant = X_test1["quantification"]
X_test1 = X_test1["peptide"].apply(convertPeptide, args=(maxLength,))







X_train1 = np.array(X_train1.to_list())
X_test1 = np.array(X_test1.to_list())

embedding_vecor_length = 128
# model = tf.keras.Sequential()
# model.add(layers.Embedding(21, embedding_vecor_length, input_length=maxLength))
# model.add(layers.LSTM(300, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
# #model.add(LSTM(300, dropout=0.4, recurrent_dropout=0.4, return_sequences=True, go_backwards=True))
# model.add(layers.LSTM(200, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
# model.add(layers.LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=True))
# model.add(layers.LSTM(50, dropout=0.3, recurrent_dropout=0.3))

# x = tf.keras.Sequential()

# main_input = tf.keras.layers.Input(shape=(40,), dtype='int32', name='main_input')
#
# x = tf.keras.layers.Embedding(21, embedding_vecor_length, input_length=maxLength)(main_input)

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

main_input = tf.keras.layers.Input(shape=(40,))
embedding_layer = TokenAndPositionEmbedding(40, 21, embed_dim)
x = embedding_layer(main_input)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
lstm_out = tf.keras.layers.GlobalAveragePooling1D()(x)

auxiliary_output = tf.keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)


auxiliary_input = tf.keras.layers.Input(shape=(1,), name='aux_input')
x = tf.keras.layers.concatenate([auxiliary_output, auxiliary_input])
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

main_output = tf.keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)

model = tf.keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])

#model.add(layers.Dense(1, activation='sigmoid'))

optimiser = tf.keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'], loss_weights=[1., 0.2])


print(model.summary())
# history = model.fit(np.asanyarray(X_train1), np.asanyarray(y_train), validation_data=(np.asarray(X_test1), np.asarray(y_test)), epochs=150, batch_size=128,
#                     sample_weight=np.array(weights))


model.fit([np.asanyarray(X_train1), np.asanyarray(X_train1_quant)], [np.asanyarray(y_train), np.asanyarray(y_train)],
          validation_data = ([np.asarray(X_test1), np.asarray(X_test1_quant)], [np.asarray(y_test),np.asarray(y_test)]),
          epochs=80, batch_size=32)



#predictions to file
df = pd.DataFrame({"peptide": X_testFinal, "quantification": X_testFinal_quant, "found": y_testFinal})
predictions = model.predict([np.array([convertPeptide(pep, maxLength) for pep in df["peptide"]]), np.array(X_testFinal_quant)])
df["predictions"] = predictions[0].flatten()
df.to_csv("output/test1.csv", index=False)
model.save("output/test1Model.h5")
