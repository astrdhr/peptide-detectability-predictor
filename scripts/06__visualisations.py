# import libraries
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.colors import ListedColormap
from textwrap import wrap
import seaborn as sns

import sklearn.model_selection
import sklearn.metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_curve

"""
Performance comparison of different preprocessing techniques on validation accuracy
for AAIndex1 (physicochemical) model
"""
# load validation scores
df_acc_aaindex1 = pd.read_table('final_data/M7_seq_aaindex1_all-val-scores_134batch_250epoch.tsv')
df_acc = pd.read_table('final_data/M7_seq_saf_aaindex1_train_134batch_250epoch.tsv')
df = pd.read_table('final_data/M7_seq_nsaf_aaindex1_var-corr05_pred_134b_250ep.tsv')

# plot training and validation accuracy
plt.clf()
epochs = range(1, len(df_acc_aaindex1['scaled_pca']) + 1)
plt.plot(epochs, df_acc_aaindex1['raw_acc'], 'tab:blue', label='Raw')
plt.plot(epochs, df_acc_aaindex1['raw_pca'], 'tab:orange', label='Raw + PCA')
plt.plot(epochs, df_acc_aaindex1['scaled'], 'tab:green', label='Scaled')
plt.plot(epochs, df_acc_aaindex1['scaled_pca'], 'tab:red', label='Scaled + PCA')
plt.title("\n".join(wrap('Effect of pre-processing technique on validation accuracy (Sequence + AAIndex1)', 60)))
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()

"""
Effect of different feature selection techniques on validation accuracy
for AAIndex1 (physicochemical) model
"""
# plot training and validation accuracy
plt.clf()
train_acc = history_dict['main_output_accuracy']
val_acc = history_dict['val_main_output_accuracy']

epochs = range(1, len(df_acc_aaindex1['scaled_pca']) + 1)

figure(figsize=(12, 6))
plt.ylim(0.66, 0.71)

plt.plot(epochs, df_acc_aaindex1['raw_acc'], 'tab:blue', label='Raw')
plt.plot(epochs, df_acc_aaindex1['var_corr_0.6'], 'tab:purple', label='Variance + Correlation > 0.6')
plt.plot(epochs, df_acc_aaindex1['var_corr_0.5'], 'tab:red', label='Variance + Correlation > 0.5')
plt.plot(epochs, df_acc_aaindex1['corr_0.8'], 'tab:pink', label='Correlation > 0.8')
plt.plot(epochs, df_acc_aaindex1['corr_0.7'], 'tab:olive', label='Correlation > 0.7')
plt.plot(epochs, df_acc_aaindex1['corr_0.5'], 'tab:orange', label='Correlation > 0.5')
plt.plot(epochs, df_acc_aaindex1['var_scaled'], 'tab:cyan', label='Scaled + Variance threshold')

plt.title('Effect of feature selection technique on validation accuracy (Sequence + AAIndex1)')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.savefig('final_figures/M7_seq_aaindex1_val_acc_feature_sel_134b_250ep', dpi=400)
plt.show()


"""
Validation accuracy scores in physicochemical model with and without quantitative features
"""
plt.clf()
val_acc = df_acc['val_main_output_accuracy']
epochs = range(1, len(df_acc_aaindex1['scaled_pca']) + 1)

plt.plot(epochs, df_acc_aaindex1['raw_acc'], 'tab:blue', label='Sequence + AAIndex1 (Raw)')
plt.plot(epochs, df_acc_aaindex1['var_corr_0.5'], 'tab:red', label='Sequence + AAIndex1 (Variance + Correlation > 0.5)')
plt.plot(epochs, df_acc_saf_aaindex['val_main_output_accuracy'], 'tab:green', label='Sequence + AAindex1 + SpC (SAF)')
plt.plot(epochs, val_acc, 'tab:orange', label='Sequence + AAindex1 + Combined SpC')

plt.title("\n".join(wrap('Performance comparison of physicochemical models with and without quantitation as feature', 60)))
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()


"""
Plot ROC curves
"""
# load datasets
df_seq = pd.read_table('final_data/M1_PRED_seq_134b_200epoch.tsv')
df_saf = pd.read_table('final_data/M2_PRED_seq_saf_134b_150ep.tsv')
df_aaindex = pd.read_table('final_data/M7_PRED_seq_aaindex1_raw_134b_250ep.tsv')
df_saf_aaindex = pd.read_table('final_data/M8_PRED_seq_saf_aaindex1_var-corr05_134b_250ep.tsv')
df_spc_aaindex = pd.read_table('final_data/M8_PRED_seq_combSpC_aaindex1_var_corr_134b_250ep.tsv')

# sequence model
fpr_seq, tpr_seq, thresholds_seq = roc_curve(df_seq['Detectability'], df_seq['Predictions'])
auc_seq = auc(fpr_seq, tpr_seq)

# sequence + SAF (quantitative) model
fpr_saf, tpr_saf, thresholds_saf = roc_curve(df_saf['Detectability'], df_saf['Predictions'])
auc_saf = auc(fpr_saf, tpr_saf)

# sequence + AAIndex1 (physicochemical) model
fpr_aaindex, tpr_aaindex, thresholds_aaindex = roc_curve(df_aaindex['Detectability'],
                                                         df_aaindex['Predictions'])
auc_aaindex = auc(fpr_aaindex, tpr_aaindex)

# sequence + SAF and physicochemical model
fpr_saf_aaindex, tpr_saf_aaindex, thresholds_saf_aaindex = roc_curve(df_saf_aaindex['Detectability'],
                                                                        df_saf_aaindex['Predictions'])
auc_saf_aaindex = auc(fpr_saf_aaindex, tpr_saf_aaindex)

# sequence + all spectral counts (SpC) + AAIndex1 (i.e. combined) model
fpr_comb, tpr_comb, thresholds_comb = roc_curve(df_spc_aaindex['Detectability'],
                                                df_spc_aaindex['Predictions'])
auc_comb = auc(fpr_comb, tpr_comb)

# plot ROC curve
plt.plot(fpr_seq, tpr_seq, 'tab:blue', label='Sequence (AUC = {:.3f})'.format(auc_seq))
plt.plot(fpr_saf, tpr_saf, 'tab:red', label='Sequence + SAF (AUC = {:.3f})'.format(auc_saf))
plt.plot(fpr_aaindex, tpr_aaindex, 'tab:orange', label='Sequence + AAIndex1 (AUC = {:.3f})'.format(auc_aaindex))
plt.plot(fpr_saf_aaindex, tpr_saf_aaindex, 'tab:green', label='Sequence + SAF + AAIndex1 (AUC = {:.3f})'.format(auc_saf_aaindex))
plt.plot(fpr_comb, tpr_comb, 'tab:pink', label='Sequence + Combined SpC + AAIndex1 (AUC = {:.3f})'.format(auc_comb))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve comparison of different features')
plt.legend(loc='best')
plt.savefig('final_figures/ROC_curve_all', dpi=200)
plt.show()


"""
Plot PR curves
"""
# sequence model
pr_seq, re_seq, thresh_seq = precision_recall_curve(df_seq['Detectability'], df_seq['Predictions'])

# sequence + SAF (quantitative) model
pr_saf, re_saf, thresh_saf = precision_recall_curve(df_saf['Detectability'], df_saf['Predictions'])

# sequence + AAIndex1 (physicochemical) model
pr_aaindex, re_aaindex, thresh_aaindex = precision_recall_curve(df_aaindex['Detectability'],
                                                                df_aaindex['Predictions'])

# sequence + all spectral counts (SpC) + AAIndex1 (i.e. combined) model
pr_saf_aaindex, re_saf_aaindex, thresh_saf_aaindex = precision_recall_curve(df_saf_aaindex['Detectability'],
                                                                               df_saf_aaindex['Predictions'])

# plot PR curve
plt.plot(re_seq, pr_seq, 'tab:blue', label='Sequence')
plt.plot(re_saf, pr_saf, 'tab:red', label='Sequence + SAF')
plt.plot(re_aaindex, pr_aaindex, 'tab:orange', label='Sequence + AAIndex1')
plt.plot(re_saf_aaindex, pr_saf_aaindex, 'tab:green', label='Sequence + SAF + AAIndex1')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('PR curve')
plt.legend(loc='best')
plt.savefig('final_figures/PR_curve_all', dpi=200)
plt.show()


"""
Histogram plot comparing prediction distributions for detected vs. undetected peptides
"""
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x=df['Predictions'], hue=df['Detectability'], stat='count', bins=25)
ax.set_title('Sequence + Combined SpC + AAIndex1 (Human)', fontsize=18)

ax.set_xlabel('Predictions', fontsize=16)
ax.set_ylabel('Count', fontsize=16)

plt.axvline(x=df['Detectability'].mean(), color='r')
plt.axvline(x=df['Predictions'].mean(), color='g', ls='--')
plt.text(0.54,2500, ('Mean = {:.3f}'.format(df['Predictions'].mean())), fontsize=13)

plt.setp(ax.get_legend().get_texts(), fontsize=14)
plt.setp(ax.get_legend().get_title(), fontsize=14)


"""
Plot histogram of detectability distributions
"""
# import dataframe
df = pd.read_table('final_data/M8_PRED_seq_nsaf_aaindex1_var_corr_134b_250ep.tsv')

# plot histogram
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x=df['Predictions'], hue=df['Detectability'], stat='count', bins=25)
ax.set_title('Sequence + SAF + Feature-selected AAIndex1 (Human)', fontsize=18)

ax.set_xlabel('Predictions', fontsize=16)
ax.set_ylabel('Count', fontsize=16)

plt.axvline(x=df['Detectability'].mean(), color='r')
plt.axvline(x=df['Predictions'].mean(), color='g', ls='--')
plt.text(0.3,3200, ('Mean = {:.3f}'.format(df['Predictions'].mean())), fontsize=13)

plt.setp(ax.get_legend().get_texts(), fontsize=14)
plt.setp(ax.get_legend().get_title(), fontsize=14)

plt.savefig('final_figures/M8_hist_seq_nsaf_aaindex1_var_corr05.png', dpi=200)


"""
Accuracy trend over peptide length (human and mouse)
"""
# import datasets
mouse_scores = pd.read_table('final_data/QUANT_PRED_ALL_MOUSE_label-free.tsv')
seq_saf_fs_aaindex_pred = pd.read_table('final_data/M8_PRED_seq_nsaf_aaindex1_var_corr_REDO_134b_250ep.tsv')

def get_confusion_matrix_scores(df, peptide_col_index, true_col_index, pred_col_index):
    scores_df = df.iloc[: , [peptide_col_index, true_col_index, pred_col_index]].copy()
    scores_df['pred_round'] = np.rint(scores_df.iloc[:, 2]).astype(int)
    scores_df['Length'] = scores_df.iloc[:, 0].str.len()
    scores_df['pred_class'] = 0

    scores_df.loc[(scores_df.iloc[:, 1] == 1) & (scores_df['pred_round'] == 1), 'pred_class'] = 'TP'
    scores_df.loc[(scores_df.iloc[:, 1] == 1) & (scores_df['pred_round'] == 0), 'pred_class'] = 'FN'
    scores_df.loc[(scores_df.iloc[:, 1] == 0) & (scores_df['pred_round'] == 0), 'pred_class'] = 'TN'
    scores_df.loc[(scores_df.iloc[:, 1] == 0) & (scores_df['pred_round'] == 1), 'pred_class'] = 'FP'

    scores_df['TP'] = 0
    scores_df['FN'] = 0
    scores_df['TN'] = 0
    scores_df['FP'] = 0

    scores_df.loc[(scores_df.iloc[:, 1] == 1) & (scores_df['pred_round'] == 1), 'TP'] = 1
    scores_df.loc[(scores_df.iloc[:, 1] == 1) & (scores_df['pred_round'] == 0), 'FN'] = 1
    scores_df.loc[(scores_df.iloc[:, 1] == 0) & (scores_df['pred_round'] == 0), 'TN'] = 1
    scores_df.loc[(scores_df.iloc[:, 1] == 0) & (scores_df['pred_round'] == 1), 'FP'] = 1

    return scores_df

def get_acc_by_peptide_len(df):
    columns = ['Length', 'TP', 'FN', 'TN', 'FP']
    peptide_len_df = pd.DataFrame(columns=columns)

    for i in range(7, 41):
        test = df[df['Peptide'].str.len() == i]

        lenTP = test['TP'].sum()
        lenFN = test['FN'].sum()
        lenTN = test['TN'].sum()
        lenFP = test['FP'].sum()

        data = [i, lenTP, lenFN, lenTN, lenFP]
        peptide_len_df.loc[i - 7] = data

    peptide_len_df['Acc'] = (peptide_len_df['TP'] + peptide_len_df['TN']) / (peptide_len_df['TP'] + peptide_len_df['TN'] +
                                                                             peptide_len_df['FP'] + peptide_len_df['FN'])

    return peptide_len_df

# example with plotting accuracy trend for each quantitative method (Homo sapiens)
plt.clf()
plt.figure(figsize=(12.5, 2.56))

plt.plot(h_saf_df['Length'], h_saf_df['Acc'], color='tab:blue', label='SAF')
plt.plot(h_nsaf_df['Length'], h_nsaf_df['Acc'], color='tab:red', label='NSAF')
plt.plot(h_cbnp_df['Length'], h_cbnp_df['Acc'], color='tab:orange', label='CBN(P)')
plt.plot(h_cbns_df['Length'], h_cbns_df['Acc'], color='tab:green', label='CBN(S)')
plt.plot(h_rsc_df['Length'], h_rsc_df['Acc'], color='tab:pink', label='RSc')

plt.title('Accuracy trend over peptide length with quantitative features (Homo sapiens)', fontsize=13)
plt.xlabel('Peptide Length (aa)', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.legend(loc='best', ncol=1)
plt.savefig('final_figures/HUMAN_QUANT-FEATURES_acc_over_peptide_length.png', dpi=200, bbox_inches='tight')
plt.show()

# example with plotting accuracy trend for different feature models (Mus musculus)
plt.clf()
plt.figure(figsize=(12.5, 2.56))

plt.plot(saf_df['Length'], saf_df['Acc'], color='tab:blue', label='SAF')
plt.plot(nsaf_df['Length'], nsaf_df['Acc'], color='tab:red', label='NSAF')
plt.plot(CBN_P_df['Length'], CBN_P_df['Acc'], color='tab:orange', label='CBN(P)')
plt.plot(CBN_S_df['Length'], CBN_S_df['Acc'], color='tab:green', label='CBN(S)')
plt.plot(RSc_df['Length'], RSc_df['Acc'], color='tab:pink', label='RSc')

plt.title('Accuracy trend over peptide length with quantitative features (Mus Musculus)', fontsize=13)
plt.xlabel('Peptide Length (aa)', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)

plt.legend(loc='best', ncol=1)
plt.savefig('final_figures/MOUSE_QUANT-FEATURES_acc_over_peptide_length.png', dpi=200, bbox_inches='tight')
plt.show()


"""
Stacked-bar confusion matrix based on peptide length
"""
# example for combined model (Homo sapiens)
stacked_df = hum_seq_saf_pred.iloc[:, 1:5].apply(lambda x: x*100/sum(x), axis=1)
stacked_df.index = range(7, 41)

stacked_df.plot(kind='bar', stacked=True,
          colormap=ListedColormap(sns.color_palette("tab20c", 10)),
          figsize=(12,6))

plt.title("Sequence + SAF + AAIndex1 (feature-selected) (Homo sapiens)", fontsize=18)
plt.xlabel("Peptide Length (aa)", fontsize=15)
plt.ylabel("Prediction Classification (%)", fontsize=15)
plt.legend(fontsize=14)
plt.xticks(rotation=360)

plt.savefig('final_figures/HUMAN_STACKED_CONF_MATRIX_over_peptide_length_seq_saf_aaindex.png', dpi=200, bbox_inches='tight')
plt.show()


"""
Grouped bar charts comparing model performance in H. sapiens vs. M. musculus
by accuracy and MCC
"""
# model performance by accuracy
labels = ['Seq', 'Seq + SAF', 'Seq + Combined SpCs', 'Seq + AAIndex1', 'Seq + SAF + AAIndex1']
mouse_acc = mouse_quant_scores['Accuracy']
human_acc = human_quant_scores['Accuracy']

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, human_acc, width, label='Homo sapiens', color='tab:cyan')
rects2 = ax.bar(x + width/2, mouse_acc, width, label='Mus musculus', color='tab:orange')

ax.set_ylabel('Accuracy')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend(loc=4)
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 4)), (p.get_x()*1.005, p.get_height()*1.005))

plt.savefig('final_figures/MOUSE_human_acc_barplot.png', dpi=200, bbox_inches='tight')


# model performance by MCC
labels = ['Seq', 'Seq + SAF', 'Seq + Combined SpCs', 'Seq + AAIndex1', 'Seq + SAF + AAIndex1']
human_mcc = human_quant_scores['MCC']
mouse_mcc = mouse_quant_scores['MCC']

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, human_mcc, width, label='Homo sapiens', color='tab:cyan')
rects2 = ax.bar(x + width/2, mouse_mcc, width, label='Mus musculus', color='tab:orange')

ax.set_ylabel('MCC')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
for p in ax.patches:
    ax.annotate(str(round(p.get_height(), 4)), (p.get_x()*1.005, p.get_height()*1.005))

ax.legend(loc=4)
plt.savefig('final_figures/MOUSE_human_MCC_barplot.png', dpi=200, bbox_inches='tight')


"""
Boxplot for peptide length distributions in H. sapiens vs M. musculus
"""
# get H. sapiens dataset
hom_detected_peptides = pd.read_table('../data/detected_peptides_all_quant_aaindex1.tsv')
hom_undetected_peptides = pd.read_table('../data/undetected_peptides_all_quant_aaindex1.tsv')

hom_detected = hom_detected_peptides.iloc[:, [0]].copy()
hom_detected['Detectability'] = 1
hom_undetected = hom_undetected_peptides.iloc[:, [0]].copy()
hom_undetected['Detectability'] = 0
hom_df = pd.concat((hom_detected, hom_undetected)).reset_index()
hom_df['Length'] = hom_df['Peptide'].str.len()

# get M. musculus dataset
mus_detected_peptides = pd.read_table('../data/detected_peptides_SpC_aaindex1_PXD027822.tsv')
mus_undetected_peptides = pd.read_table('../data/undetected_peptides_SpC_aaindex1_PXD027822.tsv')

mus_detected = mus_detected_peptides.iloc[:, [0]].copy()
mus_detected['Detectability'] = 1
mus_undetected = mus_undetected_peptides.iloc[:, [0]].copy()
mus_undetected['Detectability'] = 0
mus_df = pd.concat((mus_detected, mus_undetected)).reset_index()
mus_df['Length'] = mus_df['Peptide'].str.len()

# prepare dataframe
hom_df['Species'] = 'Homo sapiens'
mus_df['Species'] = 'Mus musculus'
hom_mus_df = pd.concat((hom_df, mus_df)).reset_index()
hom_mus_df.rename({'Peptide length (aa)': 'Peptide Length (aa)'}, axis=1, inplace=True)

# plot boxplot
colours = ['#A1C9F4', '#FFB482']
sns.set_palette(sns.color_palette(colours))
sns.boxplot(x="Species", y="Peptide Length (aa)", hue="Detectability",
            data=hom_mus_df, linewidth=0.6, fliersize=1.8)
plt.savefig('final_figures/boxplot_peptide_length.png', dpi=200)


'''
Other data exploration figures
'''
# Correlation matrix of AAIndex1 features
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sns.heatmap(X_train_aaindex1.corr(), ax=ax)
ax.set_title('Correlation matrix of calculated AAIndex1 features (train dataset)')
ax.set_xlabel('AAIndex1 Features (n=553)')
ax.set_xlabel('AAIndex1 Features (n=553)')
plt.show()
