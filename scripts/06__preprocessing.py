import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.model_selection
from sklearn.utils import shuffle


def filter_by_peptide_length(peptide_list, min_length, max_length):

    for peptide in peptide_list:
        peptide_filtered = peptide.str.len() >= min_length & peptide.str.len() <= max_length

    return peptide_filtered



def filter_by_peptide_length(detected_peptides_df, undetected_peptides_df, min_peptide_length, max_peptide_length):

    detected_peptides = pd.read_table(detected_peptides_df)
    undetected_peptides = pd.read_table(undetected_peptides_df)

    detected_peptides = detected_peptides.loc[(detected_peptides["Peptide"].str.len() >= min_peptide_length) &
                                              (detected_peptides["Peptide"].str.len() <= max_peptide_length)].reset_index(drop=True)

    undetected_peptides = undetected_peptides.loc[(undetected_peptides["Peptide"].str.len() >= min_peptide_length) &
                                                  (undetected_peptides["Peptide"].str.len() <= max_peptide_length)].reset_index(drop=True)

    return detected_peptides.shape, undetected_peptides.shape



def create_train_val_test_sets(detected_df, undetected_df):

    detected_peptides = pd.read_table(detected_df)
    undetected_peptides = pd.read_table(undetected_df)

    # filter by peptide lengths

    # undersample majority class (i.e. undetected peptides)
    undetected_peptides_balanced = undetected_peptides.sample(n=detected_peptides.shape[0],
                                                              random_state=1).reset_index(drop=True)


