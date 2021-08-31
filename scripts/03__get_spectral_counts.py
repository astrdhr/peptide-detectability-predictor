import numpy as np
import pandas as pd


def calculate_SpC(fasta_peptides_file, detected_peptides_file, undetected_peptides_file):
    """ Performs spectral counting using the NSAF method for detected and undetected peptide datasets.
    :param fasta_peptides_file_path: output file from get_fasta_peptides().
    :param detected_peptides_file_path: output file from get_detected_peptides().
    :param undetected_peptides_file_path: output file from get_undetected_peptides().
    :return: TSV files for detected and undetected datasets, with additional 'Quantification' column.
    """

    fasta_peptides = pd.read_table(fasta_peptides_file)
    detected_peptides = pd.read_table(detected_peptides_file)
    undetected_peptides = pd.read_table(undetected_peptides_file)

    # remove any peptides in detected_peptides that map to more than one different protein
    detected_peptides = detected_peptides.groupby('Peptide').filter(lambda x: x['Protein'].nunique() == 1)

    # perform spectral counting on detected peptides
    #detected_peptides['Protein'] = detected_peptides['Protein'].str.split('|').str[1] # use if protein col is in fasta format
    fasta_protein_len_dict = dict(zip(fasta_peptides.Protein, fasta_peptides.Length))
    detected_peptides['Protein_length'] = detected_peptides['Protein'].map(fasta_protein_len_dict)
    detected_peptides["PSM"] = 1
    detected_peptides_abundance = detected_peptides.groupby("Protein")["PSM"].sum()
    detected_peptides_abundance = pd.merge(detected_peptides, detected_peptides_abundance,
                                           on='Protein', how='left',
                                           suffixes=(None, '_per_protein'), indicator=True) \
                                    .query("_merge == 'both'") \
                                    .drop('_merge', 1) \
                                    .drop('PSM', 1)

    # divide number of spectral counts (SpC) by protein's length (L) -- gives SAF
    detected_peptides_abundance["SAF"] = detected_peptides_abundance["PSM_per_protein"] / \
                                         detected_peptides_abundance["Protein_length"]

    # get SpC/L of total number proteins identified
    proteins_SAF = detected_peptides_abundance.filter(["Protein", "PSM_per_protein", "SAF"], axis=1)
    proteins_SAF = proteins_SAF.drop_duplicates('Protein')

    # normalise by dividing by sum of SpC/L for all proteins in sample -- gives NSAF
    detected_peptides_abundance["NSAF"] = detected_peptides_abundance["SAF"] / proteins_SAF["SAF"].sum()

    # calculate spectral counts using CBN-based method
    SpC = detected_peptides_abundance["PSM_per_protein"]
    total_SpC = proteins_SAF["PSM_per_protein"].sum()
    total_protein = proteins_SAF.shape[0]
    f_P = 1 / total_protein
    f_t = 1 / total_SpC

    detected_peptides_abundance["CBN_P"] = (SpC / total_SpC) + f_P
    detected_peptides_abundance["CBN_S"] = (SpC / total_SpC) + f_t

    # calculate spectral counts using Rsc-based method
    detected_peptides_abundance["RSc"] = np.log2((SpC + 0.5) / (total_SpC - SpC + 0.5))

    # clean dataset for detected peptides df
    detected_peptides_abundance = detected_peptides_abundance.drop_duplicates(subset=['Peptide', 'Protein'],
                                                                              keep="first").reset_index(drop=True)
    detected_peptides_abundance = detected_peptides_abundance.drop(labels=['PSM_per_protein'], axis=1)

    # assign spectral counts from proteins in detected df to undetected df
    #undetected_peptides['Protein'] = undetected_peptides['Protein'].str.split('|').str[1]  # use if protein col is in fasta format
    detected_peptides_saf_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.SAF))
    undetected_peptides['SAF'] = undetected_peptides['Protein'].map(detected_peptides_saf_dict)

    detected_peptides_nsaf_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.NSAF))
    undetected_peptides['NSAF'] = undetected_peptides['Protein'].map(detected_peptides_nsaf_dict)

    detected_peptides_CBN_P_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.CBN_P))
    undetected_peptides['CBN_P'] = undetected_peptides['Protein'].map(detected_peptides_CBN_P_dict)

    detected_peptides_CBN_S_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.CBN_S))
    undetected_peptides['CBN_S'] = undetected_peptides['Protein'].map(detected_peptides_CBN_S_dict)

    detected_peptides_RSc_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.RSc))
    undetected_peptides['RSc'] = undetected_peptides['Protein'].map(detected_peptides_RSc_dict)

    # remove any peptides in undetected_peptides that map to more than one different protein, and drop duplicates
    undetected_peptides = undetected_peptides.groupby('Peptide').filter(lambda x: x['Protein'].nunique() == 1)
    undetected_peptides = undetected_peptides.drop_duplicates(subset=['Peptide', 'Protein'],
                                                              keep="first").reset_index(drop=True)

    # clean dataset for detected and undetected peptides df
    undetected_peptides = undetected_peptides[['Peptide', 'SAF', 'NSAF', 'CBN_P', 'CBN_S', 'RSc']]
    detected_peptides_abundance = detected_peptides_abundance[['Peptide', 'SAF', 'NSAF', 'CBN_P', 'CBN_S', 'RSc']]

    # export datasets
    detected_peptides_abundance.to_csv("detected_peptides_SpC_mouse_TMT_PXD027737.tsv", sep='\t', index=False)
    undetected_peptides.to_csv("undetected_peptides_SpC_mouse_TMT_PXD027737.tsv", sep='\t', index=False)
