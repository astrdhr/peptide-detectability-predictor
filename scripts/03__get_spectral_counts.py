import numpy as np
import pandas as pd


def calculate_nsaf(fasta_peptides_file, detected_peptides_file, undetected_peptides_file):
    """ Performs spectral counting using the NSAF method for detected and undetected peptide datasets.
    :param fasta_peptides_file: output file from get_fasta_peptides().
    :param detected_peptides_file: output file from get_detected_peptides().
    :param undetected_peptides_file: output file from get_undetected_peptides().
    :return: TSV files for detected and undetected datasets, with additional 'Quantification' column.
    """

    fasta_peptides = pd.read_table(fasta_peptides_file)
    detected_peptides = pd.read_table(detected_peptides_file)
    undetected_peptides = pd.read_table(undetected_peptides_file)

    # sort detected_peptides by PEP
    detected_peptides = detected_peptides.sort_values(by=['PEP']).reset_index(drop=True)

    # remove any peptides in detected_peptides that map to more than one different protein
    detected_peptides = detected_peptides.groupby('Peptide').filter(lambda x: x['Protein'].nunique() == 1)

    # perform spectral counting on detected peptides
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

    detected_peptides_abundance["Quantification"] = detected_peptides_abundance["PSM_per_protein"] / \
                                                    detected_peptides_abundance["Protein_length"]

    # clean dataset for detected peptides df
    detected_peptides_abundance = detected_peptides_abundance.drop_duplicates(subset=['Peptide', 'Protein'],
                                                                              keep="first").reset_index(drop=True)
    detected_peptides_abundance = detected_peptides_abundance.drop(labels=['PSM_per_protein'], axis=1)

    # assign spectral counts from proteins in detected df to undetected df
    detected_peptides_nsaf_dict = dict(zip(detected_peptides_abundance.Protein, detected_peptides_abundance.Quantification))
    undetected_peptides['Quantification'] = undetected_peptides['Protein'].map(detected_peptides_nsaf_dict)
    undetected_peptides = undetected_peptides.groupby('Peptide').filter(lambda x: x['Protein'].nunique() == 1)
    undetected_peptides = undetected_peptides.drop_duplicates(subset=['Peptide', 'Protein'],
                                                              keep="first").reset_index(drop=True)

    # clean dataset for undetected peptides df
    undetected_peptides = undetected_peptides.rename(columns={"Length": "Protein_length"})
    undetected_peptides = undetected_peptides[['Protein', 'Peptide', 'PEP', 'Protein_length', 'Quantification']]

    # export datasets
    detected_peptides_abundance.to_csv("detected_peptides_NSAF.tsv", sep='\t', index=False)
    undetected_peptides.to_csv("undetected_peptides_NSAF.tsv", sep='\t', index=False)
