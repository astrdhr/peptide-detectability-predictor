import numpy as np
import pandas as pd


def get_detected_peptides(evidence_file_path):
    """Extracts the proteins and respective peptide spectrum matches from the MaxQuant
    evidence.txt file, dataset is also cleaned.
    :param evidence_file_path: evidence.txt file (output from MaxQuant analysis).
    :return: TSV file containing protein, respective peptide spectrum matches (i.e.
    peptide sequences) and associated PEP scores.
    """

    evidence = pd.read_table(evidence_file_path)

    # clean dataset
    evidence = evidence.loc[evidence['Potential contaminant'] != '+']
    evidence = evidence.loc[evidence['Reverse'] != '+']
    evidence = evidence.loc[(evidence['Missed cleavages'] == 0)]
    evidence = evidence[~evidence['Proteins'].str.contains("CON__", na=False)]

    # keep only rows that don't contain multiple proteins in "Proteins" column
    evidence = evidence[~evidence['Proteins'].str.contains(";", na=False)]

    # extract protein and peptide sequence from evidence df
    # make the "leading razor groups" column as the protein column
    detected_peptides = evidence[['Proteins', 'Sequence', 'PEP']]
    detected_peptides = detected_peptides.rename(columns={"Proteins": "Protein", "Sequence": "Peptide"})

    detected_peptides.to_csv("detected_peptides.tsv", sep='\t', index=False)


def get_undetected_peptides(fasta_peptides_file_path, detected_peptides_file_path):
    """Gets undetected peptides by retaining proteins in fasta_peptides also present
    in detected_peptides, and finding the difference in peptides.
    :param fasta_peptides_file_path: output file from get_fasta_peptides().
    :param detected_peptides_file_path: output file from get_detected_peptides().
    :return: TSV file with columns 'Protein', 'Peptide', protein 'Length' and 'PEP' score.
    """

    fasta_peptides = pd.read_table(fasta_peptides_file_path)
    detected_peptides = pd.read_table(detected_peptides_file_path)

    expected_peptides = fasta_peptides[fasta_peptides["Protein"].isin(detected_peptides["Protein"])]
    undetected_peptides = expected_peptides[~expected_peptides["Peptide"].isin(detected_peptides["Peptide"])]
    undetected_peptides = undetected_peptides[~undetected_peptides['Peptide'].str.contains("U", na=False)]
    undetected_peptides["PEP"] = [detected_peptides["PEP"].mean()] * undetected_peptides.shape[0]

    undetected_peptides.to_csv("undetected_peptides.tsv", sep='\t', index=False)
