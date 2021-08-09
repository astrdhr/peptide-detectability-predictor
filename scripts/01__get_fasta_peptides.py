import pandas as pd
import re
from Bio import SeqIO
import csv

def get_fasta_proteins(fasta_file_path):
    """
    Extracts fasta protein name and its corresponding sequence from a raw fasta file.
    :param fasta_file_path -- path to raw fasta file.
    :return: TSV file containing columns for protein name and associated sequence.
    """

    # convert FASTA file to TSV file
    with open('fasta.tsv', 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        tsvfile.write("Sequence\tFasta headers\n")
        for record in SeqIO.parse(fasta_file_path, "fasta"):
            writer.writerow([record.seq, record.id])

    fasta_proteins = pd.read_table('fasta.tsv')

    # extract protein IDs from FASTA headers, reorder columns
    fasta_proteins['Protein'] = fasta_proteins['Fasta headers'].str.split('|').str[1]
    fasta_proteins.drop(['Fasta headers'], axis=1)
    reorder_col = ['Protein', 'Sequence']
    fasta_proteins = fasta_proteins.reindex(columns=reorder_col)

    fasta_proteins.to_csv("fasta_proteins.tsv", sep='\t', index=False)


def trp_digest(seq, returnStart):
    """
    Performs tryptic digest given a protein sequence.
    This code has been adapted and was originally provided by Esteban Gea on 20/04/2021.
    :param:
        seq -- protein sequence.
        returnStart -- return sequence position of the digested peptide: 0 for TRUE, 1 for FALSE.
    :return: list of digested peptides, along with their sequence position (if specified).
    """

    pattern = "(.(?:(?<![KR](?!P)).)*)"
    frags = list(filter(None, re.findall(pattern, seq)))
    misCleavage = 0
    min_peptide_length = 7
    max_peptide_length = 51
    peptides = []

    for i in range(0, len(frags)):
        count = 0
        if i == 0:
            start = 0
        else:
            start = len("".join(frags[:i]))

        if (len(frags[i]) >= min_peptide_length) & (len(frags[i]) <= max_peptide_length):
            if returnStart:
                peptides.append({"sequence": frags[i], "start": start})
            else:
                peptides.append(frags[i])

        if i != len(frags) - 1:
            for j in range(i + 1, len(frags)):
                if count < misCleavage:
                    count = count + 1
                    pep = "".join(frags[i:j + 1])
                    if (len(pep) >= min_peptide_length) & (len(pep) <= max_peptide_length):
                        if returnStart:
                            peptides.append({"sequence": pep, "start": start})
                        else:
                            peptides.append(pep)
                    elif len(pep) > 40:
                        break
                else:
                    break

    return peptides


def get_fasta_peptides(fasta_proteins_file_path):
    """
    Performs protein tryptic digest given a dataframe of protein sequences.
    :param fasta_proteins_file_path -- Output from get_fasta_proteins().
    :return: TSV file with additional 'Peptide' column following tryptic digest.
    """

    fasta_proteins = pd.read_table(fasta_proteins_file_path, header=0)
    fasta_peptides = fasta_proteins

    fasta_peptides['Peptide'] = fasta_peptides.apply(lambda row: trp_digest(row['Sequence'], 0), axis=1)
    fasta_peptides = fasta_peptides.explode('Peptide')

    # get sequence length for each protein, add values to a "Length" column
    # This will be used for the spectral counting calculation later on
    fasta_peptides["Length"] = fasta_peptides["Sequence"].str.len()

    # data cleaning
    fasta_peptides = fasta_peptides.drop(['Sequence'], axis=1)
    fasta_peptides.dropna(subset=["Peptide"], inplace=True)

    fasta_peptides.to_csv("fasta_peptides.tsv", sep='\t', index=False)
