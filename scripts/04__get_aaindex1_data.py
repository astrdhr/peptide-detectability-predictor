import numpy as np
import pandas as pd


def convert_aaindex1_to_df(data, output_file_name):
    """
    Converts raw AAIndex1 into Pandas DataFrame. Outputs amino acid indices with accession
    and description. Adapted from: https://github.com/tadorfer/AAIndex/blob/master/raw_to_df.py.
    Note: AAIndex1 data was downloaded from: https://www.genome.jp/ftp/db/community/aaindex/.
    """

    # define column names and initialize dataframe
    columns = ['Accession', 'Description']
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
          'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    columns = columns + aa
    df = pd.DataFrame(data=[], columns=columns)

    # conversion by parsing text file line by line
    with open(data) as f:
        for i, line in enumerate(f):
            if line[0] == 'H':
                accession = line.split()[1]
            if line[0] == 'D':
                description = line.split()[1:]
                description = ' '.join(description)
            if line[0] == 'I':
                tmp = i
            if 'tmp' in locals():
                if i == tmp + 1:
                    tmp1 = [accession] + [description] + line.split()
                if i == tmp + 2:
                    tmp2 = line.split()
                    tmp_all = tmp1 + tmp2
                    tmp_all = pd.DataFrame([tmp_all], columns=columns)
                    df = df.append([tmp_all]).reset_index(drop=True)
                    df.to_csv(output_file_name, sep='\t', index=False)

    return df


def calculate_aaindex1_features(peptide_df_file, aaindex1_df_file, output_file_name):

    peptide_df = pd.read_table(peptide_df_file)
    aaindex1_df = pd.read_table(aaindex1_df_file)

    # turn aaindex1 values into dict format so peptide AAs can be mapped to dictionary values
    print('Turning AAIndex1 values into dict format')
    del aaindex1_df['Description']
    aaindex1_df = aaindex1_df.set_index('Accession')
    aaindex1_dict = aaindex1_df.to_dict(orient='records')
    aaindex1_df.insert(loc=0, column='aa_dict', value=aaindex1_dict)

    # add aaindex1 accession columns to peptide_df
    print('Adding AAIndex1 accession columns to peptide_df')
    aaindex1_df = aaindex1_df.reset_index(level=0)
    aaindex1_accession_cols = []
    for row in aaindex1_df['Accession']:
        aaindex1_accession_cols.append(row)
    peptide_df = pd.concat([peptide_df, pd.DataFrame(columns=aaindex1_accession_cols)])

    # calculate aaindex1 values for each peptide
    print('Calculating AAIndex1 values for each peptide')
    all_peptide_vals = []

    for peptide in peptide_df['Peptide']:
        peptide_vals = []
        for aa_dict in aaindex1_df['aa_dict']:
            aaindex1_vals = sum([aa_dict[x] for x in peptide]) / len(peptide) # divide by peptide length for further normalisation
            peptide_vals.append(aaindex1_vals)
        all_peptide_vals.append(peptide_vals)

    print('Almost done, just creating the df')
    peptide_df.iloc[0:, 6:] = all_peptide_vals  # column index to add aaindex1 values will vary by df, need to specify
    peptide_df.to_csv(output_file_name, sep='\t', index=False)

    print('Finished!')
