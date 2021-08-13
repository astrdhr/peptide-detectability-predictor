import pandas as pd
import json
from Bio import SeqIO
import re


def digest(row):
    seq = row[0]
    protein = row[1]
    seq = seq.rstrip('*')
    #frags = list(filter(None, re.findall("(.*?(?:K|R|$))", seq)))
    frags = list(filter(None, re.findall(".(?:(?<![KR](?!P)).)*", seq)))
    misCleavage = 2
    peptides = []

    for i in range(0, len(frags)):
        count = 0
        if i == 0:
            start = 1
        else:
            start = len("".join(frags[:i])) + 1

        if (len(frags[i]) >= 8) & (len(frags[i]) <= 40):
            peptides.append({"peptide":frags[i], "protein":protein})

        if i != len(frags) - 1:
            for j in range(i + 1, len(frags)):
                if count < misCleavage:
                    pep = "".join(frags[i:j + 1])
                    if (len(pep) >= 8) & (len(pep) <= 40):
                        peptides.append({"peptide":pep, "protein":protein})
                        count = count + 1
                    elif len(pep) > 40:
                        break
                else:
                    break

    return pd.DataFrame(peptides)


def generateNegatives(dir):

    proteinsTsv = pd.read_csv(dir+"/protein.tsv", sep="\t")
    nonambiguousProteins = proteinsTsv[(proteinsTsv["Indistinguishable Proteins"].isna()) & (proteinsTsv["Protein Probability"]>0.9)]["Protein"]


    proteinSequences = []

    db = SeqIO.to_dict(SeqIO.parse("human_variants_proteome.fasta", "fasta"))


    for protein in nonambiguousProteins:
        proteinSequences.append({"sequence": str(db[protein].seq), "protein": protein})

    proteinSequences = pd.DataFrame(proteinSequences)

    digested = proteinSequences.apply(digest, axis = 1)

    digested = pd.concat(digested.tolist())


    allPeptides = pd.read_csv(dir+"/allPeptides.csv")["peptide"]

    negativesPeptides = digested[~digested["peptide"].isin(allPeptides)]

    negativesPeptides.to_csv(dir+"/negatives.csv", index=False)
