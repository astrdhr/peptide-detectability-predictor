import pandas as pd
from multiprocessing import Pool as ProcessPool
import json
from Bio import SeqIO






proteins  = SeqIO.to_dict(SeqIO.parse("human_variants_proteome.fasta", "fasta"))



def generateQuant(dir):


    allPeptides = pd.read_csv(dir+"/allPeptides.csv")

    proteinsDf = pd.DataFrame([{"id": id, "sequence": str(protein.seq)} for id, protein in proteins.items()])
    proteinsDf["length"] = proteinsDf["sequence"].str.len()

    # links = [({"protein": row2["id"], "peptide": row["peptide"]} for index2, row2 in proteinsDf[proteinsDf["sequence"].str.contains(row["peptide"])].iterrows())
    #           for index, row in allPeptides[allPeptides["probability"]>0.5].iterrows()]
    links = []
    for index, row in allPeptides.iterrows():
        if row["probability"]>0.5:
            matches = proteinsDf[proteinsDf["sequence"].str.contains(row["peptide"])]
            for index2, row2 in matches.iterrows():
                links.append({"protein": row2["id"], "peptide": row["peptide"]})

    pd.DataFrame(links).to_csv(dir+"/links.csv", index=False)



    # pool = ProcessPool(min(len(samplesList), 10))
    # df = pd.concat(pool.map(analyseSample, samplesList))
    # pd.DataFrame(df).to_csv("prabsMax/countIons.csv", index=False)

    links = pd.read_csv(dir+"/links.csv")

    links = links.merge(allPeptides, right_on="peptide", left_on="peptide")
    proteinAbundance = links.groupby("protein")["count"].sum()
    proteinAbundance = pd.DataFrame({"protein": proteinAbundance.index, "count": proteinAbundance}).reset_index(drop=True).merge(proteinsDf, left_on="protein", right_on="id")
    proteinAbundance["quantification"] = proteinAbundance["count"]*100/proteinAbundance["length"]
    del proteinAbundance["sequence"]
    del proteinAbundance["count"]
    del proteinAbundance["length"]
    del proteinAbundance["id"]
    proteinAbundance.to_csv(dir+"/quantification.csv", index=None)


    quantification = pd.read_csv(dir+"/quantification.csv")
    links = pd.read_csv(dir+"/links.csv")


    allQuant = allPeptides.merge(links, left_on="peptide", right_on="peptide").merge(quantification, left_on="protein", right_on="protein").groupby("peptide")["quantification"].agg("max")
    allQuant = pd.DataFrame({"peptide": allQuant.index, "quantification": allQuant}).reset_index(drop=True).merge(allPeptides, left_on="peptide", right_on="peptide")
    allQuant.to_csv(dir+"/allPeptidesQuant.csv", index=None)


    negatives = pd.read_csv(dir+"/negatives.csv")
    negativesQuant = negatives.merge(quantification, left_on="protein", right_on="protein").groupby("peptide")["quantification"].agg("max")
    negativesQuant = pd.DataFrame({"peptide": negativesQuant.index, "quantification": negativesQuant}).reset_index(drop=True).merge(negatives, left_on="peptide", right_on="peptide")
    negativesQuant.to_csv(dir+"/negativesQuant.csv", index=None)