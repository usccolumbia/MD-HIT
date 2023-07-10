"""
**************************************************************
*                                                            *
*                       MD-HIT-formula                       *
*          Dr. Jianjun Hu, Qin Li, & Nihang Fu               *
*             University of South Carolina, 2023.5           *
**************************************************************
* function: given a formula set, reduce its redundancy with  *
*           a given threshold with ElMD distance             *
**************************************************************
        Note: by default, we skipped materials with >50 atoms
"""
#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from ElMD import ElMD #https://github.com/lrcfmd/ElMD
from ElMD import elmd
from pymatgen.core import Composition

import argparse

import time
import multiprocessing as mp
from multiprocessing import set_start_method

set_start_method('fork')


# Create the argument parser
parser = argparse.ArgumentParser(description='MD-hit-formula redunancy reduction of materials composition dataset')


parser.add_argument('--inputfile', type=str, help='input formula file',default="MP_allcompounds_synthesis_totalenergy.csv")
parser.add_argument('--threshold', type=float, help='minimum distance threshold',default=5)
parser.add_argument('--similarity', type=str, help='similarity metric',default='mendeleev')
parser.add_argument('--outfile', type=str, help='output dataset file',default='MP')
parser.add_argument('--formula_column', type=str, help='formula_column name',default='pretty_formula')
parser.add_argument('--np', type=int, help='No.of parallel processes',default=10)


# Parse the command-line arguments
args = parser.parse_args()
outfile = args.outfile
formula_column = args.formula_column
np = args.np

# Access the integer parameter
threshold = float(args.threshold)
similarity = args.similarity
inputfile = args.inputfile

distancemetrics = ['mendeleev', 'petti', 'atomic', 'mod_petti','oliynyk', 
    'oliynyk_sc', 'jarvis', 'jarvis_sc', 'magpie', 'magpie_sc','cgcnn', 
    'elemnet', 'mat2vec', 'matscholar', 'megnet16']
print("similarity between CaTiO3 and SrTiO3:")
for i,s in enumerate(distancemetrics):
    print(i,"",s,end="\t\t")
    x = ElMD("CaTiO3", metric=s)
    print(x.elmd("SrTiO3"))


df=pd.read_csv(inputfile)
#df=pd.read_csv("MP_test.csv")
df = df.fillna('Na')

# threshold=5
# similarity="mendeleev" #or Magpie, Roost  check here https://github.com/lrcfmd/ElMD



# Linear: mendeleev petti atomic mod_petti
# Chemically Derived: oliynyk oliynyk_sc jarvis jarvis_sc magpie magpie_sc
# Machine Learnt: cgcnn elemnet mat2vec matscholar megnet16

# d=elmd("NaCl", "LiCl", metric=similarity)
#d=elmd("SrTiO3", "MgSiO3", metric=similarity)
#print('distance between SrTiO3 and MgSrO3:',d)
# # x = ElMD("NaCl").full_feature_vector()
# x = ElMD("NaCl", metric="magpie").feature_vector
# print(x)


# formulas = df.iloc[:,0]
formulas = df[[formula_column]]
formulas = formulas.drop_duplicates()
# formulas = formulas.iloc[:1000,:] #testing

# print(formulas.head())
print(formulas.shape)


# print(formulas)
def check_atomno(name):
    return Composition(name).num_atoms <= 50
def get_atomno(f):
    no= Composition(f).num_atoms
    return no

df_sorted = formulas[formulas['pretty_formula'].apply(check_atomno)]
# formulas = formulas[formulas['pretty_formula'].str.len() <= 50]
#sort by formual length
# df_sorted = formulas.sort_values(by='pretty_formula', key=lambda x: x.str.len(),ascending=True)
#sort by no. of atoms.
df_sorted = formulas.sort_values(by='pretty_formula', key=lambda x:x.map(get_atomno),ascending=True)


# follow CD-hit algo. CD-HIT: accelerated for clustering the next-generation sequencing data
# key is how to set the similarity threshold paramer...need to determine to separate ABO3.
# in protein sequence, they use a sequence similarity percentage. eg. 95%
# -c sequence identity threshold, default 0.9
# this is the default cd-hit's "global sequence identity"
# calculated as:
#  number of identical amino acids in alignment
#  divided by the full length of the shorter sequence
# -G use global sequence identity, default 1
#  if set to 0, then use local sequence identity, calculated as :
#  number of identical amino acids in alignment
#  divided by the length of the alignment
#  NOTE!!! don't use -G 0 unless you use alignment coverage controls
#  see options -aL, -AL, -aS, -AS
    
#start with the formula with maximum no. of atoms and start the clustering by check process.
# cluster=[df_sorted.iloc[0,0]]
cluster = ['H2O'] #start with water! as the seed material.
candidates=df_sorted['pretty_formula'].tolist()[1:]
if 'H2O' in  candidates:
    candidates.remove("H2O")


min_dist = mp.Manager().Value('i', 10000000)

def calc(tup):
    f, c = tup
    try:
        d=elmd(f, c, metric=similarity)  #Uue element error
    except:
        return   #skip this update.
    if d < min_dist.value:
        min_dist.value=d    


def main():
    total=0
    for i,f in enumerate(candidates):
        # print(i,f,end="")
        min_dist.value=1000000
        
        if len(cluster) > 1000:
            # 10 threads
            with mp.Pool(processes=np) as pool:
                pool.map(calc, [(f, c) for c in cluster])

        
        else:
            for c in cluster:
                d=elmd(f, c, metric=similarity)
                if d <min_dist.value:
                    min_dist.value=d
                
        # print("  ",min_dist.value,end=",")
        if min_dist.value>threshold:
            cluster.append(f)
            total+=1
            if total%100 ==0:
                print(f,min_dist.value,"....added..", total, i, )
        

        


    print(len(cluster), " formulas left")
    df = pd.DataFrame({'formula': cluster})
    # df = pd.DataFrame(goodstocks,columns=['symbol','lowest','highest','percentage'])
    df = df.sort_values('formula')

    # Save the DataFrame to a CSV file
    filename=f'{outfile}_formulas_nr_{len(cluster)}_{similarity}_threshold={threshold}.csv'
    df.to_csv(filename, index=False)
    print(f'check file {filename}')
    #threshold 10 ----85
    #5 ---

if __name__ == '__main__':
    main()
