"""
**************************************************************
*                                                            *
*              MD-HIT-structure Parallel version 1.0         *
*             Dr. Jianjun Hu & Qin Li, Nihang Fu             *
*                University of South Carolina, 2023.5        *
**************************************************************
* function: get non-redundant mp_ids of materials given the  *
*           XRD/OFM feature matrix datafile                  *
**************************************************************
"""

import pandas as pd
from pymatgen.core import Structure,Composition
import numpy as np
import argparse
import time
import multiprocessing as mp
import os
from multiprocessing import set_start_method

# set_start_method('fork')

# Create the argument parser
parser = argparse.ArgumentParser(description='MD-hit-structure redundancy reduction of materials dataset')

parser.add_argument('--featureFile', type=str, help='feature file', default='XRD_MP_features.csv')
parser.add_argument('--threshold', type=float, help='minimum distance threshold', default=0.8)
parser.add_argument('--similarity', type=str, help='similarity metric', default='XRD')
parser.add_argument('--np', type=int, help='no. of thresholds', default=10)
parser.add_argument('--outfile', type=str, help='output file', default='MP_structure_nr.csv')
parser.add_argument('--quiet', action='store_true', help='quiet run')



# Parse the command-line arguments
args = parser.parse_args()
nThread = args.np
threshold = args.threshold
similarity = args.similarity
outfile = args.outfile

min_dist = mp.Manager().Value('i', 10000000)


df_features1 = pd.read_csv(args.featureFile)

# series = df_features1['0']
# duplicates = series[series.duplicated()]
# print(duplicates)

df_features = df_features1.set_index("0")
df_features.index = df_features.index.astype(str)
# print(df_features.columns)
# print(df_features.index)
# df_features.head()

def get_distance(mpid1, mpid2):
    d = 0
    f1 = df_features.loc[mpid1][1:]
    f2 = df_features.loc[mpid2][1:]
    d = np.linalg.norm(f1 - f2)
    return d

dist = get_distance('mp-684654', 'mp-35683')
if not args.quiet:
    print("Bi2O3 and Na1P1F3")
    print(f"Similarity: {dist}")

def check_atomno(f):
    return Composition(f).num_atoms <= 100

def get_atomno(f):
    return Composition(f).num_atoms

df_sorted = df_features[df_features['1'].apply(check_atomno)]
df_sorted2 = df_sorted.sort_values(by='1', key=lambda x: x.map(get_atomno), ascending=True)
df_sorted2.head()
if not args.quiet:
    print('feature dimension:',df_sorted2.shape)
candidates = df_sorted2.index.tolist()


cluster = ['mp-2998']  # SrTiO3
if cluster[0] not in df_sorted2.index:
    if not args.quiet:
        print(f"Seed mpid SrTiO3 does not exist in the dataset.")
        print(f"{candidates[0]} is used as the seed material")
    cluster=[candidates[0]]
if cluster[0] in candidates:
    candidates.remove(cluster[0])


def calc(tup):
    f, c = tup
    d = get_distance(f, c)
    if d < min_dist.value:
        min_dist.value = d

def get_formula(mpid):
    return df_sorted2.loc[mpid].iloc[0]

def main():
    total = 0
    for i, f in enumerate(candidates[:]):
        min_dist.value = 1000000
        if len(cluster) > 1000:
            with mp.Pool(processes=nThread) as pool:
                pool.map(calc, [(f, c) for c in cluster])
        else:
            for c in cluster:
                d = get_distance(f, c)
                if d < min_dist.value:
                    min_dist.value = d
        if min_dist.value > threshold:
            cluster.append(f)
            total += 1
            if not args.quiet:
                print(f"{i:{6}}...{f:{10}}....added..", total)
        
    
    if not args.quiet:
        print(len(cluster), "samples left")
    
    df = pd.DataFrame({'mpid': cluster})
    df = df.sort_values('mpid')

    df['formula'] = df['mpid'].apply(get_formula)

    filename = f'{outfile.split(".")[0]}_{len(cluster)}_{similarity}_threshold={threshold}.csv'
    df.to_csv(filename, index=False)
    print(f'Check file {filename}')

if __name__ == '__main__':
    main()
