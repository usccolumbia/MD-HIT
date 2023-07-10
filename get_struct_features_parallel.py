"""
**************************************************************
*                                                            *
*              MD-HIT-structure:  featurecalculation         *
*                       Dr. Jianjun Hu & Nihang Fu           *
*                University of South Carolina, 2023.5        *
**************************************************************
* function: given cif folder, calculate                      *
*           XRD/OFM feature matrix datafile                  *
**************************************************************
"""

import numpy as np
import pandas as pd
import multiprocessing as mp
# from pymatgen.core.structure import IStructure
import pymatgen.analysis.diffraction.xrd as xrd
from scipy.stats import gaussian_kde
import itertools
import glob
import argparse
from matminer.featurizers.structure.matrix import OrbitalFieldMatrix
from pymatgen.core import Structure

ofm= OrbitalFieldMatrix()

def smooth(lt):
    X_data = list(itertools.chain.from_iterable([[x[0]] * int(x[1] * 10) for x in lt]))
    X_plot = np.linspace(0, 90, 901)
    kde = gaussian_kde(X_data, bw_method=.01)
    output = kde.evaluate(X_plot)
    return output

def convert_to_powder(cif_str, file):
    crystal_struct = Structure.from_file(file)
    powd = xrd.XRDCalculator()
    powd_patt = powd.get_pattern(crystal_struct)
    return [[x, y] for x, y in zip(powd_patt.x, powd_patt.intensity)]

def calc(tup):
    featureType = "OFM"
    i, file,featureType = tup
    #print("featureType:",featureType, '****')
    try:
        structure = Structure.from_file(file)
    except:
        print("structure reading failed")
        return []
    

    if "/" in file.split('-')[-2]:
        mpid = file.split('-')[-2].split("/")[-1]+"-"+file.split('-')[-1].split(".")[0]
    else:# SG-1-AtomNo-4-Mul-1-SiteNo-4-3-1-mp-996984.cif
        mpid = file.split('-')[-2]+"-"+file.split('-')[-1].split(".")[0]
    formula = structure.formula
    print(formula)
    try:
        
        if featureType=='XRD':
            feature = smooth(convert_to_powder(open(file, 'rb').read(), file))
        elif featureType=="OFM":
            #s = Structure.from_file(file)
            feature = ofm.featurize(structure)
        print(feature.shape)
    except:
        print("error....")
        return ()
    
    return [mpid, formula.replace(" ", ''), feature]



def main(cif_folder, output_file,featureType,np):

    onlyfiles = glob.glob(f"{cif_folder}/*.cif")
    print(f"{len(onlyfiles)} files found..")

    if featureType=="XRD":
        df = pd.DataFrame(index=range(len(onlyfiles)), columns=range(903))
    elif featureType=="OFM":
        df = pd.DataFrame(index=range(len(onlyfiles)), columns=range(1026))

    with mp.Pool(processes=np) as pool:
        results = pool.map(calc, [(i, file,featureType) for i, file in enumerate(onlyfiles)])

    tmp = 0
    for res in results:
        try:
            if res != []:
                df.iloc[tmp, 0] = res[0]
                df.iloc[tmp, 1] = res[1]
                df.iloc[tmp, 2:] = res[2]
                tmp += 1
        except:
            continue

    df.to_csv(output_file, index=None)
    print(f"check output in {output_file}")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Convert CIF files to XRD/OFM features')
    parser.add_argument('--cif_folder', help='Path to the folder containing CIF files',default="./cifs")
    parser.add_argument('--output_file', help='Output file name for CSV',default="MP_features.csv")
    parser.add_argument('--feature', help='Output file name for CSV',default="OFM") #OFM, XRD
    parser.add_argument('--np', help='No.of processes',default=5) #OFM, XRD
    args = parser.parse_args()
    featureType= args.feature
    output_file = f'{featureType}_{args.output_file}'

    main(args.cif_folder, output_file, featureType,args.np)
