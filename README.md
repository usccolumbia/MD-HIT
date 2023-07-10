# MD-HIT, Redundancy control for materials informatics
MD-HIT algorithms for reducing redundancy of materials composition or structure datasets (similar to CD-HIT in bioinformatics)

It can potentially improve the model performance on minority/OOD materials and provide more objective performance estimation.

Developed by Dr. Jianjun Hu, Nihang Fu of University of South Carolina and Qin Li from Guizhou University of Finance and Economics

```
Citation:
Q.Li, N.Fu, S.Omee, J.Hu, MD-HIT: MACHINE LEARNING FOR MATERIALS PROPERTY PREDICTION WITH DATASET REDUNDANCY CONTROL. Arxiv 2023.6

```

# How to use MD-HIT

## Generating composition non-redundant datasets
```
python MD_hit_formula_parallel.py --inputfile MP_formulas.csv --threshold 0.3 --similarity mendeleev --outfile MP_formula_mendeleev_0.3.csv --formula_column pretty_formula --np 10
```
where np is the no. of parallel thresholds to speed up the running.

## Generating structure non-redundant datasets

Step 1: generate XRD/OFM feature files from a given folder of cif structure files
```
python get_struct_feature_parallel.py --cif_folder mycifs_folder --feature XRD --output_file dataset_XRD_features.csv --np 10

python get_struct_feature_parallel.py --cif_folder mycifs_folder --feature OFM --output_file dataset_OFM_features.csv --np 10

```
These calculations may take a long time for big datasets. So be patient. You can download pre-calculated features from https://doi.org/10.6084/m9.figshare.23651568

Step 2: generate non-redundant dataset using different thresholds

```
python MD_hit_structure_parallel.py --featureFile dataset_XRD_features.csv --threshold 0.2 --similarity XRD -np 10 --outfile dataset_XRD_threshold0.2.csv

```
These calculations may take a long time for big datasets. So be patient.


## Generating datasets for materials prediction codes

prepare_label_formula.py helps to use the filtered out non-redundant formulas to find their corresponding property values to get the final composition-property pair file.

prepare_label_structure.py helps to use the filtered out non-redundant sample mp-ids to find their corresponding property values to get the final composition-property pair file.

