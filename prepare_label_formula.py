"""
**************************************************************
*                                                            *
*          MD-HIT-formula: prepare_label_formula             *
*                       Dr. Qin Li & Jianjun Hu              *
*    Guizhou University of Finance and Economics, 2023.5     *
**************************************************************
* function: use selected formulas of non-redundant set to get*
*           their property values such as formation energy   *
**************************************************************
"""
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split

# Create the argument parser
parser = argparse.ArgumentParser(description='get material properties from formulas')

# Add the input file parameters with default values
parser.add_argument('--input1', type=str, help='Path to the first input file', default='MP222_formulas_nr_12706_mendeleev_threshold=1.5.csv')
parser.add_argument('--input2', type=str, help='Path to the second input file', default='MP_materials_formulas_bandgap_forme.csv')
parser.add_argument('--testset_ratio', type=float, help='test set ratio', default=0.2)

# Add the output file parameter with a default value
parser.add_argument('--property', type=str, help='property column', default='formation_energy_per_atom')
parser.add_argument('--output_train', type=str, help='Path to the training output file', default='comp_training.csv')
parser.add_argument('--output_test', type=str, help='Path to the test output file', default='comp_test.csv')



# Parse the command-line arguments
args = parser.parse_args()
property = args.property
testset_ratio = args.testset_ratio

# Read the input files
df1 = pd.read_csv(args.input1)
df2 = pd.read_csv(args.input2)

# Rename columns in df2
df2 = df2.rename(columns={'pretty_formula': 'formula'}) 

# Merge the dataframes
merged_data = pd.merge(df1, df2, on="formula")

# Select desired columns
submerged_data = merged_data[["material_id", "formula",property]]

# Rename columns in submerged_data
submerged_data = submerged_data.rename(columns={'formula': 'composition'})

# Group by composition and find minimum energy
grouped = submerged_data.groupby('composition', as_index=False)
min_energy = grouped.apply(lambda x: x.loc[x[property].idxmin()])

# Save the resulting dataframe to the output file
# min_energy.to_csv(args.output, index=False)
train_data, test_data = train_test_split(min_energy, test_size=testset_ratio, random_state=42)

# Save the training set to a CSV file
train_data.to_csv(args.output_train, index=False)

# Save the test set to a CSV file
test_data.to_csv(args.output_test, index=False)


print(f"check output in {args.output_train} and {args.output_test}")
