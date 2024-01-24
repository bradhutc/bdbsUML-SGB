import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
csv_file_path = '/N/project/catypGC/BDBS/combined_data.csv'
df = pd.read_csv(csv_file_path)

df['g-i'] = df['gmag']-df['imag']

x_min  = df['g-i'].min()
x_min_prime = df['latent_dim1'].min()
x_max = df['g-i'].max()
x_max_prime = df['latent_dim1'].max()

y_min = df['imag'].min()
y_min_prime = df['latent_dim2'].min()
y_max = df['imag'].max()
y_max_prime = df['latent_dim2'].max()

m_x = (x_min-x_max)/(x_min_prime-x_max_prime)
b_x = x_min - m_x*x_min_prime

m_y = (y_min-y_max)/(y_min_prime-y_max_prime)
b_y = y_min - m_y*y_min_prime

df['latent_dim1'] = m_x *df['latent_dim1'] + b_x
df['latent_dim2'] = m_y * df['latent_dim2'] + b_y

matched_data = pd.read_csv('/N/project/catypGC/BDBS/simbad_matched.csv')

merged_data = pd.merge(df, matched_data, left_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], right_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], how='inner')
print(len(merged_data))

object_type_counts = merged_data['object_type'].value_counts()

print("Number of 'Star' entries:", object_type_counts.get('Star', 0))

# Map OTYPE to numerical values
otype_mapping = {otype: i for i, otype in enumerate(merged_data['object_type'].unique())}
merged_data['OTYPE_numeric'] = merged_data['object_type'].map(otype_mapping)

palette = sns.color_palette("husl", n_colors=len(merged_data['object_type'].unique()))

plt.figure(figsize=(12, 8))

marker_styles = {'RRLyrae': 's', 'HorBranch*': 'D', 'EllipVar': '^', 'RGB*': '*', 'delSctV*': 'x', 'EclBin': 'p', 'HotSubdwarf_Candidate': '*', 'ChemPec*' : 'v', 'Radio': 'o', 'LongPeriodV*_Candidate': 'o', 'LongPeriodV*': 'o','ChemPec*': '^', 'Variable*':'*', 'PulsV': 'v'}

plt.scatter(df['latent_dim1'], df['latent_dim2'], c='black', s=0.2, alpha=0.1)
for otype, marker in marker_styles.items():
    otype_data = merged_data[merged_data['object_type'] == otype]
    plt.scatter(otype_data['latent_dim1'], otype_data['latent_dim2'], label=otype, marker=marker, s=15.0, alpha=0.9)

plt.xlabel('CSFS-x')
plt.ylabel('CSFS-y')
plt.title('Composite Stellar Feature Space (CSFS) for Stars in the Southern Galactic Bulge (SGB)')
plt.legend(title="OTYPE")

plt.gca().set_facecolor('white')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.savefig('latentspace.png', format='png', dpi=500)
plt.show()