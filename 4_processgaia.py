import pandas as pd
import numpy as np

csv_file_path = '/N/project/catypGC/BDBS/gaiaresults.csv'
df = pd.read_csv(csv_file_path).drop(['table1_oid'], axis=1)

filtered_df = df[(df['parallax'] <= 0.14)]

color_indices = ['u-g', 'u-r', 'u-i', 'u-z', 'u-y', 'g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y']

for index in color_indices:
    color_band_1, color_band_2 = index.split('-')
    filtered_df[f'{index}'] = filtered_df[f'{color_band_1}mag'] - filtered_df[f'{color_band_2}mag']

for column in filtered_df.columns:
    if column not in ['bdbs_id', 'gaia_id']:
        filtered_df[column] = filtered_df[column].round(5)

print(len(filtered_df))

filtered_df['distance_parsecs'] = 1 / (filtered_df['parallax']/1000)  # Convert parallax from milliarcseconds to arcseconds

filtered_df['absolute_magnitude_G'] = filtered_df['gmag'] - 2.5 * (np.log10((filtered_df['distance_parsecs']/10)**2))

processed_file_path = 'bdbsparallaxprocessed_data.csv'
filtered_df.to_csv(processed_file_path, index=False)
print("Processed data saved to:", processed_file_path)