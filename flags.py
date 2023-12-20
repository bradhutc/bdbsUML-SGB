import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

file_path = '/N/project/catypGC/BDBS/bdbsparallaxprocessed_data.csv'
df = pd.read_csv(file_path)[['gaia_id','BDBS_ID','umag','gmag','rmag','imag','zmag','ymag']]


bands = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']

# Function to check if there's a dip between two magnitudes
def is_anomaly(row):
    for i in range(1, len(bands) - 1):
        # Check for a significant dip compared to the left band
        if row[bands[i]] < row[bands[i - 1]] and abs(row[bands[i-1]] - row[bands[i]]) >= 0.1:
            # Check for a significant increase compared to the right band
            if row[bands[i + 1]] > row[bands[i]] and abs(row[bands[i + 1]] - row[bands[i]]) >= 0.1:
                return 0  # Anomaly
    return 1  # Not an anomaly


df['label'] = df.apply(is_anomaly, axis=1)

print(len(df[df['label'] == 0]))


print(df.head())


sns.set_style("whitegrid")
stars_df = df[df['label'] == 0]
print(len(stars_df))

plt.figure(figsize=(14, 8))
plt.plot(bands, stars_df[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']].T.values, marker='o')
plt.title('Photometric Pattern of Stars Across Different Bands', fontsize=16)
plt.xlabel('Photometric Band', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.gca().invert_yaxis()
plt.show()

stars_df.to_csv('anomalies.csv', index=False)