import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.interpolate import make_interp_spline
import numpy as np
file_path = 'C:/Users/Bradl/OneDrive/Desktop/BDBS/stellaridentification/bdbsparallaxprocessed_data.csv'
df = pd.read_csv(file_path)[['gaia_id','BDBS_ID','umag','gmag','rmag','imag','zmag','ymag']]


bands = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']

# Function to check if there's a dip between two magnitudes
def is_anomaly(row):
    for i in range(1, len(bands) - 1):
        # Check for a significant dip compared to the left band
        if row[bands[i]] < row[bands[i - 1]] and abs(row[bands[i-1]] - row[bands[i]]) >= 0.25:
            # Check for a significant increase compared to the right band
            if row[bands[i + 1]] > row[bands[i]] and abs(row[bands[i + 1]] - row[bands[i]]) >= 0.25:
                return 0  # Anomaly
    return 1  # Not an anomaly


df['label'] = df.apply(is_anomaly, axis=1)

print(len(df[df['label'] == 0]))


print(df.head())


sns.set_style("whitegrid")
anom_df = df[df['label'] == 0]
print(len(anom_df))

plt.figure(figsize=(14, 8))
spl = make_interp_spline(range(len(bands)), anom_df[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']].T.values, k=2)
bands_smooth = np.linspace(0, 5, 300)
curve_smooth = spl(bands_smooth)
plt.plot(bands_smooth, curve_smooth)
plt.title('Photometric Pattern of Anomalous Stars Across Different Bands', fontsize=16)
plt.xlabel('Photometric Band', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.xticks(range(len(bands)), bands)
plt.gca().invert_yaxis()
plt.show()

reg_df = df[df['label'] == 1]
sample_df=reg_df.sample(5)

plt.figure(figsize=(14, 8))
spl = make_interp_spline(range(len(bands)), sample_df[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']].T.values, k=2)
bands_smooth = np.linspace(0, 5, 300)
curve_smooth = spl(bands_smooth)
plt.plot(bands_smooth, curve_smooth)
plt.title('Photometric Pattern of Non-Anomalous Stars Across Different Bands', fontsize=16)
plt.xlabel('Photometric Band', fontsize=14)
plt.ylabel('Magnitude', fontsize=14)
plt.xticks(range(len(bands)), bands)
plt.gca().invert_yaxis()
plt.show()




anom_df.to_csv('anomalies.csv', index=False)
reg_df.to_csv('clean.csv', index=False)