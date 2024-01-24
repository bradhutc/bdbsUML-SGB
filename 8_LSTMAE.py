import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed


file_path = '/N/project/catypGC/BDBS/bdbsparallaxprocessed_data.csv'

data = pd.read_csv(file_path)[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']]

scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
print(len(data))
data_scaled = scaler.fit_transform(data)


# Reshape data to [samples, timesteps, features] for Long-Short Term Memory (LSTM) Format.
data_scaled = data_scaled.reshape((data_scaled.shape[0], 1, data_scaled.shape[1]))

train_data, test_data = train_test_split(data_scaled, test_size=0.2, random_state=42)


model = Sequential()
model.add(LSTM(2, activation='elu', input_shape=(1, 6)))
model.add(RepeatVector(1))
model.add(LSTM(100, activation='elu', return_sequences=True))
model.add(TimeDistributed(Dense(6)))
model.compile(optimizer='adam', loss='mse')


learning_rate = 0.001
batch_size = 16

# Train the model
history = model.fit(train_data, train_data, epochs=3, batch_size=batch_size, validation_data=(test_data, test_data), verbose=1)

# model.summary()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('loss_vs_epochs.png', format='png', dpi=500)
plt.close()


# Predict and evaluate
predicted = model.predict(data_scaled, verbose=0)

# Reshape and inverse transform to original scale
predicted = predicted.reshape((predicted.shape[0], predicted.shape[2]))
predicted = scaler.inverse_transform(predicted)

test_loss = model.evaluate(test_data, test_data)
print(f"Test Loss: {test_loss}")

encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)

# Extract latent space
latent_space_representations = encoder.predict(data_scaled)

# Get the reconstructed outputs
reconstructed = model.predict(data_scaled)

reconstructed_original_scale = scaler.inverse_transform(reconstructed.reshape(-1, 6))

# Step 1: Predict the data
reconstructed_data = model.predict(data_scaled)

# Step 2: Extract latent space
latent_space_values = encoder.predict(data_scaled)

original_data_scaled_inverse = scaler.inverse_transform(data_scaled.reshape(-1, 6))
reconstructed_data_inverse = scaler.inverse_transform(reconstructed_data.reshape(-1, 6))

matched_data = pd.read_csv('/N/project/catypGC/BDBS/simbad_matched.csv')

# Step 3: Combine the data into a DataFrame
combined_data = pd.DataFrame(original_data_scaled_inverse, columns=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'])
combined_data[['umag_reconstructed', 'gmag_reconstructed', 'rmag_reconstructed', 'imag_reconstructed', 'zmag_reconstructed', 'ymag_reconstructed']] = reconstructed_data_inverse
combined_data[['latent_dim1', 'latent_dim2']] = latent_space_values.reshape(-1, 2)

# Step 4: Save to CSV file & Plot Results
output_file_path = 'combined_data.csv'
combined_data.to_csv(output_file_path, index=False)
output_dir = 'magnitude_reconstructions'
os.makedirs(output_dir, exist_ok=True)

merged_data = pd.merge(combined_data, matched_data, left_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], right_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], how='inner')

# Map OTYPE to numerical values
otype_mapping = {otype: i for i, otype in enumerate(merged_data['object_type'].unique())}
merged_data['OTYPE_numeric'] = merged_data['object_type'].map(otype_mapping)

palette = sns.color_palette("husl", n_colors=len(merged_data['object_type'].unique()))

plt.figure(figsize=(12, 8))

marker_styles = {'RRLyrae': 's', 'HorBranch*': 'D', 'EllipVar': '^', 'RGB*': '*', 'delSctV*': 'x', 'EclBin': 'p', 'HotSubdwarf_Candidate': '*', 'ChemPec*' : 'v', 'Radio': 'o', 'LongPeriodV*_Candidate': 'o', 'LongPeriodV*': 'o','ChemPec*': '^', 'Variable*':'*', 'PulsV': 'v'}
# Loop through each star type and plot with corresponding marker style
plt.scatter(combined_data['latent_dim1'], combined_data['latent_dim2'], c='grey', s=0.2, alpha=0.3)
for otype, marker in marker_styles.items():
    otype_data = merged_data[merged_data['object_type'] == otype]
    print(otype_data.columns)
    plt.scatter(otype_data['latent_dim1'], otype_data['latent_dim2'], label=otype, marker=marker, s=30, alpha=0.9)
