import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

csv_file_path = '/N/project/catypGC/BDBS/bdbsparallaxprocessed_data.csv'
stars_df = pd.read_csv(csv_file_path)
print(len(stars_df))

output_directory = '/N/project/catypGC/BDBS/plots'

os.makedirs(output_directory, exist_ok=True)

def create_hexbin_plot(x, y, xlabel, ylabel, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hexbin(stars_df[x], stars_df[y], gridsize=1000, cmap='bone', mincnt=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(False)
    fig.savefig(os.path.join(output_directory, filename), format='jpeg', dpi=1000)
    plt.close(fig)

def create_plot():
    print("Available columns for plotting: ", stars_df.columns.tolist())
    x = input("Enter the column name for the X-axis: ")
    y = input("Enter the column name for the Y-axis: ")
    if x in stars_df.columns and y in stars_df.columns:
        create_hexbin_plot(x, y, x, y, f'{y} vs {x}', f'{y}_vs_{x} for Stars in the Southern Galactic Bulge.jpeg')
    else:
        print("Invalid column names entered. Please try again.")

create_plot()
