import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
#Import data
"""data_hr = list()
root = ".\\data"
files = [ os.path.join(root, item) for item in os.listdir(root) if os.path.isfile(os.path.join(root, item)) ]
for filename in files[1:]:
    print(filename)
    data_hr.append(np.load(filename)[1:321]) #Some sims have more points, discarding them
data_hr = np.array(data_hr)
"""
#Load data and check shape
data_hr = np.load('allData.npy')
print(data_hr.shape)

# Calculating the statistics for each N and each channel
results = []
for n in range(data_hr.shape[0]):
    for c in range(data_hr.shape[-1]):
        # Extract the data for the specific N and channel
        data = data_hr[n, :, :, :, c]
        # Calculate statistics
        min_val = data.min()
        max_val = data.max()
        mean_val = data.mean()
        std_dev = data.std()
        # Store the results
        results.append([n, c, min_val, max_val, mean_val, std_dev])
# Convert results to a pandas DataFrame for better readability
df = pd.DataFrame(results, columns=['N_simulation', 'Channel', 'Min', 'Max', 'Mean', 'Std Dev'])

# Display the DataFrame
print(df)

# Plotting histograms for each simulation and each channel
values = ['u','v','p']
for n in range(data_hr.shape[0]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))
    fig.tight_layout(pad=3.0)
    for c in range(data_hr.shape[-1]):
        ax = axes[c]
        data = data_hr[n, :, :, :, c].flatten()  # Flatten the array to get a 1D array of values
        ax.hist(data, bins=60, alpha=0.75)
        # Calculating the mean and standard deviation
        mean_val = np.mean(data)
        std_dev = np.std(data)

        # Setting the limits to two standard deviations from the mean
        ax.set_xlim(mean_val - 4 * std_dev, mean_val + 4 * std_dev)

        ax.set_title(f'Histogram for Simulation {n+1}, Channel={values[c]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    plt.savefig("./temp/"+str(n+1)+".png")
    plt.close()
