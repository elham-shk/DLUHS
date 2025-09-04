
tf.config.experimental.list_physical_devices('GPU')

import os
import numpy as np
import shap
import matplotlib.pyplot as plt
import shap
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  

# Select number of samples for SHAP
num_samples =1000 # Adjust this value
X_sample = train_x[:num_samples]  # Shape: (num_samples, time_steps, channels)

X_sample = train_x[:num_samples]

print("X_sample1.shape:",X_sample.shape)



time_steps = X_sample.shape[1]  # Should match waveform input (e.g., 300 for 3s @ 0.01s)
print(f"Computing SHAP values for {num_samples} samples with {time_steps} time steps each...")

# Define SHAP explainer (DeepExplainer is suited for deep learning models)
explainer = shap.DeepExplainer(model, X_sample)


# Compute SHAP values with progress tracking
shap_values = []
for i in tqdm(range(num_samples), desc="SHAP Computation Progress"):
    shap_values.append(explainer.shap_values(X_sample[i:i+1]))  # Process sample-by-sample

shap_values = np.array(shap_values)  # Convert to NumPy array
print(f"SHAP computation completed for {num_samples} samples.")

# Fix Shape Issue
shap_values = np.array(shap_values)  # Convert list to array (should be (num_samples, time_steps, channels))
shap_values = shap_values.squeeze()  # Remove unnecessary dimensions if needed

# Aggregate across samples & channels(Take absolute mean for importance)
shap_mean = np.mean(np.abs(shap_values), axis=(0, 2))  # Shape: (time_steps,)

# Ensure x-axis and y-axis match
assert shap_mean.shape[0] == time_steps, f"Shape mismatch: shap_mean {shap_mean.shape} vs. time_steps {time_steps}"

# Plot the influence of time steps
plt.figure(figsize=(10, 5))
plt.plot(np.arange(time_steps) *step*dt, shap_mean, marker='o', linestyle='-', color='b', alpha=0.7)
plt.xlabel("Time Step (s)")
plt.ylabel("Mean SHAP Value")
plt.title(f"SHAP Importance Over Time ({num_samples} Samples)")
plt.grid(True)
plt.show()



# Extract SHAP for selected Sa(T) values
targets = [26, 67, 77, 94]  # Indices corresponding to the desired Sa(T) of 0.1s, 1.0s, 2s and 5s
colors = ['gray', '#d62728', '#2ca02c', 'orange']
periods = [0.1, 1, 2, 5]

# Compute mean absolute SHAP values across samples and features
mean_shap_all = np.mean(np.abs(shap_values), axis=(0, 1, 2))  # Shape: (111,)
print("shape of mean_shap_all:", mean_shap_all.shape)

# Compute normalization factor (sum of mean |SHAP| across all Sa(T) targets)
norm_factor = np.sum(mean_shap_all, axis=0)  # Scalar value

plt.figure(figsize=(8, 12))  # Adjust figure size for better layout

bar_width = 0.15  # Adjust bar width for clarity

for i, target in enumerate(targets):
    # Compute mean |SHAP| for selected Sa(T) target and normalize
    shap_target = np.mean(np.abs(shap_values1[:, :, :, target]), axis=(0, 2))  # Shape: (100,)
    shap_target /= norm_factor  # Normalize by sum of mean |SHAP|

    print(f"shap_target shape: {shap_target.shape}, time_axis shape: {time_axis.shape}")  # Debugging

    # Ensure time_axis matches shap_target dimensions
    if len(time_axis) != len(shap_target):
        time_axis = np.linspace(0, len(shap_target) * dt, len(shap_target))  # Adjust time_axis

    # Switch X and Y, and use `barh()` for horizontal bars**
    plt.barh(time_axis * 1, shap_target, height=bar_width, label=f"$S_a(T={periods[i]:.1f}s)$", color=colors[i], alpha=0.8)

# Swap X and Y Labels for Horizontal Bar Chart**
plt.ylabel("Ground Motion Amplitude at Different Time Steps after P-wave detection (g)", fontsize=18)
plt.xlabel("Relative Mean |SHAP|", fontsize=18)  

#plt.xticks(fontsize=17)
plt.xticks(np.arange(0, 0.21, 0.05), fontsize=17)
plt.yticks(fontsize=17)
#plt.xlim(0, 0.2)

