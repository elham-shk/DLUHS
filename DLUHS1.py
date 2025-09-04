
## input (H1,H2)  output (AvgSA of H1 & H2)
###  window 3 sec 
# train_x shape: (2464, 300, 2)
# train_y shape: (2464, 111)
# test_x shape: (617, 300, 2)
# test_y shape: (617, 111)
# MSE: 0.0039
# RMSE: 0.0625
# MAE: 0.0214
# R² Score: 0.8238
# Pearson coefficient: 0.9079




import random
from keras.models import Model
from keras import layers
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
from keras.optimizers import Adam
from keras import regularizers
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import tensorflow as tf
import keras.backend as K
import tensorflow.keras.backend as K 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scipy.stats import zscore


# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


from keras.losses import mean_squared_error

def rmse(y_true, y_pred):
    return K.sqrt(mean_squared_error(y_true, y_pred))





def CNN_model(n_timesteps, n_chns, loss=rmse, optimizer='adam', l2_reg=0.0001, modelDisplay=False):
    input_layer = Input(shape=(n_timesteps, n_chns))

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(input_layer)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(128, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(256, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(512, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l1(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(256, kernel_size=3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu',
                     kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(64, kernel_size=3, strides=1, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(32, kernel_size=3, strides=1, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(x)
    x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)

    x = layers.Conv1D(111, kernel_size=39, strides=1, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(x)

    x = layers.Conv1D(128, kernel_size=39, strides=1, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(0.0001))(x)




    # Flatten the output
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128, activation='relu')(x)
   
    # Output layers for each direction, each predicting 12 values
    Sa = layers.Dense(111, activation='linear', name='Sa_x')(x)
    model = Model(inputs=input_layer, outputs=[Sa])
    

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss=rmse, metrics=['mae', rmse])
    
    if modelDisplay:
        model.summary()
    return model

#  data
X = input_blocks_array[:,:,0:2]
Y = average_channels_case2


# Normalize the data
def normalize(X):
    mean = np.mean(X, axis=(0, 2), keepdims=True)
    std = np.std(X, axis=(0, 2), keepdims=True)
    return (X - mean) / std



random.seed(42)


train_x, test_x, train_y, test_y, train_rsn, test_rsn = train_test_split(
    X, Y, rsn_numbers, test_size=0.2, random_state=42
)

train_x_scaled = normalize(train_x)
test_x_scaled = normalize(test_x)

train_x = train_x_scaled
test_x = test_x_scaled

print("train_x shape:", train_x.shape)
print("train_y shape:", train_y.shape)
print("test_x shape:", test_x.shape)
print("test_y shape:", test_y.shape)




model = CNN_model(train_x.shape[1], train_x.shape[2], loss=rmse, optimizer='adam', l2_reg=0.0005, modelDisplay=True)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001, verbose=1)

history = model.fit(train_x, train_y, 
                     epochs=500, batch_size=128, validation_split=0.1, verbose=1, callbacks=[early_stopping, reduce_lr])





# Plotting training history
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()


# Predict and calculate metrics
predicted_y = model.predict(test_x)

# Calculate and print metrics
mse = mean_squared_error(test_y.flatten(), predicted_y.flatten())
rmse_val = np.sqrt(mse)
mae_val = mean_absolute_error(test_y.flatten(), predicted_y.flatten())
r2_val = r2_score(test_y.flatten(), predicted_y.flatten())
pearson_coef = np.corrcoef(predicted_y.flatten(), test_y.flatten())[0,1]


print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"MAE: {mae_val:.4f}")
print(f" R² Score: {r2_val:.4f}")
print(f"Pearson coefficient: {pearson_coef:.4f}")

# Plotting observed vs predicted
plt.figure(figsize=(10, 10))
plt.scatter(test_y.flatten(), predicted_y.flatten(), alpha=0.5, label='Data Points', color='blue')

# Define the range for the plot
min_val, max_val = -2, 4

# Plot 1:1 line
plt.plot([min_val, max_val], [min_val, max_val], 'k--', color='black', label='1:1 Line')

# Plot standard deviations
y_std = np.std(predicted_y_combined.flatten() - test_y.flatten())
plt.plot([min_val, max_val], [min_val + y_std, max_val + y_std], 'r--', label='+1 Std Dev')
plt.plot([min_val, max_val], [min_val - y_std, max_val - y_std], 'r--', label='-1 Std Dev')

plt.xlim([min_val, max_val])
plt.ylim([min_val, max_val])
plt.title(f'Observed vs. Predicted\nAverage R² Score: {r2_val:.4f}\nAverage RMSE: {rmse_val:.4f}')
plt.xlabel('Observed Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.grid(True)          
plt.show()
