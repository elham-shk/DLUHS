
# DLUHS  
**Explainable Deep Learning–based Uniform Hazard Spectral Acceleration Prediction**

## Overview  
This project introduces a deep learning–based framework (**DLUHS**) for on-site earthquake early warning (EEW).  
Instead of relying on manually defined parameters or only scalar metrics like PGA/PGV, the model directly uses the first **3–8 seconds of raw seismic waveforms** to predict **Uniform Hazard Spectral Acceleration (UHS)** across 111 periods (0.01–20s).


![Fig2](https://github.com/user-attachments/assets/03ea5d61-3121-4161-b611-4e13926c246c)


![Fig11](https://github.com/user-attachments/assets/5e668230-90f6-4274-8db6-910245518f97)




## Key Features  
- **DLUHS1 & DLUHS2 Architectures**: CNN-based models trained on ~17,500 NGA-West2 records.  
- **Fast & Automated**: Eliminates human bias by avoiding manually extracted features.  
- **Explainable AI**: Uses SHAP analysis to interpret how waveform features influence predictions.  
- **Performance**: Strong correlation (R² up to ~90%) and low RMSE in predicting Sa(T).  
- **Trigger Classification**: Site-specific hazard thresholds classify destructive vs. non-destructive events.  
- **Validated On**:  
  - U.S. Geological Survey Earthquake Hazard Toolbox (2023 NSHM)  
  - Japanese KiK-net & K-NET strong-motion databases  
- Provides **structurally relevant intensity measures (Sa(T))** for engineering design and seismic hazard assessment.  
- Improves **real-time EEW systems** by balancing speed and accuracy.  
- Supports decision-making for **engineers, monitoring networks, and public safety agencies**.  

## Reference

```text
Shokrgozar-Yatimdar, E., & Chen, P. (2025).
Explainable Deep Learning for Real-Time Prediction of Uniform Hazard Spectral Acceleration
for On-Site Earthquake Early Warning.Geophysical Journal International.


