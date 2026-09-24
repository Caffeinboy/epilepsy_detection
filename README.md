# Epilepsy Detection Using 1D CNN

A machine-learning project for **binary epilepsy seizure detection from EEG signal data** using a **1D Convolutional Neural Network (CNN)**.

The project uses the **Epileptic Seizure Recognition** dataset, trains a CNN model to distinguish between seizure and non-seizure samples, converts the trained TensorFlow/Keras model into a compressed **TensorFlow Lite (`.tflite`) model**, and finally converts the model into a **C header file (`.h`)** for potential deployment on embedded systems such as ESP32.

---

## Project Overview

The complete workflow is:

```text
Kaggle EEG Dataset
        ↓
Load CSV using KaggleHub
        ↓
Data preprocessing
        ↓
Binary classification
Seizure = 1
Normal   = 0
        ↓
Train/Test Split
        ↓
StandardScaler
        ↓
Reshape EEG samples
        ↓
1D CNN
        ↓
Model Evaluation
        ↓
TensorFlow Lite Conversion
        ↓
Quantized/Optimized .tflite Model
        ↓
C Header File (.h)
        ↓
Potential Embedded Deployment
```

---

## Features

* Downloads the EEG dataset directly from Kaggle using `kagglehub`
* Converts the original classification problem into binary classification
* Performs numerical feature selection
* Splits the dataset into training and testing sets
* Standardizes the EEG signal data
* Uses a **1D CNN** for EEG signal classification
* Uses dropout to reduce overfitting
* Evaluates the trained model on unseen test data
* Converts the Keras model to TensorFlow Lite
* Applies TensorFlow Lite default optimization
* Converts the `.tflite` model into a C header file
* Generates `epilepsy_model.h` for possible microcontroller deployment

---

## Dataset

This project uses the **Epileptic Seizure Recognition** dataset available through Kaggle.

Dataset:

```text
harunshimanto/epileptic-seizure-recognition
```

The dataset is loaded directly using KaggleHub:

```python
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "harunshimanto/epileptic-seizure-recognition",
    file_path
)
```

The input file used by the notebook is:

```text
Epileptic Seizure Recognition.csv
```

### Target Variable

The original dataset contains a target column called:

```text
y
```

The notebook converts this into a binary classification target:

| Original `y` | New `seizure` value | Meaning     |
| ------------ | ------------------: | ----------- |
| `1`          |                 `1` | Seizure     |
| `2–5`        |                 `0` | Non-seizure |

This is implemented using:

```python
df['seizure'] = df['y'].apply(lambda x: 1 if x == 1 else 0)
```

---

## Input Data

The EEG signal contains **178 numerical samples/features** after removing the target and non-numeric ID information.

The preprocessing pipeline extracts only numerical columns:

```python
X_df = df.drop(['y', 'seizure'], axis=1)

X = X_df.select_dtypes(include=[np.number]).values
y = df['seizure'].values
```

The resulting input is reshaped for the 1D CNN:

```text
(Samples, 178, 1)
```

The final CNN input shape is therefore:

```text
178 time steps × 1 feature
```

---

## Data Preprocessing

### 1. Train/Test Split

The dataset is divided into:

```text
80% → Training
20% → Testing
```

using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The fixed random state makes the split reproducible.

### 2. Standardization

The EEG features are standardized using `StandardScaler`:

```python
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

The scaler is fitted **only on the training data** and then applied to the test data.

### 3. CNN Reshaping

The data is reshaped from:

```text
(Samples, 178)
```

to:

```text
(Samples, 178, 1)
```

This format is required by the `Conv1D` layers.

---

# CNN Architecture

The model uses a compact 1D convolutional architecture:

```text
Input
  │
  ▼
Conv1D
32 filters
Kernel size = 3
ReLU
  │
  ▼
MaxPooling1D
Pool size = 2
  │
  ▼
Conv1D
16 filters
Kernel size = 3
ReLU
  │
  ▼
MaxPooling1D
Pool size = 2
  │
  ▼
Flatten
  │
  ▼
Dense
32 neurons
ReLU
  │
  ▼
Dropout
0.5
  │
  ▼
Dense
1 neuron
Sigmoid
  │
  ▼
Binary Classification
```

The model is implemented using:

```python
model = Sequential([
    Conv1D(
        filters=32,
        kernel_size=3,
        activation='relu',
        input_shape=(178, 1)
    ),

    MaxPooling1D(pool_size=2),

    Conv1D(
        filters=16,
        kernel_size=3,
        activation='relu'
    ),

    MaxPooling1D(pool_size=2),

    Flatten(),

    Dense(
        32,
        activation='relu'
    ),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )
])
```

---

## Model Configuration

The model uses:

| Parameter         | Value               |
| ----------------- | ------------------- |
| Architecture      | 1D CNN              |
| Optimizer         | Adam                |
| Loss              | Binary Crossentropy |
| Output activation | Sigmoid             |
| Training epochs   | 20                  |
| Batch size        | 32                  |
| Validation data   | 20% test set        |
| Dropout           | 0.5                 |
| Input shape       | `(178, 1)`          |

The model is compiled with:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

---

# Model Training

Training is performed using:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

After training, the model is evaluated using:

```python
loss, accuracy = model.evaluate(
    X_test,
    y_test
)
```

The resulting accuracy is printed as:

```text
Model Accuracy on unseen data: XX.XX%
```

The actual accuracy depends on the dataset version, preprocessing, TensorFlow/Keras version, and training run.

---

# TensorFlow Lite Conversion

After training, the Keras model is converted into TensorFlow Lite format.

```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

tflite_model = converter.convert()
```

The resulting model is saved as:

```text
epilepsy_model.tflite
```

This format is more suitable for deployment on resource-constrained devices.

---

# C Header File Generation

For embedded deployment, the TensorFlow Lite model is converted into a C header file.

First, `xxd` is installed:

```bash
!apt-get update && apt-get install -y xxd
```

Then:

```bash
!xxd -i epilepsy_model.tflite > epilepsy_model.h
```

This generates:

```text
epilepsy_model.h
```

The header contains the TensorFlow Lite model as a C byte array.

Conceptually:

```cpp
const unsigned char epilepsy_model_tflite[] = {
    0x1c, 0x00, 0x00, 0x00,
    ...
};
```

The generated header can potentially be included in an embedded C/C++ project.

---

# Generated Files

After running the complete notebook, the important output files are:

```text
epilepsy_model.tflite
epilepsy_model.h
```

### `epilepsy_model.tflite`

TensorFlow Lite representation of the trained CNN.

### `epilepsy_model.h`

C-compatible representation of the TensorFlow Lite model for use in embedded projects.

---

# Requirements

The notebook requires Python packages including:

```text
numpy
pandas
scikit-learn
tensorflow
kagglehub
```

The Kaggle dataset is installed through:

```python
!pip install kagglehub[pandas-datasets]
```

Google Colab already provides many of the required machine-learning libraries.

---

# Running the Project in Google Colab

The notebook can be opened directly in Google Colab using the repository link:

```text
https://colab.research.google.com/github/Caffeinboy/epilepsy_detection/blob/main/epilepsy_detection.ipynb
```

Alternatively, open the repository in GitHub and select the notebook.

Run the notebook cells sequentially:

```text
1. Install/import KaggleHub
2. Download and load the dataset
3. Preprocess the EEG data
4. Train the CNN
5. Evaluate the model
6. Convert the model to TensorFlow Lite
7. Generate the C header file
8. Download epilepsy_model.h
```

---

# Embedded Deployment

The generated `epilepsy_model.h` is intended for potential deployment on an embedded device.

A possible deployment architecture is:

```text
EEG Sensor
    │
    ▼
Signal Acquisition
    │
    ▼
Preprocessing
    │
    ▼
178-sample Input Window
    │
    ▼
Normalization
    │
    ▼
TensorFlow Lite Model
    │
    ▼
1D CNN Inference
    │
    ├───────────────┐
    ▼               ▼
Normal          Seizure
  0                1
```

For a microcontroller implementation, the same preprocessing used during training must be reproduced during inference.

In particular, the input should be standardized using the same scaling parameters obtained during training.

---

# Important Deployment Consideration

The notebook currently saves the TensorFlow Lite model and C header, but it **does not yet contain an embedded inference program**.

For deployment on an ESP32 or another microcontroller, the following components still need to be implemented:

1. EEG sensor interface
2. ADC/signal acquisition
3. Sampling-rate configuration
4. Signal buffering
5. 178-sample input window
6. Training-data normalization parameters
7. TensorFlow Lite Micro runtime
8. Model input tensor preparation
9. Inference
10. Seizure/normal decision logic
11. Output indication or communication

The embedded device must reproduce the preprocessing used during training. A mismatch between training preprocessing and deployment preprocessing can significantly affect inference results.

---

# Project Structure

A recommended GitHub repository structure is:

```text
epilepsy_detection/
│
├── epilepsy_detection.ipynb
├── epilepsy_model.tflite
├── epilepsy_model.h
├── README.md
└── .gitignore
```

If the `.tflite` file becomes large, Git LFS can be considered for model storage.

---

# Limitations

This project is a machine-learning demonstration and should not be interpreted as a clinically validated seizure-detection system.

Important limitations include:

* The model is trained on a specific public dataset.
* Dataset performance does not necessarily represent performance on real-world EEG recordings.
* The binary target simplifies the original multi-class dataset.
* The test split is used as validation data during training in the current notebook.
* No independent clinical validation dataset is used.
* EEG acquisition hardware and preprocessing can substantially affect model performance.
* The model's accuracy alone does not fully characterize seizure-detection performance.

For a more complete evaluation, additional metrics such as:

```text
Accuracy
Precision
Recall / Sensitivity
Specificity
F1-score
ROC-AUC
Confusion Matrix
```

should be calculated.

For seizure detection in particular, sensitivity, specificity, false-positive rate, and false-negative rate are important evaluation measures.

---

# Future Improvements

Possible extensions include:

* Add confusion matrix visualization
* Calculate precision, recall, F1-score, and specificity
* Add ROC-AUC analysis
* Use a separate validation dataset
* Implement early stopping
* Tune CNN architecture
* Compare CNN with LSTM/GRU architectures
* Implement data augmentation
* Save the trained scaler parameters
* Use representative datasets for TFLite quantization
* Reduce model size for ESP32 deployment
* Implement TensorFlow Lite Micro inference
* Connect an EEG acquisition circuit to the embedded system
* Add real-time seizure detection
* Add OLED/LCD indication
* Add wireless notification through Wi-Fi/Bluetooth

---

# Technologies Used

* **Python**
* **Google Colab**
* **Kaggle / KaggleHub**
* **NumPy**
* **Pandas**
* **Scikit-learn**
* **TensorFlow**
* **Keras**
* **TensorFlow Lite**
* **C/C++**
* **xxd**
* **GitHub**

---

# License

This repository contains code for educational and research purposes.

Check the licensing and usage terms of the underlying dataset before redistributing the dataset itself.

---

# Acknowledgment

This project uses the **Epileptic Seizure Recognition** dataset available through Kaggle and uses TensorFlow/Keras for model development and TensorFlow Lite for model conversion.

The dataset is downloaded at runtime rather than being directly stored in this repository.

---

## Author

**Caffeinboy**

GitHub:

```text
https://github.com/Caffeinboy
```
