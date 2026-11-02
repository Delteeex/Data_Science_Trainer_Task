# Fashion-MNIST Image Classification using Deep Learning

## 1. Project Overview

This project implements a deep learning-based image classification system using the **Fashion-MNIST dataset**.

The objective is to classify grayscale clothing images into 10 different fashion categories using three neural network architectures:

1. Multi-Layer Perceptron (MLP)
2. Convolutional Neural Network (CNN)
3. Improved CNN with Data Augmentation

The project compares model performance and evaluates predictions using accuracy, precision, recall, F1-score, and a confusion matrix.

## 2. Dataset Information

The Fashion-MNIST dataset contains 70,000 grayscale images of clothing items.

| Dataset | Number of Images |
|---|---:|
| Training dataset | 60,000 |
| Testing dataset | 10,000 |
| Image dimensions | 28 × 28 |
| Number of classes | 10 |
| Image format | Grayscale |

### Classification Categories

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

The dataset is loaded directly using TensorFlow/Keras.

## 3. Project Structure

```text
task3_deep_learning/
│
├── notebooks/
│   └── fashion_mnist_experiments.ipynb
│
├── models/
│   ├── mlp_model.keras
│   ├── cnn_model.keras
│   ├── improved_cnn_model.keras
│   └── improved_cnn_best.keras
│
├── outputs/
│   ├── eda/
│   ├── training_curves/
│   ├── confusion_matrices/
│   ├── predictions/
│   └── model_comparison.csv
│
├── src/
│   └── predict_image.py
│
└── README.md
```

## 4. Implementation Overview

### Step 1: Exploratory Data Analysis (EDA)

- Loaded the Fashion-MNIST dataset.
- Checked dataset shapes and data types.
- Analysed class distribution.
- Displayed sample images from all 10 classes.
- Calculated pixel intensity statistics.

### Step 2: Data Preprocessing

- Converted pixel values from 0–255 to 0–1.
- Created a stratified training and validation split.
- Used 50,000 images for training and 10,000 for validation.
- Kept the 10,000 test images separate for final evaluation.
- Reshaped images for CNN input.

### Step 3: Model 1 — MLP

Developed a baseline neural network with:
- Flatten layer
- Two hidden Dense layers
- ReLU activation
- Dropout regularization
- Softmax output layer

### Step 4: Model 2 — CNN

Developed a convolutional neural network with:
- Two convolutional blocks
- Batch normalization
- ReLU activation
- Max pooling
- Dropout
- Dense classification layer

### Step 5: Model 3 — Improved CNN

Improved the CNN architecture using:
- Additional convolutional layers
- Data augmentation
- Random translation
- Random zoom
- Horizontal flipping
- Batch normalization
- Dropout
- Learning-rate scheduling

### Step 6: Model Evaluation

Evaluated the models using:
- Training and validation accuracy
- Test accuracy
- Training and validation loss
- Precision, recall, and F1-score
- Confusion matrix
- Prediction confidence

### Step 7: Image Prediction

Created `src/predict_image.py` to support predictions from external JPG and PNG images.

The script:
- Loads a saved trained model.
- Converts an input image to grayscale.
- Resizes and preprocesses the image to 28 × 28.
- Normalizes pixel values.
- Predicts the clothing category.
- Displays the top 3 predicted classes with probabilities.

## 5. Model Performance

The following results were obtained during the final test evaluation:

| Model | Test Accuracy |
|---|---:|
| MLP | 88.38% |
| CNN | 92.04% |
| Improved CNN | 92.44% |

**Best-performing model:** Improved CNN

The Improved CNN achieved the highest test accuracy of 92.44%.

Note: The target for the improved model was 92.5%, so the achieved result is 0.06 percentage points below that target.

## 6. Requirements

Install Python and the required libraries.

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn pillow jupyter
```

## 7. How to Run the Project

### Step 1: Open the project folder

Open the `task3_deep_learning` folder in VS Code.

### Step 2: Start Jupyter Notebook

```bash
jupyter notebook
```

### Step 3: Open the notebook

Navigate to:

```text
notebooks/fashion_mnist_experiments.ipynb
```

Run the notebook cells sequentially to:
- Load the dataset.
- Perform EDA.
- Preprocess the images.
- Train all three models.
- Evaluate model performance.
- Generate visualizations.

**Note:** Training all models again may take considerable time. If the saved models are available, use them for prediction without retraining.

## 8. How to Use the Image Prediction Script

Place a JPG or PNG clothing image in a folder, for example:

```text
sample_images/shirt.jpg
```

Open the terminal from the project root and execute:

```bash
python src/predict_image.py "sample_images/shirt.jpg"
```

The script uses the default model:

```text
models/improved_cnn_model.keras
```

To use a different saved model:

```bash
python src/predict_image.py "sample_images/shirt.jpg" --model models/cnn_model.keras
```

The output displays the top 3 predicted clothing categories and their confidence probabilities.

### Important Prediction Note

Fashion-MNIST contains small, centered grayscale clothing images. Real-world photographs may have different backgrounds, lighting, angles, and object sizes. Therefore, predictions on real photographs may be less accurate than predictions on Fashion-MNIST test images.

## 9. Output Files

The project can store the following results:

- EDA visualizations
- Training and validation accuracy/loss graphs
- Confusion matrix
- Classification report
- 25-image prediction grid
- Top 10 confident incorrect predictions
- Model comparison table
- Saved trained models

Check the `outputs/` folder for generated reports and visualizations.

## 10. Key Findings

- CNN models performed better than the baseline MLP.
- The Improved CNN achieved the highest test accuracy.
- Data augmentation and a deeper architecture improved validation performance.
- Shirt and T-shirt/top were among the most confused classes.
- Coat and Shirt were also frequently confused due to similar visual characteristics.

## 11. Conclusion

This project demonstrates how deep learning can be used to classify fashion images.

Three different architectures were implemented and evaluated. The Improved CNN achieved the best test accuracy of **92.44%**.

The project also includes prediction functionality for external images, model evaluation, and visual analysis of classification results.

## 12. Future Improvements

Possible future improvements include:

- Experimenting with different CNN architectures.
- Fine-tuning augmentation parameters.
- Testing additional learning-rate schedules.
- Improving preprocessing for real-world photographs.
- Applying Grad-CAM or saliency maps to explain model predictions.
- Evaluating the model on additional clothing image datasets.

---

**Project:** Fashion-MNIST Image Classification  
**Framework:** TensorFlow / Keras  
**Language:** Python  
**Best Model:** Improved CNN  
**Final Test Accuracy:** 92.44%