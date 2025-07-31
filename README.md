# MNIST Handwritten Digit Classification

## Overview
This project implements multiple machine learning algorithms to classify handwritten digits (0-9) using the famous MNIST dataset. The project demonstrates both binary and multiclass classification approaches with comprehensive model evaluation and comparison.

## Dataset
- **Source**: MNIST dataset from OpenML
- **Features**: 784 pixel values (28x28 grayscale images)
- **Classes**: 10 digits (0-9)
- **Training Samples**: 60,000
- **Test Samples**: 10,000

## Project Structure
```
├── mnist_classification.ipynb    # Main Jupyter notebook with complete analysis
├── README.md                    # Project documentation
```

## Key Features
- **Binary Classification**: Digit 5 vs Non-5 detection
- **Multiclass Classification**: Full 10-digit classification
- **Model Comparison**: SGD, Random Forest, K-Nearest Neighbors
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Visualization**: Confusion matrices, ROC curves, precision-recall curves
- **Hyperparameter Tuning**: GridSearchCV for optimal performance

## Technologies Used
- **Python Libraries**: pandas, numpy, matplotlib, scikit-learn
- **ML Algorithms**: 
  - Stochastic Gradient Descent (SGD) Classifier
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- **Evaluation Tools**: Cross-validation, confusion matrix, ROC curves
- **Optimization**: GridSearchCV for hyperparameter tuning

## Models Implemented

### 1. Binary Classification (Digit 5 Detection)
- **SGD Classifier**: Achieved high precision with threshold optimization
- **Random Forest**: Superior ROC-AUC performance
- **Evaluation**: Precision-recall curves, ROC analysis

### 2. Multiclass Classification (All Digits)
- **Random Forest**: Best overall performance
- **KNN with GridSearch**: Optimized with different k values and weights
- **Evaluation**: Macro F1-scores, confusion matrices

## Results
- **Random Forest**: ~97% F1-score (macro average)
- **Optimized KNN**: ~97% F1-score with best parameters
- **Binary Classification ROC-AUC**: >0.95 for digit 5 detection

## Key Insights
- Random Forest consistently outperformed other models
- Proper threshold selection significantly improved precision-recall trade-off
- Cross-validation prevented overfitting and provided robust estimates
- Confusion matrix revealed specific digit pairs that are commonly misclassified

## How to Run
1. Clone the repository
2. Install required dependencies: `pip install pandas numpy matplotlib scikit-learn`
3. Open `mnist_classification.ipynb` in Jupyter Notebook
4. Run all cells sequentially (note: training may take several minutes)

## Model Performance Summary
| Model | Approach | F1-Score (Macro) | Key Strengths |
|-------|----------|------------------|---------------|
| Random Forest | Multiclass | ~0.97 | Best overall performance, handles overfitting well |
| KNN (Optimized) | Multiclass | ~0.97 | Simple, interpretable, good with proper tuning |
| SGD | Binary | High Precision | Fast training, good for binary classification |

## Visualizations
- **Image Display**: Sample digit visualization
- **ROC Curves**: Model comparison for binary classification
- **Confusion Matrices**: Detailed error analysis
- **Precision-Recall Curves**: Threshold optimization insights

## Future Improvements
- Implement deep learning approaches (CNN)
- Add data augmentation techniques
- Experiment with ensemble methods
- Optimize for inference speed
- Add noise robustness testing
