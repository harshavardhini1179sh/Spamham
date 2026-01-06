# Email Spam Detection

This project detects spam emails using machine learning.

## Prerequisites

- Python 3.7 or higher
- Required Python packages:
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - scikit-learn >= 1.2.0
  - matplotlib >= 3.6.0
  - seaborn >= 0.12.0

## Files Needed

Make sure the `Dataset/` folder contains files with these exact names:

**Training Data:**
- spam_train1.csv
- spam_train2.csv

**Test Data:**
- spam_test.csv

## How to Run the Code

### Step 1: Install Prerequisites

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Dataset Files

Make sure all required dataset files are placed in the `Dataset/` folder with the exact names listed above.

### Step 3: Execute the Script

Run the spam detection script:

```bash
python spam_detection.py
```

The script will automatically:
- Load and preprocess the training and test data
- Train 7 different machine learning models
- Evaluate each model using cross-validation
- Select the best model and make predictions on test data
- Generate visualizations comparing model performance
- Save all results to the Results and Visualizations folders

### Step 4: Check Output

After execution, check the `Results/` and `Visualizations/` folders for output files (see Output section below).

## Models Used

- Decision Tree - Simple tree-based classifier
- Random Forest - Ensemble of decision trees
- SVM - Support Vector Machine
- Naive Bayes - Probabilistic classifier
- Logistic Regression - Linear classifier
- Gradient Boosting - Sequential ensemble method
- Neural Network (MLP) - Multi-layer perceptron

## Accuracy Results

Best Model: Random Forest - 96.51%

## Output

All results are automatically saved in two folders. Both folders will be created automatically if they don't exist.

### Results Folder

The script generates output files in the `Results/` folder:
- `Results/PeriyasswamiSpam.txt` - Submission format file with predictions
- `Results/model_results.csv` - Performance metrics for all models

### Visualizations Folder

The script generates visualization files in the `Visualizations/` folder:
- `Visualizations/accuracy_comparison.png` - Bar chart comparing accuracy of all models
- `Visualizations/confusion_matrices.png` - Confusion matrices for all models
- `Visualizations/test_predictions_distribution_bar.png` - Bar chart of test prediction distribution
- `Visualizations/test_predictions_distribution_pie.png` - Pie chart of test prediction distribution
