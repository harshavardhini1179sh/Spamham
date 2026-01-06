import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import warnings
import os

warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

class SpamDetector:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.vectorizer = None
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.original_train_labels = None
        self.test_predictions = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        
    def load_and_preprocess_data(self):
        """Load and preprocess training and test datasets"""
        print("\n" + "-" * 50)
        print("Loading and preprocessing data...")
        print("-" * 50)
        
        train1_path = os.path.join(self.script_dir, 'Dataset', 'spam_train1.csv')
        train2_path = os.path.join(self.script_dir, 'Dataset', 'spam_train2.csv')
        test_data_path = os.path.join(self.script_dir, 'Dataset', 'spam_test.csv')
        
        train1 = pd.read_csv(train1_path)
        train2 = pd.read_csv(train2_path)
        test_data = pd.read_csv(test_data_path)
        
        print("\nPreprocessing Train1...")
        train1_clean = train1[['v1', 'v2']].copy()
        train1_clean.columns = ['label', 'text']
        train1_clean = train1_clean.dropna()
        train1_clean['label'] = train1_clean['label'].map({'ham': 0, 'spam': 1})
        
        print("Preprocessing Train2...")
        train2_clean = train2[['label', 'text']].copy()
        train2_clean = train2_clean.dropna()
        train2_clean['label'] = train2_clean['label'].map({'ham': 0, 'spam': 1})
        
        print("Combining training datasets...")
        combined_train = pd.concat([train1_clean, train2_clean], ignore_index=True)
        
        print("Preprocessing test data...")
        test_clean = test_data.copy()
        if 'message' in test_clean.columns:
            test_clean.columns = ['text']
        test_clean = test_clean.dropna()
        
        print("\nCleaning text data...")
        combined_train['text'] = combined_train['text'].astype(str).str.lower()
        test_clean['text'] = test_clean['text'].astype(str).str.lower()
        combined_train['text'] = combined_train['text'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
        test_clean['text'] = test_clean['text'].str.replace(r'[^a-zA-Z0-9\s]', ' ', regex=True)
        combined_train['text'] = combined_train['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        test_clean['text'] = test_clean['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
        combined_train = combined_train[combined_train['text'].str.len() > 0]
        test_clean = test_clean[test_clean['text'].str.len() > 0]
        
        print(f"\nCombined Training Data Shape: {combined_train.shape}")
        print(f"Label Distribution:\n{combined_train['label'].value_counts()}")
        print(f"\nTest Data Shape: {test_clean.shape}")
        
        X = combined_train['text']
        y = combined_train['label']
        X_test = test_clean['text']
        
        self.original_train_labels = y.copy()
        
        print("\nVectorizing text data using TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        X_vectorized = self.vectorizer.fit_transform(X)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.X_test_final = X_test_vectorized
        
        print(f"\nTraining set shape: {self.X_train.shape}")
        print(f"Validation set shape: {self.X_test.shape}")
        print(f"Final test set shape: {self.X_test_final.shape}")
        
        return self
    
    def initialize_models(self):
        """Initialize multiple ML models"""
        print("\n" + "-" * 50)
        print("Initializing models...")
        print("-" * 50)
        
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'SVM': SVC(probability=True, random_state=42),
            'Naive Bayes': MultinomialNB(),
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        print(f"\nInitialized {len(self.models)} models:")
        for name in self.models.keys():
            print(f"  - {name}")
        
        return self
    
    def train_and_evaluate_models(self):
        """Train all models and evaluate their performance"""
        print("\n" + "-" * 50)
        print("Training and evaluating models...")
        print("-" * 50)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            print("-" * 40)
            
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV Mean': cv_mean,
                'CV Std': cv_std
            })
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
            
            self.models[name] = model
            self.results[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                }
            }
        
        # Create results dataframe
        self.results_df = pd.DataFrame(results)
        self.results_df = self.results_df.sort_values('Accuracy', ascending=False)
        
        print("\n" + "-" * 50)
        print("Model Comparison Summary:")
        print("-" * 50)
        print(self.results_df.to_string(index=False))
        
        self.best_model_name = self.results_df.iloc[0]['Model']
        self.best_model = self.models[self.best_model_name]
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Best Accuracy: {self.results_df.iloc[0]['Accuracy']:.4f}")
        
        return self
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for top models"""
        print("\n" + "-" * 50)
        print("Hyperparameter tuning...")
        print("-" * 50)
        
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Neural Network': {
                'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        tuned_models = {}
        
        for model_name in ['Random Forest', 'SVM', 'Gradient Boosting', 'Neural Network']:
            if model_name not in self.models:
                continue
                
            print(f"\nTuning {model_name}...")
            print("-" * 40)
            
            base_model = self.models[model_name]
            param_grid = param_grids[model_name]
            
            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            tuned_model = grid_search.best_estimator_
            y_pred = tuned_model.predict(self.X_test)
            y_pred_proba = tuned_model.predict_proba(self.X_test)[:, 1] if hasattr(tuned_model, 'predict_proba') else None
            
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, zero_division=0)
            recall = recall_score(self.y_test, y_pred, zero_division=0)
            f1 = f1_score(self.y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            print(f"\nBest Parameters: {grid_search.best_params_}")
            print(f"Best CV Score: {grid_search.best_score_:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")
            print(f"Test F1-Score: {f1:.4f}")
            
            tuned_models[model_name] = {
                'model': tuned_model,
                'best_params': grid_search.best_params_,
                'metrics': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'roc_auc': roc_auc
                }
            }
            
            original_accuracy = self.results[model_name]['metrics']['accuracy']
            if accuracy > original_accuracy:
                self.models[model_name] = tuned_model
                self.results[model_name]['model'] = tuned_model
                self.results[model_name]['metrics']['accuracy'] = accuracy
                self.results[model_name]['metrics']['precision'] = precision
                self.results[model_name]['metrics']['recall'] = recall
                self.results[model_name]['metrics']['f1'] = f1
                self.results[model_name]['metrics']['roc_auc'] = roc_auc
                print(f"Improved from {original_accuracy:.4f} to {accuracy:.4f}")
        
        best_accuracy = max([self.results[name]['metrics']['accuracy'] for name in self.models.keys()])
        for name, result in self.results.items():
            if result['metrics']['accuracy'] == best_accuracy:
                self.best_model_name = name
                self.best_model = result['model']
                break
        
        print(f"\nBest model after tuning: {self.best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return self
    
    def create_visualizations(self):
        """Create visualizations"""
        print("\n" + "-" * 50)
        print("Creating visualizations...")
        print("-" * 50)
        
        results_dir = os.path.join(self.script_dir, 'Results')
        viz_dir = os.path.join(self.script_dir, 'Visualizations')
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(viz_dir, exist_ok=True)
        
        print("\n1. Creating accuracy comparison chart...")
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.barh(self.results_df['Model'], self.results_df['Accuracy'], color='steelblue', alpha=0.8)
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        for i, (bar, acc) in enumerate(zip(bars, self.results_df['Accuracy'])):
            ax.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("2. Creating confusion matrices...")
        top_models = self.results_df.head(3)['Model'].tolist()
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, model_name in enumerate(top_models):
            y_pred = self.results[model_name]['y_pred']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
            axes[idx].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["metrics"]["accuracy"]:.4f}',
                               fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('True Label', fontsize=10)
            axes[idx].set_xlabel('Predicted Label', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nVisualizations saved to 'Visualizations' directory")
        
        return self
    
    def create_ham_spam_visualizations(self):
        """Create ham/spam distribution visualizations"""
        print("\n" + "-" * 50)
        print("Creating prediction distribution visualizations...")
        print("-" * 50)
        
        viz_dir = os.path.join(self.script_dir, 'Visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        if self.test_predictions is not None:
            test_counts = pd.Series(self.test_predictions).value_counts()
            test_labels = ['Ham', 'Spam']
            test_values = [test_counts.get(0, 0), test_counts.get(1, 0)]
            colors = ['#4CAF50', '#F44336']
            
            print("\n1. Creating test predictions distribution (pie chart)...")
            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(test_values, labels=test_labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90,
                                              textprops={'fontsize': 12, 'fontweight': 'bold'})
            ax.set_title('Test Data Predictions Distribution\n(Ham vs Spam)', 
                        fontsize=16, fontweight='bold', pad=20)
            
            for i, (wedge, autotext) in enumerate(zip(wedges, autotexts)):
                autotext.set_text(f'{test_values[i]}\n({autotext.get_text()})')
                autotext.set_fontsize(11)
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'test_predictions_distribution_pie.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print("2. Creating test predictions distribution (bar chart)...")
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(test_labels, test_values, color=colors, alpha=0.8, 
                         edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Number of Emails', fontsize=12, fontweight='bold')
            ax.set_title('Test Data Predictions Distribution - Ham vs Spam', 
                        fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            for bar, value in zip(bars, test_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value}\n({value/sum(test_values)*100:.1f}%)',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'test_predictions_distribution_bar.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        print("\nPrediction distribution visualizations saved to 'Visualizations' directory")
        
        return self
    
    def predict_test_data(self):
        """Make predictions on test data using best model"""
        print("\n" + "-" * 50)
        print("Predicting test data...")
        print("-" * 50)
        
        print(f"\nUsing best model: {self.best_model_name}")
        print(f"Best model accuracy: {self.results[self.best_model_name]['metrics']['accuracy']:.4f}")
        
        test_predictions = self.best_model.predict(self.X_test_final)
        
        # Save with submission format name
        submission_file = os.path.join(self.script_dir, 'Results', 'PeriyasswamiSpam.txt')
        with open(submission_file, 'w') as f:
            for pred in test_predictions:
                f.write(f"{pred}\n")
        
        self.test_predictions = test_predictions
        
        print(f"\nTest Predictions Summary:")
        print(f"Total predictions: {len(test_predictions)}")
        print(f"Spam predictions: {sum(test_predictions == 1)}")
        print(f"Ham predictions: {sum(test_predictions == 0)}")
    
    def save_results(self):
        """Save all results and model"""
        print("\n" + "-" * 50)
        print("Saving results...")
        print("-" * 50)
        
        results_dir = os.path.join(self.script_dir, 'Results')
        os.makedirs(results_dir, exist_ok=True)
        
        results_csv_path = os.path.join(results_dir, 'model_results.csv')
        self.results_df.to_csv(results_csv_path, index=False)
        print("Model results saved to 'Results/model_results.csv'")
        
        return self


def main():
    """Main execution function"""
    print("\n" + "-"*50)
    print("Email Spam Detection Classifier")
    print("-"*50)
    
    detector = SpamDetector()
    
    detector.load_and_preprocess_data()
    detector.initialize_models()
    detector.train_and_evaluate_models()
    detector.hyperparameter_tuning()
    detector.create_visualizations()
    detector.predict_test_data()
    detector.create_ham_spam_visualizations()
    detector.save_results()
    
    print("\n" + "-"*50)
    print("Project completed successfully!")
    print("-"*50)
    print(f"Best Model: {detector.best_model_name}")
    print(f"Best Accuracy: {detector.results[detector.best_model_name]['metrics']['accuracy']:.4f}")


if __name__ == "__main__":
    main()

