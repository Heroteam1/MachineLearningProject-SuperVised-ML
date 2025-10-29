import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, mean_squared_error, r2_score,
    mean_absolute_error, roc_curve, precision_recall_curve, auc
)
from xgboost import XGBClassifier, XGBRegressor
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.multiclass import unique_labels
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CLASSIFICATION MODELS WITH HYPERPARAMETERS
# ============================================================================

def get_classification_models_with_params():
    """Return classification models with their hyperparameters"""
    return {
        'Logistic Regression': {
            'model': LogisticRegression,
            'params': {
                'C': [0.1, 1.0, 10.0],
                'max_iter': [100, 200, 500],
                'solver': ['liblinear', 'lbfgs']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier,
            'params': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'SVM': {
            'model': SVC,
            'params': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'probability': [True]
            }
        },
        'XGBoost': {
            'model': XGBClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier,
            'params': {
                'n_neighbors': [3, 5, 7, 10],
                'weights': ['uniform', 'distance']
            }
        },
        'Naive Bayes': {
            'model': GaussianNB,
            'params': {}
        }
    }

# ============================================================================
# COMPREHENSIVE CLASSIFICATION FUNCTION
# ============================================================================

def train_classification_model(df, target_column, model_name, test_size=0.2, random_state=42, **hyperparams):
    """
    Train a single classification model with comprehensive evaluation
    """
    try:
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state['label_encoder'] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Get model class
        models_dict = get_classification_models_with_params()
        model_class = models_dict[model_name]['model']
        
        # Create model with hyperparameters
        model = model_class(**hyperparams)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        # Calculate comprehensive metrics
        metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
        
        # Store results
        results = {
            'model': model,
            'model_name': model_name,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'metrics': metrics,
            'feature_names': X.columns.tolist()
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error training {model_name}: {str(e)}")
        return None

# ============================================================================
# COMPREHENSIVE METRICS CALCULATION
# ============================================================================

def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # ROC-AUC for binary classification
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
    
    # Cross-validation scores (approximate)
    # metrics['cv_accuracy'] = np.mean(cross_val_score(model, X, y, cv=5, scoring='accuracy'))
    
    # Additional metrics for binary classification
    if len(np.unique(y_true)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return metrics

# ============================================================================
# MODEL COMPARISON FUNCTION
# ============================================================================

def compare_classification_models(df, target_column, test_size=0.2, random_state=42):
    """
    Compare multiple classification models and return results
    """
    models_dict = get_classification_models_with_params()
    results = {}
    
    progress_bar = st.progress(0)
    total_models = len(models_dict)
    
    for i, (model_name, model_info) in enumerate(models_dict.items()):
        try:
            st.write(f"🔄 Training {model_name}...")
            
            # Use default parameters for quick comparison
            default_params = {key: values[0] for key, values in model_info['params'].items()}
            
            model_result = train_classification_model(
                df, target_column, model_name, test_size, random_state, **default_params
            )
            
            if model_result:
                results[model_name] = model_result
            
            progress_bar.progress((i + 1) / total_models)
            
        except Exception as e:
            st.warning(f"Could not train {model_name}: {str(e)}")
            continue
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot interactive confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    labels = unique_labels(y_true, y_pred)
    
    fig = px.imshow(
        cm, 
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=[f"Class {i}" for i in labels],
        y=[f"Class {i}" for i in labels],
        title=f"Confusion Matrix - {model_name}",
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        width=500,
        height=500
    )
    
    st.plotly_chart(fig)

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Plot ROC curve for binary classification"""
    if y_pred_proba is None or len(np.unique(y_true)) != 2:
        st.info("ROC curve requires probability predictions and binary classification")
        return
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC curve (AUC = {roc_auc:.3f})',
        line=dict(width=2)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(
        title=f'ROC Curve - {model_name}',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        width=600,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig)

def plot_precision_recall_curve(y_true, y_pred_proba, model_name):
    """Plot Precision-Recall curve"""
    if y_pred_proba is None or len(np.unique(y_true)) != 2:
        st.info("Precision-Recall curve requires probability predictions and binary classification")
        return
    
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
    pr_auc = auc(recall, precision)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision,
        mode='lines',
        name=f'PR curve (AUC = {pr_auc:.3f})',
        line=dict(width=2)
    ))
    
    fig.update_layout(
        title=f'Precision-Recall Curve - {model_name}',
        xaxis_title='Recall',
        yaxis_title='Precision',
        width=600,
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig)

def plot_feature_importance(model, feature_names, model_name, top_n=10):
    """Plot feature importance for tree-based models"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            fig = px.bar(
                x=importances[indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                title=f'Top {top_n} Feature Importances - {model_name}',
                labels={'x': 'Importance', 'y': 'Features'}
            )
            
            fig.update_layout(
                width=600,
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig)
        else:
            st.info(f"Feature importance not available for {model_name}")
    except Exception as e:
        st.info(f"Could not plot feature importance: {str(e)}")

# ============================================================================
# RESULTS DISPLAY FUNCTIONS
# ============================================================================

def display_single_model_results(results):
    """Display comprehensive results for a single model"""
    if not results:
        return
    
    st.header(f"📊 Results for {results['model_name']}")
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{results['metrics']['accuracy']:.3f}")
    with col2:
        st.metric("Precision", f"{results['metrics']['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{results['metrics']['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{results['metrics']['f1']:.3f}")
    
    # ROC-AUC if available
    if 'roc_auc' in results['metrics']:
        st.metric("ROC-AUC", f"{results['metrics']['roc_auc']:.3f}")
    
    # Visualizations
    st.subheader("📈 Model Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        plot_confusion_matrix(results['y_test'], results['y_pred'], results['model_name'])
    
    with col2:
        plot_roc_curve(results['y_test'], results['y_pred_proba'], results['model_name'])
    
    # Feature importance
    plot_feature_importance(results['model'], results['feature_names'], results['model_name'])
    
    # Detailed classification report
    st.subheader("📋 Detailed Classification Report")
    report_dict = classification_report(results['y_test'], results['y_pred'], output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.dataframe(report_df.style.format("{:.3f}").background_gradient(cmap='Blues'))

def display_model_comparison(results_dict):
    """Display comparison of multiple models"""
    if not results_dict:
        st.error("No models were successfully trained")
        return
    
    st.header("🏆 Model Comparison Leaderboard")
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results_dict.items():
        row = {
            'Model': model_name,
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1-Score': result['metrics']['f1']
        }
        if 'roc_auc' in result['metrics']:
            row['ROC-AUC'] = result['metrics']['roc_auc']
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy (descending)
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    
    # Display styled table
    st.dataframe(
        comparison_df.style.format("{:.3f}").background_gradient(
            subset=['Accuracy', 'Precision', 'Recall', 'F1-Score'], 
            cmap='RdYlGn'
        ).highlight_max(color='lightgreen').highlight_min(color='#ffcccc')
    )
    
    # Visual comparison
    st.subheader("📊 Performance Comparison")
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    if 'ROC-AUC' in comparison_df.columns:
        metrics_to_plot.append('ROC-AUC')
    
    fig = go.Figure()
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=comparison_df['Model'],
            y=comparison_df[metric],
            text=comparison_df[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Model Performance Comparison',
        xaxis_title='Models',
        yaxis_title='Score',
        barmode='group',
        width=800,
        height=500
    )
    
    st.plotly_chart(fig)

# ============================================================================
# ENHANCED NEURAL NETWORK CLASSIFIER
# ============================================================================

def enhanced_neural_network_classifier(df, target_column, n_hidden_layers=2, neurons_per_layer=64, 
                                     learning_rate=0.001, epochs=100, batch_size=32, test_size=0.2):
    """Enhanced neural network classifier with proper evaluation"""
    try:
        # Prepare data
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values
        
        # Encode labels if needed
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
        
        n_classes = len(np.unique(y))
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # Build model
        layers = []
        input_size = X_train.shape[1]
        
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(input_size, neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_size = neurons_per_layer
        
        layers.append(nn.Linear(input_size, n_classes))
        
        model = nn.Sequential(*layers)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        train_losses = []
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(dataloader))
            
            if (epoch + 1) % 20 == 0:
                st.write(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred_proba = model(X_test)
            y_pred = torch.argmax(y_pred_proba, dim=1)
            accuracy = (y_pred == y_test).float().mean().item()
        
        # Convert to numpy for metrics
        y_test_np = y_test.numpy()
        y_pred_np = y_pred.numpy()
        y_pred_proba_np = torch.softmax(y_pred_proba, dim=1).numpy()
        
        # Calculate metrics
        metrics = calculate_classification_metrics(y_test_np, y_pred_np, y_pred_proba_np)
        
        results = {
            'model': model,
            'model_name': 'Neural Network',
            'X_test': X_test,
            'y_test': y_test_np,
            'y_pred': y_pred_np,
            'y_pred_proba': y_pred_proba_np,
            'metrics': metrics,
            'feature_names': df.drop(columns=[target_column]).columns.tolist(),
            'train_losses': train_losses
        }
        
        return results
        
    except Exception as e:
        st.error(f"Error training neural network: {str(e)}")
        return None

# ============================================================================
# KEEP YOUR EXISTING REGRESSION FUNCTIONS (they're good!)
# ============================================================================

# Your existing regression functions remain the same...
# [Keep all your existing regression code here]

# Update your regression models dictionary to match the classification structure
regression_models = {
    'Linear Regression': LinearRegression,
    'Ridge Regression': Ridge,
    'Lasso Regression': Lasso,
    'Decision Tree': DecisionTreeRegressor,
    'Random Forest': RandomForestRegressor,
    'XGBoost': XGBRegressor
}

def regression_model(df, target_column, test_size=0.2, random_state=42):
    """Enhanced regression model function"""
    try:
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        model_results = {}
        for model_name, model_class in regression_models.items():
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            model_results[model_name] = {
                'model': model,
                'MSE': mse, 
                'R2': r2, 
                'MAE': mae,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
        return model_results
        
    except Exception as e:
        st.error(f"Regression error: {str(e)}")
        return None
