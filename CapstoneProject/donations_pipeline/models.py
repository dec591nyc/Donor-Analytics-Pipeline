#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Models Module
Handles model training, evaluation, and persistence
"""

from pathlib import Path
from typing import Dict, Tuple, List, Any
import joblib
import json

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    r2_score
)

from .config import Config
from .logger import PipelineLogger


class ModelTrainer:
    """Base class for model training"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def build_preprocessor(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str]
    ) -> ColumnTransformer:
        """Build sklearn preprocessing pipeline"""
        
        transformers = []
        
        if numeric_cols:
            transformers.append((
                'num',
                SimpleImputer(strategy='median'),
                numeric_cols
            ))
        
        if categorical_cols:
            transformers.append((
                'cat',
                Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]),
                categorical_cols
            ))
        
        return ColumnTransformer(transformers)
    
    def save_model(
        self,
        model: Pipeline,
        model_name: str,
        output_dir: Path
    ) -> None:
        """Save trained model to disk"""
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f"{model_name}_model.pkl"
        
        try:
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved: {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
    
    def save_metrics(
        self,
        metrics: Dict[str, Any],
        model_name: str,
        output_dir: Path
    ) -> None:
        """Save model metrics to JSON"""
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / f"{model_name}_metrics.json"
        
        try:
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, default=str)
            self.logger.debug(f"Metrics saved: {metrics_path}")
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")


class PropensityModeler(ModelTrainer):
    """Trains models for re-donation propensity"""
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_cols: List[str],
        output_dir: Path
    ) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
        """
        Train multiple classification models
        
        Returns:
            (trained_models_dict, metrics_dataframe)
        """
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
        
        preprocessor = self.build_preprocessor(numeric_cols, categorical_cols)
        
        # Define models
        model_configs = {
            'LogisticRegression': LogisticRegression(
                max_iter=self.config.models.lr_max_iter,
                solver=self.config.models.lr_solver,
                class_weight='balanced',
                random_state=self.config.analysis.random_state
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=self.config.models.rf_n_estimators,
                max_depth=self.config.models.rf_max_depth,
                class_weight='balanced',
                random_state=self.config.analysis.random_state,
                n_jobs=self.config.models.rf_n_jobs
            )
        }
        
        # Try to add LightGBM
        try:
            from lightgbm import LGBMClassifier
            model_configs['LightGBM'] = LGBMClassifier(
                n_estimators=self.config.models.lgbm_n_estimators,
                learning_rate=self.config.models.lgbm_learning_rate,
                class_weight='balanced',
                random_state=self.config.analysis.random_state
            )
        except ImportError:
            self.logger.warning("LightGBM not available, skipping")
        
        trained_models = {}
        metrics_list = []
        
        # donations_pipeline_Claude/models.py

        # Train each model
        for model_name, base_model in model_configs.items():
            with self.logger.timed_operation(f"Training {model_name}"):
                try:
                    # Build pipeline
                    self.logger.info(f"[{model_name}] Building pipeline...")
                    pipeline = Pipeline([
                        ('preprocessor', preprocessor),
                        ('model', base_model)
                    ])
                    
                    # Train
                    self.logger.info(f"[{model_name}] Starting model fitting (this is the main training step)...")
                    pipeline.fit(X_train, y_train)
                    self.logger.info(f"[{model_name}] Model fitting complete.")
                    
                    # Evaluate
                    self.logger.info(f"[{model_name}] Starting model evaluation on test data...")
                    metrics = self._evaluate_classifier(
                        pipeline, X_test, y_test, model_name
                    )
                    self.logger.info(f"[{model_name}] Model evaluation complete.")
                    
                    # Store & Save
                    self.logger.info(f"[{model_name}] Storing results and saving model artifacts to disk...")
                    trained_models[model_name] = pipeline
                    metrics_list.append(metrics)
                    
                    self.save_model(pipeline, f"propensity_{model_name}", output_dir)
                    self.save_metrics(metrics, f"propensity_{model_name}", output_dir)
                    self.logger.info(f"[{model_name}] Artifacts saved successfully.")
                    
                except Exception as e:
                    self.logger.error(f"Failed to train {model_name}: {e}")
        
        print(f"metrics_list content: {metrics_list}")
        print(f"metrics_list length: {len(metrics_list)}")
        if metrics_list:
            print(f"First item keys: {metrics_list[0].keys()}")

        # Then modify line 185 to handle the error gracefully:
        if metrics_list and 'auc' in metrics_list[0]:
            metrics_df = pd.DataFrame(metrics_list).sort_values('auc', ascending=False)
        else:
            # Handle the case where AUC is missing
            metrics_df = pd.DataFrame(metrics_list) if metrics_list else pd.DataFrame()
            self.logger.warning(f"AUC column missing. Available columns: {metrics_df.columns.tolist()}")
        metrics_df = pd.DataFrame(metrics_list).sort_values('auc', ascending=False)
        
        return trained_models, metrics_df
    
    def _evaluate_classifier(
        self,
        model: Pipeline,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str
    ) -> Dict[str, Any]:
        """Evaluate classification model"""
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # AUC
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan
            self.logger.warning(f"{model_name}: Cannot compute AUC (single class)")
        
        metrics = {
            'model': model_name,
            'auc': float(auc) if not np.isnan(auc) else None,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'support': int(report['weighted avg']['support'])
        }
        
        self.logger.info(
            f"{model_name}: AUC={metrics['auc']:.3f}, F1={metrics['f1']:.3f}"
        )
        
        return metrics


class MajorGiftModeler(PropensityModeler):
    """Trains models for major gift propensity"""
    
    def train_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_cols: List[str],
        output_dir: Path
    ) -> Tuple[Dict[str, Pipeline], pd.DataFrame]:
        """Train major gift propensity models"""
        
        # Use parent class implementation but with different naming
        trained_models, metrics_df = super().train_models(
            X_train, X_test, y_train, y_test, categorical_cols, output_dir
        )
        
        # Re-save with major_gift prefix
        for model_name, model in trained_models.items():
            self.save_model(model, f"major_gift_{model_name}", output_dir)
        
        return trained_models, metrics_df


class AmountRegressor(ModelTrainer):
    """Trains regression model for donation amount prediction"""
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_cols: List[str],
        output_dir: Path
    ) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Train Random Forest regressor for amount prediction
        
        Returns:
            (trained_model, metrics_dict)
        """
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
        preprocessor = self.build_preprocessor(numeric_cols, categorical_cols)
        
        with self.logger.timed_operation("Training Amount Regressor"):
            # Build pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=self.config.models.amount_rf_n_estimators,
                    random_state=self.config.analysis.random_state,
                    n_jobs=self.config.models.rf_n_jobs
                ))
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'MAE': float(mae),
                'R2': float(r2),
                'mean_actual': float(y_test.mean()),
                'mean_predicted': float(y_pred.mean())
            }
            
            self.logger.info(f"Amount Model: MAE=${mae:.2f}, R²={r2:.3f}")
            
            # Save
            self.save_model(model, "amount_predictor", output_dir)
            self.save_metrics(metrics, "amount_predictor", output_dir)
        
        return model, metrics


class TimeToNextRegressor(ModelTrainer):
    """Trains regression model for time until next donation"""
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_cols: List[str],
        output_dir: Path
    ) -> Tuple[Pipeline, Dict[str, float], pd.Series]:
        """
        Train model to predict days until next donation
        
        Returns:
            (trained_model, metrics_dict, test_residuals)
        """
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
        preprocessor = self.build_preprocessor(numeric_cols, categorical_cols)
        
        with self.logger.timed_operation("Training Time-to-Next Regressor"):
            # Build pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', RandomForestRegressor(
                    n_estimators=self.config.models.time_rf_n_estimators,
                    random_state=self.config.analysis.random_state,
                    n_jobs=self.config.models.rf_n_jobs
                ))
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'MAE_days': float(mae),
                'R2': float(r2),
                'residual_median': float(np.median(residuals)),
                'residual_p20': float(np.quantile(residuals, 0.2)),
                'residual_p80': float(np.quantile(residuals, 0.8))
            }
            
            self.logger.info(
                f"Time Model: MAE={mae:.1f} days, R²={r2:.3f}"
            )
            
            # Save
            self.save_model(model, "time_to_next_predictor", output_dir)
            self.save_metrics(metrics, "time_to_next_predictor", output_dir)
        
        return model, metrics, residuals


class PreferenceClassifier(ModelTrainer):
    """Trains multiclass classifier for fund/appeal preference"""
    
    def train_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        categorical_cols: List[str],
        output_dir: Path,
        target_name: str = "preference"
    ) -> Tuple[Pipeline, Dict[str, float]]:
        """
        Train multiclass classifier for preference prediction
        
        Returns:
            (trained_model, metrics_dict)
        """
        numeric_cols = [c for c in X_train.columns if c not in categorical_cols]
        preprocessor = self.build_preprocessor(numeric_cols, categorical_cols)
        
        # Try LightGBM first, fallback to RandomForest
        try:
            from lightgbm import LGBMClassifier
            base_model = LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                random_state=self.config.analysis.random_state
            )
            model_name = "LightGBM"
        except ImportError:
            base_model = RandomForestClassifier(
                n_estimators=400,
                random_state=self.config.analysis.random_state,
                class_weight='balanced',
                n_jobs=self.config.models.rf_n_jobs
            )
            model_name = "RandomForest"
        
        with self.logger.timed_operation(f"Training Preference Classifier ({model_name})"):
            # Build pipeline
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('model', base_model)
            ])
            
            # Train
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            metrics = {
                'model': model_name,
                'target': target_name,
                'macro_f1': report.get('macro avg', {}).get('f1-score', np.nan),
                'weighted_f1': report.get('weighted avg', {}).get('f1-score', np.nan),
                'accuracy': report.get('accuracy', np.nan),
                'n_classes': len(np.unique(y_train))
            }
            
            self.logger.info(
                f"Preference Model: {target_name}, "
                f"Weighted F1={metrics['weighted_f1']:.3f}, "
                f"{metrics['n_classes']} classes"
            )
            
            # Save
            self.save_model(model, f"preference_{target_name}", output_dir)
            self.save_metrics(metrics, f"preference_{target_name}", output_dir)
        
        return model, metrics


class FeatureImportanceExtractor:
    """Extracts and exports feature importance from trained models"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def extract_importance(
        self,
        model: Pipeline,
        feature_names: List[str],
        categorical_cols: List[str]
    ) -> pd.DataFrame:
        """
        Extract feature importance from tree-based model
        
        Returns:
            DataFrame with columns: [feature, importance, field]
        """
        # Get the trained model
        base_model = model.named_steps['model']
        
        if not hasattr(base_model, 'feature_importances_'):
            raise ValueError("Model does not have feature_importances_ attribute")
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        
        # Numeric features
        numeric_cols = [c for c in feature_names if c not in categorical_cols]
        
        # Categorical features (after one-hot encoding)
        cat_features = []
        if categorical_cols:
            try:
                cat_transformer = preprocessor.named_transformers_['cat']
                ohe = cat_transformer.named_steps['onehot']
                cat_features = list(ohe.get_feature_names_out(categorical_cols))
            except Exception as e:
                self.logger.warning(f"Could not extract categorical features: {e}")
        
        all_features = numeric_cols + cat_features
        importances = base_model.feature_importances_
        
        # Create DataFrame
        fi_df = pd.DataFrame({
            'feature': all_features,
            'importance': importances
        })
        
        # Map to original field names
        fi_df['field'] = fi_df['feature'].apply(
            lambda x: self._map_to_field(x, categorical_cols)
        )
        
        return fi_df
    
    @staticmethod
    def _map_to_field(feature_name: str, categorical_cols: List[str]) -> str:
        """Map one-hot encoded feature back to original field"""
        # Check if it's a one-hot encoded feature
        for cat_col in categorical_cols:
            if feature_name.startswith(f"{cat_col}_"):
                return cat_col
        
        # Otherwise, it's a numeric feature
        return feature_name
    
    def export_field_importance(
        self,
        fi_df: pd.DataFrame,
        model_name: str,
        output_dir: Path
    ) -> None:
        """Export field-level aggregated importance"""
        
        field_importance = (
            fi_df.groupby('field', as_index=False)['importance']
            .sum()
            .sort_values('importance', ascending=False)
        )
        
        output_path = output_dir / f"field_importance_{model_name}.csv"
        field_importance.to_csv(output_path, index=False)
        
        self.logger.info(f"Feature importance exported: {output_path}")