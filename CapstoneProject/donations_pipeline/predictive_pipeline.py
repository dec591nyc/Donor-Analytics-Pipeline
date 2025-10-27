#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predictive Analytics Pipeline - FIXED VERSION

KEY FIXES:
1. Removed all age-related features (Age, AgeGroup, AgeAtDonation, AgeGroupAtDonation)
2. Fixed "columns are missing" error by ensuring consistent features in training and prediction
3. Removed date columns from feature sets to prevent sklearn errors
"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from .config import Config
from .logger import PipelineLogger
from .preprocessor import FeatureEngineer, DataSplitter
from .models import (
    PropensityModeler,
    MajorGiftModeler,
    AmountRegressor,
    TimeToNextRegressor,
    PreferenceClassifier,
    FeatureImportanceExtractor
)


class PredictivePipeline:
    """Orchestrates all predictive modeling"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
        self.feature_engineer = FeatureEngineer(config, logger)
        self.data_splitter = DataSplitter(config, logger)
    
    def run(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Run all predictive models"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Re-donation propensity
        self._train_propensity_models(df, output_dir)
        
        # 2. Major gift propensity
        self._train_major_gift_models(df, output_dir)
        
        # 3. Donation amount prediction
        amount_model = self._train_amount_model(df, output_dir)
        
        # 4. Time to next donation
        time_model, residuals = self._train_time_model(df, output_dir)
        
        # 5. Fund/Appeal preference
        pref_model, target_name = self._train_preference_model(df, output_dir)
        
        # 6. Generate combined playbook
        self._generate_playbook(
            df, output_dir,
            amount_model, time_model, residuals, pref_model, target_name
        )
    
    def _train_propensity_models(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Train re-donation propensity models"""
        try:
            with self.logger.timed_operation("Training Re-donation Propensity Models"):
                X_with_id, y, cat_cols = self._build_redonate_panel(df)
                
                if len(X_with_id) < self.config.analysis.min_samples_for_training:
                    self.logger.warning(f"Insufficient samples for propensity model: {len(X_with_id)}")
                    return
                
                if y.sum() == 0:
                    self.logger.warning("No positive labels for propensity model")
                    return
                
                X = X_with_id.drop(columns=[self.config.columns.donor_id], errors='ignore')
                
                X_train, X_test, y_train, y_test = self.data_splitter.time_based_split(X, y)
                modeler = PropensityModeler(self.config, self.logger)
                models, metrics_df = modeler.train_models(
                    X_train, X_test, y_train, y_test, cat_cols, output_dir
                )
                
                metrics_df.to_csv(output_dir / "propensity_model_comparison.csv", index=False)
                self._export_feature_importance(models, X_train.columns.tolist(), cat_cols, output_dir, "propensity")
                
        except Exception as e:
            self.logger.error(f"Propensity modeling failed: {e}", exc_info=True)
    
    def _train_major_gift_models(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Train major gift propensity models"""
        try:
            with self.logger.timed_operation("Training Major Gift Propensity Models"):
                X_with_id, y, cat_cols = self._build_major_gift_panel(df)
                
                if len(X_with_id) < self.config.analysis.min_samples_for_training:
                    self.logger.warning(f"Insufficient samples for major gift model: {len(X_with_id)}")
                    return
                
                if y.sum() == 0:
                    self.logger.warning("No major gifts in training data")
                    return
                
                X = X_with_id.drop(columns=[self.config.columns.donor_id], errors='ignore')
                
                X_train, X_test, y_train, y_test = self.data_splitter.time_based_split(X, y)
                
                modeler = MajorGiftModeler(self.config, self.logger)
                models, metrics_df = modeler.train_models(
                    X_train, X_test, y_train, y_test, cat_cols, output_dir
                )
                
                metrics_df.to_csv(output_dir / "major_gift_model_comparison.csv", index=False)
                self._export_feature_importance(models, X_train.columns.tolist(), cat_cols, output_dir, "major_gift")
                
        except Exception as e:
            self.logger.error(f"Major gift modeling failed: {e}", exc_info=True)
    
    def _train_amount_model(self, df: pd.DataFrame, output_dir: Path) -> Optional[object]:
        """Train donation amount prediction model"""
        try:
            with self.logger.timed_operation("Training Amount Prediction Model"):
                X, y, cat_cols = self._build_amount_dataset(df)
                
                if len(X) < 50:
                    self.logger.warning("Insufficient samples for amount model")
                    return None
                
                X_train, X_test, y_train, y_test = self.data_splitter.random_split(X, y, stratify=False)
                
                regressor = AmountRegressor(self.config, self.logger)
                model, metrics = regressor.train_model(
                    X_train, X_test, y_train, y_test, cat_cols, output_dir
                )
                
                return model
                
        except Exception as e:
            self.logger.error(f"Amount modeling failed: {e}", exc_info=True)
            return None
    
    def _train_time_model(self, df: pd.DataFrame, output_dir: Path) -> tuple[Optional[object], Optional[pd.Series]]:
        """Train time-to-next-donation model"""
        try:
            with self.logger.timed_operation("Training Time-to-Next Model"):
                X_with_id, y, cat_cols = self._build_time_dataset(df)
                
                if len(X_with_id) < 50:
                    self.logger.warning("Insufficient samples for time model")
                    return None, None
                
                X = X_with_id.drop(columns=[self.config.columns.donor_id], errors='ignore')
                
                X_train, X_test, y_train, y_test = self.data_splitter.time_based_split(X, y)
                
                regressor = TimeToNextRegressor(self.config, self.logger)
                model, metrics, residuals = regressor.train_model(
                    X_train, X_test, y_train, y_test, cat_cols, output_dir
                )
                
                return model, residuals
                
        except Exception as e:
            self.logger.error(f"Time modeling failed: {e}", exc_info=True)
            return None, None
    
    def _train_preference_model(self, df: pd.DataFrame, output_dir: Path) -> tuple[Optional[object], Optional[str]]:
        """Train fund/appeal preference model"""
        try:
            with self.logger.timed_operation("Training Preference Model"):
                X_with_id, y, cat_cols, target_name = self._build_preference_dataset(df)
                
                if len(X_with_id) < 100:
                    self.logger.warning("Insufficient samples for preference model")
                    return None, None
                
                X = X_with_id.drop(columns=[self.config.columns.donor_id], errors='ignore')
                
                X_train, X_test, y_train, y_test = self.data_splitter.time_based_split(X, y)
                if y_train.nunique() < 2:
                    self.logger.warning(
                        f"Preference model: Training data (y_train) has only {y_train.nunique()} unique class. "
                        "Model cannot be trained. Skipping."
                    )
                    return None, None
                classifier = PreferenceClassifier(self.config, self.logger)
                model, metrics = classifier.train_model(
                    X_train, X_test, y_train, y_test, cat_cols, output_dir, target_name
                )
                
                return model, target_name
                
        except Exception as e:
            self.logger.error(f"Preference modeling failed: {e}", exc_info=True)
            return None, None
    

    def run_training_only(self, df: pd.DataFrame, output_dir: Path) -> dict:
        """Run all predictive models WITHOUT generating playbook"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trained_models = {}
        
        # 1. Re-donation propensity
        self._train_propensity_models(df, output_dir)
        
        # 2. Major gift propensity
        self._train_major_gift_models(df, output_dir)
        
        # 3. Donation amount prediction
        trained_models['amount_model'] = self._train_amount_model(df, output_dir)
        
        # 4. Time to next donation
        time_model, residuals = self._train_time_model(df, output_dir)
        trained_models['time_model'] = time_model
        trained_models['residuals'] = residuals
        
        # 5. Fund/Appeal preference
        pref_model, target_name = self._train_preference_model(df, output_dir)
        trained_models['pref_model'] = pref_model
        trained_models['target_name'] = target_name
        
        return trained_models


    def generate_playbook_from_models(
        self, 
        df: pd.DataFrame, 
        output_dir: Path, 
        trained_models: dict
    ) -> None:
        """Generate prospecting playbook from trained models"""
        self._generate_playbook(
            df, output_dir,
            trained_models.get('amount_model'),
            trained_models.get('time_model'),
            trained_models.get('residuals'),
            trained_models.get('pref_model'),
            trained_models.get('target_name')
        )

    def _generate_playbook(
        self, df: pd.DataFrame, output_dir: Path,
        amount_model, time_model, residuals, pref_model, target_name
    ) -> None:
        """Generate combined prospecting playbook"""
        try:
            with self.logger.timed_operation("Generating Prospecting Playbook"):
                cols = self.config.columns
                playbook = pd.DataFrame({cols.donor_id: df[cols.donor_id].unique()})
                
                if amount_model:
                    amount_preds = self._predict_amounts(df, amount_model)
                    playbook = playbook.merge(amount_preds, on=cols.donor_id, how='left')
                
                if time_model and residuals is not None:
                    time_preds = self._predict_next_date(df, time_model, residuals)
                    playbook = playbook.merge(time_preds, on=cols.donor_id, how='left')
                
                if pref_model and target_name:
                    pref_preds = self._predict_preference(df, pref_model, target_name)
                    playbook = playbook.merge(pref_preds, on=cols.donor_id, how='left')
                
                playbook = self._calculate_priority_scores(playbook)
                playbook = playbook.sort_values('priority_score', ascending=False)
                playbook.to_csv(output_dir / "prospecting_playbook.csv", index=False)
                
                self.logger.info(f"Playbook generated: {len(playbook):,} donors")
                
        except Exception as e:
            self.logger.error(f"Playbook generation failed: {e}", exc_info=True)
    
    # ===== DATASET BUILDING METHODS =====
    def _get_redonate_labels(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Label function for create_panel_dataset: Calculates 'ReDonate'"""
        cols = self.config.columns
        analysis = self.config.analysis
        
        window_end = cutoff + pd.Timedelta(days=analysis.redonate_window_days)
        future = df[(df[cols.date] > cutoff) & (df[cols.date] <= window_end)]
        
        labels = future.groupby(cols.donor_id).size().rename('ReDonate')
        labels = (labels > 0).astype(int)
        
        return labels.reset_index()

    def _get_major_gift_labels(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Generate major gift labels for panel creation."""
        cols = self.config.columns
        analysis = self.config.analysis
        threshold = analysis.major_gift_threshold

        window_end = cutoff + pd.Timedelta(days=analysis.redonate_window_days)
        future = df[
            (df[cols.date] > cutoff)
            & (df[cols.date] <= window_end)
            & (df[cols.amount] >= threshold)
        ]

        labels = future.groupby(cols.donor_id).size().rename('MajorGift')
        labels = (labels > 0).astype(int)
        return labels.reset_index()

    def _get_time_labels(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        """Generate time-to-next donation labels for panel creation."""
        cols = self.config.columns

        future = df[df[cols.date] > cutoff]
        next_date = future.groupby(cols.donor_id)[cols.date].min().rename('NextGiftDate')

        if next_date.empty:
            return pd.DataFrame(columns=[cols.donor_id, 'NextGiftDate', 'DaysToNext'])

        labels = next_date.to_frame().reset_index()
        labels['DaysToNext'] = (labels['NextGiftDate'] - cutoff).dt.days
        labels = labels[labels['DaysToNext'] > 0]
        return labels

    def _get_preference_labels(
        self,
        df: pd.DataFrame,
        cutoff: pd.Timestamp,
        target_col: str
    ) -> pd.DataFrame:
        """Generate preference labels (fund/appeal) for panel creation."""
        cols = self.config.columns
        analysis = self.config.analysis

        window_end = cutoff + pd.Timedelta(days=analysis.redonate_window_days)
        future = df[
            (df[cols.date] > cutoff)
            & (df[cols.date] <= window_end)
            & (df[target_col].notna())
        ].sort_values([cols.donor_id, cols.date])

        next_pref = future.groupby(cols.donor_id)[target_col].first().rename('TargetCategory')
        return next_pref.reset_index()


    def _get_redonate_labels(self, df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
        cols = self.config.columns
        analysis = self.config.analysis
        
        window_end = cutoff + pd.Timedelta(days=analysis.redonate_window_days)
        future = df[(df[cols.date] > cutoff) & (df[cols.date] <= window_end)]
        
        labels = future.groupby(cols.donor_id).size().rename('ReDonate')
        labels = (labels > 0).astype(int)

        return labels.to_frame().reset_index()


    def _build_redonate_panel(self, df):
        """Build panel dataset for re-donation propensity"""
        cols = self.config.columns
        analysis = self.config.analysis
        
        max_date = df[cols.date].max()
        min_date = df[cols.date].min()
        
        start_date = min_date + pd.Timedelta(days=365)
        end_date = max_date - pd.Timedelta(days=analysis.redonate_window_days)
        
        if start_date >= end_date:
            raise ValueError("Insufficient date range for panel creation")
        
        cutoff_dates = pd.date_range(start=start_date, end=end_date, freq=analysis.panel_step)
        self.logger.info(f"Building re-donation panel: {len(cutoff_dates)} cutoffs")
        
        X_with_labels, y_series = self.feature_engineer.create_panel_dataset(
            df,
            cutoff_dates,
            label_func=self._get_redonate_labels,
            label_column_name='ReDonate',
            label_merge_how='left'
        )

        y = y_series.fillna(0).astype(int).reset_index(drop=True)
        X = X_with_labels.reset_index(drop=True)
        
        # REMOVED: Age, AgeGroup, AgeAtDonation, AgeGroupAtDonation
        cat_cols = [
            cols.gender, cols.occupation_desc, cols.religion, 
            cols.state, cols.education,
            cols.current_parent, cols.past_parent, cols.past_student
        ]
        cat_cols = [c for c in cat_cols if c and c in X.columns]
        self.logger.info(f"Panel created: {len(X):,} observations, {y.sum():,} positive labels ({y.mean()*100:.1f}%)")
        
        for col in X.columns:
            if col in cat_cols:
                X[col] = X[col].fillna('Missing').astype(str)
            elif col not in ['CutoffDate', cols.donor_id]:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = self._convert_na_to_nan(X)
        return X, y, cat_cols
    
    def _build_major_gift_panel(self, df):
        """Build panel dataset for major gift propensity"""
        cols = self.config.columns
        analysis = self.config.analysis
        threshold = analysis.major_gift_threshold
        
        max_date = df[cols.date].max()
        min_date = df[cols.date].min()
        
        start_date = min_date + pd.Timedelta(days=365)
        end_date = max_date - pd.Timedelta(days=analysis.redonate_window_days)
        
        if start_date >= end_date:
            raise ValueError("Insufficient date range for panel creation")
        
        cutoff_dates = pd.date_range(start=start_date, end=end_date, freq=analysis.panel_step)
        
        self.logger.info(f"Building major gift panel: {len(cutoff_dates)} cutoffs, threshold=${threshold:,.0f}")
        
        X_with_labels, y_series = self.feature_engineer.create_panel_dataset(
            df,
            cutoff_dates,
            label_func=self._get_major_gift_labels,
            label_column_name='MajorGift',
            label_merge_how='left'
        )

        y = y_series.fillna(0).astype(int).reset_index(drop=True)
        X = X_with_labels.reset_index(drop=True)

        cat_cols = [
            cols.gender, cols.occupation_desc, cols.religion,
            cols.state, cols.education,
            cols.current_parent, cols.past_parent, cols.past_student
        ]
        cat_cols = [c for c in cat_cols if c and c in X.columns]
        
        self.logger.info(f"Major gift panel: {len(X):,} observations, {y.sum():,} major gifts ({y.mean()*100:.1f}%)")
        
        for col in X.columns:
            if col in cat_cols:
                X[col] = X[col].fillna('').astype(str)
            elif col not in ['CutoffDate', cols.donor_id]:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = self._convert_na_to_nan(X)
        return X, y, cat_cols
    
    def _build_amount_dataset(self, df):
        """Build dataset for donation amount prediction"""
        cols = self.config.columns
        analysis = self.config.analysis
        
        max_date = df[cols.date].max()
        cutoff = max_date - pd.Timedelta(days=analysis.redonate_window_days)
        
        self.logger.info(f"Building amount dataset with cutoff: {cutoff.date()}")
        
        hist = df[df[cols.date] <= cutoff]
        feats = self.feature_engineer.create_donor_features(hist, cutoff)
        feats = self.feature_engineer.add_demographic_features(feats, df)
        
        future = df[df[cols.date] > cutoff]
        target = future.groupby(cols.donor_id)[cols.amount].median().rename('FutureMedianAmount')
        
        data = feats.merge(target.to_frame(), on=cols.donor_id, how='inner')
        
        y = data['FutureMedianAmount'].astype(float)
        X = data.drop(columns=['FutureMedianAmount', cols.donor_id], errors='ignore')
        
        cat_cols = [
            cols.gender, cols.occupation_desc, cols.religion,
            cols.state, cols.education,
            cols.current_parent, cols.past_parent, cols.past_student
        ]
        cat_cols = [c for c in cat_cols if c and c in X.columns]
        
        self.logger.info(f"Amount dataset: {len(X):,} donors, median target=${y.median():.2f}")
        
        for col in X.columns:
            if col in cat_cols:
                X[col] = X[col].fillna('').astype(str)
            else:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = self._convert_na_to_nan(X)
        return X, y, cat_cols
    
    def _build_time_dataset(self, df):
        """Build dataset for time-to-next-donation prediction"""
        cols = self.config.columns
        analysis = self.config.analysis
        
        max_date = df[cols.date].max()
        min_date = df[cols.date].min()
        
        start_date = pd.to_datetime(analysis.min_panel_start)
        if start_date < min_date:
            start_date = min_date + pd.Timedelta(days=365)
        
        end_date = max_date - pd.Timedelta(days=7)
        
        if start_date >= end_date:
            raise ValueError("Insufficient date range for time dataset")
        
        cutoff_dates = pd.date_range(start=start_date, end=end_date, freq=analysis.panel_step)
        
        self.logger.info(f"Building time dataset: {len(cutoff_dates)} cutoffs")
        
        X_with_labels, y_series = self.feature_engineer.create_panel_dataset(
            df,
            cutoff_dates,
            label_func=self._get_time_labels,
            label_column_name='DaysToNext',
            label_merge_how='inner'
        )

        if X_with_labels.empty:
            raise ValueError("No valid observations for time dataset")

        y = y_series.astype(float).reset_index(drop=True)
        X = X_with_labels.drop(columns=['NextGiftDate', cols.donor_id], errors='ignore').reset_index(drop=True)
        
        cat_cols = [
            cols.gender, cols.occupation_desc, cols.religion,
            cols.state, cols.education,
            cols.current_parent, cols.past_parent, cols.past_student
        ]
        cat_cols = [c for c in cat_cols if c and c in X.columns]
        
        self.logger.info(f"Time dataset: {len(X):,} observations, median={y.median():.1f} days")
        
        for col in X.columns:
            if col in cat_cols:
                X[col] = X[col].fillna('').astype(str)
            elif col not in ['CutoffDate']:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = self._convert_na_to_nan(X)
        return X, y, cat_cols
    
    def _build_preference_dataset(self, df):
        """Build dataset for fund/appeal preference prediction"""
        cols = self.config.columns
        analysis = self.config.analysis
        
        if cols.fund and cols.fund in df.columns and df[cols.fund].notna().any():
            target_col = cols.fund
            target_name = 'fund'
        elif cols.appeal and cols.appeal in df.columns and df[cols.appeal].notna().any():
            target_col = cols.appeal
            target_name = 'appeal'
        else:
            raise ValueError("Neither Fund nor Appeal columns available")
        
        max_date = df[cols.date].max()
        min_date = df[cols.date].min()
        
        start_date = pd.to_datetime(analysis.min_panel_start)
        if start_date < min_date:
            start_date = min_date + pd.Timedelta(days=365)
        
        end_date = max_date - pd.Timedelta(days=analysis.redonate_window_days)
        
        if start_date >= end_date:
            raise ValueError("Insufficient date range for preference dataset")
        
        cutoff_dates = pd.date_range(start=start_date, end=end_date, freq=analysis.panel_step)
        
        self.logger.info(f"Building preference dataset for {target_name}: {len(cutoff_dates)} cutoffs")

        X_with_labels, y_series = self.feature_engineer.create_panel_dataset(
            df,
            cutoff_dates,
            label_func=lambda data, cutoff: self._get_preference_labels(data, cutoff, target_col),
            label_column_name='TargetCategory',
            label_merge_how='inner'
        )

        if X_with_labels.empty:
            raise ValueError("No valid observations for preference dataset")

        y = y_series.dropna().astype(str)
        valid_index = y.index
        X = X_with_labels.loc[valid_index].drop(columns=[cols.donor_id], errors='ignore')
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        
        cat_cols = [
            cols.gender, cols.occupation_desc, cols.religion,
            cols.state, cols.education,
            cols.current_parent, cols.past_parent, cols.past_student
        ]
        cat_cols = [c for c in cat_cols if c and c in X.columns]
        
        n_classes = y.nunique()
        self.logger.info(f"Preference dataset: {len(X):,} observations, {n_classes} {target_name} categories")
        
        for col in X.columns:
            if col in cat_cols:
                X[col] = X[col].fillna('').astype(str)
            elif col not in ['CutoffDate']:
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        X = self._convert_na_to_nan(X)
        return X, y, cat_cols, target_name
    
    # ===== PREDICTION METHODS (FIXED) =====
    
    def _predict_amounts(self, df, model):
        """Generate amount predictions - FIXED to match training features"""
        cols = self.config.columns
        cutoff = df[cols.date].max()
        
        self.logger.debug("\n" + "="*80)
        self.logger.debug("DEBUG: _predict_amounts")
        self.logger.debug("="*80)
        
        # Step 1: Create features
        feats = self.feature_engineer.create_donor_features(df, cutoff)
        self.logger.debug("\nAfter create_donor_features:")
        self.logger.debug("   Columns: {feats.columns.tolist()}")
        self.logger.debug("   Dtypes:\n{feats.dtypes}")
    
        # Check for pd.NA
        for col in feats.columns:
            try:
                has_pd_na = feats[col].apply(lambda x: x is pd.NA).any()
                if has_pd_na:
                    self.logger.debug("   ⚠️  Column '{col}' has pd.NA!")
            except:
                pass
            
        # Step 2: Add demographics
        feats = self.feature_engineer.add_demographic_features(feats, df)
        self.logger.debug("\nAfter add_demographic_features:")
        self.logger.debug("   Columns: {feats.columns.tolist()}")
        self.logger.debug("   Dtypes:\n{feats.dtypes}")
    
        # Check for pd.NA again
        for col in feats.columns:
            try:
                has_pd_na = feats[col].apply(lambda x: x is pd.NA).any()
                if has_pd_na:
                    self.logger.debug("   ⚠️  Column '{col}' has pd.NA!")
            except:
                pass
        
        donor_ids = feats[cols.donor_id].copy()
        
        # Step 3: Drop donor_id
        # Drop ALL non-feature columns (donor_id only, no date columns)
        X = feats.drop(columns=[cols.donor_id], errors='ignore')
        self.logger.debug("\nAfter dropping donor_id:")
        self.logger.debug("   Shape: {X.shape}")
        self.logger.debug("   Columns: {X.columns.tolist()}")
        self.logger.debug("   Dtypes:\n{X.dtypes}")

        # Check for pd.NA again
        for col in X.columns:
            try:
                has_pd_na = X[col].apply(lambda x: x is pd.NA).any()
                if has_pd_na:
                    self.logger.debug("   ⚠️  Column '{col}' has pd.NA!")
                    # Show sample
                    sample = X[col].head(20)
                    self.logger.debug("      Sample: {sample.tolist()}")
            except:
                pass

        # Step 4: Convert NA
        X_before = X.copy()

        X = self._convert_na_to_nan(X)
        
        self.logger.debug("\nAfter _convert_na_to_nan:")
        self.logger.debug("   Dtypes:\n{X.dtypes}")
        
        # Final check
        for col in X.columns:
            try:
                has_pd_na = X[col].apply(lambda x: x is pd.NA).any()
                if has_pd_na:
                    self.logger.debug("   🔴 ERROR: Column '{col}' STILL has pd.NA!")
                    self.logger.debug("      Before conversion: {X_before[col].head(10).tolist()}")
                    self.logger.debug("      After conversion: {X[col].head(10).tolist()}")
            except:
                pass
    
        # Try sklearn test
        self.logger.debug("\nTesting sklearn compatibility:")
        try:
            for col in X.columns:
                if X[col].dtype == 'object':
                    _ = X[col] != X[col]
            self.logger.debug("   PASSED!")
        except TypeError as e:
            self.logger.debug("   ❌ FAILED: {e}")
            self.logger.debug("\n   Checking which column failed...")
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        _ = X[col] != X[col]
                    except TypeError:
                        self.logger.debug("      🔴 Column '{col}' failed the test!")
                        self.logger.debug("         Sample values: {X[col].head(20).tolist()}")
        
        self.logger.debug("\nCalling model.predict...")
        try:
            predictions = model.predict(X)
            self.logger.debug("   SUCCESS!")
        except Exception as e:
            self.logger.debug("   ❌ FAILED: {e}")
            raise
        
        self.logger.debug("="*80 + "\n")
        
        return pd.DataFrame({
            cols.donor_id: donor_ids,
            'pred_amount': predictions
        })
    
    def _predict_next_date(self, df, model, residuals):
        """Generate next donation date predictions"""
        cols = self.config.columns
        cutoff = df[cols.date].max()
        
        feats = self.feature_engineer.create_donor_features(df, cutoff)
        feats = self.feature_engineer.add_demographic_features(feats, df)
        
        donor_ids = feats[cols.donor_id].copy()
        X = feats.drop(columns=[cols.donor_id], errors='ignore')
        X = self._convert_na_to_nan(X)
        pred_days = model.predict(X)
        
        p20 = np.quantile(residuals, 0.2)
        p50 = np.median(residuals)
        p80 = np.quantile(residuals, 0.8)
        
        # Convert to dates and format as date only (no time)
        pred_date_point = (pd.to_datetime(cutoff) + pd.to_timedelta(pred_days, unit='D')).date
        pred_date_p20 = (pd.to_datetime(cutoff) + pd.to_timedelta(pred_days + p20, unit='D')).date
        pred_date_p50 = (pd.to_datetime(cutoff) + pd.to_timedelta(pred_days + p50, unit='D')).date
        pred_date_p80 = (pd.to_datetime(cutoff) + pd.to_timedelta(pred_days + p80, unit='D')).date
        
        return pd.DataFrame({
            cols.donor_id: donor_ids,
            'pred_days_to_next': pred_days,
            'pred_date_point': pred_date_point,
            'pred_date_p20': pred_date_p20,
            'pred_date_p50': pred_date_p50,
            'pred_date_p80': pred_date_p80
        })
    
    def _predict_preference(self, df, model, target_name):
        """Generate fund/appeal preference predictions"""
        cols = self.config.columns
        cutoff = df[cols.date].max()
        
        feats = self.feature_engineer.create_donor_features(df, cutoff)
        feats = self.feature_engineer.add_demographic_features(feats, df)
        
        donor_ids = feats[cols.donor_id].copy()
        X = feats.drop(columns=[cols.donor_id], errors='ignore')
        X = self._convert_na_to_nan(X)
        try:
            proba = model.predict_proba(X)
            classes = model.named_steps['model'].classes_
            
            rows = []
            for did, prob_row in zip(donor_ids, proba):
                sorted_indices = np.argsort(prob_row)[::-1][:3]
                
                rec = {cols.donor_id: did}
                for i, idx in enumerate(sorted_indices, 1):
                    rec[f'top{i}_{target_name}'] = classes[idx]
                    rec[f'top{i}_prob'] = float(prob_row[idx])
                
                rows.append(rec)
            
            return pd.DataFrame(rows)
            
        except Exception:
            predictions = model.predict(X)
            return pd.DataFrame({
                cols.donor_id: donor_ids,
                f'top1_{target_name}': predictions
            })
    
    def _calculate_priority_scores(self, playbook: pd.DataFrame) -> pd.DataFrame:
        """Calculate composite priority scores"""
        df = playbook.copy()
        
        def normalize(series):
            if series.isna().all():
                return series
            min_val = series.min()
            max_val = series.max()
            if max_val == min_val:
                return pd.Series(0.5, index=series.index)
            return (series - min_val) / (max_val - min_val)
        
        amount_score = normalize(df['pred_amount'].fillna(0)) if 'pred_amount' in df.columns else 0
        time_score = 1 - normalize(df['pred_days_to_next'].fillna(df['pred_days_to_next'].max())) if 'pred_days_to_next' in df.columns else 0
        pref_score = df['top1_prob'].fillna(0) if 'top1_prob' in df.columns else 0
        
        df['priority_score'] = (0.4 * amount_score + 0.4 * time_score + 0.2 * pref_score)
        df['action_bucket'] = df.apply(self._assign_action_bucket, axis=1)
        
        return df
    
    def _assign_action_bucket(self, row) -> str:
        """Assign operational action bucket"""
        amount = row.get('pred_amount', np.nan)
        days = row.get('pred_days_to_next', np.nan)
        prob = row.get('top1_prob', np.nan)
        
        if (pd.notna(amount) and amount >= 1000 and
            pd.notna(days) and days <= 60 and
            pd.notna(prob) and prob >= 0.7):
            return "VIP_target"
        
        if (pd.notna(amount) and amount >= 500 and
            pd.notna(days) and days <= 120):
            return "Upgrade_push"
        
        if pd.notna(days) and days > 180:
            return "Reactivation"
        
        return "Nurture"
    
    def _export_feature_importance(
        self, models: dict, feature_names: list,
        categorical_cols: list, output_dir: Path, prefix: str
    ) -> None:
        """Export feature importance for tree-based models"""
        extractor = FeatureImportanceExtractor(self.config, self.logger)
        
        for model_name in ['RandomForest', 'LightGBM']:
            if model_name not in models:
                continue
            
            try:
                fi_df = extractor.extract_importance(
                    models[model_name],
                    feature_names,
                    categorical_cols
                )
                
                extractor.export_field_importance(
                    fi_df,
                    f"{prefix}_{model_name}",
                    output_dir
                )
                
            except Exception as e:
                self.logger.warning(f"Could not extract importance for {model_name}: {e}")

    def _convert_na_to_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pandas NA to numpy nan for sklearn compatibility
        
        sklearn's SimpleImputer cannot handle pd.NA 
        This method converts all pd.NA values to np.nan
        """
        result = df.copy()
        
        for col in result.columns:
            # For numeric columns, convert pd.NA to np.nan
            if pd.api.types.is_numeric_dtype(result[col]):
                result[col] = result[col].fillna(np.nan)
            # For string columns, imputer handles with 'most_frequent'
        
        return result