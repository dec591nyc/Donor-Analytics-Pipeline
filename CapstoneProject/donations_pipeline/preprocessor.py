#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Module
Handles feature engineering and data transformations
"""

from pathlib import Path
import pandas as pd
import numpy as np

from .config import Config
from .logger import PipelineLogger


class DataPreprocessor:
    """Transforms raw data into analysis-ready format"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main preprocessing pipeline
        
        Steps:
        1. Add temporal features
        2. Process relationship flags
        3. Normalize categorical variables
        4. Clean amount column
        5. Detect and report anomalies
        
        NOTE: Age/demographic features removed per user request
        """
        self.logger.info("Starting data preprocessing")
        
        # Work on a copy to avoid modifying original
        result = df.copy()
        
        # Sequential transformations
        result = self._add_temporal_features(result)
        result = self._process_relationship_flags(result)
        result = self._normalize_categoricals(result)
        result = self._clean_amount_column(result)
        
        # Report statistics
        self._report_preprocessing_stats(df, result)
        
        return result
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add year, month, and period columns"""
        cols = self.config.columns
        
        if cols.date not in df.columns:
            self.logger.warning("Date column not found, skipping temporal features")
            return df
        
        if not np.issubdtype(df[cols.date].dtype, np.datetime64):
            df[cols.date] = pd.to_datetime(df[cols.date], errors='coerce')

        df["Year"] = df[cols.date].dt.year
        df["MonthPeriod"] = df[cols.date].dt.to_period("M")
        df["Month"] = df["MonthPeriod"].astype(str)  # YYYY-MM
        self.logger.debug("Added temporal features: Year, Month, MonthPeriod")
        return df
    
    def _normalize_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize categorical columns (strip whitespace, handle nulls)"""
        cols = self.config.columns
        
        categorical_cols = [
            getattr(cols, "gender", None),
            getattr(cols, "suburb", None),
            getattr(cols, "state", None),
            getattr(cols, "postcode", None),
            getattr(cols, "occupation_desc", None),
            getattr(cols, "religion", None),
            getattr(cols, "education", None),
            getattr(cols, "appeal", None),
            getattr(cols, "fund", None),
            getattr(cols, "appeal_desc", None),
            getattr(cols, "fund_desc", None),
        ]
        
        for col in categorical_cols:
            if col and col in df.columns:
                df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"": "Missing", "nan": "Missing"})

        self.logger.debug(f"Normalized {len([c for c in categorical_cols if c])} categorical columns")
        return df
    
    def _process_relationship_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert relationship flags to 0/1 and detect anomalies"""
        cols = self.config.columns
        
        rel_cols = [
            getattr(cols, "current_parent", None),
            getattr(cols, "future_parent", None),
            getattr(cols, "past_parent", None),
            getattr(cols, "past_student", None),
        ]

        truthy = {"1","y","yes","true","t"}
        falsy  = {"0","n","no","false","f",""}

        anomalies = []
        for col in rel_cols:
            if not col or col not in df.columns:
                continue

            s = df[col].astype(str).str.strip().str.lower()
            mapped = pd.Series(pd.NA, index=s.index, dtype="object")
            mapped[s.isin(truthy)] = 1
            mapped[s.isin(falsy)]  = 0

            num = pd.to_numeric(df[col], errors="coerce").astype('float64')
            mapped = mapped.fillna(num)

            cleaned = pd.to_numeric(mapped, errors="coerce").astype('float64')
            invalid_mask = ~cleaned.isin([0,1]) & cleaned.notna()

            if invalid_mask.any():
                bad = df.loc[invalid_mask, [cols.donor_id, col]].copy()
                bad["FlagColumn"] = col
                anomalies.append(bad)
                self.logger.warning(f"{col}: Found {invalid_mask.sum():,} rows with invalid values")

            df[col] = cleaned.fillna(0).astype(int) 

        if anomalies:
            self._export_relationship_anomalies(pd.concat(anomalies, ignore_index=True))
        
        return df
    
    def _clean_amount_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean amount column and remove invalid donations"""
        cols = self.config.columns
        
        if cols.amount not in df.columns:
            raise ValueError(f"Amount column '{cols.amount}' not found")
        
        original_count = len(df)
        df[cols.amount] = pd.to_numeric(df[cols.amount], errors='coerce')

        df = df[df[cols.amount] > 0]

        removed = original_count - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed:,} rows with non-positive amounts")
        return df
    
    def _export_relationship_anomalies(self, anomalies_df: pd.DataFrame) -> None:
        """Export detected relationship flag anomalies to CSV"""
        output_path = self.config.output_path / "relationship_flag_anomalies.csv"
        
        try:
            anomalies_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            self.logger.info(
                f"Exported {len(anomalies_df):,} relationship flag anomalies to: {output_path}"
            )
        except Exception as e:
            self.logger.error(f"Failed to export anomalies: {e}")
    
    def _report_preprocessing_stats(self, original: pd.DataFrame, processed: pd.DataFrame) -> None:
        """Report preprocessing statistics"""
        self.logger.info(
            f"Preprocessing complete: "
            f"{len(original):,} → {len(processed):,} rows "
            f"({len(original.columns)} → {len(processed.columns)} columns)"
        )
        
        # Show new columns
        new_cols = set(processed.columns) - set(original.columns)
        if new_cols:
            self.logger.debug(f"Added columns: {', '.join(sorted(new_cols))}")


class FeatureEngineer:
    """Advanced feature engineering for modeling"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def create_donor_features(
        self, 
        df: pd.DataFrame, 
        cutoff_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Create donor-level features up to a cutoff date
        
        Features:
        - Recency: days since last donation
        - Frequency: total number of donations
        - Monetary: total donation amount
        - AvgAmount: average donation amount
        - LastAmount: most recent donation amount
        
        Args:
            df: Transaction-level dataframe
            cutoff_date: Feature calculation cutoff
        
        Returns:
            Donor-level feature dataframe
        """
        cols = self.config.columns
        if cols.date not in df.columns:
            self.logger.warning("Date column not found for feature creation")
            return pd.DataFrame()

        if not np.issubdtype(df[cols.date].dtype, np.datetime64):
            df = df.copy()
            df[cols.date] = pd.to_datetime(df[cols.date], errors='coerce')
        
        # Filter to history only
        hist = df[df[cols.date] <= cutoff_date].copy()
        
        if len(hist) == 0:
            self.logger.warning(f"No data before cutoff: {cutoff_date}")
            return pd.DataFrame()
        
        # Sort for correct "last" calculations
        hist = hist.sort_values([cols.donor_id, cols.date])
        
        # Aggregate features (removed date columns)
        features = hist.groupby(cols.donor_id).agg(
            Recency=(cols.date, lambda s: (cutoff_date - s.max()).days if len(s) > 0 else np.nan),
            Frequency=(cols.date, 'count'),
            Monetary=(cols.amount, 'sum'),
            AvgAmount=(cols.amount, 'mean'),
            LastAmount=(cols.amount, lambda s: s.iloc[-1] if len(s) > 0 else np.nan)
        ).reset_index()
        
        return features
    
    def add_demographic_features(
        self, 
        features: pd.DataFrame, 
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add demographic features to donor-level features
        Uses the most recent demographic information per donor
        
        NOTE: Age-related features removed per user request
        """
        cols = self.config.columns
        if cols.donor_id not in df.columns:
            return features
        
        if getattr(cols, "date", None) in df.columns and not np.issubdtype(df[cols.date].dtype, np.datetime64):
            df = df.copy()
            df[cols.date] = pd.to_datetime(df[cols.date], errors='coerce')

        sort_cols = [c for c in [cols.donor_id, getattr(cols, "date", None)] if c]
        df_sorted = df.sort_values(sort_cols)
        latest_demo = df_sorted.groupby(cols.donor_id).tail(1)

        # Removed Age and AgeGroup from demographic columns
        demo_cols = [
            getattr(cols, "gender", None),
            getattr(cols, "occupation_desc", None),
            getattr(cols, "religion", None),
            getattr(cols, "state", None),
            getattr(cols, "education", None),
            getattr(cols, "current_parent", None),
            getattr(cols, "past_parent", None),
            getattr(cols, "past_student", None),
        ]
        available_demo_cols = [c for c in demo_cols if c and c in latest_demo.columns]
        if not available_demo_cols:
            self.logger.warning("No demographic columns available")
            return features
        
        demo_data = latest_demo[[cols.donor_id] + available_demo_cols]
        result = features.merge(demo_data, on=cols.donor_id, how='left')
        self.logger.debug(f"Added {len(available_demo_cols)} demographic features")
        return result
    
    def create_panel_dataset(
        self,
        df: pd.DataFrame,
        cutoff_dates: pd.DatetimeIndex,
        label_func: callable
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Create panel dataset with multiple time periods
        
        Args:
            df: Transaction data
            cutoff_dates: List of cutoff dates for feature calculation
            label_func: Function(df, cutoff_date) -> Series of labels
        
        Returns:
            (features_df, labels_series)
        """
        panels = []
        
        for cutoff in cutoff_dates:
            # Create features
            feats = self.create_donor_features(df, cutoff)
            
            if feats.empty:
                continue
            
            # Add demographics
            feats = self.add_demographic_features(feats, df)
            
            # Add cutoff date
            feats['CutoffDate'] = cutoff
            
            # Calculate labels
            labels = label_func(df, cutoff)
            donor_col = self.config.columns.donor_id

            if isinstance(labels, pd.Series):
                if labels.index.name != donor_col:
                    labels.index.name = donor_col
                if not labels.name:
                    labels.name = 'label'
                labels = labels.to_frame().reset_index()

            elif isinstance(labels, pd.DataFrame):
                if donor_col not in labels.columns and labels.index.name == donor_col:
                    labels = labels.reset_index()

            # Merge labels
            feats = feats.merge(
                labels,
                on=self.config.columns.donor_id,
                how='left'
            )
            
            panels.append(feats)
        
        if not panels:
            raise ValueError("No valid panels created")
        
        # Combine all panels
        panel_df = pd.concat(panels, ignore_index=True)
        
        # Separate features and labels
        label_col = labels.name if hasattr(labels, 'name') else 'label'
        y = panel_df[label_col]
        X = panel_df.drop(columns=[label_col])
        
        self.logger.info(
            f"Created panel dataset: "
            f"{len(cutoff_dates)} periods, "
            f"{len(X):,} observations"
        )
        
        return X, y


class DataSplitter:
    """Handles train/test splitting with temporal awareness"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def time_based_split(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        time_column: str = 'CutoffDate',
        train_ratio: float = 0.8
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data based on time to avoid leakage
        
        Args:
            X: Features
            y: Labels
            time_column: Column containing time information
            train_ratio: Proportion of data for training
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if time_column not in X.columns:
            self.logger.warning(
                f"Time column '{time_column}' not found, using random split"
            )
            return self.random_split(X, y)
        
        if not np.issubdtype(X[time_column].dtype, np.datetime64):
            X = X.copy()
            X[time_column] = pd.to_datetime(X[time_column], errors='coerce')

        # Find split point
        split_date = X[time_column].quantile(train_ratio)
        
        # Split
        train_mask = X[time_column] <= split_date
        test_mask = X[time_column] > split_date
        
        X_train = X.loc[train_mask].drop(columns=[time_column])
        X_test = X.loc[test_mask].drop(columns=[time_column])
        y_train = y.loc[train_mask]
        y_test = y.loc[test_mask]
        
        self.logger.info(
            f"Time-based split: "
            f"train={len(X_train):,} ({len(X_train)/len(X)*100:.1f}%), "
            f"test={len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)"
        )
        
        return X_train, X_test, y_train, y_test
    
    def random_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = None,
        stratify: bool = True
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Random train/test split with optional stratification"""
        from sklearn.model_selection import train_test_split
        
        if test_size is None:
            test_size = self.config.analysis.train_test_split_ratio
        
        stratify_arg = y if stratify and y.nunique() < 10 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.config.analysis.random_state,
            stratify=stratify_arg
        )
        
        self.logger.info(
            f"Random split: "
            f"train={len(X_train):,}, test={len(X_test):,}"
        )
        
        return X_train, X_test, y_train, y_test