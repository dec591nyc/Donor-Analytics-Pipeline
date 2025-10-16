#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actionable Lists Module
Generates targeting and reactivation lists for operational use
"""

from typing import Dict
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

from .config import Config
from .logger import PipelineLogger


class ActionableListGenerator:
    """Generates operational targeting lists"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def generate_lists(
        self,
        df: pd.DataFrame,
        rfm: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate all actionable lists
        
        Args:
            df: Transaction-level data
            rfm: Donor-level RFM data
        
        Returns:
            Dictionary of list_name -> dataframe
        """
        # Create donor summary
        donor_summary = self._create_donor_summary(df)
        
        lists = {}
        
        # Lapsed donors (>12 months)
        lists['lapsed_12m'] = self._generate_lapsed_list(donor_summary)
        
        # High-potential sleepers
        lists['high_potential_sleeper'] = self._generate_sleeper_list(donor_summary)
        
        # Upgrade candidates
        lists['upgrade_candidates'] = self._generate_upgrade_list(donor_summary)
        
        # New donors
        lists['new_donors_90d'] = self._generate_new_donor_list(donor_summary)
        
        # Log summary
        for name, table in lists.items():
            self.logger.info(f"Generated {name}: {len(table)} donors")
        
        return lists
    
    def _create_donor_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive donor summary"""
        cols = self.config.columns
        
        # Get reference date
        ref_date = df[cols.date].max()
        
        # Ensure ref_date is Timestamp
        if pd.isna(ref_date):
            self.logger.warning("No valid dates found, using today as reference")
            ref_date = pd.Timestamp.now()
        
        ref_date = ref_date + DateOffset(days=1)
        
        # Sort for correct last calculations
        df_sorted = df.sort_values([cols.donor_id, cols.date])
        
        # Aggregate all metrics
        summary = df_sorted.groupby(cols.donor_id).agg(
            LastGiftDate=(cols.date, 'max'),
            FirstGiftDate=(cols.date, 'min'),
            Frequency=(cols.date, 'count'),
            Monetary=(cols.amount, 'sum'),
            AvgAmount=(cols.amount, 'mean'),
            LastAmount=(cols.amount, lambda s: s.iloc[-1])
        ).reset_index()
        
        # Ensure date columns are datetime
        summary['LastGiftDate'] = pd.to_datetime(summary['LastGiftDate'], errors='coerce')
        summary['FirstGiftDate'] = pd.to_datetime(summary['FirstGiftDate'], errors='coerce')
        
        # Calculate recency
        summary['Recency'] = (ref_date - summary['LastGiftDate']).dt.days
        
        # Calculate 12-month windows
        cut_12m = ref_date - DateOffset(days=365)
        cut_prev12 = ref_date - DateOffset(days=2*365)
        
        # Recent 12 months
        recent_12m = df[df[cols.date] > cut_12m].groupby(cols.donor_id).agg(
            Gifts12=(cols.date, 'size'),
            Amt12=(cols.amount, 'sum')
        )
        
        # Previous 12 months
        prev_12m = df[
            (df[cols.date] <= cut_12m) & 
            (df[cols.date] > cut_prev12)
        ].groupby(cols.donor_id).agg(
            GiftsPrev12=(cols.date, 'size'),
            AmtPrev12=(cols.amount, 'sum')
        )
        
        # Merge all
        summary = summary.join(recent_12m, on=cols.donor_id, how='left')
        summary = summary.join(prev_12m, on=cols.donor_id, how='left')
        
        # Fill NaN with 0 for numeric columns only
        numeric_cols = ['Gifts12', 'Amt12', 'GiftsPrev12', 'AmtPrev12']
        for col in numeric_cols:
            if col in summary.columns:
                summary[col] = summary[col].fillna(0)
        
        return summary
    
    def _generate_lapsed_list(self, summary: pd.DataFrame) -> pd.DataFrame:
        """
        Lapsed donors: No gift in >12 months, but some history
        """
        lapsed_threshold = 365
        min_monetary = 100.0
        
        mask = (
            (summary['Recency'] >= lapsed_threshold) &
            (summary['Monetary'] >= min_monetary)
        )
        
        lapsed = summary[mask].copy()
        
        # Sort by monetary value (prioritize high-value lapsed)
        lapsed = lapsed.sort_values(['Monetary', 'Recency'], ascending=[False, True])
        
        # Select relevant columns
        cols = [
            self.config.columns.donor_id,
            'Recency',
            'Frequency',
            'Monetary',
            'AvgAmount',
            'LastAmount',
            'LastGiftDate',
            'FirstGiftDate',
            'Gifts12',
            'Amt12',
            'GiftsPrev12',
            'AmtPrev12'
        ]
        
        return lapsed[[c for c in cols if c in lapsed.columns]]
    
    def _generate_sleeper_list(self, summary: pd.DataFrame) -> pd.DataFrame:
        """
        High-potential sleepers: Strong history but dormant
        """
        sleeper_threshold = 365
        min_monetary = 500.0
        
        mask = (
            (summary['Recency'] >= sleeper_threshold) &
            (summary['Monetary'] >= min_monetary)
        )
        
        sleepers = summary[mask].copy()
        sleepers = sleepers.sort_values(['Monetary', 'Recency'], ascending=[False, True])
        
        cols = [
            self.config.columns.donor_id,
            'Recency',
            'Frequency',
            'Monetary',
            'AvgAmount',
            'LastAmount',
            'LastGiftDate',
            'FirstGiftDate',
            'Gifts12',
            'Amt12',
            'GiftsPrev12',
            'AmtPrev12'
        ]
        
        return sleepers[[c for c in cols if c in sleepers.columns]]
    
    def _generate_upgrade_list(self, summary: pd.DataFrame) -> pd.DataFrame:
        """
        Upgrade candidates: Recent gift ≥ 1.5x average, frequency ≥ 2
        """
        upgrade_multiplier = self.config.analysis.upgrade_multiplier
        min_frequency = self.config.analysis.upgrade_min_frequency
        recent_days = 365
        
        # Calculate reference date - handle mixed types
        if summary['LastGiftDate'].dtype == 'object':
            summary['LastGiftDate'] = pd.to_datetime(summary['LastGiftDate'], errors='coerce')
        
        # Get max date safely
        valid_dates = summary['LastGiftDate'].dropna()
        if len(valid_dates) == 0:
            self.logger.warning("No valid LastGiftDate found for upgrade list")
            return pd.DataFrame()
        
        max_date = valid_dates.max()
        recent_cutoff = max_date - pd.Timedelta(days=recent_days)
        
        mask = (
            (summary['LastGiftDate'] >= recent_cutoff) &
            (summary['LastAmount'] >= upgrade_multiplier * summary['AvgAmount']) &
            (summary['Frequency'] >= min_frequency)
        )
        
        upgrades = summary[mask].copy()
        upgrades = upgrades.sort_values('LastAmount', ascending=False)
        
        cols = [
            self.config.columns.donor_id,
            'Recency',
            'Frequency',
            'Monetary',
            'AvgAmount',
            'LastAmount',
            'LastGiftDate',
            'FirstGiftDate',
            'Gifts12',
            'Amt12',
            'GiftsPrev12',
            'AmtPrev12'
        ]
        
        return upgrades[[c for c in cols if c in upgrades.columns]]
    
    def _generate_new_donor_list(self, summary: pd.DataFrame) -> pd.DataFrame:
        """
        New donors: First gift in last 90 days
        """
        new_donor_days = self.config.analysis.new_donor_days
        
        # Ensure FirstGiftDate is datetime
        if summary['FirstGiftDate'].dtype == 'object':
            summary['FirstGiftDate'] = pd.to_datetime(summary['FirstGiftDate'], errors='coerce')
        
        # Calculate reference date safely
        valid_dates = summary['FirstGiftDate'].dropna()
        if len(valid_dates) == 0:
            self.logger.warning("No valid FirstGiftDate found for new donor list")
            return pd.DataFrame()
        
        max_date = valid_dates.max()
        new_cutoff = max_date - pd.Timedelta(days=new_donor_days)
        
        mask = summary['FirstGiftDate'] >= new_cutoff
        
        new_donors = summary[mask].copy()
        new_donors = new_donors.sort_values('FirstGiftDate', ascending=False)
        
        cols = [
            self.config.columns.donor_id,
            'Recency',
            'Frequency',
            'Monetary',
            'AvgAmount',
            'LastAmount',
            'LastGiftDate',
            'FirstGiftDate',
            'Gifts12',
            'Amt12'
        ]
        
        return new_donors[[c for c in cols if c in new_donors.columns]]