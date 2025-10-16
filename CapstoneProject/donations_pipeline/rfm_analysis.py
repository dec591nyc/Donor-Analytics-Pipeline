#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFM Analysis Module
Handles Recency, Frequency, Monetary segmentation
"""

from pathlib import Path
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset

from .config import Config
from .logger import PipelineLogger


class RFMAnalyzer:
    """Performs RFM segmentation and scoring"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def compute_rfm(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main RFM computation method
        
        Steps:
        1. Validate data
        2. Determine reference date
        3. Aggregate RFM metrics
        4. Calculate RFM scores
        5. Assign segments
        
        Returns:
            Donor-level RFM dataframe
        """
        with self.logger.timed_operation("RFM Computation"):
            # Validate
            df_valid = self._validate_and_prepare(df)
            
            # Reference date
            ref_date = self._get_reference_date(df_valid)
            
            # Aggregate metrics
            rfm = self._aggregate_rfm_metrics(df_valid, ref_date)
            
            # Calculate scores
            rfm = self._calculate_rfm_scores(rfm)
            
            # Assign segments
            rfm = self._assign_segments(rfm)
            
            self.logger.info(f"RFM computed for {len(rfm):,} donors")
            
            return rfm
    
    def _validate_and_prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data and prepare for RFM computation"""
        cols = self.config.columns
        
        # Drop rows with missing dates
        df_valid = df.dropna(subset=[cols.date]).copy()
        
        if df_valid.empty:
            raise ValueError("No valid dates found for RFM computation")
        
        # Sort by donor and date
        df_valid = df_valid.sort_values([cols.donor_id, cols.date])
        
        return df_valid
    
    def _get_reference_date(self, df: pd.DataFrame) -> pd.Timestamp:
        """Determine reference date for Recency calculation"""
        cols = self.config.columns
        
        if self.config.rfm_reference:
            ref_date = pd.to_datetime(self.config.rfm_reference)
            self.logger.info(f"Using configured reference date: {ref_date.date()}")
        else:
            ref_date = df[cols.date].max() + DateOffset(days=1)
            self.logger.info(f"Using computed reference date: {ref_date.date()}")
        
        return ref_date
    
    def _aggregate_rfm_metrics(
        self, 
        df: pd.DataFrame, 
        ref_date: pd.Timestamp
    ) -> pd.DataFrame:
        """Aggregate RFM metrics per donor"""
        cols = self.config.columns
        
        rfm = df.groupby(cols.donor_id).agg(
            Recency=(cols.date, lambda s: (ref_date - s.max()).days),
            Frequency=(cols.date, 'count'),
            Monetary=(cols.amount, 'sum'),
            LastAmount=(cols.amount, lambda s: s.iloc[-1]),
            FirstGiftDate=(cols.date, 'min'),
            LastGiftDate=(cols.date, 'max')
        ).reset_index()
        
        return rfm
    
    def _calculate_rfm_scores(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Calculate quintile-based RFM scores (1-5)"""
        quantiles = self.config.analysis.rfm_quantiles
        
        # R score (lower is better, so reverse)
        rfm['R_score'] = self._quantile_score(rfm['Recency'], quantiles, reverse=True)
        
        # F score (higher is better)
        rfm['F_score'] = self._quantile_score(rfm['Frequency'], quantiles, reverse=False)
        
        # M score (higher is better)
        rfm['M_score'] = self._quantile_score(rfm['Monetary'], quantiles, reverse=False)
        
        # Combined RFM score
        rfm['RFM_Score'] = rfm[['R_score', 'F_score', 'M_score']].sum(axis=1)
        
        self.logger.debug(
            f"RFM Score distribution: "
            f"min={rfm['RFM_Score'].min():.0f}, "
            f"mean={rfm['RFM_Score'].mean():.1f}, "
            f"max={rfm['RFM_Score'].max():.0f}"
        )
        
        return rfm
    
    @staticmethod
    def _quantile_score(
        series: pd.Series,
        quantiles: list[float],
        reverse: bool = False
    ) -> pd.Series:
        """
        Convert values to quintile scores (1-5)
        
        Args:
            series: Values to score
            quantiles: Quantile thresholds (e.g., [0.2, 0.4, 0.6, 0.8])
            reverse: If True, lower values get higher scores
        
        Returns:
            Series of scores (1-5)
        """
        series_copy = series.copy()
        
        if reverse:
            series_copy = -series_copy
        
        valid_mask = series_copy.notna()
        
        # Calculate bins
        valid_values = series_copy[valid_mask]
        bins = [-np.inf] + [valid_values.quantile(q) for q in quantiles] + [np.inf]
        
        # Assign scores
        scores = pd.Series(index=series.index, dtype=float)
        scores.loc[valid_mask] = pd.cut(
            series_copy[valid_mask],
            bins=bins,
            labels=[1, 2, 3, 4, 5]
        ).astype(float)
        
        # Handle missing values (assign middle score)
        scores.loc[~valid_mask] = 3.0
        
        return scores.astype(int)
    
    def _assign_segments(self, rfm: pd.DataFrame) -> pd.DataFrame:
        """Assign descriptive segments based on RFM scores"""
        
        def segment_logic(row):
            r, f, m = row['R_score'], row['F_score'], row['M_score']
            
            # Champions: High on all metrics
            if r >= 4 and f >= 4 and m >= 4:
                return "Champion"
            
            # Loyal: Recent and frequent
            if r >= 4 and f >= 3:
                return "Loyal"
            
            # At Risk: Low recent, low frequency, low value
            if r <= 2 and f <= 2 and m <= 2:
                return "At Risk / Lapsed"
            
            # High Potential: Recent and high value
            if r >= 4 and m >= 4:
                return "High Potential"
            
            # Default
            return "Regular"
        
        rfm['RFM_Segment'] = rfm.apply(segment_logic, axis=1)
        
        # Log segment distribution
        segment_counts = rfm['RFM_Segment'].value_counts()
        self.logger.info("RFM Segment distribution:")
        for segment, count in segment_counts.items():
            pct = count / len(rfm) * 100
            self.logger.info(f"  {segment}: {count:,} ({pct:.1f}%)")
        
        return rfm
    
    def identify_lapsed_donors(
        self,
        rfm: pd.DataFrame,
        recency_threshold: int = None,
        min_monetary: float = None
    ) -> pd.DataFrame:
        """
        Identify lapsed donors for reactivation
        
        Args:
            rfm: RFM dataframe
            recency_threshold: Days since last gift (default from config)
            min_monetary: Minimum historical giving (default from config)
        
        Returns:
            Dataframe of lapsed donors
        """
        if recency_threshold is None:
            recency_threshold = self.config.analysis.lapsed_recency_days
        
        if min_monetary is None:
            min_monetary = self.config.analysis.lapsed_min_monetary
        
        lapsed = rfm[
            (rfm['Recency'] >= recency_threshold) &
            (rfm['Monetary'] >= min_monetary)
        ].copy()
        
        # Sort by historical value
        lapsed = lapsed.sort_values('Monetary', ascending=False)
        
        self.logger.info(
            f"Identified {len(lapsed):,} lapsed donors "
            f"(>{recency_threshold} days, ≥${min_monetary:.0f} lifetime)"
        )
        
        return lapsed[[
            self.config.columns.donor_id,
            'Recency',
            'Frequency',
            'Monetary',
            'RFM_Segment'
        ]]


class RelationshipAnalyzer:
    """Analyzes school relationship patterns"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def analyze_relationships(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Analyze donor-school relationships
        
        Returns:
            Dictionary of analysis results:
            - relationship_summary: Aggregate by relationship type
            - relationship_trend: Trends over years
            - relationship_agegroup: Cross-analysis with age
        """
        cols = self.config.columns
        
        # Identify relationship columns
        rel_cols = [
            cols.current_parent,
            cols.past_parent,
            cols.past_student,
            cols.future_parent
        ]
        rel_cols = [c for c in rel_cols if c and c in df.columns]
        
        if not rel_cols:
            self.logger.warning("No relationship columns found")
            return {}
        
        results = {}
        
        # Summary by relationship type
        results['relationship_summary'] = self._summarize_by_relationship(
            df, rel_cols
        )
        
        # Trends over time
        if 'Year' in df.columns:
            results['relationship_trend'] = self._relationship_trends(
                df, rel_cols
            )
        
        # Age group analysis
        if 'AgeGroupAtDonation' in df.columns:
            results['relationship_agegroup'] = self._relationship_by_age(
                df, rel_cols
            )
        
        return results
    
    def _summarize_by_relationship(
            self,
            df: pd.DataFrame,
            rel_cols: list[str]
        ) -> pd.DataFrame:
            """Aggregate donations by relationship type"""
            cols = self.config.columns
            
            summary = df.groupby(rel_cols, dropna=False).agg(
                donors=(cols.donor_id, pd.Series.nunique),
                donation_count=(cols.amount, 'size'),
                total_amount=(cols.amount, 'sum'),
                avg_amount=(cols.amount, 'mean'),
                median_amount=(cols.amount, 'median')
            ).reset_index()
            
            # Sort by relationship flags in binary order (descending)
            # Binary order: 1,1,1,1 (15) -> 1,1,1,0 (14) -> ... -> 0,0,0,0 (0)
            
            if len(rel_cols) >= 4:
                # Assuming rel_cols order: [CurrentParent, PastParent, PastStudent, FutureParent]
                summary['_binary_key'] = (
                    summary[rel_cols[0]].astype(int) * 8 +
                    summary[rel_cols[1]].astype(int) * 4 +
                    summary[rel_cols[2]].astype(int) * 2 +
                    summary[rel_cols[3]].astype(int) * 1
                )
                
                # Sort by binary key descending (highest combination first)
                summary = summary.sort_values(
                    by='_binary_key',
                    ascending=False,
                    na_position='last'
                )
                
                # Drop the helper column
                summary = summary.drop(columns=['_binary_key'])
            else:
                # Fallback: sort by total_amount if fewer than 4 columns
                summary = summary.sort_values('total_amount', ascending=False)
            
            summary = summary.reset_index(drop=True)
            
            self.logger.info(
                f"Relationship summary: {len(summary)} unique combinations"
            )
            
            return summary
    
    def _relationship_trends(
            self,
            df: pd.DataFrame,
            rel_cols: list[str]
        ) -> pd.DataFrame:
            """Analyze relationship trends over years"""
            cols = self.config.columns
            
            trend = df.groupby(['Year'] + rel_cols).agg(
                total_amount=(cols.amount, 'sum'),
                donation_count=(cols.amount, 'size')
            ).reset_index()
            
            # Sort by Year (descending), then by relationship flags in binary order
            # Binary order: 1,1,1,1 (15) -> 1,1,1,0 (14) -> ... -> 0,0,0,0 (0)
            
            # Create binary combination key from relationship columns
            if len(rel_cols) >= 4:
                # Assuming rel_cols order: [CurrentParent, PastParent, PastStudent, FutureParent]
                trend['_binary_key'] = (
                    trend[rel_cols[0]].astype(int) * 8 +
                    trend[rel_cols[1]].astype(int) * 4 +
                    trend[rel_cols[2]].astype(int) * 2 +
                    trend[rel_cols[3]].astype(int) * 1
                )
                
                # Sort by Year descending, then by binary key descending
                trend = trend.sort_values(
                    by=['Year', '_binary_key'],
                    ascending=[False, False],
                    na_position='last'
                )
                
                # Drop the helper column
                trend = trend.drop(columns=['_binary_key'])
            else:
                # Fallback if relationship columns are fewer than 4
                trend = trend.sort_values('Year', ascending=False)
            
            trend = trend.reset_index(drop=True)
            
            return trend
    
    def _relationship_by_age(
        self,
        df: pd.DataFrame,
        rel_cols: list[str]
    ) -> pd.DataFrame:
        """Cross-analyze relationships with age groups"""
        cols = self.config.columns
        
        age_rel = df.groupby(rel_cols + ['AgeGroupAtDonation']).agg(
            donor_count=(cols.donor_id, 'nunique'),
            donation_count=(cols.amount, 'size'),
            total_amount=(cols.amount, 'sum')
        ).reset_index()
        
        # Calculate averages
        age_rel['avg_amount_per_donation'] = (
            age_rel['total_amount'] / age_rel['donation_count']
        )
        age_rel['avg_amount_per_donor'] = (
            age_rel['total_amount'] / age_rel['donor_count']
        )
        
        return age_rel