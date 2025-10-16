#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descriptive Analytics Module
Handles trends, distributions, and campaign performance analysis
"""

from typing import Dict
import pandas as pd
import numpy as np

from .config import Config
from .logger import PipelineLogger


class DescriptiveAnalyzer:
    """Performs descriptive analytics on donation data"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Run all descriptive analyses
        
        Returns:
            Dictionary mapping filename to dataframe
        """
        results = {}
        
        # Temporal trends
        results.update(self._analyze_trends(df))
        
        # Demographic distributions
        results.update(self._analyze_distributions(df))
        
        # Campaign performance
        results.update(self._analyze_campaigns(df))
        
        self.logger.info(f"Generated {len(results)} descriptive reports")
        
        return results
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze donation trends over time"""
        cols = self.config.columns
        results = {}
        
        # Yearly trends
        if 'Year' in df.columns:
            trend_year = self._create_yearly_trend(df)
            results['trend_year.csv'] = trend_year
            
            self.logger.info(
                f"Yearly trend: {len(trend_year)} years, "
                f"total ${trend_year['total_amount'].sum():,.0f}"
            )
        
        # Monthly trends
        if 'MonthPeriod' in df.columns:
            trend_month = self._create_monthly_trend(df)
            results['trend_month.csv'] = trend_month
            
            self.logger.info(
                f"Monthly trend: {len(trend_month)} months"
            )
        
        return results
    
    def _create_yearly_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create yearly aggregated trend"""
        cols = self.config.columns
        
        trend = df.groupby('Year', as_index=False).agg(
            total_amount=(cols.amount, 'sum'),
            donors=(cols.donor_id, pd.Series.nunique),
            donation_count=(cols.amount, 'size')
        )
        
        # Calculate averages
        trend['avg_amount_per_donation'] = (
            trend['total_amount'] / trend['donation_count']
        )
        trend['avg_amount_per_donor'] = (
            trend['total_amount'] / trend['donors']
        )
        trend['avg_donations_per_donor'] = (
            trend['donation_count'] / trend['donors']
        )
        
        # Calculate YoY growth rates
        trend = trend.sort_values('Year')
        
        trend['YoY_total_growth_%'] = (
            trend['total_amount'].pct_change() * 100
        )
        trend['YoY_donor_growth_%'] = (
            trend['donors'].pct_change() * 100
        )
        trend['YoY_avg_donation_growth_%'] = (
            trend['avg_amount_per_donation'].pct_change() * 100
        )
        
        # Sort newest first for output
        return trend.sort_values('Year', ascending=False)
    
    def _create_monthly_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monthly aggregated trend"""
        cols = self.config.columns
        
        trend = df.groupby('MonthPeriod', as_index=False).agg(
            total_amount=(cols.amount, 'sum'),
            donors=(cols.donor_id, pd.Series.nunique),
            donation_count=(cols.amount, 'size')
        )
        
        # Convert period to string
        trend['Month'] = trend['MonthPeriod'].astype(str)
        
        # Calculate averages
        trend['avg_amount_per_donation'] = (
            trend['total_amount'] / trend['donation_count']
        )
        trend['avg_amount_per_donor'] = (
            trend['total_amount'] / trend['donors']
        )
        trend['avg_donations_per_donor'] = (
            trend['donation_count'] / trend['donors']
        )
        
        # Select and order columns
        columns = [
            'Month',
            'total_amount',
            'donors',
            'donation_count',
            'avg_amount_per_donation',
            'avg_amount_per_donor',
            'avg_donations_per_donor'
        ]
        
        return trend[columns].sort_values('Month', ascending=False)
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze demographic distributions"""
        results = {}
        
        distributions = [
            (self.config.columns.gender, 'dist_gender.csv', 50),
            ('AgeGroup', 'dist_agegroup.csv', 50),
            (self.config.columns.occupation_desc, 'dist_occupation.csv', 50),
            (self.config.columns.religion, 'dist_religion.csv', 50),
            (self.config.columns.suburb, 'dist_suburb.csv', 50)
        ]
        
        for col, filename, top_n in distributions:
            if col and col in df.columns:
                dist_df = self._create_distribution(df, col, top_n)
                if not dist_df.empty:
                    results[filename] = dist_df
                    
                    self.logger.debug(
                        f"{col}: {len(dist_df)} categories, "
                        f"top=${dist_df['total_amount'].iloc[0]:,.0f}"
                    )
        
        return results
    
    def _create_distribution(
        self,
        df: pd.DataFrame,
        group_col: str,
        top_n: int = 50
    ) -> pd.DataFrame:
        """Create distribution by category"""
        cols = self.config.columns
        
        dist = df.groupby(group_col).agg(
            total_amount=(cols.amount, 'sum'),
            donors=(cols.donor_id, pd.Series.nunique),
            donation_count=(cols.amount, 'size')
        ).reset_index()
        
        # Calculate averages
        dist['avg_amount_per_donation'] = (
            dist['total_amount'] / dist['donation_count']
        )
        dist['avg_amount_per_donor'] = (
            dist['total_amount'] / dist['donors']
        )
        
        # Sort by total amount and donation count
        dist = dist.sort_values(
            ['total_amount', 'donation_count'],
            ascending=False
        ).head(top_n)
        
        return dist
    
    def _analyze_campaigns(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Analyze appeal and fund performance"""
        results = {}
        cols = self.config.columns
        
        # Appeal performance
        if cols.appeal and cols.appeal in df.columns:
            appeal_perf = df.groupby(cols.appeal).agg(
                total_amount=(cols.amount, 'sum'),
                donors=(cols.donor_id, pd.Series.nunique),
                donation_count=(cols.amount, 'size')
            ).reset_index()
            
            # Calculate averages
            appeal_perf['avg_amount_per_donation'] = (
                appeal_perf['total_amount'] / appeal_perf['donation_count']
            )
            appeal_perf['avg_amount_per_donor'] = (
                appeal_perf['total_amount'] / appeal_perf['donors']
            )
            
            appeal_perf = appeal_perf.sort_values('total_amount', ascending=False)
            results['appeal_performance.csv'] = appeal_perf
            
            self.logger.info(
                f"Appeal analysis: {len(appeal_perf)} appeals, "
                f"top: {appeal_perf.iloc[0][cols.appeal]}"
            )
        
        # Fund performance
        if cols.fund and cols.fund in df.columns:
            fund_perf = df.groupby(cols.fund).agg(
                total_amount=(cols.amount, 'sum'),
                donors=(cols.donor_id, pd.Series.nunique),
                donation_count=(cols.amount, 'size')
            ).reset_index()
            
            # Calculate averages
            fund_perf['avg_amount_per_donation'] = (
                fund_perf['total_amount'] / fund_perf['donation_count']
            )
            fund_perf['avg_amount_per_donor'] = (
                fund_perf['total_amount'] / fund_perf['donors']
            )
            
            fund_perf = fund_perf.sort_values('total_amount', ascending=False)
            results['fund_performance.csv'] = fund_perf
            
            self.logger.info(
                f"Fund analysis: {len(fund_perf)} funds, "
                f"top: {fund_perf.iloc[0][cols.fund]}"
            )
        
        return results