#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Loading and SQL Extraction Module
Handles CSV reading and SQL database extraction
"""

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .config import Config, SQLConfig
from .logger import PipelineLogger


class DataLoader:
    """Handles data loading from CSV or SQL sources"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def load_data(self) -> pd.DataFrame:
        """
        Main entry point for data loading
        Automatically routes to SQL or CSV based on config
        """
        if self.config.sql.use_sql:
            return self._load_from_sql()
        else:
            return self._load_from_csv()
    
    def _load_from_csv(self) -> pd.DataFrame:
        """Load data from CSV file"""
        csv_path = Path(self.config.input_file)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"Input file not found: {csv_path}")
        
        self.logger.info(f"Loading data from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            
            self.logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
            return self._apply_type_conversions(df)
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}", exc_info=True)
            raise
    
    def _load_from_sql(self) -> pd.DataFrame:
        """Load data from SQL Server"""
        try:
            import pyodbc
        except ImportError:
            raise RuntimeError(
                "pyodbc is required for SQL extraction. "
                "Install with: pip install pyodbc"
            )
        
        self.logger.info(
            f"Connecting to SQL Server: {self.config.sql.server}\\{self.config.sql.database}"
        )
        
        # Determine output CSV path
        out_csv = self._get_sql_export_path()
        
        try:
            conn_str = self.config.sql.get_connection_string()
            sql_query = self._build_sql_query()
            
            # Remove old export if exists
            if out_csv.exists():
                out_csv.unlink()
                self.logger.debug(f"Removed old export: {out_csv}")
            
            # Extract data in chunks
            df = self._extract_sql_data(conn_str, sql_query, out_csv)
            
            # Update config to use exported CSV
            self.config.input_file = str(out_csv)
            
            self.logger.info(f"SQL export completed: {len(df):,} rows")
            return self._apply_type_conversions(df)
            
        except Exception as e:
            self.logger.error(f"SQL extraction failed: {e}", exc_info=True)
            raise
    
    def _extract_sql_data(
        self, 
        conn_str: str, 
        sql_query: str, 
        out_csv: Path
    ) -> pd.DataFrame:
        """Execute SQL query and export to CSV in chunks"""
        import pyodbc
        
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        first_chunk = True
        
        try:
            with pyodbc.connect(conn_str) as conn:
                for i, chunk in enumerate(
                    pd.read_sql(
                        sql_query, 
                        conn, 
                        chunksize=self.config.sql.chunk_size
                    )
                ):
                    # Write chunk to CSV
                    chunk.to_csv(
                        out_csv,
                        mode='w' if first_chunk else 'a',
                        header=first_chunk,
                        index=False,
                        encoding='utf-8'
                    )
                    
                    if first_chunk:
                        self.logger.info(f"First chunk written ({len(chunk):,} rows)")
                    
                    first_chunk = False
                    
                    if (i + 1) % 10 == 0:
                        self.logger.debug(f"Processed {(i + 1) * self.config.sql.chunk_size:,} rows")
            
            # Read complete CSV
            return pd.read_csv(out_csv, dtype=str, low_memory=False)
            
        except Exception as e:
            # Clean up partial file on error
            if out_csv.exists():
                out_csv.unlink()
            raise
    
    def _apply_type_conversions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply appropriate type conversions to columns"""
        cols = self.config.columns
        
        # Numeric: Amount
        if cols.amount in df.columns:
            df[cols.amount] = pd.to_numeric(df[cols.amount], errors='coerce')
        
        # String: Donor ID (preserve leading zeros)
        if cols.donor_id in df.columns:
            df[cols.donor_id] = df[cols.donor_id].astype(str).str.strip()
        
        # Dates
        date_columns = [
            (cols.date, "donation date"),
            (cols.birthdate, "birth date")
        ]
        
        for col, description in date_columns:
            if col and col in df.columns:
                original_count = df[col].notna().sum()
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                converted_count = df[col].notna().sum()
                
                if original_count > converted_count:
                    lost = original_count - converted_count
                    self.logger.warning(
                        f"{description.title()}: {lost:,} values failed to parse"
                    )
        
        return df
    
    def _get_sql_export_path(self) -> Path:
        """Determine where SQL export should be saved"""
        input_path = Path(self.config.input_file)
        
        if input_path.suffix.lower() == '.csv':
            return input_path.resolve()
        else:
            return Path("FullDataTable.csv").resolve()
    
    def _build_sql_query(self) -> str:
        """Build SQL query for data extraction"""
        # You can parameterize this further if needed
        return """
            SELECT
                a.ReceiptSeq,
                a.ID,
                a.IDJoint,
                a.ReceiptNo,
                a.PledgeSeq,
                CAST(a.Date AS date) AS Date,
                a.Appeal,
                a.Fund,
                a.Amount,
                a.AnonymousFlag,
                a.RecognitionFlag,
                a.Comments,
                a.PostingNumber,
                a.ReceiptNumber,
                a.GLJournalSeq,
                a.PaymentMethodCode,
                a.HoldPledgeInstalmentFrequency,
                a.HoldPledgeInstalmentNumber,
                a.HoldPledgeInstalmentNextAmount,
                CAST(a.HoldPledgeInstalmentNextDate AS date) AS HoldPledgeInstalmentNextDate,
                b.Gender,
                CAST(b.BirthDate AS date) AS BirthDate,
                b.MaritalStatus,
                b.CurrentParent,
                b.FutureParent,
                b.PastParent,
                b.PastStudent,
                b.Suburb,
                b.State,
                b.PostCode,
                b.Education,
                b.OccupCode,
                b.OccupCodeDesc,
                b.OccupDesc,
                b.ReligionCode,
                c.Code as AppealCode,
                c.Description as AppealDescription,
                c.ActiveFlag as AppealActiveFlag,
                CAST(c.ModifiedDate AS date) AS AppealModifiedDate,
                d.Code as FundCode,
                d.Description as FundDescription,
                d.TaxDeductableFlag as FundTaxDeductableFlag,
                d.ActiveFlag as FundActiveFlag
            FROM DonorReceipts a
            LEFT JOIN DonorAttributes b ON a.ID = b.ID
            LEFT JOIN luAppeal c ON a.Appeal = c.Code
            LEFT JOIN luFund d ON a.Fund = d.Code
            ORDER BY a.ID ASC, a.Date ASC
        """


class DataValidator:
    """Validates loaded data meets minimum requirements"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def validate(self, df: pd.DataFrame) -> None:
        """
        Run all validation checks
        Raises ValueError if validation fails
        """
        self._check_required_columns(df)
        self._check_data_quality(df)
        self._check_date_ranges(df)
        self._check_relationship_flags(df)
    
    def _check_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure required columns exist"""
        required = [
            self.config.columns.date,
            self.config.columns.donor_id,
            self.config.columns.amount
        ]
        
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        self.logger.info("✓ All required columns present")
    
    def _check_data_quality(self, df: pd.DataFrame) -> None:
        """Check basic data quality metrics"""
        cols = self.config.columns
        
        # Check for empty dataframe
        if len(df) == 0:
            raise ValueError("DataFrame is empty")
        
        # Check amount column
        if cols.amount in df.columns:
            amount_col = df[cols.amount]
            null_count = amount_col.isna().sum()
            negative_count = (amount_col < 0).sum()
            zero_count = (amount_col == 0).sum()
            
            if null_count > 0:
                self.logger.warning(f"Amount: {null_count:,} null values")
            if negative_count > 0:
                self.logger.warning(f"Amount: {negative_count:,} negative values")
            if zero_count > 0:
                self.logger.warning(f"Amount: {zero_count:,} zero values")
        
        self.logger.info(f"✓ Data quality checks passed ({len(df):,} rows)")
    
    def _check_date_ranges(self, df: pd.DataFrame) -> None:
        """Validate date ranges are reasonable"""
        date_col = self.config.columns.date
        
        if date_col in df.columns:
            valid_dates = df[date_col].dropna()
            
            if len(valid_dates) == 0:
                raise ValueError("No valid dates found")
            
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            span_years = (max_date - min_date).days / 365.25
            
            self.logger.info(
                f"✓ Date range: {min_date.date()} to {max_date.date()} "
                f"({span_years:.1f} years)"
            )
            
            # Warn if date range seems unusual
            if span_years < 1:
                self.logger.warning("Date range is less than 1 year")
            elif span_years > 50:
                self.logger.warning("Date range exceeds 50 years - verify data")
    
    def _check_relationship_flags(self, df: pd.DataFrame) -> None:
        """Check relationship flag columns contain only 0/1"""
        cols = self.config.columns
        rel_cols = [
            cols.current_parent,
            cols.future_parent,
            cols.past_parent,
            cols.past_student
        ]
        
        issues = []
        
        for col in rel_cols:
            if col and col in df.columns:
                # Convert to numeric
                values = pd.to_numeric(df[col], errors='coerce')
                
                # Check for invalid values
                invalid = values[~values.isin([0, 1, np.nan])]
                
                if len(invalid) > 0:
                    issues.append(f"{col}: {len(invalid):,} invalid values")
        
        if issues:
            self.logger.warning(
                f"Relationship flag issues detected:\n" + 
                "\n".join(f"  - {issue}" for issue in issues)
            )
        else:
            self.logger.info("✓ Relationship flags validated")