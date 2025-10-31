#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Management Module
Centralizes all configuration parameters
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import pandas as pd

@dataclass
class ColumnConfig:
    """Data column configuration"""
    date: str = "Date"
    donor_id: str = "ID"
    amount: str = "Amount"
    gender: Optional[str] = "Gender"
    birthdate: Optional[str] = "BirthDate"
    suburb: Optional[str] = "Suburb"
    state: Optional[str] = "State"
    postcode: Optional[str] = "PostCode"
    occupation_desc: Optional[str] = "OccupDesc"
    religion: Optional[str] = "ReligionCode"
    education: Optional[str] = "Education"
    appeal: Optional[str] = "Appeal"
    fund: Optional[str] = "Fund"
    appeal_desc: Optional[str] = "AppealDescription"
    fund_desc: Optional[str] = "FundDescription"
    current_parent: Optional[str] = "CurrentParent"
    future_parent: Optional[str] = "FutureParent"
    past_parent: Optional[str] = "PastParent"
    past_student: Optional[str] = "PastStudent"


@dataclass
class AnalysisConfig:
    """Analysis parameters"""
    rfm_quantiles: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    year_cutoff: int = 2000
    redonate_window_days: int = 365
    panel_step: str = "2Q"
    min_panel_start: str = "2000-01-01"
    major_gift_threshold: float = 1000.0
    lapsed_recency_days: int = 730
    lapsed_min_monetary: float = 200.0
    new_donor_days: int = 90
    upgrade_multiplier: float = 1.5
    upgrade_min_frequency: int = 2
    train_test_split_ratio: float = 0.25
    random_state: int = 42
    min_samples_for_training: int = 200


@dataclass
class ModelConfig:
    """Machine learning model configurations"""
    lr_max_iter: int = 5000
    lr_solver: str = "lbfgs"
    rf_n_estimators: int = 300
    rf_max_depth: Optional[int] = None
    rf_n_jobs: int = -1
    lgbm_n_estimators: int = 300
    lgbm_learning_rate: float = 0.05
    time_rf_n_estimators: int = 500
    amount_rf_n_estimators: int = 200


@dataclass
class SQLConfig:
    """SQL connection configuration"""
    use_sql: bool = False
    server: Optional[str] = None
    database: Optional[str] = None
    trusted_connection: bool = True
    odbc_driver: str = "ODBC Driver 17 for SQL Server"
    user: Optional[str] = None
    password: Optional[str] = None
    chunk_size: int = 100000
    
    def get_connection_string(self) -> str:
        """Build SQL Server connection string"""
        if self.trusted_connection:
            return (
                f"DRIVER={{{self.odbc_driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                "Trusted_Connection=yes;"
                "Encrypt=yes;"
                "TrustServerCertificate=yes"
            )
        else:
            if not self.user or not self.password:
                raise ValueError("Username and password required for SQL authentication")
            return (
                f"DRIVER={{{self.odbc_driver}}};"
                f"SERVER={self.server};"
                f"DATABASE={self.database};"
                f"UID={self.user};"
                f"PWD={self.password};"
                "Encrypt=yes;"
                "TrustServerCertificate=yes"
            )


@dataclass
class Config:
    """Master configuration object"""
    # Paths
    input_file: str = ""
    out_dir: str = "outputs"
    excel_name: str = "Donations_DataModellingResult.xlsx"
    rfm_reference: Optional[str] = None
    
    # Sub-configurations
    columns: ColumnConfig = field(default_factory=ColumnConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    sql: SQLConfig = field(default_factory=SQLConfig)
    
    @property
    def output_path(self) -> Path:
        """Get output directory as Path object"""
        return Path(self.out_dir)
    
    def ensure_output_dir(self) -> None:
        """Create output directory if it doesn't exist"""
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def from_args(cls, args) -> 'Config':
        """Create Config from command line arguments"""
        config = cls()
        
        # ===== Basic Configuration =====
        config.input_file = args.input_file
        config.out_dir = args.out_dir
        config.excel_name = args.excel_name
        
        # ===== Analysis Configuration =====
        config.analysis.redonate_window_days = args.window_days
        config.analysis.major_gift_threshold = args.major_threshold
        config.analysis.year_cutoff = args.year_cutoff
        
        # ===== SQL Configuration =====
        config.sql.use_sql = args.use_sql.lower() == 'true'
        if config.sql.use_sql:
            config.sql.server = args.sql_server
            config.sql.database = args.sql_db
            config.sql.trusted_connection = args.sql_trusted.lower() == 'true'
            config.sql.odbc_driver = args.sql_odbc_driver
        
        return config