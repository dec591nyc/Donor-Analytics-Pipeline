#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Donations Analytics Pipeline - Main Entry Point
"""

import sys
import warnings
from pathlib import Path
import argparse
import traceback

import pandas as pd

# Import modules
from donations_pipeline.config import Config
from donations_pipeline.logger import setup_logger
from donations_pipeline.data_loader import DataLoader, DataValidator
from donations_pipeline.preprocessor import DataPreprocessor
from donations_pipeline.rfm_analysis import RFMAnalyzer, RelationshipAnalyzer
from donations_pipeline.descriptive_analytics import DescriptiveAnalyzer
from donations_pipeline.actionable_lists import ActionableListGenerator
from donations_pipeline.predictive_pipeline import PredictivePipeline
from donations_pipeline.output_manager import OutputManager
from donations_pipeline.email_sender import EmailSender
from donations_pipeline.env_loader import EnvLoader

warnings.filterwarnings('ignore', category=FutureWarning)


class DonationsPipeline:
    """Main orchestrator for the donations analytics pipeline"""
    
    def __init__(self, config: Config):
        self.config = config
        self.config.ensure_output_dir()
        
        # Initialize logger
        self.logger = setup_logger(self.config.output_path)
        self.logger.info("="*80)
        self.logger.info("Donations Analytics Pipeline - Starting")
        self.logger.info("="*80)
        
        # Initialize components
        self.data_loader = DataLoader(config, self.logger)
        self.validator = DataValidator(config, self.logger)
        self.preprocessor = DataPreprocessor(config, self.logger)
        self.output_manager = OutputManager(config, self.logger)
        self.email_sender = EmailSender(config, self.logger)
        
        # Track success/failure
        self.pipeline_success = False
        self.error_message = None
        self.output_dir = None
    
    def run(self) -> None:
        """Execute the complete analytics pipeline"""
        
        try:
            # Stage 1: Data Loading
            with self.logger.timed_operation("[1/6] Data Loading"):
                raw_df = self.data_loader.load_data()
                self.validator.validate(raw_df)
            
            # Stage 2: Preprocessing
            with self.logger.timed_operation("[2/6] Data Preprocessing"):
                processed_df = self.preprocessor.preprocess(raw_df)
                self._export_cleaned_data(processed_df)
            
            # Stage 3-6: Analysis
            self._run_analysis_for_slices(processed_df)
            
            # Save timing report
            timing_report_path = self.config.output_path / "pipeline_timing.txt"
            self.logger.save_timing_report(timing_report_path)
            
            # Mark as successful
            self.pipeline_success = True
            
            self.logger.info("="*80)
            self.logger.info("Pipeline completed successfully!")
            self.logger.info(f"Results saved to: {self.config.output_path}")
            self.logger.info("="*80)
            
        except Exception as e:
            self.pipeline_success = False
            self.error_message = f"{str(e)}\n\n{traceback.format_exc()}"
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        
        finally:
            pass
            # Stage 7: Send email notification (always execute)
            # self._send_notification()
    
    def _send_notification(self) -> None:
        """Send email notification about pipeline completion"""
        try:
            with self.logger.timed_operation("[7/6] Sending Email Notification"):
                # Read timing report
                timing_report_path = self.config.output_path / "pipeline_timing.txt"
                timing_report = ""
                
                if timing_report_path.exists():
                    with open(timing_report_path, 'r', encoding='utf-8') as f:
                        timing_report = f.read()
                
                # Determine output directory
                if self.output_dir is None:
                    self.output_dir = self.config.output_path / f"since_{self.config.analysis.year_cutoff}"
                
                # Send email
                self.email_sender.send_completion_email(
                    success=self.pipeline_success,
                    output_dir=self.output_dir,
                    timing_report=timing_report,
                    error_message=self.error_message
                )
        
        except Exception as e:
            self.logger.error(f"Failed to send notification email: {e}", exc_info=True)
    
    def _export_cleaned_data(self, df: pd.DataFrame) -> None:
        """Export cleaned dataset"""
        output_path = self.config.output_path / "cleaned_donations.csv"
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        self.logger.info(f"Cleaned data exported: {output_path}")
    
    def _run_analysis_for_slices(self, df: pd.DataFrame) -> None:
        """Run analysis for different time slices"""
        
        year_cutoff = self.config.analysis.year_cutoff
        df_recent = df[df['Year'] >= year_cutoff].copy()
        
        if not df_recent.empty:
            self.logger.info("\n" + "="*80)
            self.logger.info(f"Analyzing: Since {year_cutoff}")
            self.logger.info("="*80)
            
            self.output_dir = self.config.output_path / f"since_{year_cutoff}"
            self.output_dir.mkdir(exist_ok=True)
            
            self._run_analysis_slice(df_recent, self.output_dir, f"since_{year_cutoff}")
            
            # Stage 6: Package outputs
            with self.logger.timed_operation("[6/6] Packaging Results"):
                self.output_manager.bundle_to_excel(self.output_dir)
                self.output_manager.archive_intermediate_files(self.output_dir)
     
    def _run_analysis_slice(
        self, 
        df: pd.DataFrame, 
        output_dir: Path,
        slice_name: str
    ) -> None:
        """Run complete analysis for a data slice"""
        
        # Stage 3: Basic Data Analytics
        with self.logger.timed_operation(f"[3/6] Basic Data Analytics - {slice_name}"):
            desc_analyzer = DescriptiveAnalyzer(self.config, self.logger)
            desc_results = desc_analyzer.analyze(df)
            
            for filename, table in desc_results.items():
                table.to_csv(output_dir / filename, index=False, encoding='utf-8-sig')
            
            rfm_analyzer = RFMAnalyzer(self.config, self.logger)
            rfm_df = rfm_analyzer.compute_rfm(df)
            rfm_df.to_csv(output_dir / "rfm_table.csv", index=False, encoding='utf-8-sig')
            
            lapsed_df = rfm_analyzer.identify_lapsed_donors(rfm_df)
            lapsed_df.to_csv(output_dir / "lapsed_targets.csv", index=False, encoding='utf-8-sig')
        
            rel_analyzer = RelationshipAnalyzer(self.config, self.logger)
            rel_results = rel_analyzer.analyze_relationships(df)
            
            for filename, table in rel_results.items():
                table.to_csv(output_dir / f"{filename}.csv", index=False, encoding='utf-8-sig')
        
        # Stage 4: Predictive Analytics
        with self.logger.timed_operation(f"[4/6] Predictive Analytics - {slice_name}"):
            predictive = PredictivePipeline(self.config, self.logger)
            trained_models = predictive.run_training_only(df, output_dir)

        # Stage 5: Actionable Lists & Playbook
        with self.logger.timed_operation(f"[5/6] Actionable Lists - {slice_name}"):
            predictive.generate_playbook_from_models(df, output_dir, trained_models)
            
            list_generator = ActionableListGenerator(self.config, self.logger)
            actionable_lists = list_generator.generate_lists(df, rfm_df)
            
            for list_name, table in actionable_lists.items():
                table.to_csv(output_dir / f"{list_name}.csv", index=False, encoding='utf-8-sig')


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Donations Analytics Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--input_file",
        required=True,
        help="Path to input CSV file"
    )
    
    # Output settings
    parser.add_argument(
        "--out_dir",
        default="outputs",
        help="Output directory"
    )
    
    parser.add_argument(
        "--excel_name",
        default="Donations_DataModellingResult.xlsx",
        help="Output Excel filename"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--rfm_reference",
        default=None,
        help="RFM reference date (YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--window_days",
        type=int,
        default=365,
        help="Prediction window in days"
    )
    
    parser.add_argument(
        "--major_threshold",
        type=float,
        default=1000.0,
        help="Major gift threshold amount"
    )
    
    parser.add_argument(
        "--year_cutoff",
        type=int,
        default=2000,
        help="Analysis year cutoff"
    )
    
    # SQL parameters
    parser.add_argument(
        "--use_sql",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Extract data from SQL Server"
    )
    
    parser.add_argument(
        "--sql_server",
        default=None,
        help="SQL Server address"
    )
    
    parser.add_argument(
        "--sql_db",
        default=None,
        help="SQL Server database name"
    )
    
    parser.add_argument(
        "--sql_trusted",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Use Windows Authentication"
    )
    
    parser.add_argument(
        "--sql_odbc_driver",
        default="ODBC Driver 17 for SQL Server",
        help="ODBC driver name"
    )
    
    # Email parameters
    parser.add_argument(
        "--email_enabled",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Enable email notifications"
    )
    
    parser.add_argument(
        "--email_to",
        default="",
        help="Comma-separated TO addresses"
    )
    
    parser.add_argument(
        "--email_cc",
        default=None,
        nargs='*',  # Accepts zero or more arguments
        help="Comma-separated CC addresses (optional)"
    )
    
    parser.add_argument(
        "--email_from",
        default="",
        help="From email address"
    )
    
    parser.add_argument(
        "--smtp_server",
        default="smtp.office365.com",
        help="SMTP server address"
    )
    
    parser.add_argument(
        "--smtp_port",
        type=int,
        default=587,
        help="SMTP server port"
    )
    
    parser.add_argument(
        "--smtp_username",
        default="",
        help="SMTP username"
    )
    
    # Note: Password should NOT be in args, only in .env
    
    return parser.parse_args()


def main():
    """Main entry point"""
    try:
        # Load environment variables first
        EnvLoader.load_env_file()
        
        # Parse arguments
        args = parse_arguments()
        
        # Create configuration
        config = Config.from_args(args)
        
        # Run pipeline
        pipeline = DonationsPipeline(config)
        pipeline.run()
        
    except SystemExit as e:
        if e.code != 0:
            print(f"\nError: Argument parsing failed. Exit code: {e.code}")
            print("Please check your arguments and try again.")
        sys.exit(e.code)
    
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()