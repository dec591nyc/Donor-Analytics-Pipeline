#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Output Management Module
Handles results packaging, Excel bundling, and archiving
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd

from .config import Config
from .logger import PipelineLogger


class OutputManager:
    """Manages output packaging and archiving"""
    
    def __init__(self, config: Config, logger: PipelineLogger):
        self.config = config
        self.logger = logger
    
    def bundle_to_excel(self, output_dir: Path) -> None:
        """
        Bundle all CSV/JSON files into a single Excel workbook
        
        Args:
            output_dir: Directory containing output files
        """
        excel_name = self.config.excel_name
        
        # Auto-version if file exists
        excel_path = self._get_versioned_path(output_dir / excel_name)
        
        self.logger.info(f"Bundling results to Excel: {excel_path.name}")
        
        try:
            # Collect files
            csv_files = self._collect_csv_files(output_dir)
            json_files = self._collect_json_files(output_dir)
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Write CSVs
                toc_entries = self._write_csvs_to_excel(csv_files, writer)
                
                # Write JSONs
                json_entries = self._write_jsons_to_excel(json_files, writer)
                toc_entries.extend(json_entries)
                
                # Write table of contents
                self._write_toc(toc_entries, writer)
            
            self.logger.info(
                f"Excel bundle created: {len(toc_entries)} sheets, "
                f"{excel_path.stat().st_size / 1024 / 1024:.1f} MB"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create Excel bundle: {e}", exc_info=True)
    
    def archive_intermediate_files(self, output_dir: Path) -> None:
        """
        Move CSV/JSON files to dated archive folder
        
        Args:
            output_dir: Directory containing files to archive
        """
        # Create archive directory
        archive_base = output_dir / f"archive_{datetime.now().strftime('%Y%m%d')}"
        archive_dir = self._get_unique_dir(archive_base)
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        moved_count = 0

        # Move model files
        model_dir = archive_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        for file in output_dir.glob("*.pkl"):
            try:
                shutil.move(str(file), str(model_dir / file.name))
            except Exception as e:
                self.logger.warning(f"Could not move {file.name}: {e}")

        # Move CSV and JSON files
        for ext in ['.csv', '.json']:
            for file_path in output_dir.glob(f"*{ext}"):
                if file_path.is_file():
                    try:
                        shutil.move(str(file_path), str(archive_dir / file_path.name))
                        moved_count += 1
                    except Exception as e:
                        self.logger.warning(f"Could not move {file_path.name}: {e}")
        
        if moved_count > 0:
            self.logger.info(f"Archived {moved_count} files to: {archive_dir.name}")
    
    def _get_versioned_path(self, base_path: Path) -> Path:
        """
        Get path with version suffix if file exists
        
        Example: report.xlsx -> report_v2.xlsx -> report_v3.xlsx
        """
        if not base_path.exists():
            return base_path
        
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        
        version = 1
        while True:
            version += 1
            new_path = parent / f"{stem}_v{version}{suffix}"
            if not new_path.exists():
                return new_path
    
    def _get_unique_dir(self, base_dir: Path) -> Path:
        """Get unique directory name with numeric suffix if needed"""
        if not base_dir.exists():
            return base_dir
        
        version = 1
        while True:
            version += 1
            new_dir = Path(str(base_dir) + f"_{version}")
            if not new_dir.exists():
                return new_dir
    
    def _collect_csv_files(self, output_dir: Path) -> List[Path]:
        """
        Collect CSV files in preferred order
        
        Priority files appear first, then alphabetical
        """
        all_csvs = list(output_dir.glob("*.csv"))
        
        # Preferred order
        priority = [
            "prospecting_playbook.csv",
            "rfm_table.csv",
            "lapsed_targets.csv",
            "high_potential_sleeper.csv",
            "upgrade_candidates.csv",
            "new_donors_90d.csv",
            "dist_suburb.csv",
            "relationship_summary.csv",
            "relationship_agegroup.csv",
            "relationship_trend.csv",
            "trend_year.csv",
            "trend_year_major.csv",
            "trend_month.csv",
            "dist_gender.csv",
            "dist_agegroup.csv",
            "dist_occupation.csv",
            "dist_religion.csv",
            "appeal_performance.csv",
            "fund_performance.csv",
            "lapsed_12m.csv",
            "propensity_model_comparison.csv",
            "major_gift_model_comparison.csv",
        ]
        
        # Sort: priority first, then alphabetical
        name_to_path = {p.name: p for p in all_csvs}
        
        ordered = []
        for name in priority:
            if name in name_to_path:
                ordered.append(name_to_path[name])
        
        # Add remaining files alphabetically
        remaining = sorted([p for p in all_csvs if p.name not in priority])
        ordered.extend(remaining)
        
        return ordered
    
    def _collect_json_files(self, output_dir: Path) -> List[Path]:
        """Collect all JSON files"""
        return sorted(output_dir.glob("*.json"))
    
    def _write_csvs_to_excel(
        self,
        csv_files: List[Path],
        writer: pd.ExcelWriter
    ) -> List[tuple]:
        """
        Write CSV files to Excel sheets
        
        Returns:
            List of (filename, sheet_name, note) tuples for TOC
        """
        toc_entries = []
        used_sheet_names = set()
        
        for csv_path in csv_files:
            try:
                # Read CSV
                df = pd.read_csv(csv_path)
                
                # Generate unique sheet name (max 31 chars)
                sheet_name = self._generate_sheet_name(
                    csv_path.stem,
                    used_sheet_names
                )
                used_sheet_names.add(sheet_name)
                
                # Write to Excel
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Record for TOC
                toc_entries.append((
                    csv_path.name,
                    sheet_name,
                    f"{len(df):,} rows × {len(df.columns)} cols"
                ))
                
            except Exception as e:
                self.logger.warning(f"Could not export {csv_path.name}: {e}")
        
        return toc_entries
    
    def _write_jsons_to_excel(
        self,
        json_files: List[Path],
        writer: pd.ExcelWriter
    ) -> List[tuple]:
        """
        Flatten and write JSON files to Excel
        
        Returns:
            List of (filename, sheet_name, note) tuples for TOC
        """
        if not json_files:
            return []
        
        # Flatten all JSONs into one dataframe
        all_data = []
        
        for json_path in json_files:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Flatten nested structure
                flat_data = pd.json_normalize(data)
                flat_data.insert(0, 'source_file', json_path.name)
                
                all_data.append(flat_data)
                
            except Exception as e:
                self.logger.warning(f"Could not read {json_path.name}: {e}")
        
        if not all_data:
            return []
        
        # Combine all JSONs
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Write to Excel
        sheet_name = "json_metrics"
        combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return [(
            "*.json (combined)",
            sheet_name,
            f"{len(combined_df):,} records from {len(json_files)} files"
        )]
    
    def _write_toc(
        self,
        entries: List[tuple],
        writer: pd.ExcelWriter
    ) -> None:
        """
        Write table of contents sheet
        
        Args:
            entries: List of (filename, sheet_name, note) tuples
            writer: Excel writer object
        """
        toc_df = pd.DataFrame(
            entries,
            columns=['Source File', 'Sheet Name', 'Description']
        )
        
        toc_df.to_excel(writer, sheet_name="Contents", index=False)
    
    def _generate_sheet_name(
        self,
        base_name: str,
        used_names: set,
        max_length: int = 31
    ) -> str:
        """
        Generate unique Excel sheet name
        
        Excel sheet names must be:
        - ≤ 31 characters
        - Unique within workbook
        
        Args:
            base_name: Desired sheet name
            used_names: Set of already used names
            max_length: Maximum allowed length
        
        Returns:
            Valid, unique sheet name
        """
        # Truncate to max length
        name = base_name[:max_length]
        
        if name not in used_names:
            return name
        
        # Add numeric suffix if duplicate
        suffix = 1
        while True:
            suffix += 1
            # Reserve space for suffix
            truncated = base_name[:max_length - len(f"_{suffix}")]
            candidate = f"{truncated}_{suffix}"
            
            if candidate not in used_names:
                return candidate
    
    def create_execution_summary(
        self,
        output_dir: Path,
        start_time: datetime,
        end_time: datetime,
        record_count: int
    ) -> None:
        """
        Create execution summary report
        
        Args:
            output_dir: Output directory
            start_time: Pipeline start time
            end_time: Pipeline end time
            record_count: Number of records processed
        """
        duration = (end_time - start_time).total_seconds()
        
        summary = {
            'execution_date': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': duration,
            'duration_formatted': self._format_duration(duration),
            'records_processed': record_count,
            'records_per_second': record_count / duration if duration > 0 else 0,
            'output_directory': str(output_dir),
            'config': {
                'year_cutoff': self.config.analysis.year_cutoff,
                'major_gift_threshold': self.config.analysis.major_gift_threshold,
                'window_days': self.config.analysis.redonate_window_days
            }
        }
        
        # Write to JSON
        summary_path = output_dir / "execution_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Execution summary saved: {summary_path}")
    
    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"