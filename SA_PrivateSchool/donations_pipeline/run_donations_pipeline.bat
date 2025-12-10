@echo off
REM ===========================================================
REM  SA Private School – Donations Analytics Pipeline (v5)
REM  Author: Yichi Nien (Ethan)
REM  Date:   14-10-2025
REM ===========================================================

REM === Change working directory to project root ===
cd /d "C:\Users\sysuni01\Documents\Python"

REM === Run the pipeline (module form) ===
python -m donations_pipeline.main ^
  --use_sql true ^
  --sql_server "SP****02" ^
  --sql_db "UniSA_DonorAnalysis" ^
  --input_file "C:\Users\sysuni01\Documents\Python\Data\Input\FullDataTable.csv" ^
  --out_dir "C:\Users\sysuni01\Documents\Python\Data\Output\IDR_v1" ^
  --excel_name "Donations_DataModellingResult.xlsx" ^
  --window_days 365 ^
  --year_cutoff 2000 ^
  --major_threshold 1000

REM === Deactivate environment ===
deactivate

REM === Show completion message ===
echo ===========================================================
echo  Pipeline run completed. Check output folder:
echo  C:\Users\sysuni01\Documents\Python\Data\Output\IDR_v1
echo ===========================================================
pause
