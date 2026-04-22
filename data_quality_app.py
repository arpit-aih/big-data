import os, datetime, json, io
import pandas as pd
from data_cleaner import DataCleaner


global_state = {
    'original_data': None,
    'cleaned_data': None,
    'data_cleaner': None,
    'cleaning_report': None,
    'data_quality_issues': None,
    'file_name': None,
    'llm': None,
    'current_step': 1,
    'data_analysis_report': None,
    'quality_report': None
}

def generate_data_analysis_report(df, focus="comprehensive"):
    """Generate a comprehensive data analysis report for non-technical users"""
    
    from data_analysis_agents import generate_agent_report
    return generate_agent_report(df, focus)


def clean_data(df, cleaning_options=None):
    """Clean data based on specified options"""
    if global_state['data_cleaner'] is None:
        global_state['data_cleaner'] = DataCleaner()
    
    if cleaning_options is None:
        
        cleaning_options = {
            "remove_duplicates": True,
            "handle_missing": True,
            "missing_strategy": "auto",
            "handle_outliers": True,
            "outlier_strategy": "auto",
            "outlier_columns": [],
            "fix_data_types": True,
            "use_llm_cleaning": True
        }
        
        
        if global_state['data_quality_issues'] and "outliers" in global_state['data_quality_issues']:
            cleaning_options["outlier_columns"] = list(global_state['data_quality_issues']["outliers"].keys())
    
    print("Cleaning data...")
    cleaned_df, report = global_state['data_cleaner'].clean_data(df, cleaning_options)
    
    if cleaned_df is not None:
        
        global_state['cleaned_data'] = cleaned_df
        global_state['cleaning_report'] = report
        
        
        print("\nCleaning Report:")
        print(f"Rows: {report['rows_before']} → {report['rows_after']} ({report['rows_removed']} removed)")
        print(f"Columns: {report['columns_before']} → {report['columns_after']} ({report['columns_removed']} removed)")
        
        if report["changes"]:
            print("\nChanges Made:")
            for change in report["changes"]:
                print(f"- {change}")
        
        return cleaned_df, report
    else:
        print("Error during data cleaning.")
        return None, None


def export_data(df, formats=None, base_filename=None):
    """Export data to specified formats"""
    if df is None:
        print("No data to export.")
        return False
    
    if formats is None:
        formats = ["csv", "excel", "parquet"]
    
    if base_filename is None:
        if global_state['file_name']:
            base_filename = os.path.splitext(global_state['file_name'])[0]
            base_filename = f"cleaned_{base_filename}"
        else:
            base_filename = "cleaned_data"
    
    try:
        
        export_dir = os.path.join(os.getcwd(), "exports", base_filename)
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = {}
        
        
        if "csv" in formats:
            print("Exporting CSV file...")
            csv_path = os.path.join(export_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            exported_files["csv"] = csv_path
        
        
        if "excel" in formats:
            print("Exporting Excel file...")
            excel_path = os.path.join(export_dir, f"{base_filename}.xlsx")
            df.to_excel(excel_path, index=False)
            exported_files["excel"] = excel_path
        
        
        if "parquet" in formats:
            print("Exporting Parquet file...")
            parquet_path = os.path.join(export_dir, f"{base_filename}.parquet")
            df.to_parquet(parquet_path, index=False)
            exported_files["parquet"] = parquet_path
        
        
        metadata = {
            "original_file": global_state['file_name'],
            "export_timestamp": datetime.datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "formats_exported": formats,
            "export_directory": export_dir
        }
        
        with open(os.path.join(export_dir, "export_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        exported_files["metadata"] = os.path.join(export_dir, "export_metadata.json")
        
        print(f"\nAll formats successfully exported to: {export_dir}")
        for fmt, path in exported_files.items():
            print(f"- {fmt.upper()}: {path}")
        
        return exported_files
    
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return False


def create_zip_archive(df, base_filename=None):
    """Create a ZIP archive with all export formats"""
    if df is None:
        print("No data to export.")
        return False
    
    if base_filename is None:
        if global_state['file_name']:
            base_filename = os.path.splitext(global_state['file_name'])[0]
            base_filename = f"cleaned_{base_filename}"
        else:
            base_filename = "cleaned_data"
    
    try:
        import zipfile
        import tempfile
        
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as temp_zip:
            zip_path = temp_zip.name
        
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            zipf.writestr(f"{base_filename}.csv", csv_data)
            
            
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer) as writer:
                df.to_excel(writer, index=False)
            zipf.writestr(f"{base_filename}.xlsx", excel_buffer.getvalue())
            
            
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            zipf.writestr(f"{base_filename}.parquet", parquet_buffer.getvalue())
            
            readme_content = f"""

            Original file: {global_state['file_name']}
            Export date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            Rows: {len(df)}
            Columns: {len(df.columns)}

            This archive contains the cleaned data in multiple formats:
            - CSV: {base_filename}.csv
            - Excel: {base_filename}.xlsx
            - Parquet: {base_filename}.parquet

            Generated by Data Quality & Cleaning Assistant
            """
            zipf.writestr("README.md", readme_content)
        
        print(f"ZIP archive created at: {zip_path}")
        return zip_path
    
    except Exception as e:
        print(f"Error creating ZIP archive: {str(e)}")
        return False


def export_data(df, formats=None, base_filename=None):
    """Export data to specified formats"""
    if df is None:
        print("No data to export.")
        return False
    
    if formats is None:
        formats = ["csv", "excel", "parquet"]
    
    if base_filename is None:
        if global_state['file_name']:
            base_filename = os.path.splitext(global_state['file_name'])[0]
            base_filename = f"cleaned_{base_filename}"
        else:
            base_filename = "cleaned_data"
    
    try:
        
        export_dir = os.path.join(os.getcwd(), "exports", base_filename)
        os.makedirs(export_dir, exist_ok=True)
        
        exported_files = {}
        
        
        if "csv" in formats:
            print("Exporting CSV file...")
            csv_path = os.path.join(export_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            exported_files["csv"] = csv_path
        
        
        if "excel" in formats:
            print("Exporting Excel file...")
            excel_path = os.path.join(export_dir, f"{base_filename}.xlsx")
            df.to_excel(excel_path, index=False)
            exported_files["excel"] = excel_path
        
        
        if "parquet" in formats:
            print("Exporting Parquet file...")
            parquet_path = os.path.join(export_dir, f"{base_filename}.parquet")
            df.to_parquet(parquet_path, index=False)
            exported_files["parquet"] = parquet_path
        
        
        metadata = {
            "original_file": global_state['file_name'],
            "export_timestamp": datetime.datetime.now().isoformat(),
            "row_count": len(df),
            "column_count": len(df.columns),
            "formats_exported": formats,
            "export_directory": export_dir
        }
        
        with open(os.path.join(export_dir, "export_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        exported_files["metadata"] = os.path.join(export_dir, "export_metadata.json")
        
        print(f"\nAll formats successfully exported to: {export_dir}")
        for fmt, path in exported_files.items():
            print(f"- {fmt.upper()}: {path}")
        
        return exported_files
    
    except Exception as e:
        print(f"Error during export: {str(e)}")
        return False


def generate_quality_report(df):
    """Generate a comprehensive data quality report using SmartDataframe"""
    
    from data_analysis_agents import generate_quality_report
    return generate_quality_report(df)


def analyze_data_quality(df):
    """Analyze data quality and return issues"""
    if global_state['data_cleaner'] is None:
        global_state['data_cleaner'] = DataCleaner()
    
    print("Analyzing data quality...")
    quality_issues = global_state['data_cleaner'].check_data_quality(df)
    global_state['data_quality_issues'] = quality_issues
    
    print("\nData Quality Overview:")
    
    missing_values = quality_issues["missing_values"]
    if missing_values:
        total_missing = sum(data["count"] for data in missing_values.values())
        total_cells = len(df) * len(df.columns)
        missing_percent = round((total_missing / total_cells) * 100, 2)
        print(f"\nMissing Values: {missing_percent}% of all cells")
        
        
        quality_issues["total_missing_percent"] = missing_percent
        
        missing_df = pd.DataFrame(
            [(col, data["count"], data["percent"]) for col, data in missing_values.items()],
            columns=["Column", "Missing Count", "Missing %"]
        ).sort_values(by="Missing Count", ascending=False).head(5)
        
        print(missing_df)
    else:
        print("\nMissing Values: 0% - No missing values")
        quality_issues["total_missing_percent"] = 0.0
    
    
    outliers = quality_issues["outliers"]
    if outliers:
        total_outliers = sum(data["count"] for data in outliers.values())
        numeric_columns = df.select_dtypes(include=['number']).columns
        total_numeric_cells = len(df) * len(numeric_columns)
        outlier_percent = round((total_outliers / total_numeric_cells) * 100, 2) if total_numeric_cells > 0 else 0
        
        
        quality_issues["total_outlier_percent"] = outlier_percent
        
        print(f"\nOutliers: {outlier_percent}% of numeric values")
        
        
        outlier_df = pd.DataFrame(
            [(col, data["count"], data["percent"]) for col, data in outliers.items()],
            columns=["Column", "Outlier Count", "Outlier %"]
        ).sort_values(by="Outlier Count", ascending=False).head(5)
        
        print(outlier_df)
    else:
        print("\nOutliers: 0% - No outliers detected")
        quality_issues["total_outlier_percent"] = 0.0
    
    
    dup_count = quality_issues["duplicates"]["count"]
    dup_percent = quality_issues["duplicates"]["percent"]
    
    
    quality_issues["total_duplicate_percent"] = dup_percent
    
    print(f"\nDuplicate Rows: {dup_count:,} rows ({dup_percent}%)")
    
    
    type_issues = quality_issues["data_type_issues"]
    if type_issues:
        print(f"\nData Type Issues: {len(type_issues)} issues detected")
        
        
        if isinstance(type_issues, list):
            type_issues_df = pd.DataFrame(type_issues)
            print(type_issues_df)
    else:
        print("\nData Type Issues: 0 - No data type issues")
    
    return quality_issues