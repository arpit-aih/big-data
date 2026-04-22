import warnings
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from core.llm_config import get_llm
from typing import Dict, Any, Tuple, List


warnings.filterwarnings('ignore', message='`PandasAI` .* is deprecated')


def safe_is_valid(obj):
    """Safely check if an object is valid (not None and has data)"""
    if obj is None:
        return False
    
    try:
        
        if hasattr(obj, 'dataframe'):
            return True
        return False
    except:
        return False


def release_resources():
    """Placeholder to release file handles without forcing gc.collect() overhead"""
    pass


class DataCleaner:
    """Class to handle data cleaning operations using PandasAI"""
    
    def __init__(self):
        """Initialize the DataCleaner with PandasAI"""
        try:
            self.llm = get_llm()
            if not self.llm:
                st.error("Azure OpenAI configuration is incomplete. Check your environment variables.")
        except Exception as e:
            st.error(f"Error initializing LLM for data cleaning: {str(e)}")
            self.llm = None
    
    def _create_smart_dataframe(self, df):
        """Create a SmartDataframe from a pandas DataFrame"""
        if self.llm is None:
            return None

        try:
            release_resources()
            
            df_copy = df.copy(deep=False)
            
            return SmartDataframe(df_copy, config={"llm": self.llm, "verbose": True, "enforce_privacy": True, "enable_cache": True})
        except Exception as e:
            st.warning(f"Error creating SmartDataframe: {str(e)}")
            return None
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        
        quality_issues = {
            "missing_values": {},
            "duplicates": {"count": 0, "percent": 0},
            "outliers": {},
            "inconsistent_data": [],
            "data_type_issues": []
        }
        
        
        missing_count = df.isnull().sum()
        missing_percent = (missing_count / len(df) * 100).round(2)
        quality_issues["missing_values"] = {
            col: {"count": int(count), "percent": float(pct)} 
            for col, count, pct in zip(df.columns, missing_count, missing_percent) 
            if count > 0
        }
        
        duplicates = df.duplicated().sum()
        quality_issues["duplicates"]["count"] = int(duplicates)
        quality_issues["duplicates"]["percent"] = float(round((duplicates / len(df)) * 100, 2)) if len(df) > 0 else 0
        
        
        for col in df.select_dtypes(include=['number']).columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                quality_issues["outliers"][col] = {
                    "count": int(outliers),
                    "percent": float(round((outliers / df[col].count()) * 100, 2)),
                    "lower_bound": float(lower_bound),
                    "upper_bound": float(upper_bound)
                }
        
        
        smart_df = self._create_smart_dataframe(df)
        if safe_is_valid(smart_df):
            try:
                prompt = """
                Analyze this DataFrame for data type issues. Look for:
                1. Text columns that should be numerical
                2. Text columns that should be dates
                3. Inconsistent formats (like different date formats)
                4. Mixed data types in the same column
                
                IMPORTANT: Be very careful with your code. Do not use string methods on non-string columns.
                Always check column data types before applying operations.
                If a column might have mixed types, use safe type checking with isinstance() for each value.
                
                Return the result as a list of dictionaries with this format:
                [
                    {
                        "column": "column_name",
                        "issue": "brief description of the issue",
                        "recommendation": "recommended data type or fix"
                    }
                ]
                
                If there are no issues, return an empty list.
                """
                
                
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        type_issues = smart_df.chat(prompt)
                        break  
                    except (AttributeError, ValueError, TypeError, NameError) as e:
                        
                        if attempt == max_retries - 1:
                            st.warning(f"Data type analysis failed after {max_retries} attempts: {str(e)}")
                            quality_issues["data_type_issues"] = [{"column": "general", "issue": "Analysis failed", "recommendation": "Manual review needed"}]
                            return quality_issues
                        else:
                            
                            prompt = """
                            Analyze this DataFrame and return a list of columns that might have data type issues.
                            Return a very simple result as a list of dictionaries:
                            [{"column": "column_name", "issue": "issue description", "recommendation": "fix recommendation"}]
                            """
                    except Exception as e:
                        quality_issues["data_type_issues"] = [{"column": "general", "issue": f"Analysis error: {str(e)}", "recommendation": "Manual review needed"}]
                        return quality_issues
                    
                
                if isinstance(type_issues, list):
                    quality_issues["data_type_issues"] = type_issues
                elif isinstance(type_issues, str):
                    
                    import json
                    try:
                        if "```" in type_issues:
                            code_block = type_issues.split("```")[1]
                            if code_block.startswith("json"):
                                code_block = code_block[4:]
                            quality_issues["data_type_issues"] = json.loads(code_block)
                        else:
                            quality_issues["data_type_issues"] = json.loads(type_issues)
                    except (json.JSONDecodeError, IndexError):
                        quality_issues["data_type_issues"] = [{"column": "general", "issue": type_issues, "recommendation": "Review manually"}]
                    except Exception as e:
                        quality_issues["data_type_issues"] = [{"column": "general", "issue": f"Parsing error: {str(e)}", "recommendation": "Review manually"}]
            except Exception as e:
                st.warning(f"Could not analyze data type issues with LLM: {str(e)}")
                quality_issues["data_type_issues"] = [{"column": "general", "issue": "LLM analysis failed", "recommendation": "Manual inspection required"}]
        
        return quality_issues
    
    def clean_data(self, df: pd.DataFrame, options: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean the data based on specified options
        
        Args:
            df (pd.DataFrame): Input dataframe to clean
            options (Dict[str, Any]): Cleaning options
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and cleaning report
        """
        
        df = df.copy(deep=False)
        
        original_df = df.copy(deep=False)
        cleaning_report = {
            "rows_before": len(df),
            "columns_before": len(df.columns),
            "changes": []
        }
        
        
        if options.get("remove_duplicates", False):
            dup_count_before = df.duplicated().sum()
            df = df.drop_duplicates(keep='first')
            dup_removed = dup_count_before - df.duplicated().sum()
            if dup_removed > 0:
                cleaning_report["changes"].append(f"Removed {dup_removed} duplicate rows")
        
        
        if options.get("handle_missing", False):
            missing_strategy = options.get("missing_strategy", "auto")
            
            if missing_strategy == "drop_rows":
                rows_before = len(df)
                df = df.dropna()
                rows_removed = rows_before - len(df)
                if rows_removed > 0:
                    cleaning_report["changes"].append(f"Removed {rows_removed} rows with missing values")
            
            elif missing_strategy == "drop_columns":
                threshold = options.get("missing_threshold", 50) / 100
                cols_before = len(df.columns)
                missing_percent = df.isnull().mean()
                cols_to_drop = missing_percent[missing_percent > threshold].index.tolist()
                df = df.drop(columns=cols_to_drop)
                if len(cols_to_drop) > 0:
                    cleaning_report["changes"].append(f"Dropped {len(cols_to_drop)} columns with more than {threshold*100}% missing values: {', '.join(cols_to_drop)}")
            
            elif missing_strategy == "auto" or missing_strategy in ["mean", "median", "mode", "constant"]:
                
                for col in df.columns:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        
                        if missing_strategy == "auto":
                            if pd.api.types.is_numeric_dtype(df[col]):
                                
                                if col in options.get("outlier_columns", []):
                                    df[col] = df[col].fillna(df[col].median())
                                    method = "median"
                                else:
                                    df[col] = df[col].fillna(df[col].mean())
                                    method = "mean"
                            elif pd.api.types.is_datetime64_dtype(df[col]):
                                
                                try:
                                    df[col] = df[col].interpolate(method='time').ffill().bfill()
                                    method = "interpolate"
                                except (NotImplementedError, ValueError, TypeError):
                                    
                                    df[col] = df[col].ffill().bfill()
                                    method = "forward/backward fill"
                                except Exception as e:
                                    
                                    most_frequent = df[col].mode()[0] if not df[col].mode().empty else pd.NaT
                                    df[col] = df[col].fillna(most_frequent)
                                    method = "mode (interpolation failed)"
                            else:
                                
                                most_frequent = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                                df[col] = df[col].fillna(most_frequent)
                                method = "mode"
                        else:
                            
                            if missing_strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].mean())
                                method = "mean"
                            elif missing_strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
                                df[col] = df[col].fillna(df[col].median())
                                method = "median"
                            elif missing_strategy == "mode":
                                most_frequent = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                                df[col] = df[col].fillna(most_frequent)
                                method = "mode"
                            elif missing_strategy == "constant":
                                fill_value = options.get("constant_value", 0)
                                df[col] = df[col].fillna(fill_value)
                                method = f"constant ({fill_value})"
                            else:
                                
                                most_frequent = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
                                df[col] = df[col].fillna(most_frequent)
                                method = "mode"
                        
                        cleaning_report["changes"].append(f"Filled {missing_count} missing values in '{col}' using {method}")
        
        
        if options.get("handle_outliers", False):
            outlier_strategy = options.get("outlier_strategy", "auto")
            outlier_columns = options.get("outlier_columns", [])
            
            
            if not outlier_columns:
                outlier_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            for col in outlier_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    outliers = ((df[col] < lower_bound) | (df[col] > upper_bound))
                    outlier_count = outliers.sum()
                    
                    if outlier_count > 0:
                        if outlier_strategy == "remove":
                            rows_before = len(df)
                            df = df[~outliers]
                            rows_removed = rows_before - len(df)
                            cleaning_report["changes"].append(f"Removed {rows_removed} rows with outliers in '{col}'")
                        
                        elif outlier_strategy == "clip":
                            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                            cleaning_report["changes"].append(f"Clipped {outlier_count} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                        
                        elif outlier_strategy == "auto":
                            
                            
                            if outlier_count / len(df) > 0.05:
                                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                                cleaning_report["changes"].append(f"Clipped {outlier_count} outliers in '{col}' to range [{lower_bound:.2f}, {upper_bound:.2f}]")
                            else:
                                rows_before = len(df)
                                df = df[~outliers]
                                rows_removed = rows_before - len(df)
                                cleaning_report["changes"].append(f"Removed {rows_removed} rows with outliers in '{col}'")
        
        
        if options.get("fix_data_types", False) and self.llm:
            try:
                
                smart_df = self._create_smart_dataframe(df)
                if safe_is_valid(smart_df):
                    prompt = """
                    Fix the data types in this DataFrame. Specifically:
                    1. Convert string columns to appropriate numerical types where possible
                    2. Convert string columns to datetime types where they contain dates or times
                    3. Return the modified DataFrame with the correct data types
                    
                    Make sure to:
                    - Preserve the data values when converting types
                    - Handle type conversion errors gracefully
                    - Return a valid DataFrame
                    - IMPORTANT: Use safe type checking and conversion to avoid errors
                    - Avoid assuming string methods are available on all columns
                    
                    The returned DataFrame should have the same columns and rows as the input, but with corrected data types.
                    """
                    
                    
                    max_retries = 3
                    fixed_df = None
                    
                    for attempt in range(max_retries):
                        try:
                            fixed_df = smart_df.chat(prompt)
                            
                            
                            if isinstance(fixed_df, pd.DataFrame) and fixed_df.shape == df.shape:
                                
                                changed_dtypes = []
                                for col in fixed_df.columns:
                                    if df[col].dtype != fixed_df[col].dtype:
                                        changed_dtypes.append(f"'{col}': {df[col].dtype} → {fixed_df[col].dtype}")
                                
                                if changed_dtypes:
                                    df = fixed_df
                                    cleaning_report["changes"].append(f"Fixed data types: {', '.join(changed_dtypes)}")
                                break  
                            elif isinstance(fixed_df, pd.DataFrame):
                                
                                st.warning(f"Data type conversion returned mismatched shape: {fixed_df.shape} vs {df.shape}. Trying again...")
                            else:
                                
                                st.warning(f"Data type conversion didn't return a DataFrame. Trying again...")
                                
                        except (AttributeError, ValueError, TypeError, NameError) as e:
                            
                            if attempt == max_retries - 1:
                                st.warning(f"Data type conversion failed after {max_retries} attempts: {str(e)}")
                            else:
                                
                                prompt = """
                                Do a very simple data type conversion on this DataFrame:
                                1. Only convert obvious text columns to numbers where ALL values are numeric
                                2. Only convert obvious text columns to dates where ALL values are dates
                                3. Do not use complex operations, just simple pd.to_numeric and pd.to_datetime
                                4. Return the modified DataFrame
                                """
                        except Exception as e:
                            st.warning(f"Unexpected error in data type conversion: {str(e)}")
                            break  
            except Exception as e:
                st.warning(f"Could not fix data types using LLM: {str(e)}")
        
        
        if options.get("use_llm_cleaning", False) and self.llm:
            try:
                
                smart_df = self._create_smart_dataframe(df)
                if safe_is_valid(smart_df):
                    
                    analysis_prompt = """
                    Analyze this dataframe for cleanliness issues that have not yet been fixed.
                    Look for:
                    1. Inconsistent text formatting (e.g., "Male" vs "male" vs "M")
                    2. Values that need normalization
                    3. Erroneous values that should be corrected
                    4. String columns that need to be stripped of whitespace
                    
                    IMPORTANT: Never use string methods on non-string columns. Always check if a column
                    contains strings before applying string operations.
                    
                    Return a Python dictionary where keys are column names and values are
                    descriptions of issues that need to be fixed.
                    """
                    
                    
                    max_retries = 3
                    analysis_result = None
                    
                    for attempt in range(max_retries):
                        try:
                            analysis_result = smart_df.chat(analysis_prompt)
                            break
                        except (AttributeError, ValueError, TypeError, NameError) as e:
                            if attempt == max_retries - 1:
                                st.warning(f"Data analysis for cleaning failed after {max_retries} attempts: {str(e)}")
                                cleaning_report["changes"].append("Advanced cleaning skipped due to analysis errors")
                                return df, cleaning_report
                            else:
                                
                                analysis_prompt = """
                                Look for basic data quality issues in string columns only.
                                Return a simplified dictionary of column names and issues.
                                """
                        except Exception as e:
                            st.warning(f"Unexpected error in cleaning analysis: {str(e)}")
                            cleaning_report["changes"].append("Advanced cleaning skipped due to unexpected errors")
                            return df, cleaning_report
                    
                    
                    if isinstance(analysis_result, dict) and len(analysis_result) > 0:
                        cleaning_prompt = f"""
                        Clean the dataframe based on these identified issues:
                        {analysis_result}
                        
                        Apply appropriate cleaning techniques for each issue, such as:
                        1. Normalizing inconsistent categories
                        2. Stripping whitespace
                        3. Fixing formatting issues
                        4. Correcting clear errors
                        
                        IMPORTANT SAFETY RULES:
                        - Always check column data types before applying operations
                        - Never use string methods on non-string data
                        - Always handle exceptions in your code
                        - Preserve null values rather than introducing errors
                        - Don't drop any rows or columns
                        
                        Return the cleaned dataframe.
                        """
                        
                        
                        for attempt in range(max_retries):
                            try:
                                cleaned_df = smart_df.chat(cleaning_prompt)
                                
                                
                                if isinstance(cleaned_df, pd.DataFrame):
                                    if cleaned_df.shape[0] == df.shape[0]:
                                        
                                        df = cleaned_df
                                        issue_columns = list(analysis_result.keys())
                                        if len(issue_columns) <= 5:
                                            cleaning_report["changes"].append(f"Applied advanced cleaning to columns: {', '.join(issue_columns)}")
                                        else:
                                            cleaning_report["changes"].append(f"Applied advanced cleaning to {len(issue_columns)} columns")
                                        break
                                    else:
                                        
                                        st.warning("Cleaning operation changed row count - reverting to original")
                                else:
                                    
                                    st.warning("Cleaning operation did not return a valid DataFrame")
                                    
                            except (AttributeError, ValueError, TypeError, NameError) as e:
                                if attempt == max_retries - 1:
                                    st.warning(f"Advanced cleaning failed after {max_retries} attempts: {str(e)}")
                                    cleaning_report["changes"].append("Some advanced cleaning steps skipped due to errors")
                                else:
                                    
                                    cleaning_prompt = """
                                    Perform very basic cleaning:
                                    1. Strip whitespace from string columns
                                    2. Convert text to lowercase where appropriate
                                    3. Use simple, safe operations only
                                    4. Return the cleaned DataFrame
                                    """
                            except Exception as e:
                                st.warning(f"Unexpected error in cleaning operation: {str(e)}")
                                cleaning_report["changes"].append("Advanced cleaning incomplete due to errors")
                                break
                    else:
                        cleaning_report["changes"].append("No additional cleaning issues identified by LLM")
            
            except Exception as e:
                st.warning(f"LLM-based advanced cleaning failed: {str(e)}")
                cleaning_report["changes"].append("Advanced cleaning skipped due to initialization errors")
        
        
        cleaning_report["rows_after"] = len(df)
        cleaning_report["columns_after"] = len(df.columns)
        cleaning_report["rows_removed"] = cleaning_report["rows_before"] - cleaning_report["rows_after"]
        cleaning_report["columns_removed"] = cleaning_report["columns_before"] - cleaning_report["columns_after"]
        
        return df, cleaning_report
    
    def generate_cleaning_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate data cleaning recommendations using SmartDataframe"""
        
        if not self.llm:
            return [{"type": "error", "message": "LLM not initialized correctly"}]
        
        try:
            
            smart_df = self._create_smart_dataframe(df)
            if not safe_is_valid(smart_df):
                return [{"type": "error", "message": "Could not create SmartDataframe"}]
                
            prompt = """
            Analyze this DataFrame and provide specific data cleaning recommendations.
            For each recommendation, provide:
            1. The issue type (e.g., "missing_values", "outliers", "inconsistent_data", "data_types")
            2. The affected columns
            3. A detailed description of the issue
            4. A recommended cleaning approach
            
            Format the recommendations as a list of dictionaries with the following keys:
            - issue_type: string
            - columns: list of column names
            - description: detailed description of the issue
            - recommended_action: specific cleaning action to take
            - priority: "high", "medium", or "low"
            
            Make sure your recommendations are concrete, actionable, and specific to this dataset.
            Avoid using string methods on non-string columns to prevent AttributeError.
            """
            
            
            max_retries = 3
            retry_count = 0
            last_error = None
            
            while retry_count < max_retries:
                try:
                    recommendations = smart_df.chat(prompt)
                    break
                except (AttributeError, NameError, ValueError, TypeError) as e:
                    retry_count += 1
                    last_error = e
                    
                    if retry_count == max_retries - 1:
                        prompt = """
                        Analyze this DataFrame and provide basic data cleaning recommendations.
                        Return a simple list of dictionaries with these keys:
                        - issue_type: string (e.g., "missing_values")
                        - columns: list of column names
                        - description: brief description
                        - recommended_action: simple action to take
                        - priority: "high", "medium", or "low"
                        """
                except Exception as e:
                    
                    return [{"type": "error", "message": f"Error generating recommendations: {str(e)}"}]
            
            
            if retry_count == max_retries:
                return [{"type": "error", "message": f"Failed after {max_retries} attempts: {str(last_error)}"}]
            
            
            if isinstance(recommendations, list):
                return recommendations
            elif isinstance(recommendations, str):
                
                import json
                if "```" in recommendations:
                    json_str = recommendations.split("```")[1]
                    if json_str.startswith("json"):
                        json_str = json_str[4:]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        
                        return [{"issue_type": "unknown", "columns": [], 
                                "description": recommendations, 
                                "recommended_action": "Review recommendations manually", 
                                "priority": "medium"}]
                else:
                    try:
                        return json.loads(recommendations)
                    except json.JSONDecodeError:
                        return [{"type": "error", "message": "Could not parse recommendations", "details": recommendations}]
            else:
                return [{"type": "error", "message": "Unexpected response type", "details": str(type(recommendations))}]
        
        except Exception as e:
            return [{"type": "error", "message": f"Error generating recommendations: {str(e)}"}]