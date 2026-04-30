import os, uuid, datetime, json, tiktoken
import pandas as pd
from core.llm_config import get_llm

from pandasai.smart_dataframe import SmartDataframe
from pandasai.smart_datalake import SmartDatalake


data = None
file_name = None
query_history = []
current_response = None
last_error = None
smart_df = None
smart_datalake = None
data_collection = [] 

INPUT_COST_PER_1M = 1.25
OUTPUT_COST_PER_1M = 10.00

DUCKDB_RESERVED_FUNCTIONS = {
    "abs", "acos", "age", "alias", "any", "approx_count", "approx_quantile",
    "arg_max", "arg_min", "array", "ascii", "asin", "atan", "atan2",
    "avg", "bar", "base64", "bin", "bit", "bitstring", "bool",
    "cardinality", "cbrt", "ceil", "ceiling", "chr", "coalesce",
    "concat", "contains", "corr", "cos", "cot", "count", "current_date",
    "current_schema", "current_time", "current_timestamp", "date", "date_diff",
    "date_part", "date_sub", "date_trunc", "day", "dayname", "dayofmonth",
    "dayofweek", "dayofyear", "decode", "degrees", "dexp", "div",
    "encode", "epoch", "epoch_ms", "error", "exp", "extract",
    "filter", "first", "flatten", "floor", "format", "from_base64",
    "gamma", "generate_series", "get", "greatest", "group", "hash",
    "hex", "hour", "ifnull", "ilike", "index", "instr", "interval",
    "isfinite", "isinf", "isnan", "julian", "lag", "last", "last_day",
    "lead", "least", "left", "len", "length", "lgamma", "like",
    "list", "ln", "log", "log10", "log2", "lower", "lpad", "ltrim",
    "maketime", "map", "max", "mean", "median", "microsecond",
    "millennium", "millisecond", "min", "minute", "mod", "mode",
    "month", "monthname", "now", "nullif", "oid", "ord", "parse",
    "pi", "position", "pow", "power", "printf", "quarter", "radians",
    "random", "range", "rank", "read", "regexp", "regexp_matches",
    "regexp_replace", "repeat", "replace", "reverse", "right", "round",
    "row", "row_number", "rpad", "rtrim", "second", "sign", "sin",
    "sqrt", "startswith", "strftime", "string", "strlen", "strpos",
    "struct", "substr", "substring", "sum", "tan", "time", "timestamp",
    "timezone", "to_base64", "today", "trim", "trunc", "typeof",
    "union", "unnest", "upper", "uuid", "variance", "vin", "week",
    "weekday", "weekofyear", "year",
}


def _calculate_token_usage(input_text, output_text, df=None):
    
    try:
        enc = tiktoken.encoding_for_model("gpt-4")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")

    # Build input: user query + dataframe context that PandasAI sends internally
    full_input = str(input_text) if input_text else ""
    if df is not None:
        try:
            # PandasAI sends column info, dtypes, and sample rows to the LLM
            df_context = f"\nDataFrame columns: {list(df.columns)}\n"
            df_context += f"DataFrame dtypes: {df.dtypes.to_dict()}\n"
            df_context += f"DataFrame shape: {df.shape}\n"
            df_context += f"Sample data:\n{df.head(5).to_string()}\n"
            full_input += df_context
        except Exception:
            pass

    input_tokens = len(enc.encode(full_input)) if full_input else 0
    output_tokens = len(enc.encode(str(output_text))) if output_text else 0
    total_tokens = input_tokens + output_tokens

    input_cost = round(input_tokens * INPUT_COST_PER_1M / 1_000_000, 8)
    output_cost = round(output_tokens * OUTPUT_COST_PER_1M / 1_000_000, 8)
    total_cost = round(input_cost + output_cost, 5)

    return {
        "total_token_counts": total_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost
    }


def load_data(file_path):
    """Load data from file path with chunking for large files"""
    try:
        file_size = os.path.getsize(file_path)
        
        use_chunking = file_size > 100 * 1024 * 1024
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.csv':
            try:
                if use_chunking:
                    chunks = []
                    for chunk in pd.read_csv(file_path, chunksize=100000):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                
                import chardet
                with open(file_path, 'rb') as f:
                    result = chardet.detect(f.read())
                
                if use_chunking:
                    chunks = []
                    for chunk in pd.read_csv(file_path, encoding=result['encoding'], chunksize=100000):
                        chunks.append(chunk)
                    df = pd.concat(chunks, ignore_index=True)
                else:
                    df = pd.read_csv(file_path, encoding=result['encoding'])
                    
        elif file_ext in ['.xlsx', '.xls']:
            
            if use_chunking:
                
                df = pd.read_excel(file_path, nrows=500000)
                print("Large Excel file detected! Only the first 500,000 rows were loaded. Consider converting to CSV or Parquet for full analysis.")
            else:
                df = pd.read_excel(file_path)
                
        elif file_ext == '.parquet':
            
            df = pd.read_parquet(file_path)
        else:
            print(f"Unsupported file format: {file_ext}")
        
        df = optimize_dataframe_memory(df)
        
        return df
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None


def optimize_dataframe_memory(df):
    """Optimize dataframe memory usage by downcasting numeric types"""
    
    float_cols = df.select_dtypes(include=['float']).columns
    if not float_cols.empty:
        df[float_cols] = df[float_cols].apply(pd.to_numeric, downcast='float')
    
    
    int_cols = df.select_dtypes(include=['integer']).columns
    if not int_cols.empty:
        df[int_cols] = df[int_cols].apply(pd.to_numeric, downcast='integer')
    
    
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        if df[col].nunique() < 0.5 * len(df):
            df[col] = df[col].astype('category')
            
    return df


def analyze_multiple_files(file_paths=None, query=None, selected_columns_map=None, user_id="default_user", dataframes=None, file_names=None):
    
    global data_collection, smart_datalake
    
    
    data_collection = []
    smart_datalake = None

    
    if dataframes is not None and file_names is not None:
        loaded_dfs = dataframes
        loaded_names = file_names
    else:
        
        if not file_paths or not isinstance(file_paths, list) or len(file_paths) == 0:
            print("No files provided for analysis.")
            return None
        
        loaded_dfs = []
        loaded_names = []
        
        for file_path in file_paths:
            print(f"Loading data from {file_path}...")
            df_and_name = load_data(file_path, user_id)
            
            if not df_and_name or df_and_name[0] is None:
                print(f"Failed to load data from {file_path}. Skipping this file.")
            else:
                df, name = df_and_name
                
                if isinstance(df, dict):
                    for sheet_name, sheet_df in df.items():
                        loaded_dfs.append(sheet_df)
                        loaded_names.append(f"{name} ({sheet_name})") 
                else:
                    loaded_dfs.append(df)
                    loaded_names.append(name)
                print(f"Successfully loaded: {name}")
    
    if not loaded_dfs:
        print("No valid datasets were loaded.")
        return None
    
    
    print(f"\nLoaded {len(loaded_dfs)} datasets:")
    for i, (df, name) in enumerate(zip(loaded_dfs, loaded_names)):
        summary = generate_dataset_summary(df)
        print(f"\nDataset {i+1}: {name}")
        print(f"Rows: {summary['row_count']:,}")
        print(f"Columns: {summary['column_count']}")
        print(f"Missing values: {summary['missing_percentage']}%")
    
    
    result = process_query_with_pandasai_datalake(query, loaded_dfs, loaded_names, user_id, selected_columns_map)
    
    
    if result:
        json_path = save_results_to_json(result, user_id)
        if json_path:
            print(f"\nJSON output saved to: {json_path}")
        
        
        if result.get("chart_files"):
            print(f"\nGenerated charts:")
            for chart_path in result["chart_files"]:
                full_path = os.path.join("GRAPH", user_id, chart_path)
                print(f"- {chart_path} (Full path: {os.path.abspath(full_path)})")
    
    return result


def save_results_to_json(result, user_id):
    """Save analysis results to a JSON file by user ID"""
    if not result:
        return None

    chart_files = result.get("visualization_paths", [])

    if chart_files:
        user_charts_dir = os.path.join("GRAPH", user_id)
        os.makedirs(user_charts_dir, exist_ok=True)
    
    text_response = result.get("response_text", "")
    visualization_text = result.get("visualization_text", "")
    
    if visualization_text:
        text_response = visualization_text
    
    output_json = {
        "user_id": user_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": result.get("query", ""),
        "response": text_response or result.get("response_text", ""),
        "chart_paths": chart_files,  
        "visualization_paths": chart_files  
    }
    
    if "cost_analysis" in result:
        output_json["cost_analysis"] = result["cost_analysis"]

    output_path = f"chat_output/{user_id}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, indent=2)

    history_file = os.path.join("chat_history", f"{user_id}.json")

    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except:
            history = []

    question_number = len(history) + 1

    history_entry = {
        "question_number": question_number,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": result.get("query", ""),
        "response": text_response or result.get("response_text", ""),
        "cost_analysis": result.get("cost_analysis", {}),
        "chart_paths": chart_files,  
        "visualization_paths": chart_files  
    }
    
    history.append(history_entry)
    
    with open(history_file, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    
    print(f"Results saved to {output_path}")
    print(f"History updated in {history_file}")
    return output_path



def analyze_data(file_path, query, user_id="default_user"):
    """Analyze data from file with a specific query and save results to JSON by user ID"""
    global data, file_name, smart_df
    
    data = None
    file_name = None
    smart_df = None
    
    user_charts_dir = os.path.join("GRAPH", user_id)
    os.makedirs(user_charts_dir, exist_ok=True)
    
    print(f"Loading data from {file_path}...")
    df_and_name = load_data(file_path, user_id)
    loaded, file_name = df_and_name
    
    if isinstance(loaded, dict):
        
        sheet_dfs = loaded
        return analyze_multiple_files(
            file_paths=None,
            query=query,
            selected_columns_map=None,
            user_id=user_id,
            
            dataframes=list(sheet_dfs.values()),
            file_names=list(sheet_dfs.keys())
        )
    
    if loaded is None:
        print("Failed to load data.")
        return None

    data = loaded
    
    if not df_and_name or df_and_name[0] is None:
        print("Failed to load data.")
        return None
    
    data, file_name = df_and_name
    print(f"Successfully loaded: {file_name}")
    
    summary = generate_dataset_summary(data)
    print(f"\nDataset Summary:")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Columns: {summary['column_count']}")
    print(f"Missing values: {summary['missing_percentage']}%")
    print(f"Numeric columns: {', '.join(summary['numeric_columns'][:5])}{'...' if len(summary['numeric_columns']) > 5 else ''}")
    print(f"Categorical columns: {', '.join(summary['categorical_columns'][:5])}{'...' if len(summary['categorical_columns']) > 5 else ''}")
    
    result = process_query_with_pandasai(query, data, user_id)
    
    if result:
        json_path = save_results_to_json(result, user_id)
        if json_path:
            print(f"\nJSON output saved to: {json_path}")
        
        if result.get("chart_files"):
            print(f"\nGenerated charts:")
            for chart_path in result["chart_files"]:
                full_path = os.path.join("GRAPH", user_id, chart_path)
                print(f"- {chart_path} (Full path: {os.path.abspath(full_path)})")
    
    return result


def generate_dataset_summary(df):
    """Generate summary statistics for the dataset"""
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "memory_usage": df.memory_usage(deep=True).sum(),
        "missing_values": df.isnull().sum().sum(),
        "missing_percentage": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "datetime_columns": [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])],
        "column_types": {col: str(df[col].dtype) for col in df.columns}
    }
    
    summary["numeric_stats"] = {}
    for col in summary["numeric_columns"]:
        summary["numeric_stats"][col] = {
            "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
            "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
            "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
            "median": float(df[col].median()) if not pd.isna(df[col].median()) else None,
            "std": float(df[col].std()) if not pd.isna(df[col].std()) else None
        }
    
    for col in summary["categorical_columns"]:
        try:
            if col not in df.columns:
                continue

            if df[col].nunique() < 50:
                value_counts = df[col].value_counts().head(5).to_dict()

                summary.setdefault("categorical_stats", {})[col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in value_counts.items()}
                }
        except Exception as e:
            print(f"Skipping column {col}: {e}")

    return summary


def _extract_text_response(smart_obj, response):
    """Extract text response and visualization text from a PandasAI smart object."""
    text_response = str(response)
    visualization_text = None
    last_result = None

    
    if hasattr(smart_obj, '_agent') and hasattr(smart_obj._agent, 'last_result'):
        last_result = smart_obj._agent.last_result
    
    elif hasattr(smart_obj, 'last_result'):
        last_result = smart_obj.last_result

    if isinstance(last_result, dict):
        if 'explanation' in last_result:
            text_response = last_result['explanation']
        elif 'text' in last_result:
            text_response = last_result['text']
        if 'text' in last_result:
            visualization_text = last_result['text']

    return text_response, visualization_text


def process_query_with_pandasai(query, df=None, user_id="default_user", selected_columns=None):
    """Process user query using PandasAI and return text response and visualizations"""
    global data, smart_df, current_response, query_history
    
    if not query or (df is None and data is None):
        return None
    
    if df is not None:
        data = df
    
    try:
        print("Analyzing data and generating response...")

        if smart_df is None:
            smart_df = initialize_pandasai(data, user_id)
            
        if smart_df is None:
            print("Failed to initialize PandasAI. Please check your configuration.")
            return None

        user_charts_dir = os.path.abspath(os.path.join("GRAPH", user_id))
        os.makedirs(user_charts_dir, exist_ok=True)

        before_charts = set(os.listdir(user_charts_dir))

        chart_filename = f"{uuid.uuid4()}.html"
        chart_save_path = os.path.join(user_charts_dir, chart_filename).replace("\\", "/")
        
        column_focus = ""
        if selected_columns and isinstance(selected_columns, list) and len(selected_columns) > 0:
            valid_columns = [col for col in selected_columns if col in data.columns]
            if valid_columns:
                column_focus = f"Focus ONLY on the following columns: {', '.join(valid_columns)}. "
                column_focus += "Ignore all other columns in your analysis. "
        
        viz_instruction = (
            f"If creating a visualization, use Plotly and save it strictly with:\n"
            f"fig.write_html('{chart_save_path}')\n"
            f"DO NOT return the plot object. "
            f"CRITICAL: Always generate visually attractive, colorful plots using vibrant color palettes (e.g., Plotly Express 'color' parameter or 'px.colors.qualitative.Plotly'). NEVER generate plain or monochrome/black-and-white plots! "
        )
        enhanced_query = (
            f"{query}\n\n"
            f"{column_focus}"
            f"IMPORTANT: You must analyze the data and create a detailed text answer with statistics and numbers. "
            f"Avoid Float Drift: Do NOT calculate averages using the raw FLOAT/DOUBLE columns. "
            f"Cast to Decimal: In your SQL, you MUST cast the numeric column to DECIMAL for the calculation. Example: SELECT AVG(CAST(salary AS DECIMAL(18,2))) FROM table."
            f"NEVER estimate, guess, or infer numbers."
            f"If the query asks for a visualization (plot, chart, graph, show me), also create one before returning the text. "
            f"{viz_instruction}"
            f"If data is not available to answer, use 'Data not available' as the value. "
            f"Your Python code MUST end with EXACTLY this format:\n"
            f"result = {{'type': 'string', 'value': 'your detailed answer here'}}\n"
            f"NEVER use 'text' as the type. ALWAYS use 'string'. NEVER return a plain string."
        )
        
        response = smart_df.chat(enhanced_query)
        
        text_response, visualization_text = _extract_text_response(smart_df, response)
        cost_analysis = _calculate_token_usage(enhanced_query, text_response, df=data)
        
        
        after_charts = set(os.listdir(user_charts_dir))
        new_charts = list(after_charts - before_charts)

        new_charts = [f for f in new_charts if f.endswith(".html")]
       
        if new_charts:
            new_charts = _ensure_html_charts(user_charts_dir, new_charts, text_response)
        
        
        result = {
            "success": True,
            "response_text": text_response,
            "visualization_text": visualization_text,
            "query": query,
            "user_id": user_id,
            "cost_analysis": cost_analysis,
            "chart_files": new_charts,
            "visualization_paths": [os.path.join("GRAPH", user_id, chart).replace("\\", "/") for chart in new_charts]
        }
        
        
        if query not in [item['query'] for item in query_history]:
            query_history.append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_id': user_id,
                'has_chart': len(new_charts) > 0
            })
        
        
        print(f"\nQuery: {query}")
        if new_charts:
            print(f"Generated {len(new_charts)} chart(s). Saved to: {', '.join(new_charts)}")
            if visualization_text:
                print(f"\nVisualization explanation:\n{visualization_text}")
        print(f"\nResponse:\n{text_response}")
        print(f"\nToken Usage: {cost_analysis['total_token_counts']} tokens, Input Cost: ${cost_analysis['input_cost']:.8f}, Output Cost: ${cost_analysis['output_cost']:.8f}")
        
        return result
        
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        last_error = str(e)
        return None


def sanitize_dataframe_columns(df: pd.DataFrame) -> tuple:
    
    rename_map = {}       
    reverse_map = {}    
    new_columns = []
    for col in df.columns:
        if col.lower() in DUCKDB_RESERVED_FUNCTIONS:
            safe_col = f"{col}_col"
            
            while safe_col in df.columns or safe_col in reverse_map.values():
                safe_col = f"{safe_col}_"
            rename_map[safe_col] = col
            reverse_map[col] = safe_col
            new_columns.append(safe_col)
            print(f"[sanitize_dataframe_columns] Renamed column '{col}' -> '{safe_col}' "
                  f"(conflicts with DuckDB built-in function)")
        else:
            new_columns.append(col)
    sanitized_df = df.copy()
    sanitized_df.columns = new_columns
    return sanitized_df, rename_map


def initialize_pandasai(df, user_id="default_user"):
    
    try:
        llm = get_llm()
        if not llm:
            print("Failed to get LLM from configuration.")
            return None

        user_charts_dir = os.path.join("GRAPH", user_id)
        os.makedirs(user_charts_dir, exist_ok=True)
        print(f"Charts will be saved to: {os.path.abspath(user_charts_dir)}")
        
        df, _col_rename_map = sanitize_dataframe_columns(df)

        smart_df = SmartDataframe(
            df,
            config={
                "llm": llm,
                "verbose": True,
                "enable_cache": True,
                "save_charts": False,  
                "save_charts_path": user_charts_dir,  
                "enforce_privacy": True,
                "max_retries": 3,
                "use_error_correction_framework": True,
                "save_logs": True,
                "whitelisted_libraries": ["plotly"],
                "open_charts": False,
                "data_viz_library": "plotly",
            }
        )
        
        return smart_df
        
    except Exception as e:
        print(f"Error initializing PandasAI: {str(e)}")
        return None


def process_query_with_pandasai_datalake(query, dataframes=None, file_names=None, user_id="default_user", selected_columns_map=None):
    """Process user query using PandasAI SmartDatalake and return text response and visualizations"""
    global data_collection, smart_datalake, current_response, query_history
    
    if not query or (dataframes is None and not data_collection):
        return None
    
    if dataframes is not None:
        data_collection = dataframes
    
    try:
        print("Analyzing multiple datasets and generating response...")
        
        if smart_datalake is None:
            smart_datalake = initialize_pandasai_datalake(data_collection, user_id)
            
        if smart_datalake is None:
            print("Failed to initialize PandasAI SmartDatalake. Please check your configuration.")
            return None
        
        user_charts_dir = os.path.join("GRAPH", user_id)
        os.makedirs(user_charts_dir, exist_ok=True)
        
        before_charts = set(os.listdir(user_charts_dir))
        
        chart_filename = f"{uuid.uuid4()}.html"
        chart_save_path = os.path.join(user_charts_dir, chart_filename).replace("\\", "/")
        
        column_focus = ""
        if selected_columns_map and isinstance(selected_columns_map, dict) and len(selected_columns_map) > 0:
            column_instructions = []
            
            for i, df_name in enumerate(file_names):
                if df_name in selected_columns_map:
                    columns = selected_columns_map[df_name]
                    if columns and len(columns) > 0:
                        
                        current_df = dataframes[i]
                        valid_columns = [col for col in columns if col in current_df.columns]
                        if valid_columns:
                            column_instructions.append(f"For dataframe '{df_name}', focus ONLY on these columns: {', '.join(valid_columns)}.")
            
            if column_instructions:
                column_focus = "Column focus instructions:\n" + "\n".join(column_instructions) + "\nIgnore all other columns in your analysis. "
        
        viz_instruction = (
            f"If creating a visualization, use Plotly and save it strictly with:\n"
            f"fig.write_html('{chart_save_path}')\n"
            f"DO NOT return the plot object. "
            f"CRITICAL: Always generate visually attractive, colorful plots using vibrant color palettes (e.g., Plotly Express 'color' parameter or 'px.colors.qualitative.Plotly'). NEVER generate plain or monochrome/black-and-white plots! "
        )
        enhanced_query = (
            f"{query}\n\n"
            f"{column_focus}"
            f"IMPORTANT: You must analyze the data and create a detailed text answer with statistics and numbers. "
            f"Your Python code MUST finish by returning your detailed text answer as a string. "
            f"If the query asks for a visualization (plot, chart, graph, show me), also create one before returning the text. "
            f"{viz_instruction}"
            f"If data is not available to answer, return 'Data not available'."
        )

        response = smart_datalake.chat(enhanced_query)
        
        after_charts = set(os.listdir(user_charts_dir))
        new_charts = list(after_charts - before_charts)

        new_charts = [f for f in new_charts if f.endswith(".html")]
        
        text_response, visualization_text = _extract_text_response(smart_datalake, response)
        cost_analysis = _calculate_token_usage(enhanced_query, text_response, df=data_collection[0] if data_collection else None)
        
        if new_charts:
            new_charts = _ensure_html_charts(user_charts_dir, new_charts, text_response)
        
        result = {
            "success": True,
            "response_text": text_response,
            "visualization_text": visualization_text,
            "query": query,
            "user_id": user_id,
            "cost_analysis": cost_analysis,
            "file_names": file_names,
            "chart_files": new_charts,
            "visualization_paths": [os.path.join("GRAPH", user_id, chart).replace("\\", "/") for chart in new_charts]
        }
        
        if query not in [item['query'] for item in query_history]:
            query_history.append({
                'query': query,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'user_id': user_id,
                'files': file_names,
                'has_chart': len(new_charts) > 0
            })

        current_response = result
        
        print(f"\nQuery: {query}")
        if new_charts:
            print(f"Generated {len(new_charts)} chart(s). Saved to: {', '.join(new_charts)}")
            if visualization_text:
                print(f"\nVisualization explanation:\n{visualization_text}")
        print(f"\nResponse:\n{text_response}")
        print(f"\nToken Usage: {cost_analysis['total_token_counts']} tokens, Input Cost: ${cost_analysis['input_cost']:.8f}, Output Cost: ${cost_analysis['output_cost']:.8f}")

        return result
        
    except Exception as e:
        print(f"Error processing query with SmartDatalake: {str(e)}")
        last_error = str(e)
        return None


def _ensure_html_charts(user_charts_dir, new_charts, text_response):
    import base64
    updated_charts = []

    escaped_text = (text_response or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    for chart in new_charts:
        chart_path = os.path.join(user_charts_dir, chart)

        if chart.lower().endswith('.png'):
            html_filename = chart.rsplit('.', 1)[0] + '.html'
            html_path = os.path.join(user_charts_dir, html_filename)

            try:
                with open(chart_path, 'rb') as img_file:
                    img_b64 = base64.b64encode(img_file.read()).decode('utf-8')

                html_content = f"""<!DOCTYPE html>
                <html lang="en">
                <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width,initial-scale=1.0">
                <title>Analysis Result</title>
                <style>
                body {{ font-family:'Segoe UI',Tahoma,sans-serif; margin:0; padding:20px; background:#f5f7fa; color:#333; }}
                .container {{ max-width:1200px; margin:0 auto; }}
                .section {{ background:#fff; border-radius:12px; padding:24px; margin-bottom:20px; box-shadow:0 2px 8px rgba(0,0,0,.08); }}
                .section h2 {{ margin-top:0; padding-bottom:10px; }}
                .chart h2 {{ color:#2c3e50; border-bottom:2px solid #27ae60; }}
                .chart img {{ max-width:100%; height:auto; border-radius:8px; }}
                </style>
                </head>
                <body>
                <div class="container">
                <div class="section chart">
                <h2>📈 Visualization</h2>
                <img src="data:image/png;base64,{img_b64}" />
                </div>
                </div>
                </body>
                </html>"""

                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)

                os.remove(chart_path)
                updated_charts.append(html_filename)

            except Exception as e:
                print(f"Failed to convert {chart}: {e}")
                updated_charts.append(chart)

        elif chart.lower().endswith('.html'):
            updated_charts.append(chart)

        else:
            updated_charts.append(chart)

    return updated_charts


def initialize_pandasai_datalake(dataframes, user_id="default_user"):
    """Initialize SmartDatalake for analysis of multiple dataframes.
    Uses get_llm() from core.llm_config for consistent LLM initialization."""
    try:
        llm = get_llm()
        if not llm:
            print("Failed to get LLM from configuration.")
            return None

        user_charts_dir = os.path.join("GRAPH", user_id)
        os.makedirs(user_charts_dir, exist_ok=True)
        print(f"SmartDatalake charts will be saved to: {os.path.abspath(user_charts_dir)}")

        
        sanitized_dfs = []
        for df in dataframes:
            sanitized_df, _ = sanitize_dataframe_columns(df)
            sanitized_dfs.append(sanitized_df)

        smart_datalake = SmartDatalake(
            sanitized_dfs,
            config={
                "llm": llm,
                "verbose": True,
                "enable_cache": True,
                "save_charts": False,
                "save_charts_path": user_charts_dir,
                "enforce_privacy": True,
                "max_retries": 3,
                "use_error_correction_framework": True,
                "save_logs": True,
                "whitelisted_libraries": ["plotly"],
                "open_charts": False,
                "data_viz_library": "plotly",
                "memory_size": 5
            }
        )

        return smart_datalake

    except Exception as e:
        print(f"Error initializing PandasAI SmartDatalake: {str(e)}")
        return None


def analyze_data_with_columns(file_path, query, selected_columns=None, user_id="default_user"):

    global data, file_name, smart_df
    
    data = None
    file_name = None
    smart_df = None
    
    user_charts_dir = os.path.join("GRAPH", user_id)
    os.makedirs(user_charts_dir, exist_ok=True)
    
    print(f"Loading data from {file_path}...")
    df_and_name = load_data(file_path, user_id)
    
    if not df_and_name or df_and_name[0] is None:
        print("Failed to load data.")
        return None
    
    full_df, file_name = df_and_name
    print(f"Successfully loaded: {file_name}")
    
    
    if selected_columns and isinstance(selected_columns, list) and len(selected_columns) > 0:
        
        valid_columns = [col for col in selected_columns if col in full_df.columns]
        
        if not valid_columns:
            print("None of the specified columns exist in the dataset.")
            print(f"Available columns: {', '.join(full_df.columns)}")
            return None
        
        if len(valid_columns) < len(selected_columns):
            missing_columns = set(selected_columns) - set(valid_columns)
            print(f"Warning: Some columns were not found in the dataset: {', '.join(missing_columns)}")
        
        
        data = full_df[valid_columns].copy()
        print(f"Analysis will be performed on {len(valid_columns)} selected columns: {', '.join(valid_columns)}")
    else:
        
        data = full_df
        print(f"Analysis will be performed on all {len(data.columns)} columns")
    
    
    summary = generate_dataset_summary(data)
    print(f"\nDataset Summary (Selected Columns):")
    print(f"Rows: {summary['row_count']:,}")
    print(f"Columns: {summary['column_count']}")
    print(f"Missing values: {summary['missing_percentage']}%")
    print(f"Numeric columns: {', '.join(summary['numeric_columns'][:5])}{'...' if len(summary['numeric_columns']) > 5 else ''}")
    print(f"Categorical columns: {', '.join(summary['categorical_columns'][:5])}{'...' if len(summary['categorical_columns']) > 5 else ''}")
    
    result = process_query_with_pandasai(query, data, user_id, selected_columns=selected_columns)
    
    if result:
        json_path = save_results_to_json(result, user_id)
        if json_path:
            print(f"\nJSON output saved to: {json_path}")
        
        
        if result.get("chart_files"):
            print(f"\nGenerated charts:")
            for chart_path in result["chart_files"]:
                full_path = os.path.join("GRAPH", user_id, chart_path)
                print(f"- {chart_path} (Full path: {os.path.abspath(full_path)})")
    
    return result