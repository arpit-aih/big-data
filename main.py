import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient
from typing import List, Optional
from urllib.parse import quote_plus
from langchain_core.tools import tool
import os, json, shutil, uuid, traceback, re
from sqlalchemy import create_engine, text
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from fastapi import FastAPI, HTTPException, UploadFile, Form, File
from langchain_community.callbacks.manager import get_openai_callback, OpenAICallbackHandler

from visualization import DuckDBManager, build_prompt, generate_chart_with_summary
from tools import create_sql_tools, create_mongo_tools
from text_data_analysis import load_data
from data_quality_app import clean_data, create_zip_archive, export_data, analyze_data_quality, generate_quality_report
from cost_analysis import calculate_cost, calculate_token_cost, track_analysis_report_tokens
from text_data_analysis import (
    analyze_data as text_analyze_data,
    analyze_data_with_columns,
    analyze_multiple_files
)
from schemas import (
    CleaningRequest,
    AnalysisRequest,
    ExportRequest,
    ProcessAllRequest,
    PostgresConnectRequest,
    MySQLConnectRequest,
    MongoConnectRequest,
    ChatRequest,
    DataQualityRequest
)


load_dotenv()

app = FastAPI()


llm = AzureChatOpenAI(
    azure_deployment="gpt-5.1",
    api_version="2024-12-01-preview",
    temperature=0
)


client = MongoClient(os.getenv("MONGO_URI"))
db = client["data_analyzer"] 
chat_collection = db["chat_history"]

sessions = {}
user_data = {}

duckdb_manager = DuckDBManager()


def get_session(user_id: str):
    if user_id not in sessions:
        sessions[user_id] = {
            "connections": {
                "postgres": None,
                "mysql": None,
                "mongodb": None
            },
            "agents": {
                "postgres": None,
                "mysql": None,
                "mongodb": None
            },
            "schemas": {  
                "postgres": None,
                "mysql": None,
                "mongodb": None
            }
        }
    return sessions[user_id]


def _execute_file_query(df: pd.DataFrame, question: str) -> pd.DataFrame:
    """
    Use LLM to generate pandas code, then execute it safely.
    Returns a result DataFrame.
    """
    schema_str = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "sample": df.head(5).to_dict(orient="records")
    }

    code_prompt = f"""
        You are a pandas expert.

        DataFrame schema:
        {json.dumps(schema_str, indent=2)}

        User question: "{question}"

        Write a single Python expression using the variable `df` to answer this question.
        - For aggregations: use df.groupby(...)[...].mean() or .sum() or .count()
        - Always reset_index() after groupby
        - Return a DataFrame with exactly 2 columns: one categorical, one numeric
        - DO NOT import anything
        - Return ONLY the expression, no explanation, no variable assignment, no markdown

        Example:
        df.groupby("department")["salary"].mean().reset_index()"""

    response = llm.invoke(code_prompt)
    code = response.content.strip().strip("```python").strip("```").strip()

    try:
        result = eval(code, {"df": df, "pd": pd})
        if isinstance(result, pd.DataFrame):
            return result
        elif isinstance(result, pd.Series):
            return result.reset_index()
        else:
            return None
    except Exception as e:
        print(f"[FILE] pandas eval error: {e} | code: {code}")
        return None


def create_file_tools(file_path: str):
    """
    Creates a LangChain @tool for querying a file (CSV, Excel, Parquet).
    Mirrors the create_sql_tools pattern — returns a list of tools.
    """

    df_store = {}  

    def _load_df() -> pd.DataFrame:
        if "df" not in df_store:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".csv":
                df_store["df"] = pd.read_csv(file_path)
            elif ext in (".xlsx", ".xls"):
                df_store["df"] = pd.read_excel(file_path)
            elif ext == ".parquet":
                df_store["df"] = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")
        return df_store["df"]

    @tool
    def query_file(question: str) -> str:
        """
        Query a structured file (CSV, Excel, Parquet).
        Pass a natural language question — the tool will analyze
        the file schema and data to answer it.
        Returns a JSON string with table_name, columns, preview, row_count.
        """
        try:
            df = _load_df()

            schema_info = {
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "row_count": len(df),
                "sample": df.head(20).to_dict(orient="records")
            }

            result_df = _execute_file_query(df, question)

            if result_df is None or result_df.empty:
                return json.dumps({
                    "table_name": None,
                    "columns": [],
                    "preview": [],
                    "row_count": 0
                })

            table_name = duckdb_manager.store_dataframe(result_df)

            return json.dumps({
                "table_name": table_name,
                "columns": list(result_df.columns),
                "preview": result_df.head(5).to_dict(orient="records"),
                "row_count": len(result_df),
                "file_schema": schema_info  
            })

        except Exception as e:
            return f"File Error: {str(e)}"

    return [query_file]


def extract_sql_schema(sql_db: SQLDatabase) -> dict:
    """Extract table names and columns from a SQL database."""
    schema = {}
    try:
        table_names = sql_db.get_usable_table_names()
        for table in table_names:
            info = sql_db.get_table_info(table_names=[table])
            schema[table] = info
    except Exception as e:
        print(f"Schema extraction error: {e}")
    return schema


@app.post("/upload", summary="Upload a data file")
async def upload_data(
    user_id: str = Form(...),
    file: UploadFile = File(...)
):
    try:
        base_dir = "temp_uploads"

        user_folder = os.path.join(base_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)

        filename, file_ext = os.path.splitext(file.filename)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        new_filename = f"{filename}_{timestamp}{file_ext}"
        temp_path = os.path.join(user_folder, new_filename)

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = load_data(temp_path)

        if df is None:
            raise HTTPException(status_code=400, detail="Failed to load data file")

        user_data[user_id] = {
            "original_data": df,
            "file_name": new_filename,
            "file_path": temp_path,
            "cleaned_data": None,
            "quality_issues": None,
            "quality_report": None,
            "cleaning_report": None,
            "analysis_report": None,
            "token_usage": {}
        }

        return {
            "user_id": user_id,
            "file_name": new_filename,
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "missing_values": int(df.isnull().sum().sum()),
            "preview": json.loads(df.head(10).to_json(orient="records"))
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )


@app.post("/analyze-quality", summary="Analyze data quality")
async def analyze_quality(request: DataQualityRequest):
    
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail="User data not found. Please upload data first.")
    
    user = user_data[request.user_id]
    df = user["original_data"]
    
    try:
        
        quality_cb = OpenAICallbackHandler()
        report_cb = OpenAICallbackHandler()
        
        with get_openai_callback() as cb:
            quality_issues = analyze_data_quality(df)
            quality_cb = cb
        
        user["quality_issues"] = quality_issues
        
        with get_openai_callback() as cb:
            quality_report = generate_quality_report(df)
            report_cb = cb
        
        user["quality_report"] = quality_report
        
        quality_token_usage = calculate_token_cost(quality_cb)
        report_token_usage = calculate_token_cost(report_cb)
        
        total_token_usage = {
            "total_tokens": quality_token_usage["total_tokens"] + report_token_usage["total_tokens"],
            "prompt_tokens": quality_token_usage["prompt_tokens"] + report_token_usage["prompt_tokens"],
            "completion_tokens": quality_token_usage["completion_tokens"] + report_token_usage["completion_tokens"],
            "input_cost": round(quality_token_usage["input_cost"] + report_token_usage["input_cost"], 6),
            "output_cost": round(quality_token_usage["output_cost"] + report_token_usage["output_cost"], 6),
            "total_cost": round(quality_token_usage["total_cost"] + report_token_usage["total_cost"], 6)
        }
        
        user["token_usage"]["analyze_quality"] = total_token_usage
        
        quality_summary = {
            "total_missing_percent": quality_issues.get("total_missing_percent", 0.0),
            "total_outlier_percent": quality_issues.get("total_outlier_percent", 0.0),
            "total_duplicate_percent": quality_issues.get("total_duplicate_percent", 0.0),
            "data_type_issues": quality_issues.get("data_type_issues", [])
        }
        
        return {
            "user_id": request.user_id,
            "quality_issues": quality_issues,
            "quality_summary": quality_summary,
            "quality_report": quality_report,
            "token_usage": total_token_usage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing data quality: {str(e)}")


@app.post("/clean-data", summary="Clean the data")
async def clean_data_endpoint(request: CleaningRequest):
    
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail="User data not found. Please upload data first.")
    
    user = user_data[request.user_id]
    df = user["original_data"]
    
    try:
        options = request.options.dict()
        
        if options["missing_strategy"] == "constant" and options["constant_value"] is not None:
            options["constant_value"] = options.pop("constant_value")
        elif options["missing_strategy"] == "drop_columns" and options["missing_threshold"] is not None:
            options["missing_threshold"] = options.pop("missing_threshold")
        
        options = {k: v for k, v in options.items() if v is not None}
        
        
        with get_openai_callback() as cb:
            cleaned_df, report = clean_data(df, options)
            
        if cleaned_df is None:
            raise HTTPException(status_code=500, detail="Data cleaning failed")
        
        token_usage = calculate_token_cost(cb)
        
        user["cleaned_data"] = cleaned_df
        user["cleaning_report"] = report
        user["token_usage"]["clean_data"] = token_usage
        
        return {
            "user_id": request.user_id,
            "cleaning_report": report,
            "preview": json.loads(cleaned_df.head(5).to_json(orient="records")),
            "rows": len(cleaned_df),
            "columns": len(cleaned_df.columns),
            "missing_values": int(cleaned_df.isnull().sum().sum()),
            "token_usage": token_usage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error cleaning data: {str(e)}")


@app.post("/analyze-data", summary="Generate data analysis report")
async def analyze_data(request: AnalysisRequest):
    """
    Generate a data analysis report for the cleaned data.
    
    Returns the analysis report.
    """
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail="User data not found. Please upload data first.")
    
    user = user_data[request.user_id]
    
    if user["cleaned_data"] is None:
        raise HTTPException(status_code=400, detail="No cleaned data available. Please clean the data first.")
    
    try:
        
        analysis_report, token_usage = track_analysis_report_tokens(
            user["cleaned_data"], 
            focus=request.analysis_type
        )
        
        user["analysis_report"] = analysis_report
        user["token_usage"]["analyze_data"] = token_usage
        
        
        return {
            "user_id": request.user_id,
            "analysis_report": analysis_report,
            "analysis_type": request.analysis_type,
            "token_usage": token_usage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating analysis report: {str(e)}")


@app.post("/export-data", summary="Export the cleaned data")
async def export_data_endpoint(request: ExportRequest):
    
    if request.user_id not in user_data:
        raise HTTPException(status_code=404, detail="User data not found. Please upload data first.")
    
    user = user_data[request.user_id]
    
    if user["cleaned_data"] is None:
        raise HTTPException(status_code=400, detail="No cleaned data available. Please clean the data first.")
    
    try:
        
        base_filename = os.path.splitext(user["file_name"])[0]
        base_filename = f"cleaned_{base_filename}"
        
        exported_files = export_data(user["cleaned_data"], formats=request.formats, base_filename=base_filename)
        
        result = {
            "user_id": request.user_id,
            "exported_files": exported_files
        }
        
        if request.create_zip:
            zip_path = create_zip_archive(user["cleaned_data"], base_filename=base_filename)
            if zip_path:
                result["zip_archive"] = zip_path
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.post("/text-analyze-data", summary="Generate text-based data analysis")
async def text_analyze_data_endpoint(
    user_id: str = Form(...),
    query: str = Form(...),
    mode: Optional[str] = Form(None),
    columns: Optional[str] = Form(None),  
    files: List[UploadFile] = File(...)
):
    try:
        saved_paths = []

        for file in files:
            file_ext = os.path.splitext(file.filename)[1]
            temp_filename = f"temp_{uuid.uuid4().hex}{file_ext}"
            temp_path = os.path.join("temp_uploads", temp_filename)

            os.makedirs("temp_uploads", exist_ok=True)

            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_paths.append(temp_path)

        import json
        parsed_columns = None
        if columns:
            try:
                parsed_columns = json.loads(columns)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON format for 'columns'"
                )

        if not mode:
            if len(saved_paths) > 1:
                if parsed_columns and isinstance(parsed_columns, dict):
                    mode = "multiple-columns"
                else:
                    mode = "multiple"
            else:
                if parsed_columns and isinstance(parsed_columns, list):
                    mode = "single-columns"
                else:
                    mode = "single"

        print(f"Resolved mode: {mode}")

        try:
            if mode == "single":
                result = text_analyze_data(
                    saved_paths[0], query, user_id
                )

            elif mode == "single-columns":
                result = analyze_data_with_columns(
                    saved_paths[0],
                    query,
                    selected_columns=parsed_columns,
                    user_id=user_id
                )

            elif mode == "multiple":
                result = analyze_multiple_files(
                    saved_paths,
                    query,
                    selected_columns_map=None,
                    user_id=user_id
                )

            elif mode == "multiple-columns":
                result = analyze_multiple_files(
                    saved_paths,
                    query,
                    selected_columns_map=parsed_columns,
                    user_id=user_id
                )

            else:
                raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}")

        except Exception as processing_error:
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {str(processing_error)}"
            )

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Analysis returned no result"
            )

        # 🔹 Load output JSON if exists
        output_path = f"chat_output/{user_id}.json"

        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    output_data = json.load(f)
            except:
                output_data = result
        else:
            output_data = result

        return {
            "user_id": user_id,
            "analysis_type": "query_response",
            "mode": mode,
            "result": output_data,
            "output_json_path": output_path,
            "files_processed": [file.filename for file in files]
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error generating text analysis: {str(e)}"
        )


@app.post("/process-all", summary="Process data through the entire pipeline")
async def process_all(request: ProcessAllRequest):
    """
    Process data through the entire pipeline: upload, analyze quality, clean, analyze, and export.
    
    Returns results from all steps.
    """
    try:
        
        quality_cb = OpenAICallbackHandler()
        report_cb = OpenAICallbackHandler()
        clean_cb = OpenAICallbackHandler()
        
        
        df = load_data(request.file_path)
        
        if df is None:
            raise HTTPException(status_code=400, detail="Failed to load data file")
        
        
        user_data[request.user_id] = {
            "original_data": df,
            "file_name": os.path.basename(request.file_path),
            "file_path": request.file_path,
            "cleaned_data": None,
            "quality_issues": None,
            "quality_report": None,
            "cleaning_report": None,
            "analysis_report": None,
            "token_usage": {}
        }
        
        
        with get_openai_callback() as cb:
            quality_issues = analyze_data_quality(df)
            quality_cb = cb
            
        with get_openai_callback() as cb:
            quality_report = generate_quality_report(df)
            report_cb = cb
        
        
        cleaning_options = request.cleaning_options.dict() if request.cleaning_options else None
        with get_openai_callback() as cb:
            cleaned_df, cleaning_report = clean_data(df, cleaning_options)
            clean_cb = cb
        
        if cleaned_df is None:
            raise HTTPException(status_code=500, detail="Data cleaning failed")
        
        
        user_data[request.user_id]["cleaned_data"] = cleaned_df
        user_data[request.user_id]["cleaning_report"] = cleaning_report
        
        
        analysis_report, analysis_token_usage = track_analysis_report_tokens(
            cleaned_df, 
            focus=request.analysis_type
        )
            
        user_data[request.user_id]["analysis_report"] = analysis_report
        
        
        quality_token_usage = calculate_token_cost(quality_cb)
        report_token_usage = calculate_token_cost(report_cb)
        clean_token_usage = calculate_token_cost(clean_cb)
        
        
        total_token_usage = {
            "total_tokens": (quality_token_usage["total_tokens"] + report_token_usage["total_tokens"] + 
                            clean_token_usage["total_tokens"] + analysis_token_usage["total_tokens"]),
            "prompt_tokens": (quality_token_usage["prompt_tokens"] + report_token_usage["prompt_tokens"] + 
                             clean_token_usage["prompt_tokens"] + analysis_token_usage["prompt_tokens"]),
            "completion_tokens": (quality_token_usage["completion_tokens"] + report_token_usage["completion_tokens"] + 
                                 clean_token_usage["completion_tokens"] + analysis_token_usage["completion_tokens"]),
            "total_cost": round(quality_token_usage["total_cost"] + report_token_usage["total_cost"] + 
                              clean_token_usage["total_cost"] + analysis_token_usage["total_cost"], 6)
        }
        
        
        user_data[request.user_id]["token_usage"] = {
            "analyze_quality": quality_token_usage,
            "quality_report": report_token_usage,
            "clean_data": clean_token_usage,
            "analyze_data": analysis_token_usage,
            "total": total_token_usage
        }
        
        quality_summary = {
            "total_missing_percent": quality_issues.get("total_missing_percent", 0.0),
            "total_outlier_percent": quality_issues.get("total_outlier_percent", 0.0),
            "total_duplicate_percent": quality_issues.get("total_duplicate_percent", 0.0),
            "data_type_issues": quality_issues.get("data_type_issues", [])
        }
        
        base_filename = os.path.splitext(os.path.basename(request.file_path))[0]
        base_filename = f"cleaned_{base_filename}"
        
        exported_files = export_data(cleaned_df, formats=request.export_formats, base_filename=base_filename)
        zip_path = create_zip_archive(cleaned_df, base_filename=base_filename)
        
        
        return {
            "user_id": request.user_id,
            "file_info": {
                "file_name": os.path.basename(request.file_path),
                "original_rows": len(df),
                "original_columns": len(df.columns)
            },
            "quality_analysis": {
                "quality_issues": quality_issues,
                "quality_summary": quality_summary,
                "quality_report": quality_report,
                "token_usage": {
                    "analyze_quality": quality_token_usage,
                    "quality_report": report_token_usage
                }
            },
            "cleaning": {
                "cleaning_report": cleaning_report,
                "cleaned_rows": len(cleaned_df),
                "cleaned_columns": len(cleaned_df.columns),
                "token_usage": clean_token_usage
            },
            "analysis": {
                "analysis_type": request.analysis_type,
                "analysis_report": analysis_report,
                "token_usage": analysis_token_usage
            },
            "export": {
                "exported_files": exported_files,
                "zip_archive": zip_path
            },
            "token_usage": total_token_usage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")


@app.post("/connect/postgres")
async def connect_postgres(req: PostgresConnectRequest):
    try:
        session = get_session(req.user_id)
        password = quote_plus(req.password)

        uri = f"postgresql://{req.username}:{password}@{req.host}:{req.port}/{req.database}"
        engine = create_engine(uri)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        sql_db = SQLDatabase(engine)
        tools = create_sql_tools(sql_db)
        agent = create_agent(llm, tools)

        session["connections"]["postgres"] = sql_db
        session["agents"]["postgres"] = agent

        session["schemas"]["postgres"] = extract_sql_schema(sql_db)

        return {"message": "PostgreSQL connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/connect/mysql")
async def connect_mysql(req: MySQLConnectRequest):
    try:
        session = get_session(req.user_id)
        password = quote_plus(req.password)

        uri = f"mysql+pymysql://{req.username}:{password}@{req.host}:{req.port}/{req.database}"
        engine = create_engine(uri)

        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        sql_db = SQLDatabase(engine)
        tools = create_sql_tools(sql_db)
        agent = create_agent(llm, tools)

        session["connections"]["mysql"] = sql_db
        session["agents"]["mysql"] = agent
        session["schemas"]["mysql"] = extract_sql_schema(sql_db) 

        return {"message": "MySQL connected"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/connect/mongodb")
async def connect_mongodb(req: MongoConnectRequest):
    try:
        session = get_session(req.user_id)
        client = MongoClient(req.uri)
        db = client[req.database]
        db.list_collection_names()

        tools = create_mongo_tools(db)
        agent = create_agent(llm, tools)

        session["connections"]["mongodb"] = db
        session["agents"]["mongodb"] = agent

        session["schemas"]["mongodb"] = extract_mongo_schema(db)

        return {"message": "MongoDB connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def extract_mongo_schema(db) -> dict:
    """Sample one document per collection to infer field names."""
    schema = {}
    try:
        for col_name in db.list_collection_names():
            sample = db[col_name].find_one({}, {"_id": 0})
            schema[col_name] = list(sample.keys()) if sample else []
    except Exception as e:
        print(f"Mongo schema extraction error: {e}")
    return schema


def extract_chart_data_from_reply(reply_text: str) -> pd.DataFrame:
    """
    Fallback: parse structured data from LLM reply text when
    the table_name points to raw data instead of aggregated result.
    Handles formats like:
    - MODEL 3: 68234
    - d001: 79,879.18
    """
    try:
        pattern = r'-\s*([^–—\-\n]+?)\s*[–—\-:]+\s*([\d,\.]+)'
        matches = re.findall(pattern, reply_text)

        if not matches:
            return pd.DataFrame()

        rows = []
        for label, value in matches:
            clean_value = float(value.replace(",", ""))
            rows.append({"label": label.strip(), "value": clean_value})

        return pd.DataFrame(rows)

    except Exception as e:
        print(f"Reply parsing error: {e}")
        return pd.DataFrame()


@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        session = get_session(req.user_id)

        has_dbs  = bool(req.db_types)
        has_files = bool(req.file_paths)

        if not has_dbs and not has_files:
            raise HTTPException(status_code=400, detail="Provide at least one db_type or file_paths")

        if has_files:
            invalid_files = [fp for fp in req.file_paths if not os.path.exists(fp)]
            if invalid_files:
                raise HTTPException(
                    status_code=400,
                    detail=f"Files not found: {invalid_files}"
        )

        all_data     = []
        final_answer = None
        chart_type   = "none"
        x_label      = None
        y_label      = None
        df           = pd.DataFrame()

        total_input_cost  = 0
        total_output_cost = 0
        total_cost        = 0

        agents_to_run = [] 

        if has_dbs:
            selected_dbs = route_databases(req.query, req.db_types)
            for db in selected_dbs:
                agent = session.get("agents", {}).get(db)
                if not agent:
                    continue
                schema = session.get("schemas", {}).get(db, {})
                agents_to_run.append((db, agent, schema))

        if has_files:
            for file_path in req.file_paths:
                try:
                    file_tools = create_file_tools(file_path)
                    file_agent = create_agent(llm, file_tools)

                    file_schema = {
                        "source": "file",
                        "file": os.path.basename(file_path)
                    }

                    agents_to_run.append(
                        (os.path.basename(file_path), file_agent, file_schema)
                    )

                except Exception as e:
                    print(f"Error creating agent for {file_path}: {e}")

        for source_name, agent, schema in agents_to_run:
            prompt = build_prompt(req.query, schema)

            response = agent.invoke({
                "messages": [HumanMessage(content=prompt)]
            })

            cost_info = calculate_cost(response)
            total_input_cost  += cost_info["input_cost_usd"]
            total_output_cost += cost_info["output_cost_usd"]
            total_cost        += cost_info["total_cost_usd"]

            raw_output = response["messages"][-1].content

            try:
                parsed = json.loads(raw_output)
            except Exception:
                try:
                    cleaned = re.sub(r"```(?:json)?", "", raw_output).strip().rstrip("```").strip()
                    parsed  = json.loads(cleaned)
                except Exception:
                    continue

            data = parsed.get("data", [])
            if data:
                for row in data:
                    row["source"] = source_name
                all_data.extend(data)

            if not final_answer:
                final_answer = parsed.get("answer")

            if chart_type == "none":
                chart_type = parsed.get("chart_type", "none")

            x_label = parsed.get("x_label") or x_label
            y_label = parsed.get("y_label") or y_label

        x_label = x_label or "Category"
        y_label = y_label or "Value"

        normalized_data = normalize_chart_data(all_data)

        if normalized_data:
            df = pd.DataFrame(normalized_data)

            if not df.empty:
                if df["value"].notnull().sum() > 0:
                    df = df.dropna(subset=["value"])
                    df = df.groupby("label", as_index=False)["value"].mean()

                else:
                    df = df.groupby("label", as_index=False).size()
                    df = df.rename(columns={"size": "value"})

        answer = (
            generate_detailed_answer(df, x_label, y_label, req.query)
            if not df.empty
            else final_answer or "No data found."
        )

        chart_path, summary = None, None
        gemini_cost = 0.0

        if chart_type != "none" and not df.empty:
            chart_path, summary, gemini_cost = await generate_chart_with_summary(
                df.to_dict(orient="records"),
                chart_type,
                req.user_id,
                x_label=x_label,
                y_label=y_label
            )
        
        chat_record = {
            "user_id": req.user_id,
            "query": req.query,
            "answer": answer,
            "filepath": chart_path,
            "summary": summary
        }

        total_cost += gemini_cost

        try:
            chat_collection.insert_one(chat_record)
        except Exception as db_error:
            print("MongoDB insert error:", db_error)

        return {
            "query":      req.query,
            "answer":     answer,
            "chart_path": chart_path,
            "summary":    summary,
            "sources":    list({row.get("source") for row in all_data if row.get("source")}),
            "cost": {
                "input_cost_usd":  round(total_input_cost,  6),
                "output_cost_usd": round(total_output_cost, 6),
                "total_cost_usd":  round(total_cost,        6)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def route_databases(query: str, available_dbs: list[str]) -> list[str]:
    prompt = f"""
    You are a database router.

    Available databases:
    {available_dbs}

    User query:
    "{query}"

    Return ONLY a JSON list of relevant databases from the available ones.
    Example:
    ["postgres", "mongodb"]

    If unsure, return all.
    """

    response = llm.invoke(prompt)

    try:
        dbs = json.loads(response.content)
        return [db for db in dbs if db in available_dbs]
    except Exception:
        return available_dbs  


def normalize_chart_data(data):
    normalized = []

    for row in data:
        if not isinstance(row, dict) or not row:
            continue

        label = None
        value = None

        numeric_fields = []
        categorical_fields = []

        for k, v in row.items():
            if isinstance(v, (int, float)):
                numeric_fields.append((k, v))
            elif isinstance(v, str):
                categorical_fields.append((k, v))

        if categorical_fields:
            label = categorical_fields[0][1]

        if numeric_fields:
            key, val = numeric_fields[0]

            if "id" not in key.lower():
                value = val

        normalized.append({
            "label": label,
            "value": value,  
            "source": row.get("source")
        })

    return normalized


def generate_detailed_answer(df, x_label, y_label, original_query: str = ""):
    try:
        if df is None or df.empty:
            return "No data available."

        data_str = df.to_string(index=False)

        prompt = f"""
        You are a data analyst. A user asked the following question:
        "{original_query}"

        Here is the relevant data retrieved:
        {data_str}

        X-axis represents: {x_label}
        Y-axis represents: {y_label}

        Write a clear, concise, natural language answer to the user's question based on the data above.
        Do not repeat the raw data. Just answer the question directly and meaningfully.
        """

        response = llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print("Answer generation error:", e)
        return "Could not generate detailed answer."


@app.get("/chat-history/{user_id}")
def get_chat_history(user_id: str):
    try:
        records = list(
            chat_collection.find(
                {"user_id": user_id},
                {"_id": 0}  
            ).sort("_id", -1) 
        )

        if not records:
            return {
                "user_id": user_id,
                "history": [],
                "message": "No chat history found."
            }

        return {
            "user_id": user_id,
            "history": records
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)