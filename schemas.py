from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class OutlierStrategy(str, Enum):
    auto = "auto"
    clip = "clip"
    remove = "remove"


class MissingStrategy(str, Enum):
    auto = "auto"
    mean = "mean"
    median = "median"
    mode = "mode"
    constant = "constant"
    drop_rows = "drop_rows"
    drop_columns = "drop_columns"


class CleaningOptions(BaseModel):
    remove_duplicates: bool = Field(True, description="Whether to remove duplicate rows")
    handle_missing: bool = Field(True, description="Whether to handle missing values")
    missing_strategy: MissingStrategy = Field(MissingStrategy.auto, description="Strategy for handling missing values")
    constant_value: Optional[float] = Field(None, description="Value to use when missing_strategy is 'constant'")
    missing_threshold: Optional[int] = Field(None, description="Threshold for dropping columns with missing values (percentage)")
    handle_outliers: bool = Field(True, description="Whether to handle outliers")
    outlier_strategy: OutlierStrategy = Field(OutlierStrategy.auto, description="Strategy for handling outliers")
    outlier_columns: List[str] = Field([], description="Columns to check for outliers")
    fix_data_types: bool = Field(True, description="Whether to fix data types")
    use_llm_cleaning: bool = Field(True, description="Whether to use LLM for advanced cleaning")


class CleaningRequest(BaseModel):
    user_id: str = Field(..., description="User ID to identify the uploaded data")
    options: CleaningOptions = Field(..., description="Cleaning options")


class AnalysisType(str, Enum):
    comprehensive = "comprehensive"
    business = "business"
    technical = "technical"
    executive = "executive" 


class AnalysisRequest(BaseModel):
    user_id: str = Field(..., description="User ID to identify the cleaned data")
    analysis_type: AnalysisType = Field(AnalysisType.comprehensive, description="Type of analysis to perform")


class ExportRequest(BaseModel):
    user_id: str = Field(..., description="User ID to identify the cleaned data")
    formats: List[str] = Field(["csv", "excel", "parquet"], description="Export formats")
    create_zip: bool = Field(True, description="Whether to create a ZIP archive")


class ProcessAllRequest(BaseModel):
    user_id: str = Field(..., description="User ID for tracking the data processing")
    file_path: str = Field(..., description="Path to the data file")
    cleaning_options: Optional[CleaningOptions] = Field(None, description="Cleaning options")
    analysis_type: AnalysisType = Field(AnalysisType.comprehensive, description="Type of analysis to perform")
    export_formats: List[str] = Field(["csv", "excel", "parquet"], description="Export formats")


class PostgresConnectRequest(BaseModel):
    user_id: str
    host: str
    port: int
    username: str
    password: str
    database: str


class MySQLConnectRequest(BaseModel):
    user_id: str
    host: str
    port: int
    username: str
    password: str
    database: str


class MongoConnectRequest(BaseModel):
    user_id: str
    uri: str
    database: str


class ChatRequest(BaseModel):
    user_id: str
    db_types: list[str] = []
    query: str
    file_path: str | None = None


class ChatResponse(BaseModel):
    reply: str


class DataQualityRequest(BaseModel):
    user_id: str = Field(..., description="User ID to identify the uploaded data")