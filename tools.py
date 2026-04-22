import json, os
import pandas as pd
from typing import Dict, Any
from sqlalchemy import  text
from langchain_core.tools import tool
from visualization import DuckDBManager


duckdb_manager = DuckDBManager()


def create_sql_tools(sql_db):

    @tool
    def query_sql(query: str) -> str:
        """
         Execute a SAFE SQL SELECT query.
         Stores result in DuckDB and returns table reference + preview.
         """
        try:
            forbidden = ["drop", "delete", "truncate", "update", "insert", "alter"]
            if any(word in query.lower() for word in forbidden):
                return "Error: Only SELECT queries are allowed"

            query = query.strip().rstrip(";")

            MAX_ROWS = 10_000

            with sql_db._engine.connect() as conn:
                result = conn.execute(text(query))
                rows = result.fetchmany(MAX_ROWS)
                columns = list(result.keys())
                was_capped = result.fetchone() is not None 

            df = pd.DataFrame(rows, columns=columns)

            if df.empty:
                return json.dumps({
                    "table_name": None,
                    "columns": [],
                    "preview": [],
                    "row_count": 0,
                    "capped": False
                })

            table_name = duckdb_manager.store_dataframe(df)

            return json.dumps({
                "table_name": table_name,
                "columns": columns,
                "preview": df.head(5).to_dict(orient="records"),
                "row_count": len(df),
                "capped": was_capped
            })

        except Exception as e:
            return f"SQL Error: {str(e)}"

    return [query_sql]


def create_mongo_tools(mongo_db):

    @tool
    def query_mongo(collection_name: str, query_filter: Dict[str, Any]) -> str:
        """
        Query a MongoDB collection and return results.
        
        Args:
            collection_name: Name of the collection to query.
            query_filter: MongoDB filter dict. Use {} to fetch all documents.
                          Examples:
                          - {} → fetch all documents
                          - {"status": "active"} → filter by field
                          - {"age": {"$gt": 30}} → use MongoDB operators
        
        Stores results in DuckDB and returns table reference + preview.
        """

        try:
            if collection_name not in mongo_db.list_collection_names():
                return f"Error: Collection '{collection_name}' does not exist"

            collection = mongo_db[collection_name]
            cursor = collection.find(query_filter, {"_id": 0})
            results = list(cursor)

            if not results:
                return json.dumps({
                    "table_name": None,
                    "columns": [],
                    "preview": [],
                    "row_count": 0
                })

            df = pd.json_normalize(results)
            table_name = duckdb_manager.store_dataframe(df)

            return json.dumps({
                "table_name": table_name,
                "columns": list(df.columns),
                "preview": df.head(5).to_dict(orient="records"),
                "row_count": len(df)
            })

        except Exception as e:
            return f"Mongo Error: {str(e)}"

    return [query_mongo]

