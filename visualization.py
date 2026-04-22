import os
import uuid
import duckdb
import pandas as pd
import plotly.express as px
from threading import Lock
from decimal import Decimal


CHART_DIR = "graph"
os.makedirs(CHART_DIR, exist_ok=True)


# def generate_chart(data: list, chart_type: str, session_id: str) -> str:
#     try:
#         df = pd.DataFrame(data)

#         if df.empty or len(df.columns) < 2:
#             return None

#         x_col = df.columns[0]
#         y_col = df.columns[1]

#         if chart_type == "bar":
#             fig = px.bar(
#                 df, x=x_col, y=y_col,
#                 color=x_col,
#                 color_discrete_sequence=px.colors.qualitative.Bold
#             )
#         elif chart_type == "line":
#             fig = px.line(
#                 df, x=x_col, y=y_col,
#                 color_discrete_sequence=px.colors.qualitative.Bold
#             )
#         elif chart_type == "pie":
#             fig = px.pie(
#                 df, names=x_col, values=y_col,
#                 color_discrete_sequence=px.colors.qualitative.Bold
#             )
#         elif chart_type == "scatter":
#             fig = px.scatter(
#                 df, x=x_col, y=y_col,
#                 color=x_col,
#                 color_discrete_sequence=px.colors.qualitative.Bold
#             )
#         else:
#             return None

#         fig.update_layout(
#             plot_bgcolor="white",
#             paper_bgcolor="white",
#             font=dict(family="Arial", size=13),
#             showlegend=True,
#             legend=dict(
#                 title=dict(text=x_col.replace("_", " ").title()),
#                 orientation="v",
#                 x=1.02,
#                 y=1,
#                 xanchor="left",
#                 yanchor="top",
#                 bgcolor="rgba(255,255,255,0.8)",
#                 bordercolor="#cccccc",
#                 borderwidth=1
#             ),
#             xaxis=dict(showgrid=False),
#             yaxis=dict(gridcolor="#eeeeee"),
#             margin=dict(r=150)
#         )

#         session_chart_dir = os.path.join(CHART_DIR, session_id)
#         os.makedirs(session_chart_dir, exist_ok=True)

#         filename = f"{uuid.uuid4().hex}.html"
#         filepath = os.path.join(session_chart_dir, filename)
#         fig.write_html(filepath)

#         return filepath

#     except Exception as e:
#         print("Chart generation error:", e)
#         return None


def generate_chart(data: list, chart_type: str, session_id: str, x_label="Category", y_label="Value") -> str:
    try:
        df = pd.DataFrame(data)

        if df.empty or len(df.columns) < 2:
            return None

        x_col = df.columns[0]
        y_col = df.columns[1]

        # Rename columns for clean axis labels
        df = df.rename(columns={
            x_col: x_label,
            y_col: y_label
        })

        x_col = x_label
        y_col = y_label

        # 🔥 Use a strong color palette
        colors = px.colors.qualitative.Bold

        if chart_type == "bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=x_col,  # ✅ THIS MAKES IT COLORFUL
                color_discrete_sequence=colors
            )

        elif chart_type == "line":
            fig = px.line(
                df,
                x=x_col,
                y=y_col,
                markers=True,
                color_discrete_sequence=colors
            )

        elif chart_type == "pie":
            fig = px.pie(
                df,
                names=x_col,
                values=y_col,
                color_discrete_sequence=colors
            )

        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                color=x_col,
                size=y_col,
                color_discrete_sequence=colors
            )

        else:
            return None

        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Arial", size=13),
            xaxis_title=x_col,
            yaxis_title=y_col,
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#eeeeee"),
            legend_title_text=x_col,
            margin=dict(r=120)
        )

        # Save chart
        session_chart_dir = os.path.join(CHART_DIR, session_id)
        os.makedirs(session_chart_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.html"
        filepath = os.path.join(session_chart_dir, filename)

        filepath = os.path.abspath(filepath)
        fig.write_html(filepath)

        return filepath

    except Exception as e:
        print("Chart generation error:", e)
        return None
        

class DuckDBManager:
    def __init__(self, db_path="analytics.duckdb"):
        self.conn = duckdb.connect(db_path)
        self.lock = Lock()

    def store_dataframe(self, df: pd.DataFrame) -> str:
        table_name = f"t_{uuid.uuid4().hex}"

        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: float(x) if isinstance(x, Decimal) else x
                )

        with self.lock:
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

        return table_name

    def get_dataframe(self, table_name: str) -> pd.DataFrame:
        with self.lock:
            return self.conn.execute(f"SELECT * FROM {table_name}").fetchdf()

    def get_preview(self, table_name: str, limit: int = 5):
        with self.lock:
            return self.conn.execute(
                f"SELECT * FROM {table_name} LIMIT {limit}"
            ).fetchdf().to_dict(orient="records")

    def drop_table(self, table_name: str):
        with self.lock:
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")


def build_prompt(user_query: str, schema: dict):
    return f"""
    You are a professional data analyst for a BI system.

    SCHEMA:
    {schema}

    TOOLS:
    - SQL: query_sql(query: str)
    - Mongo: query_mongo(collection: str, filter: dict)

    CRITICAL ENFORCEMENT RULES:
    - You MUST call a tool to retrieve data BEFORE answering
    - DO NOT guess, estimate, or fabricate values
    - DO NOT answer from general knowledge
    - If no data is found, return empty data

    AGGREGATION RULES:
    - "average" → AVG()
    - "total" → SUM()
    - "count" → COUNT()
    - Per-category questions MUST use GROUP BY

    JOIN RULES:
    - ALWAYS resolve IDs to human-readable names via JOIN
    - NEVER return raw IDs

    DATA RULES:
    - ALWAYS return structured "data"
    - Max 50 rows
    - Format:
      {{ "label": "<name>", "value": <number> }}

    RESPONSE RULES:
    - "answer" = 1–2 line insight ONLY
    - NO raw data inside answer

    CHART RULES:
    - bar: categorical comparison
    - line: time series
    - pie: ≤ 10 categories
    - scatter: correlation
    - none: if not applicable

    OUTPUT FORMAT (STRICT JSON ONLY):
    {{
        "answer": "Short insight",
        "chart_type": "bar|line|pie|scatter|none",
        "x_label": "Dimension",
        "y_label": "Metric",
        "data": [
            {{"label": "<name>", "value": <number>}}
        ]
    }}

    QUESTION:
    {user_query}
    """