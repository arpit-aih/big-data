import os, json, asyncio, tiktoken
import uuid
import duckdb
import logging
import pandas as pd
from google import genai
from google.genai import types
from threading import Lock
import plotly.express as px
from decimal import Decimal
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

_executor = ThreadPoolExecutor(max_workers=4)

logger = logging.getLogger(__name__)

GEMINI_MODELS = ["gemini-3-flash-preview", "gemini-3-pro-preview"]

encoding = tiktoken.get_encoding("cl100k_base")

INPUT_COST_PER_TOKEN = 0.50 / 1_000_000
OUTPUT_COST_PER_TOKEN = 3.00 / 1_000_000


client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY"),
    http_options={'api_version': 'v1beta'}
)

CHART_DIR = "graph"
os.makedirs(CHART_DIR, exist_ok=True)


def count_tokens(text: str) -> int:
    return len(encoding.encode(text))


def generate_chart(data: list, chart_type: str, session_id: str, x_label="Category", y_label="Value") -> str:
    try:
        df = pd.DataFrame(data)

        if df.empty or len(df.columns) < 2:
            return None

        x_col = df.columns[0]
        y_col = df.columns[1]

        df = df.rename(columns={
            x_col: x_label,
            y_col: y_label
        })

        x_col = x_label
        y_col = y_label

        colors = px.colors.qualitative.Bold

        if chart_type == "bar":
            fig = px.bar(
                df,
                x=x_col,
                y=y_col,
                color=x_col,  
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

        session_chart_dir = os.path.join(CHART_DIR, session_id)
        os.makedirs(session_chart_dir, exist_ok=True)

        filename = f"{uuid.uuid4().hex}.html"
        filepath = os.path.join(session_chart_dir, filename)

        filepath = os.path.abspath(filepath)
        fig.write_html(filepath)

        return filepath, fig

    except Exception as e:
        print("Chart generation error:", e)
        return None, None


async def generate_chart_with_summary(data, chart_type, session_id, x_label, y_label):
    loop = asyncio.get_event_loop()

    html_path, fig = await loop.run_in_executor(
        _executor,
        lambda: generate_chart(data, chart_type, session_id, x_label, y_label)
    )

    if not html_path or fig is None:
        return None, None, 0

    try:
        fig_json = fig.to_json()

        prompt = f"""
        You are a data analyst assistant.
        Analyze this Plotly chart JSON and provide a detailed summary including:
        - What the chart represents
        - Key trends or patterns
        - Any correlations between variables
        - Notable outliers
        - Business insights

        Chart JSON:
        {fig_json}

        Return a JSON object with exactly this field:
        - "summary": A detailed string summary of the chart analysis.
        """

        input_tokens = count_tokens(prompt)

        for model_name in GEMINI_MODELS:
            try:
                logger.info(f"Attempting chart summary with model: {model_name}")

                response = await loop.run_in_executor(
                    _executor,
                    lambda: client.models.generate_content(
                        model=model_name,
                        contents=[prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.1
                        )
                    )
                )

                response_text = response.text
                output_tokens = count_tokens(response_text)

                input_cost = input_tokens * INPUT_COST_PER_TOKEN
                output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
                total_cost = input_cost + output_cost

                result_json = json.loads(response_text)
                summary = result_json.get("summary")

                if summary:
                    logger.info(f"Successfully got summary from {model_name}")
                    return html_path, summary, total_cost
                else:
                    logger.warning(f"{model_name} returned JSON without summary field")
                    continue

            except Exception as e:
                logger.error(f"Gemini API request failed with {model_name}: {e}")
                if model_name == GEMINI_MODELS[-1]:
                    logger.error("All Gemini models failed for chart summary")
                    return html_path, None
                else:
                    logger.info(f"Falling back to next model...")
                    continue

    except Exception as e:
        logger.error(f"Chart summary generation error: {e}")
        return html_path, None, 0
        

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