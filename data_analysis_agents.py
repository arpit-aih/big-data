import os
import json
from aih_automaton import Agent, Task as BaseTask, LinearSyncPipeline
from aih_automaton.tasks.task_literals import OutputType

class Task(BaseTask):
    """Custom Task class to restore functions and function_call support"""
    def __init__(self, **kwargs):
        self.functions = kwargs.pop('functions', None)
        self.function_call = kwargs.pop('function_call', None)
        super().__init__(**kwargs)
        # Re-assign _execute_task to our method because super().__init__ shadows it
        self._execute_task = self._custom_execute_task

    def _create_task_execution_method(self):
        # Do nothing to prevent base class from setting its own lambdas
        pass

    def _custom_execute_task(self):
        # Custom execution logic that supports functions
        # Simplified phrasing to avoid triggering content filters (jailbreak detection)
        system_persona = f"Role: {self.agent.role}\n{self.agent.prompt_persona}"
        prompt = self.instructions
        
        if self.functions:
            return self.model.generate_text(
                task_id=self.task_id,
                system_persona=system_persona,
                prompt=f"{prompt}  Input: {self.previous_output} {self.default_input}",
                functions=self.functions,
                function_call=self.function_call
            )
        
        # Fallback to original base class behavior for other output types
        if self.output_type == OutputType.IMAGE:
            return self._generate_image(f"{system_persona} {prompt}")
        if self.output_type == OutputType.TOOL:
            if self.tool is not None:
                return self._execute_tool(system_persona, prompt)
            else:
                # If no tool but output_type is TOOL, fallback to text or error
                pass
        
        return self._generate_text(system_persona=system_persona, prompt=prompt)
        
from dotenv import load_dotenv
from azure.identity import ClientSecretCredential

from pandasai_openai import AzureOpenAI


try:
    from azure_openai import AzureOpenAIModel
except ImportError:
    
    class AzureOpenAIModel:
        def __init__(self, azure_endpoint, azure_api_key, azure_api_version):
            self.azure_endpoint = azure_endpoint
            self.azure_api_key = azure_api_key
            self.azure_api_version = azure_api_version
            self.deployment = "nexa1-gpt-4.1-global-std"  
            
            
            from openai import AzureOpenAI
            load_dotenv()
            engine=os.getenv("Engine")
            
            tenant_id=os.getenv("tenant_id")
            client_id=os.getenv("client_id")
            client_secret=os.getenv("Secret_Value")

  
            credential = ClientSecretCredential(
    tenant_id,client_id,client_secret
    )

            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("End_point"),
                api_version=os.getenv("API_version"),
                deployment_name=engine,
                azure_ad_token=credential.get_token("https://cognitiveservices.azure.com/.default").token
            )

        
        def chat_completion(self, **kwargs):
            kwargs['temperature'] = kwargs.get('temperature', 0)
            return self.client.chat.completions.create(
                model=self.deployment,
                **kwargs
            )


azure_api_key = os.getenv("API_Key")
azure_endpoint = os.getenv("End_point")
azure_api_version = os.getenv("API_version")
base_url = os.getenv("End_point")


client = AzureOpenAIModel(
    azure_api_key=azure_api_key,
    azure_api_version=azure_api_version,
    azure_endpoint=base_url
)


comprehensive_report_schema = {
    "name": "generate_comprehensive_report",
    "description": "Generate a comprehensive data analysis report",
    "parameters": {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "A high-level summary of the dataset"
            },
            "key_statistics": {
                "type": "string",
                "description": "Key statistical findings from the dataset"
            },
            "patterns": {
                "type": "string",
                "description": "Important patterns and trends identified in the data"
            },
            "insights": {
                "type": "string",
                "description": "Meaningful insights derived from the data analysis"
            },
            "recommendations": {
                "type": "string",
                "description": "Actionable recommendations based on the analysis"
            }
        },
        "required": ["summary", "key_statistics", "patterns", "insights", "recommendations"]
    }
}

technical_details_schema = {
    "name": "generate_technical_details",
    "description": "Generate technical data analysis details",
    "parameters": {
        "type": "object",
        "properties": {
            "technical_summary": {
                "type": "string",
                "description": "A technical summary of the dataset structure and quality"
            },
            "statistical_analysis": {
                "type": "string",
                "description": "Detailed statistical analysis of the dataset"
            },
            "data_distribution": {
                "type": "string",
                "description": "Analysis of data distributions and outliers"
            },
            "correlation_analysis": {
                "type": "string",
                "description": "Analysis of correlations between variables"
            },
            "technical_recommendations": {
                "type": "string",
                "description": "Technical recommendations for further analysis"
            }
        },
        "required": ["technical_summary", "statistical_analysis", "data_distribution", "correlation_analysis", "technical_recommendations"]
    }
}


def generate_executive_summary(df):
    """Generate an executive summary report using an agent"""
    
    data_summary = create_data_summary(df)
    
    
    system_content = """
    You are 'Executive Analytics Advisor', a specialized AI for generating concise executive summaries of data analysis.
    Your task is to analyze the provided dataset summary and create a focused, high-level executive summary
    that highlights only the most critical insights and strategic recommendations.

    Your analysis should be:
    1. Extremely concise and focused
    2. Based solely on the data provided
    3. Written for C-level executives with limited time
    4. Focused only on strategic-level findings and implications
    5. Free of technical details and jargon

    Use the generate_executive_summary function to structure your response."""

    prompt = f"""
    Please provide an executive summary of data analysis using the generate_executive_summary schema.

    Here is the dataset summary:
    {json.dumps(data_summary, indent=2)}

    Your executive summary should include:
    - A concise 2-3 sentence overview of the data analysis
    - Only the 1-2 most critical findings that require executive attention
    - Strategic implications of these findings for the organization
    - Clear, high-level strategic recommendations (no more than 2)

    Use direct, assertive language appropriate for C-level executives. The entire summary should be readable in under 1 minute.
    """

    agent = Agent(role="executive analytics advisor", prompt_persona=prompt)
    task = Task(
        model=client,
        agent=agent,
        function_call='auto',
        functions=[executive_summary_schema],
        output_type=OutputType.TOOL,
        instructions=system_content
    )
    
    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Executive Summary Generated")
    result = pipeline.run()
    
    
    output_data = result[0]['task_output'].arguments
    report_data = json.loads(output_data)
    
    
    markdown_report = f"""

    {report_data['executive_summary']}


    {report_data['key_findings']}


    {report_data['strategic_implications']}


    {report_data['strategic_recommendations']}
    """
    
    return markdown_report


executive_summary_schema = {
    "name": "generate_executive_summary",
    "description": "Generate an executive summary of data analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "executive_summary": {
                "type": "string",
                "description": "A concise executive summary of the data analysis"
            },
            "key_findings": {
                "type": "string",
                "description": "The most critical findings for executive attention"
            },
            "strategic_implications": {
                "type": "string", 
                "description": "Strategic implications of the data analysis"
            },
            "strategic_recommendations": {
                "type": "string",
                "description": "High-level strategic recommendations"
            }
        },
        "required": ["executive_summary", "key_findings", "strategic_implications", "strategic_recommendations"]
    }
}


business_insights_schema = {
    "name": "generate_business_insights",
    "description": "Generate business-focused insights from data analysis",
    "parameters": {
        "type": "object",
        "properties": {
            "business_summary": {
                "type": "string",
                "description": "A business-oriented summary of the dataset"
            },
            "key_metrics": {
                "type": "string",
                "description": "Key business metrics and KPIs identified in the data"
            },
            "business_impact": {
                "type": "string",
                "description": "Analysis of business impact based on the data"
            },
            "action_items": {
                "type": "string",
                "description": "Specific business action items recommended"
            },
            "competitive_advantage": {
                "type": "string",
                "description": "Opportunities for competitive advantage based on the data"
            }
        },
        "required": ["business_summary", "key_metrics", "business_impact", "action_items"]
    }
}


def create_data_summary(df):
    """Create a summary of the dataframe to include in the prompt"""
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "numeric_columns": df.select_dtypes(include=['number']).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
        "temporal_columns": df.select_dtypes(include=['datetime']).columns.tolist()
    }
    
    if summary["numeric_columns"]:
        summary["numeric_stats"] = df[summary["numeric_columns"]].describe().to_dict()
    
    
    if summary["categorical_columns"]:
        cat_info = {}
        for col in summary["categorical_columns"]:
            value_counts = df[col].value_counts()
            if len(value_counts) <= 10:  
                cat_info[col] = value_counts.to_dict()
            else:
                cat_info[col] = {"unique_values": df[col].nunique()}
        summary["categorical_info"] = cat_info
    
    return summary


def generate_business_insights(df):
    """Generate a business-focused insights report using an agent"""
    # Create data summary
    data_summary = create_data_summary(df)
    
    # System prompt
    system_content = """
    You are 'Business Analytics Strategist', a specialized AI for generating business-focused data insights.
    Your task is to analyze the provided dataset summary and extract key business insights, metrics,
    and actionable recommendations that would be valuable to business stakeholders.

    Your analysis should be:
    1. Focused on business value and impact
    2. Based solely on the data provided
    3. Written in business language, avoiding technical jargon
    4. Actionable with specific business recommendations
    5. Structured to highlight competitive advantages and opportunities

    Use the generate_business_insights function to structure your response.
    """

        # User prompt with data summary
    prompt = f"""
    Please provide a business insights report using the generate_business_insights schema.

    Here is the dataset summary:
    {json.dumps(data_summary, indent=2)}

    Your report should include:
    - A business-oriented summary of the dataset
    - Key business metrics and KPIs identified
    - Analysis of business impact based on the data
    - Specific business action items recommended
    - Opportunities for competitive advantage

    Focus on business outcomes, revenue impact, customer metrics, operational efficiency,
    and market positioning. Avoid technical statistics terminology.
    """

    # Create agent, task and pipeline
    agent = Agent(role="business analytics strategist", prompt_persona=prompt)
    task = Task(
        model=client,
        agent=agent,
        function_call='auto',
        functions=[business_insights_schema],
        output_type=OutputType.TOOL,
        instructions=system_content
    )
    
    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Business Insights Generated")
    result = pipeline.run()
    
    # Process the output
    output_data = result[0]['task_output'].arguments
    report_data = json.loads(output_data)
    
    # Format the report as Markdown
    markdown_report = f"""
    # Business Insights Report

    ## Business Summary
    {report_data['business_summary']}

    ## Key Business Metrics
    {report_data['key_metrics']}

    ## Business Impact Analysis
    {report_data['business_impact']}

    ## Recommended Action Items
    {report_data['action_items']}
    """

    # Add competitive advantage section if available
    if 'competitive_advantage' in report_data and report_data['competitive_advantage']:
        markdown_report += f"\n## Competitive Advantage Opportunities\n{report_data['competitive_advantage']}"
    
    return markdown_report


def generate_technical_details(df):
    """Generate a technical data analysis report using an agent"""
    
    data_summary = create_data_summary(df)
    
    system_content = """
    You are 'Technical Data Scientist', a specialized AI for generating detailed technical data analysis.
    Your task is to analyze the provided dataset summary and create a comprehensive technical report
    with statistical analyses, distribution insights, correlation examinations, and technical recommendations.

    Your analysis should be:
    1. Technically precise and detailed
    2. Based solely on the data provided
    3. Include relevant statistical terminology and measures
    4. Provide technical recommendations for further analysis
    5. Structured for data professionals and analysts

    Use the generate_technical_details function to structure your response.
    """
    
    prompt = f"""
    Please provide a technical data analysis report using the generate_technical_details schema.

    Here is the dataset summary:
    {json.dumps(data_summary, indent=2)}

    Your report should include:
    - A technical summary of the dataset structure and quality
    - Detailed statistical analysis (including measures of central tendency, dispersion, etc.)
    - Analysis of data distributions, skewness, kurtosis, and outliers
    - Correlation analysis between variables and potential relationships
    - Technical recommendations for further data analysis, transformations, or modeling approaches

    Include appropriate statistical terminology and technical details that would be relevant for a data scientist.
    """
    
    agent = Agent(role="technical data scientist", prompt_persona=prompt)
    task = Task(
        model=client,
        agent=agent,
        function_call='auto',
        functions=[technical_details_schema],
        output_type=OutputType.TOOL,
        instructions=system_content
    )
    
    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Technical Details Generated")
    result = pipeline.run()
    
    
    output_data = result[0]['task_output'].arguments
    report_data = json.loads(output_data)
    
    
    markdown_report = f"""

    {report_data['technical_summary']}


    {report_data['statistical_analysis']}


    {report_data['data_distribution']}


    {report_data['correlation_analysis']}


    {report_data['technical_recommendations']}
    """
    
    return markdown_report


def generate_comprehensive_report(df):
    """Generate a comprehensive data analysis report using an agent"""
    
    data_summary = create_data_summary(df)
    
    
    system_content = """
    You are 'Data Analysis Expert', a specialized AI for generating comprehensive data analysis reports.
    Your task is to analyze the provided dataset summary and create a detailed, insightful report that
    covers statistical findings, patterns, insights, and actionable recommendations.

    Your analysis should be:
    1. Thorough and comprehensive
    2. Based solely on the data provided
    3. Presented in a clear, organized structure
    4. Actionable with specific, concrete recommendations
    5. Written in a professional but accessible style

    Use the generate_comprehensive_report function to structure your response.
    """
    
    prompt = f"""
    Please provide a comprehensive analysis report using the generate_comprehensive_report schema.

    Here is the dataset summary:
    {json.dumps(data_summary, indent=2)}

    Your report should include:
    - A high-level summary of the dataset 
    - Key statistical findings (distributions, central tendencies, outliers)
    - Important patterns and trends identified
    - Meaningful insights derived from the data
    - Actionable recommendations based on your analysis

    Structure your response with clear sections and formatting for readability.
    """
    
    agent = Agent(role="data analysis expert", prompt_persona=prompt)
    task = Task(
        model=client,
        agent=agent,
        function_call='auto',
        functions=[comprehensive_report_schema],
        output_type=OutputType.TOOL,
        instructions=system_content
    )
    
    pipeline = LinearSyncPipeline(tasks=[task], completion_message="Comprehensive Report Generated")
    result = pipeline.run()
    
    
    output_data = result[0]['task_output'].arguments
    report_data = json.loads(output_data)
    
    
    markdown_report = f"""

    {report_data['summary']}


    {report_data['key_statistics']}


    {report_data['patterns']}


    {report_data['insights']}


    {report_data['recommendations']}
    """
    
    return markdown_report


def generate_agent_report(df, focus="comprehensive"):
    """Generate a data analysis report using specialized agents based on focus type"""
    if focus == "business":
        return generate_business_insights(df)
    elif focus == "technical":
        return generate_technical_details(df)
    elif focus == "executive":
        return generate_executive_summary(df)
    else:  
        return generate_comprehensive_report(df)