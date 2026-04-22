from langchain_community.callbacks.manager import OpenAICallbackHandler
from data_quality_app import generate_data_analysis_report

TOKEN_COSTS = {
    "gpt-4.1": 0.002,  
    "gpt-4.1-completion": 0.008,  
    "cached-input": 0.0005  
}

INPUT_COST_PER_TOKEN = 1.38e-06
OUTPUT_COST_PER_TOKEN = 1.1e-05


def calculate_cost(response: dict) -> dict:
    """Extract token usage and calculate cost from agent response."""
    try:
        messages = response.get("messages", [])
        
        input_tokens = 0
        output_tokens = 0

        for msg in messages:
            if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                input_tokens += msg.usage_metadata.get("input_tokens", 0)
                output_tokens += msg.usage_metadata.get("output_tokens", 0)
            # Fallback: response_metadata
            elif hasattr(msg, "response_metadata") and msg.response_metadata:
                token_usage = msg.response_metadata.get("token_usage", {})
                input_tokens += token_usage.get("prompt_tokens", 0)
                output_tokens += token_usage.get("completion_tokens", 0)

        input_cost = input_tokens * INPUT_COST_PER_TOKEN
        output_cost = output_tokens * OUTPUT_COST_PER_TOKEN
        total_cost = input_cost + output_cost

        return {
            "input_cost_usd": round(input_cost, 6),
            "output_cost_usd": round(output_cost, 6),
            "total_cost_usd": round(total_cost, 6)
        }

    except Exception as e:
        print(f"Cost calculation error: {e}")
        return {
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "total_cost_usd": 0.0
        }


def calculate_token_cost(cb):
    """Calculate token cost based on usage"""
    token_usage = {
        "total_tokens": cb.total_tokens,
        "prompt_tokens": cb.prompt_tokens,
        "completion_tokens": cb.completion_tokens,
        "input_cost": round(cb.prompt_tokens * TOKEN_COSTS["gpt-4.1"] / 1000, 6),
        "output_cost": round(cb.completion_tokens * TOKEN_COSTS["gpt-4.1-completion"] / 1000, 6),
        "total_cost": round((cb.prompt_tokens * TOKEN_COSTS["gpt-4.1"] + 
                           cb.completion_tokens * TOKEN_COSTS["gpt-4.1-completion"]) / 1000, 6)
    }
    
    
    if hasattr(cb, "cached_tokens"):
        token_usage["cached_tokens"] = getattr(cb, "cached_tokens", 0)
        token_usage["cached_cost"] = round(token_usage["cached_tokens"] * TOKEN_COSTS["cached-input"] / 1000, 6)
        token_usage["total_cost"] += token_usage["cached_cost"]
    
    return token_usage


def track_analysis_report_tokens(df, focus="comprehensive"):
    """Generate a data analysis report with token tracking"""
    
    cb = OpenAICallbackHandler()
    
    report = generate_data_analysis_report(df, focus=focus)
    
    if cb.total_tokens == 0:
        
        estimated_tokens = len(report) // 4
        
        token_usage = {
            "total_tokens": estimated_tokens,
            "prompt_tokens": estimated_tokens // 3,  
            "completion_tokens": estimated_tokens * 2 // 3,
            "input_cost": round((estimated_tokens // 3) * TOKEN_COSTS["gpt-4.1"] / 1000, 6),
            "output_cost": round((estimated_tokens * 2 // 3) * TOKEN_COSTS["gpt-4.1-completion"] / 1000, 6),
            "total_cost": round(
                ((estimated_tokens // 3) * TOKEN_COSTS["gpt-4.1"] + 
                 (estimated_tokens * 2 // 3) * TOKEN_COSTS["gpt-4.1-completion"]) / 1000, 
                6
            ),
            "is_estimated": True  
        }
    else:
        
        token_usage = calculate_token_cost(cb)
        token_usage["is_estimated"] = False
    
    return report, token_usage