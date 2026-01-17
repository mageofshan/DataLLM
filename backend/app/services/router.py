import json
from typing import Dict, Any, Optional
from app.services.llm_service import LLMService

class QueryRouter:
    def __init__(self):
        self.llm = LLMService()

    async def route_query(self, query: str, dataset_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Decides how to handle the query.
        If dataset_id is provided, uses the agentic tool-using LLM flow.
        Otherwise, falls back to simple chat.
        """
        
        if dataset_id:
            # Use the new agentic flow with tool calling and structured outputs
            analysis_result = await self.llm.analyze_dataset(dataset_id, query)
            
            return {
                "route": "agentic_analysis",
                "response": analysis_result.user_output.summary,
                "data": {
                    "insights": analysis_result.user_output.key_insights,
                    "python_code": analysis_result.python_output.code_snippet if analysis_result.python_output else None,
                    "python_params": json.loads(analysis_result.python_output.parameters_json) if analysis_result.python_output and analysis_result.python_output.parameters_json else None,
                    "visualization": analysis_result.visualization.model_dump() if analysis_result.visualization else None
                }
            }
        else:
            # Fallback for no dataset (general chat)
            response = await self.llm.generate_response(prompt=query)
            return {
                "route": "general_chat",
                "response": response,
                "data": None
            }
