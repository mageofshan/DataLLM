import os
import json
import io
import contextlib
import traceback
import pandas as pd
from openai import AsyncOpenAI
from typing import Optional, AsyncGenerator, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from app.services.storage import StorageService

class UserUnderstanding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    summary: str = Field(description="A clear, concise summary of the data analysis for the user.")
    key_insights: List[str] = Field(description="List of key insights derived from the data.")

class PythonAnalysisData(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code_snippet: Optional[str] = Field(default=None, description="Python code snippet if explicitly requested or needed for external execution.")
    parameters_json: Optional[str] = Field(default=None, description="JSON string of parameters if code is provided.")

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    user_output: UserUnderstanding
    python_output: Optional[PythonAnalysisData] = Field(default=None)

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "DataLLM",
            }
        ) if self.api_key else None
        self.mock_mode = not bool(self.api_key)

    async def generate_response(
        self, 
        prompt: str, 
        system_prompt: str = "You are a helpful data analysis assistant. Answer based on the 4 pilars of data analysis, identify which you used, and let the user what you used: descriptive, diagnostic, predictive, and prescriptive. When answering, ALWAYS provide the region of data which was used to calcualte the request, what process was used, and the final answer.",
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 1024
    ) -> str:
        """
        Generates a response from the LLM.
        """
        if self.mock_mode:
            return f"[MOCK LLM RESPONSE] Set OPENROUTER_API_KEY. Prompt: {prompt[:50]}..."

        try:
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM: {str(e)}"

    async def stream_response(
        self, 
        prompt: str, 
        system_prompt: str = "You are a helpful data analysis assistant. Always respond in markdown format.",
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 1024
    ) -> AsyncGenerator[str, None]:
        """
        Streams a response from the LLM.
        """
        if self.mock_mode:
            yield "[MOCK STREAM] Set OPENROUTER_API_KEY. "
            for word in prompt.split():
                yield word + " "
            return

        try:
            stream = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"Error streaming LLM: {str(e)}"

    async def analyze_dataset(
        self,
        dataset_id: str,
        query: str,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 1000
    ) -> AnalysisResult:
        """
        Analyzes a dataset using tool calling and returns structured output.
        """
        if self.mock_mode:
            return AnalysisResult(
                user_output=UserUnderstanding(summary="Mock summary", key_insights=["Mock insight"]),
                python_output=None
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_dataset_sample",
                    "description": "Get a sample of the dataset to understand its structure and content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "n_rows": {
                                "type": "integer",
                                "description": "Number of rows to retrieve (default 5)",
                                "default": 5
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_code",
                    "description": "Execute Python code on the full dataset to calculate statistics or answer questions. The dataset is available as 'df'. Print the result to stdout.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Valid python code to execute. Assume 'df' is already loaded. Use print() to output the answer."
                            }
                        },
                        "required": ["code"]
                    }
                }
            }
        ]

        messages = [
            {"role": "system", "content": "You are a data analysis assistant. You MUST use the provided tools to inspect the dataset and calculate answers. \n1. Always call 'get_dataset_sample' first to understand the data.\n2. Then, if the user asks for a value (e.g., average, count, correlation), use 'execute_code' to calculate it on the full 'df'.\n3. Finally, provide the answer in the 'summary'. \n4. Do NOT populate 'python_output' unless the user explicitly asks for the code or a script. If you calculated the answer internally, just report it."},
            {"role": "user", "content": f"Dataset ID: {dataset_id}. Query: {query}"}
        ]

        # Loop to handle multiple tool calls (e.g. sample -> code -> answer)
        for _ in range(3):  # Max 3 turns
            response = await self.client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )

            message = response.choices[0].message
            messages.append(message)

            if not message.tool_calls:
                # No more tools, generate final structured output
                break
            
            for tool_call in message.tool_calls:
                tool_output = ""
                if tool_call.function.name == "get_dataset_sample":
                    args = json.loads(tool_call.function.arguments)
                    n_rows = args.get("n_rows", 5)
                    df = StorageService.load_dataset(dataset_id)
                    if df is not None:
                        sample = df.head(n_rows).to_markdown(index=False)
                        columns = str(df.dtypes.to_dict())
                        tool_output = f"Columns: {columns}\nSample Data:\n{sample}"
                    else:
                        tool_output = "Error: Dataset not found."
                
                elif tool_call.function.name == "execute_code":
                    args = json.loads(tool_call.function.arguments)
                    code = args.get("code", "")
                    df = StorageService.load_dataset(dataset_id)
                    if df is not None:
                        # Safe-ish execution environment
                        local_env = {'df': df, 'pd': pd}
                        output_buffer = io.StringIO()
                        try:
                            with contextlib.redirect_stdout(output_buffer):
                                exec(code, {"__builtins__": __builtins__, "pd": pd}, local_env)
                            tool_output = f"Execution Output:\n{output_buffer.getvalue()}"
                        except Exception as e:
                            tool_output = f"Execution Error: {traceback.format_exc()}"
                    else:
                        tool_output = "Error: Dataset not found."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_output
                })

        # Final call to generate structured output
        completion = await self.client.beta.chat.completions.parse(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            response_format=AnalysisResult
        )
        
        return completion.choices[0].message.parsed
