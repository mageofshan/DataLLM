"""
LangChain-based LLM Service for DataLLM

This service integrates LangChain agents with custom dataset analysis tools.
It provides automatic tool selection based on user queries.

Usage:
    service = LangChainLLMService()
    result = await service.analyze_dataset(dataset_id="abc123", query="What's the average price?")
"""

import os
from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain import agents
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough

from app.services.dataset_tools import (
    set_dataset_context,
    get_dataset_info,
    calculate_descriptive_statistics,
    calculate_correlation,
    analyze_missing_data,
    detect_outliers,
    group_and_aggregate,
    calculate_value_counts,
    filter_data,
    ALL_TOOLS
)


class LangChainLLMService:
    """
    LangChain-powered LLM service for dataset analysis.
    
    Features:
    - Automatic tool selection based on user queries
    - Multi-step reasoning for complex questions
    - Structured output with citations
    - Support for conversation history
    """
    
    def __init__(self):
        """Initialize the LangChain service with OpenRouter."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.mock_mode = not bool(self.api_key)
        
        if not self.mock_mode:
            # Initialize LangChain ChatOpenAI with OpenRouter
            self.llm = ChatOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                model="openai/gpt-4o-mini",
                temperature=0,  # Deterministic for data analysis
                default_headers={
                    "HTTP-Referer": "http://localhost:3000",
                    "X-Title": "DataLLM",
                }
            )
            
            # Define available tools
            self.tools = ALL_TOOLS
            
            # Bind tools to LLM
            self.llm_with_tools = self.llm.bind_tools(self.tools)
    
    def _create_prompt(self) -> str:
        """Create the system prompt for the agent."""
        return """You are an expert data analyst assistant. You have access to various tools to analyze datasets.

Your workflow:
1. **Understand the question**: Carefully read what the user is asking
2. **Plan your approach**: Decide which tool(s) will help answer the question
3. **Execute tools**: Use the appropriate tools to gather information
4. **Interpret results**: Analyze the tool outputs
5. **Provide clear answer**: Give a concise, accurate response with specific numbers

Guidelines:
- ALWAYS use tools to calculate actual values - NEVER make up statistics
- If you need to understand the data structure first, use get_dataset_info
- For questions about averages, use calculate_descriptive_statistics
- For questions about relationships, use calculate_correlation
- For questions about groups/categories, use group_and_aggregate or calculate_value_counts
- For questions about filtering/counting, use filter_data
- Cite specific numbers from your tool results
- If a tool returns an error, explain it clearly to the user
- Be concise but thorough in your explanations

Example interactions:

User: "What's the average price?"
You: Use calculate_descriptive_statistics on the price column, then report the mean value.

User: "How many products are in each category?"
You: Use calculate_value_counts on the category column to get the distribution.

User: "Are price and rating correlated?"
You: Use calculate_correlation with columns=['price', 'rating'] to check their relationship.

Remember: Precision and accuracy are paramount. Always verify your answers with tool outputs."""
    
    async def analyze_dataset(
        self,
        dataset_id: str,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze dataset using LangChain with tools.
        
        Args:
            dataset_id: ID of the dataset to analyze
            query: User's question about the data
            conversation_history: Optional list of previous messages
        
        Returns:
            Dictionary containing:
            - answer: The agent's response
            - tool_calls: List of tools that were used
            - intermediate_steps: Detailed execution trace
        """
        if self.mock_mode:
            return {
                "answer": f"[MOCK MODE] Set OPENROUTER_API_KEY to enable analysis. Query: {query}",
                "tool_calls": [],
                "intermediate_steps": []
            }
        
        # Set dataset context for tools
        set_dataset_context(dataset_id)
        
        try:
            # Create system message with tool instructions
            system_message = self._create_prompt()
            
            # Build messages
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": query}
            ]
            
            # Add conversation history if provided
            if conversation_history:
                history_messages = []
                for msg in conversation_history:
                    history_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                messages = [messages[0]] + history_messages + [messages[1]]
            
            # Invoke LLM with tools
            response = await self.llm_with_tools.ainvoke(messages)
            
            # Check if tools were called
            tool_calls = []
            intermediate_steps = []
            
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    tool_name = tool_call['name']
                    tool_args = tool_call['args']
                    
                    # Find and execute the tool
                    tool = next((t for t in self.tools if t.name == tool_name), None)
                    if tool:
                        try:
                            tool_result = tool.invoke(tool_args)
                            tool_calls.append({
                                "tool": tool_name,
                                "input": tool_args,
                                "output": tool_result
                            })
                            intermediate_steps.append((tool_call, tool_result))
                        except Exception as e:
                            tool_calls.append({
                                "tool": tool_name,
                                "input": tool_args,
                                "output": f"Error: {str(e)}"
                            })
                
                # Get final response with tool results
                messages.append({"role": "assistant", "content": str(response.content)})
                for tc in tool_calls:
                    messages.append({
                        "role": "function",
                        "name": tc["tool"],
                        "content": str(tc["output"])
                    })
                
                # Get final answer
                final_response = await self.llm.ainvoke(messages)
                answer = final_response.content
            else:
                # No tools called, use direct response
                answer = response.content
            
            return {
                "answer": answer,
                "tool_calls": tool_calls,
                "intermediate_steps": intermediate_steps
            }
            
        except Exception as e:
            return {
                "answer": f"I encountered an error while analyzing the data: {str(e)}",
                "tool_calls": [],
                "intermediate_steps": [],
                "error": str(e)
            }
    
    async def get_available_tools(self) -> List[Dict[str, str]]:
        """
        Get list of available tools with descriptions.
        
        Returns:
            List of tool metadata (name, description)
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "args_schema": tool.args_schema.schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
            }
            for tool in self.tools
        ]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_usage():
    """Example of how to use the LangChain LLM service."""
    
    # Initialize service
    service = LangChainLLMService()
    
    # Example 1: Simple statistical query
    result1 = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="What's the average price in the dataset?"
    )
    print("Query 1:", result1["answer"])
    print("Tools used:", [tc["tool"] for tc in result1["tool_calls"]])
    print()
    
    # Example 2: Grouping query
    result2 = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="How many items are in each category?"
    )
    print("Query 2:", result2["answer"])
    print("Tools used:", [tc["tool"] for tc in result2["tool_calls"]])
    print()
    
    # Example 3: Correlation query
    result3 = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="Is there a correlation between price and rating?"
    )
    print("Query 3:", result3["answer"])
    print("Tools used:", [tc["tool"] for tc in result3["tool_calls"]])
    print()
    
    # Example 4: Complex multi-step query
    result4 = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="What's the average price for products with rating above 4?"
    )
    print("Query 4:", result4["answer"])
    print("Tools used:", [tc["tool"] for tc in result4["tool_calls"]])
    print()
    
    # Example 5: With conversation history
    conversation_history = [
        {"role": "user", "content": "What columns are in this dataset?"},
        {"role": "assistant", "content": "The dataset has columns: product_name, category, price, rating, stock_count"}
    ]
    result5 = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="What's the average of the third column?",
        conversation_history=conversation_history
    )
    print("Query 5 (with context):", result5["answer"])
    print()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
