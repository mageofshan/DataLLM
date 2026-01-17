# LangChain Tool Integration - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                  │
│                         (Frontend - React/TypeScript)                        │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │ HTTP Request
                                 │ POST /api/analyze
                                 │ { dataset_id, query }
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│                           API ENDPOINT                                       │
│                      (FastAPI - chat.py)                                     │
│                                                                              │
│  • Receives user query                                                       │
│  • Validates request                                                         │
│  • Routes to LLM service                                                     │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 │
                    ┌────────────▼────────────┐
                    │   Feature Flag Check    │
                    │  USE_LANGCHAIN_AGENT?   │
                    └────────┬────────┬───────┘
                             │        │
                    ┌────────▼──┐  ┌──▼────────┐
                    │   TRUE    │  │   FALSE   │
                    └────────┬──┘  └──┬────────┘
                             │        │
         ┌───────────────────▼────────▼───────────────────┐
         │                                                 │
         │                                                 │
┌────────▼──────────────┐                  ┌──────────────▼────────────┐
│  LangChainLLMService  │                  │  LLMService (Current)     │
│  (NEW SYSTEM)         │                  │  (Fallback/Legacy)        │
│                       │                  │                           │
│  • Agent Executor     │                  │  • Manual tool calling    │
│  • Tool Selection     │                  │  • 2 hardcoded tools      │
│  • Multi-step         │                  │  • Basic execution        │
└───────────┬───────────┘                  └───────────────────────────┘
            │
            │ 1. Set dataset context
            │ 2. Create agent input
            │ 3. Execute agent
            │
┌───────────▼──────────────────────────────────────────────────────────────┐
│                        LANGCHAIN AGENT EXECUTOR                           │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  LLM (OpenRouter - GPT-4o-mini)                                 │    │
│  │  • Receives user query                                          │    │
│  │  • Analyzes available tools                                     │    │
│  │  • Decides which tool(s) to call                                │    │
│  │  • Generates tool arguments                                     │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
│                           │ Tool Selection                               │
│                           │                                              │
│  ┌────────────────────────▼────────────────────────────────────────┐    │
│  │                    TOOL REGISTRY                                │    │
│  │  ┌──────────────────────────────────────────────────────────┐  │    │
│  │  │  Available Tools (8+):                                   │  │    │
│  │  │  1. get_dataset_info                                     │  │    │
│  │  │  2. calculate_descriptive_statistics                     │  │    │
│  │  │  3. calculate_correlation                                │  │    │
│  │  │  4. analyze_missing_data                                 │  │    │
│  │  │  5. detect_outliers                                      │  │    │
│  │  │  6. group_and_aggregate                                  │  │    │
│  │  │  7. calculate_value_counts                               │  │    │
│  │  │  8. filter_data                                          │  │    │
│  │  │  ... (easily extensible)                                 │  │    │
│  │  └──────────────────────────────────────────────────────────┘  │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
│                           │ Execute Tool(s)                              │
│                           │                                              │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                         TOOL EXECUTION                                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Tool Function (e.g., calculate_descriptive_statistics)         │    │
│  │                                                                  │    │
│  │  1. Get current dataset from context                            │    │
│  │  2. Validate inputs (Pydantic schema)                           │    │
│  │  3. Perform calculation                                         │    │
│  │  4. Return structured result                                    │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
└───────────────────────────┼──────────────────────────────────────────────┘
                            │
                            │ Access Dataset
                            │
┌───────────────────────────▼──────────────────────────────────────────────┐
│                      STORAGE SERVICE                                      │
│                                                                           │
│  • load_dataset(dataset_id) → DataFrame                                  │
│  • save_dataset(dataset_id, df)                                          │
│  • Dataset context management                                            │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Dataset Storage (File System / Database)                       │    │
│  │  • CSV files                                                     │    │
│  │  • Parquet files                                                 │    │
│  │  • Database tables                                               │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                              DATA FLOW EXAMPLE
═══════════════════════════════════════════════════════════════════════════════

User Query: "What's the average price in the Electronics category?"

1. API receives query
2. LangChainLLMService.analyze_dataset() called
3. Agent analyzes query and decides on tools:
   
   Step 1: Use filter_data to get Electronics products
   ┌─────────────────────────────────────────────────────────────┐
   │ Tool: filter_data                                           │
   │ Input: {column: "category", operator: "==",                │
   │         value: "Electronics"}                               │
   │ Output: {matching_rows: 200, percentage: 20.0, ...}        │
   └─────────────────────────────────────────────────────────────┘
   
   Step 2: Use calculate_descriptive_statistics on price
   ┌─────────────────────────────────────────────────────────────┐
   │ Tool: calculate_descriptive_statistics                      │
   │ Input: {columns: ["price"], include_percentiles: true}     │
   │ Output: {statistics: {price: {mean: 245.67, ...}}}         │
   └─────────────────────────────────────────────────────────────┘

4. Agent synthesizes answer:
   "The average price for products in the Electronics category is $245.67.
    This is based on 200 products (20% of the dataset)."

5. Return to user with:
   - answer: The synthesized response
   - tool_calls: List of tools used
   - intermediate_steps: Detailed execution trace


═══════════════════════════════════════════════════════════════════════════════
                            KEY COMPONENTS
═══════════════════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────────────┐
│  TOOL DEFINITION (@tool decorator)                                      │
│                                                                          │
│  from langchain.tools import tool                                        │
│  from pydantic import BaseModel, Field                                   │
│                                                                          │
│  class ToolInput(BaseModel):                                            │
│      param: str = Field(description="Parameter description")            │
│                                                                          │
│  @tool(args_schema=ToolInput)                                           │
│  def my_tool(param: str) -> dict:                                       │
│      """Tool description for LLM."""                                    │
│      # Implementation                                                    │
│      return {"result": "value"}                                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  AGENT CREATION                                                          │
│                                                                          │
│  from langchain.agents import create_tool_calling_agent                 │
│  from langchain_openai import ChatOpenAI                                │
│                                                                          │
│  llm = ChatOpenAI(api_key=..., base_url="openrouter.ai/api/v1")        │
│  agent = create_tool_calling_agent(llm, tools, prompt)                 │
│  executor = AgentExecutor(agent, tools, max_iterations=5)              │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  EXECUTION                                                               │
│                                                                          │
│  result = await executor.ainvoke({"input": user_query})                │
│  # Returns: {                                                            │
│  #   "output": "The answer...",                                         │
│  #   "intermediate_steps": [(action, observation), ...]                │
│  # }                                                                     │
└─────────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════════════
                         BENEFITS OF THIS ARCHITECTURE
═══════════════════════════════════════════════════════════════════════════════

✓ Automatic Tool Selection
  → LLM decides which tools to use based on query

✓ Multi-Step Reasoning
  → Agent can chain multiple tools for complex queries

✓ Type Safety
  → Pydantic schemas validate all inputs

✓ Extensibility
  → Add new tools with simple @tool decorator

✓ Error Handling
  → Comprehensive error messages guide users

✓ Testability
  → Each tool can be tested independently

✓ Observability
  → Detailed execution traces for debugging

✓ Flexibility
  → Easy to add domain-specific tools
```
