# LangChain Tool Integration Research for DataLLM

## Executive Summary

This document provides comprehensive research on integrating LangChain tools for dynamic dataset analysis. The goal is to create a flexible, general-purpose tool system that can be automatically selected and invoked based on user requests for various types of dataset operations.

## Current State Analysis

### Your Existing Implementation

Your current `llm_service.py` uses:
- **OpenRouter API** with OpenAI client
- **Manual tool definitions** (JSON schema format)
- **Two hardcoded tools**:
  1. `get_dataset_sample` - retrieves dataset preview
  2. `execute_code` - runs Python code on the dataset
- **Tool calling loop** with max 3 iterations
- **Structured output** using Pydantic models

**Strengths:**
- Already implements tool calling pattern
- Uses Pydantic for validation
- Safe code execution environment
- Structured output separation (user vs. developer)

**Limitations:**
- Tools are hardcoded, not dynamically extensible
- Limited to 2 basic operations
- No semantic tool selection
- Manual tool schema definitions

---

## Recommended Architecture: LangChain Integration

### 1. **Core Approach: Custom Tools with LangChain**

Instead of manually defining tools in JSON, leverage LangChain's `@tool` decorator and agent framework for:
- **Dynamic tool registration**
- **Automatic schema generation** from Python type hints
- **Better LLM understanding** through enhanced descriptions
- **Extensibility** - easily add new tools

### 2. **General-Purpose Dataset Tools**

Create a comprehensive toolkit of operations that apply to most datasets:

#### **Statistical Analysis Tools**
```python
from langchain.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd

class DescriptiveStatsInput(BaseModel):
    """Input schema for descriptive statistics."""
    columns: Optional[List[str]] = Field(
        default=None, 
        description="List of column names to analyze. If None, analyze all numeric columns."
    )
    include_percentiles: bool = Field(
        default=True,
        description="Whether to include percentile calculations (25%, 50%, 75%)"
    )

@tool(args_schema=DescriptiveStatsInput)
def calculate_descriptive_statistics(columns: Optional[List[str]] = None, include_percentiles: bool = True) -> dict:
    """
    Calculate comprehensive descriptive statistics for dataset columns.
    
    Returns mean, median, mode, standard deviation, variance, min, max, and optionally percentiles.
    Useful for understanding data distribution and central tendencies.
    """
    # Implementation will access dataset from context
    pass

class CorrelationInput(BaseModel):
    """Input schema for correlation analysis."""
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to include in correlation analysis. If None, use all numeric columns."
    )
    method: str = Field(
        default="pearson",
        description="Correlation method: 'pearson', 'spearman', or 'kendall'"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Only return correlations with absolute value above this threshold"
    )

@tool(args_schema=CorrelationInput)
def calculate_correlation(columns: Optional[List[str]] = None, method: str = "pearson", threshold: Optional[float] = None) -> dict:
    """
    Calculate correlation matrix between numeric columns.
    
    Identifies relationships between variables. High correlation (>0.7 or <-0.7) indicates strong relationships.
    Use this to find which variables move together.
    """
    pass
```

#### **Data Quality Tools**
```python
class MissingDataInput(BaseModel):
    """Input schema for missing data analysis."""
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to check. If None, check all columns."
    )

@tool(args_schema=MissingDataInput)
def analyze_missing_data(columns: Optional[List[str]] = None) -> dict:
    """
    Analyze missing values in the dataset.
    
    Returns count and percentage of missing values per column.
    Helps identify data quality issues and decide on imputation strategies.
    """
    pass

@tool
def detect_outliers(column: str, method: str = "iqr") -> dict:
    """
    Detect outliers in a specific column using IQR or Z-score method.
    
    Args:
        column: Name of the column to analyze
        method: Detection method - 'iqr' (Interquartile Range) or 'zscore'
    
    Returns outlier indices, values, and statistics.
    """
    pass

@tool
def check_data_types() -> dict:
    """
    Analyze and validate data types for all columns.
    
    Identifies potential type mismatches (e.g., numbers stored as strings).
    Returns suggested type conversions for better analysis.
    """
    pass
```

#### **Aggregation & Grouping Tools**
```python
class GroupByInput(BaseModel):
    """Input schema for group-by operations."""
    group_columns: List[str] = Field(
        description="Column(s) to group by"
    )
    agg_column: str = Field(
        description="Column to aggregate"
    )
    agg_function: str = Field(
        default="mean",
        description="Aggregation function: 'mean', 'sum', 'count', 'min', 'max', 'median', 'std'"
    )

@tool(args_schema=GroupByInput)
def group_and_aggregate(group_columns: List[str], agg_column: str, agg_function: str = "mean") -> dict:
    """
    Group data by one or more columns and apply aggregation function.
    
    Example: Group by 'category' and calculate average 'price'.
    Returns grouped results with aggregated values.
    """
    pass

@tool
def calculate_value_counts(column: str, top_n: int = 10, normalize: bool = False) -> dict:
    """
    Count unique values in a column and return top N most frequent.
    
    Args:
        column: Column name to analyze
        top_n: Number of top values to return
        normalize: If True, return percentages instead of counts
    
    Useful for categorical data analysis and frequency distributions.
    """
    pass
```

#### **Filtering & Querying Tools**
```python
class FilterDataInput(BaseModel):
    """Input schema for data filtering."""
    column: str = Field(description="Column to filter on")
    operator: str = Field(
        description="Comparison operator: '>', '<', '>=', '<=', '==', '!=', 'contains', 'in'"
    )
    value: str = Field(description="Value to compare against (will be type-converted)")
    
@tool(args_schema=FilterDataInput)
def filter_data(column: str, operator: str, value: str) -> dict:
    """
    Filter dataset rows based on a condition.
    
    Returns count of matching rows and sample of filtered data.
    Use this to answer questions like "how many rows have price > 100?"
    """
    pass

@tool
def query_data(query_string: str) -> dict:
    """
    Execute a pandas query string on the dataset.
    
    Args:
        query_string: Pandas query expression (e.g., "age > 30 and city == 'NYC'")
    
    More flexible than filter_data for complex conditions.
    """
    pass
```

#### **Time Series Tools** (if applicable)
```python
@tool
def detect_time_column() -> dict:
    """
    Automatically detect datetime columns in the dataset.
    
    Returns identified time columns and their formats.
    """
    pass

@tool
def calculate_time_series_stats(date_column: str, value_column: str, frequency: str = "D") -> dict:
    """
    Calculate time-series specific statistics.
    
    Args:
        date_column: Column containing dates/timestamps
        value_column: Column with values to analyze over time
        frequency: Resampling frequency ('D'=daily, 'W'=weekly, 'M'=monthly)
    
    Returns trends, seasonality indicators, and rolling statistics.
    """
    pass
```

#### **Comparison Tools**
```python
@tool
def compare_groups(group_column: str, value_column: str, groups: List[str]) -> dict:
    """
    Compare statistics between different groups.
    
    Args:
        group_column: Column containing group labels
        value_column: Column with values to compare
        groups: List of specific groups to compare
    
    Returns comparative statistics and indicates significant differences.
    """
    pass
```

---

## 3. **Implementation Strategy**

### Phase 1: LangChain Integration

**Install Dependencies:**
```bash
pip install langchain langchain-openai langchain-experimental
```

**Create Tool Module** (`backend/app/services/dataset_tools.py`):
```python
from langchain.tools import tool
from langchain_core.tools import BaseTool
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import pandas as pd
from app.services.storage import StorageService

# Global context for current dataset (thread-safe alternative needed for production)
_current_dataset_id: Optional[str] = None

def set_dataset_context(dataset_id: str):
    """Set the current dataset context for tool execution."""
    global _current_dataset_id
    _current_dataset_id = dataset_id

def get_current_dataframe() -> pd.DataFrame:
    """Retrieve the current dataset as a DataFrame."""
    if not _current_dataset_id:
        raise ValueError("No dataset context set")
    df = StorageService.load_dataset(_current_dataset_id)
    if df is None:
        raise ValueError(f"Dataset {_current_dataset_id} not found")
    return df

# Define all tools here using @tool decorator
# ... (tools from section 2)
```

### Phase 2: Agent Creation

**Update LLM Service** (`backend/app/services/llm_service.py`):
```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.services.dataset_tools import (
    set_dataset_context,
    calculate_descriptive_statistics,
    calculate_correlation,
    analyze_missing_data,
    detect_outliers,
    group_and_aggregate,
    calculate_value_counts,
    filter_data,
    # ... import all tools
)

class LangChainLLMService:
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        
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
        self.tools = [
            calculate_descriptive_statistics,
            calculate_correlation,
            analyze_missing_data,
            detect_outliers,
            group_and_aggregate,
            calculate_value_counts,
            filter_data,
            # ... add all tools
        ]
        
        # Create agent prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis expert assistant. You have access to various tools to analyze datasets.

When a user asks a question about their data:
1. First, understand what information they need
2. Select the most appropriate tool(s) to gather that information
3. Execute the tools and interpret the results
4. Provide a clear, concise answer with key insights

Always use tools to calculate actual values - never make up statistics.
If you need to see the data structure first, you can request a sample.
Be precise and cite specific numbers from your tool results."""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,
            handle_parsing_errors=True
        )
    
    async def analyze_dataset(
        self,
        dataset_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Analyze dataset using LangChain agent with tools."""
        # Set dataset context for tools
        set_dataset_context(dataset_id)
        
        try:
            # Execute agent
            result = await self.agent_executor.ainvoke({
                "input": query
            })
            
            return {
                "answer": result["output"],
                "intermediate_steps": result.get("intermediate_steps", [])
            }
        except Exception as e:
            return {
                "answer": f"Error during analysis: {str(e)}",
                "intermediate_steps": []
            }
```

### Phase 3: Dynamic Tool Selection

The LangChain agent automatically handles tool selection based on:
1. **Tool descriptions** - Clear docstrings guide the LLM
2. **Input schemas** - Pydantic models define expected parameters
3. **User query semantics** - LLM matches intent to appropriate tools

**No manual routing needed!** The agent reasons about which tools to use.

---

## 4. **Advanced Patterns**

### Multi-Step Reasoning

The agent can chain multiple tools:
```
User: "What's the average price for products in the Electronics category?"

Agent reasoning:
1. Use filter_data to get Electronics rows
2. Use calculate_descriptive_statistics on price column
3. Synthesize answer from results
```

### Semantic Routing (Optional Enhancement)

For very large tool sets, implement semantic routing:

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Create embeddings of tool descriptions
tool_descriptions = [tool.description for tool in tools]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(tool_descriptions, embeddings)

# At query time, find most relevant tools
relevant_tools = vectorstore.similarity_search(user_query, k=5)
# Pass only relevant tools to agent
```

### Plan-and-Execute Pattern

For complex queries, use a planning agent:

```python
from langchain.agents import create_plan_and_execute_agent

planner = create_plan_and_execute_agent(
    llm=llm,
    tools=tools,
    verbose=True
)

# Agent first creates a plan, then executes steps
result = planner.invoke({"input": "Compare sales trends across regions and identify top performers"})
```

---

## 5. **Best Practices**

### Tool Design
1. **Single Responsibility** - Each tool does one thing well
2. **Clear Descriptions** - Detailed docstrings with examples
3. **Type Safety** - Use Pydantic for all inputs
4. **Error Handling** - Return informative errors, don't crash
5. **Consistent Output Format** - Always return dict with predictable structure

### Security
1. **Sandboxing** - If executing user code, use restricted environments
2. **Input Validation** - Validate all parameters before execution
3. **Rate Limiting** - Prevent abuse of expensive operations
4. **Audit Logging** - Log all tool invocations

### Performance
1. **Caching** - Cache expensive calculations (correlations, stats)
2. **Lazy Loading** - Only load data when needed
3. **Sampling** - For large datasets, offer sampling options
4. **Async Operations** - Use async tools for I/O operations

### Testing
```python
# Test each tool independently
def test_descriptive_statistics():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
    result = calculate_descriptive_statistics.invoke({"columns": ["A"]})
    assert result["mean"] == 3.0
    assert result["std"] > 0
```

---

## 6. **Migration Path from Current Implementation**

### Option A: Gradual Migration
1. Keep existing `analyze_dataset` method
2. Add new `analyze_dataset_with_langchain` method
3. Add feature flag to switch between implementations
4. Test thoroughly, then deprecate old method

### Option B: Hybrid Approach
1. Use LangChain tools but keep OpenRouter client
2. Convert tools to OpenRouter function calling format
3. Benefit from tool organization without full LangChain dependency

### Option C: Full Migration
1. Replace entire LLM service with LangChain
2. Migrate all existing functionality to tools
3. Update API endpoints to use new service
4. Most powerful but requires more testing

**Recommendation: Start with Option A** for safety and gradual rollout.

---

## 7. **Example Tool Implementations**

Here's a complete, production-ready tool example:

```python
from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

class DescriptiveStatsInput(BaseModel):
    """Input schema for descriptive statistics calculation."""
    
    columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to analyze. If not provided, analyzes all numeric columns."
    )
    include_percentiles: bool = Field(
        default=True,
        description="Whether to include 25th, 50th, and 75th percentiles in the output."
    )
    
    @validator('columns')
    def validate_columns(cls, v):
        if v is not None and len(v) == 0:
            raise ValueError("If columns are specified, list cannot be empty")
        return v

@tool(args_schema=DescriptiveStatsInput)
def calculate_descriptive_statistics(
    columns: Optional[List[str]] = None,
    include_percentiles: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for numeric columns in the dataset.
    
    This tool computes:
    - Count of non-null values
    - Mean (average)
    - Standard deviation
    - Minimum value
    - Maximum value
    - 25th, 50th (median), 75th percentiles (if include_percentiles=True)
    
    Use this tool when users ask about:
    - "What's the average/mean of X?"
    - "Show me statistics for column Y"
    - "What's the distribution of Z?"
    - "Summarize the data"
    
    Args:
        columns: List of column names to analyze. If None, analyzes all numeric columns.
        include_percentiles: Whether to include percentile calculations.
    
    Returns:
        Dictionary with statistics for each column, including:
        - count: Number of non-null values
        - mean: Average value
        - std: Standard deviation
        - min: Minimum value
        - max: Maximum value
        - 25%/50%/75%: Percentiles (if requested)
    
    Example output:
        {
            "price": {
                "count": 100,
                "mean": 49.99,
                "std": 15.23,
                "min": 10.00,
                "max": 99.99,
                "25%": 35.00,
                "50%": 48.50,
                "75%": 65.00
            }
        }
    """
    try:
        df = get_current_dataframe()
        
        # Select columns
        if columns:
            # Validate columns exist
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                return {
                    "error": f"Columns not found in dataset: {missing_cols}",
                    "available_columns": df.columns.tolist()
                }
            df_subset = df[columns]
        else:
            df_subset = df.select_dtypes(include=[np.number])
        
        if df_subset.empty:
            return {
                "error": "No numeric columns found in dataset",
                "available_columns": df.columns.tolist(),
                "column_types": df.dtypes.to_dict()
            }
        
        # Calculate statistics
        stats = df_subset.describe(
            percentiles=[0.25, 0.5, 0.75] if include_percentiles else []
        ).to_dict()
        
        # Format output
        result = {}
        for col, col_stats in stats.items():
            result[col] = {
                k: float(v) if not pd.isna(v) else None
                for k, v in col_stats.items()
            }
        
        return {
            "statistics": result,
            "columns_analyzed": list(result.keys()),
            "total_rows": len(df)
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate statistics: {str(e)}",
            "error_type": type(e).__name__
        }
```

---

## 8. **Recommended Tool Suite for DataLLM**

### Core Tools (Implement First)
1. ✅ `get_dataset_info` - Basic metadata (rows, columns, types)
2. ✅ `calculate_descriptive_statistics` - Mean, median, std, etc.
3. ✅ `calculate_correlation` - Correlation matrix
4. ✅ `analyze_missing_data` - Missing value analysis
5. ✅ `calculate_value_counts` - Frequency distributions
6. ✅ `filter_data` - Basic filtering
7. ✅ `group_and_aggregate` - Group-by operations

### Extended Tools (Add as Needed)
8. `detect_outliers` - Outlier detection
9. `compare_groups` - Group comparisons
10. `calculate_percentiles` - Custom percentile calculations
11. `find_unique_values` - Unique value analysis
12. `calculate_range` - Min/max ranges
13. `detect_data_types` - Type inference and validation

### Advanced Tools (Future)
14. `perform_hypothesis_test` - Statistical testing
15. `calculate_regression` - Linear regression
16. `detect_time_patterns` - Time series analysis
17. `suggest_visualizations` - Recommend chart types
18. `generate_summary_report` - Comprehensive analysis

---

## 9. **Code Examples**

### Complete Integration Example

See the attached file: `langchain_integration_example.py` (to be created separately)

---

## 10. **Resources & References**

### Official Documentation
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [Custom Tools Guide](https://python.langchain.com/docs/modules/tools/custom_tools)
- [Pydantic Documentation](https://docs.pydantic.dev/)

### Best Practices Articles
- "Building Production-Ready LangChain Agents" - Medium
- "LangChain Tool Calling Patterns" - LangChain Blog
- "Robust AI Agents with ACID Principles" - Engineering Blogs

### Community Examples
- LangChain GitHub Examples
- LangChain Community Discussions
- Stack Overflow LangChain Tag

---

## 11. **Next Steps**

1. **Review this research** with your team
2. **Choose migration strategy** (Option A, B, or C)
3. **Implement core tools** (7 tools from Core Tools list)
4. **Create test suite** for each tool
5. **Update API endpoints** to use new service
6. **Test with real datasets** and user queries
7. **Monitor performance** and iterate
8. **Add extended tools** based on user needs

---

## Conclusion

LangChain provides a robust framework for building dynamic, tool-based dataset analysis systems. By leveraging:
- **@tool decorator** for easy tool creation
- **Pydantic schemas** for type safety
- **Agent executors** for automatic tool selection
- **Clear descriptions** for LLM understanding

You can create a flexible, extensible system that handles a wide variety of dataset analysis tasks without hardcoding specific operations.

The key is to design tools that are:
1. **General-purpose** - Work with any dataset structure
2. **Well-documented** - Clear descriptions for LLM
3. **Type-safe** - Pydantic validation
4. **Composable** - Can be chained together
5. **Reliable** - Proper error handling

This approach will significantly enhance your DataLLM platform's capabilities while maintaining code quality and user experience.
