# Quick Reference: Adding New Tools to DataLLM

## Template for Creating a New Tool

### Simple Tool (No Input Parameters)

```python
from langchain.tools import tool
from typing import Dict, Any

@tool
def my_simple_tool() -> Dict[str, Any]:
    """
    Brief description of what this tool does.
    
    Use this when users ask:
    - "Example question 1"
    - "Example question 2"
    
    Returns a dictionary with the results.
    """
    try:
        df = get_current_dataframe()
        
        # Your logic here
        result = df.some_operation()
        
        return {
            "result": result,
            "metadata": "additional info"
        }
    except Exception as e:
        return {"error": f"Failed to execute: {str(e)}"}
```

### Tool with Input Parameters (Recommended)

```python
from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List

class MyToolInput(BaseModel):
    """Input schema for my tool."""
    
    required_param: str = Field(
        description="Clear description of this required parameter"
    )
    optional_param: Optional[int] = Field(
        default=10,
        description="Clear description of this optional parameter"
    )
    
    @validator('required_param')
    def validate_required_param(cls, v):
        if not v:
            raise ValueError("Parameter cannot be empty")
        return v

@tool(args_schema=MyToolInput)
def my_tool_with_params(
    required_param: str,
    optional_param: int = 10
) -> Dict[str, Any]:
    """
    Brief description of what this tool does.
    
    Detailed explanation of the tool's purpose and behavior.
    
    Use this when users ask:
    - "Example question 1"
    - "Example question 2"
    
    Args:
        required_param: Description of what this parameter does
        optional_param: Description of what this parameter does
    
    Returns:
        Dictionary containing:
        - key1: Description of this return value
        - key2: Description of this return value
    
    Example output:
        {
            "key1": "value1",
            "key2": 123
        }
    """
    try:
        df = get_current_dataframe()
        
        # Validate inputs
        if required_param not in df.columns:
            return {
                "error": f"Column '{required_param}' not found",
                "available_columns": df.columns.tolist()
            }
        
        # Your logic here
        result = df[required_param].some_operation()
        
        return {
            "result": result,
            "parameter_used": required_param,
            "optional_param_value": optional_param
        }
    except Exception as e:
        return {"error": f"Failed to execute: {str(e)}"}
```

---

## Step-by-Step: Adding a New Tool

### 1. Define the Tool

Add to `backend/app/services/dataset_tools.py`:

```python
# At the top with other imports
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# Define input schema (if needed)
class MyNewToolInput(BaseModel):
    """Input schema for my new tool."""
    column: str = Field(description="Column to analyze")
    threshold: float = Field(default=0.5, description="Threshold value")

# Define the tool
@tool(args_schema=MyNewToolInput)
def my_new_tool(column: str, threshold: float = 0.5) -> Dict[str, Any]:
    """
    One-line description for LLM.
    
    Detailed description...
    
    Use this when users ask:
    - "Question pattern 1"
    - "Question pattern 2"
    """
    try:
        df = get_current_dataframe()
        # Implementation
        return {"result": "success"}
    except Exception as e:
        return {"error": str(e)}
```

### 2. Register the Tool

Add to the `ALL_TOOLS` list at the bottom of `dataset_tools.py`:

```python
ALL_TOOLS = [
    get_dataset_info,
    calculate_descriptive_statistics,
    # ... existing tools ...
    my_new_tool,  # Add your new tool here
]
```

### 3. Import in LLM Service

Update `backend/app/services/langchain_llm_service.py`:

```python
from app.services.dataset_tools import (
    set_dataset_context,
    get_dataset_info,
    # ... existing imports ...
    my_new_tool,  # Add your new tool
    ALL_TOOLS
)
```

### 4. Test the Tool

Create a test in `tests/test_dataset_tools.py`:

```python
def test_my_new_tool(sample_dataset):
    """Test my new tool."""
    result = my_new_tool.invoke({
        "column": "price",
        "threshold": 0.5
    })
    
    assert "result" in result or "error" in result
    # Add more specific assertions
```

### 5. Run Tests

```bash
cd backend
pytest tests/test_dataset_tools.py::test_my_new_tool -v
```

### 6. Test with Agent

```python
# In Python REPL or test script
from app.services.langchain_llm_service import LangChainLLMService
import asyncio

async def test():
    service = LangChainLLMService()
    result = await service.analyze_dataset(
        dataset_id="test_dataset",
        query="Use my new tool on the price column"
    )
    print(result["answer"])

asyncio.run(test())
```

---

## Best Practices Checklist

When creating a new tool, ensure:

- [ ] **Clear, concise description** - First line is a one-sentence summary
- [ ] **Use cases listed** - Include example questions that trigger this tool
- [ ] **Type hints** - All parameters and return type are typed
- [ ] **Pydantic schema** - For tools with parameters
- [ ] **Input validation** - Check column existence, data types, etc.
- [ ] **Error handling** - Try/except with informative error messages
- [ ] **Consistent output** - Always return a dictionary
- [ ] **Documentation** - Docstring with Args and Returns sections
- [ ] **Examples** - Include example output in docstring
- [ ] **Tests** - Unit test for the tool
- [ ] **Performance** - Consider large datasets (use sampling if needed)

---

## Common Patterns

### Pattern 1: Column Validation

```python
@tool
def my_tool(column: str) -> Dict[str, Any]:
    """Tool description."""
    try:
        df = get_current_dataframe()
        
        # Validate column exists
        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found",
                "available_columns": df.columns.tolist()
            }
        
        # Validate column type
        if not pd.api.types.is_numeric_dtype(df[column]):
            return {
                "error": f"Column '{column}' must be numeric",
                "actual_type": str(df[column].dtype)
            }
        
        # Your logic here
        result = df[column].mean()
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
```

### Pattern 2: Multiple Column Operations

```python
class MultiColumnInput(BaseModel):
    columns: List[str] = Field(description="List of columns to analyze")

@tool(args_schema=MultiColumnInput)
def analyze_multiple_columns(columns: List[str]) -> Dict[str, Any]:
    """Analyze multiple columns."""
    try:
        df = get_current_dataframe()
        
        # Validate all columns exist
        missing = set(columns) - set(df.columns)
        if missing:
            return {
                "error": f"Columns not found: {list(missing)}",
                "available_columns": df.columns.tolist()
            }
        
        # Process each column
        results = {}
        for col in columns:
            results[col] = df[col].describe().to_dict()
        
        return {"results": results}
    except Exception as e:
        return {"error": str(e)}
```

### Pattern 3: Conditional Logic

```python
@tool
def smart_analysis(column: str) -> Dict[str, Any]:
    """Automatically choose analysis based on data type."""
    try:
        df = get_current_dataframe()
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        col_data = df[column]
        
        # Different analysis for different types
        if pd.api.types.is_numeric_dtype(col_data):
            result = {
                "type": "numeric",
                "mean": float(col_data.mean()),
                "std": float(col_data.std())
            }
        elif pd.api.types.is_string_dtype(col_data):
            result = {
                "type": "categorical",
                "unique_count": int(col_data.nunique()),
                "top_values": col_data.value_counts().head(5).to_dict()
            }
        else:
            result = {
                "type": str(col_data.dtype),
                "info": "Unsupported type for detailed analysis"
            }
        
        return result
    except Exception as e:
        return {"error": str(e)}
```

### Pattern 4: Sampling for Large Datasets

```python
@tool
def expensive_operation(column: str, sample_size: int = 10000) -> Dict[str, Any]:
    """Operation that's expensive on large datasets."""
    try:
        df = get_current_dataframe()
        
        # Use sampling for large datasets
        if len(df) > sample_size:
            df_sample = df.sample(n=sample_size, random_state=42)
            used_sample = True
        else:
            df_sample = df
            used_sample = False
        
        # Perform expensive operation on sample
        result = df_sample[column].some_expensive_operation()
        
        return {
            "result": result,
            "used_sample": used_sample,
            "sample_size": len(df_sample),
            "total_size": len(df)
        }
    except Exception as e:
        return {"error": str(e)}
```

---

## Example: Complete Tool Implementation

Here's a complete example of a well-designed tool:

```python
from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

class PercentileInput(BaseModel):
    """Input schema for percentile calculation."""
    
    column: str = Field(
        description="Name of the numeric column to calculate percentiles for"
    )
    percentiles: Optional[str] = Field(
        default="25,50,75",
        description="Comma-separated percentile values (0-100). Example: '25,50,75,90'"
    )
    
    @validator('percentiles')
    def validate_percentiles(cls, v):
        try:
            values = [float(x.strip()) for x in v.split(',')]
            if any(x < 0 or x > 100 for x in values):
                raise ValueError("Percentiles must be between 0 and 100")
            return v
        except ValueError as e:
            raise ValueError(f"Invalid percentiles format: {e}")

@tool(args_schema=PercentileInput)
def calculate_percentiles(
    column: str,
    percentiles: str = "25,50,75"
) -> Dict[str, Any]:
    """
    Calculate specific percentiles for a numeric column.
    
    Percentiles help understand the distribution of data. For example:
    - 50th percentile (median) is the middle value
    - 25th percentile means 25% of data is below this value
    - 90th percentile is useful for finding high values
    
    Use this when users ask:
    - "What's the median of X?"
    - "What's the 90th percentile of Y?"
    - "Show me percentiles for Z"
    - "What value do 75% of items fall below?"
    
    Args:
        column: Name of the numeric column to analyze
        percentiles: Comma-separated percentile values (0-100)
    
    Returns:
        Dictionary containing:
        - percentiles: Dict mapping percentile to value
        - column: Column name analyzed
        - count: Number of non-null values
    
    Example output:
        {
            "percentiles": {
                "25": 10.5,
                "50": 25.0,
                "75": 42.3
            },
            "column": "price",
            "count": 1000
        }
    """
    try:
        df = get_current_dataframe()
        
        # Validate column exists
        if column not in df.columns:
            return {
                "error": f"Column '{column}' not found in dataset",
                "available_columns": df.columns.tolist()
            }
        
        # Get column data
        col_data = df[column].dropna()
        
        # Validate numeric type
        if not pd.api.types.is_numeric_dtype(col_data):
            return {
                "error": f"Column '{column}' is not numeric (type: {df[column].dtype})",
                "suggestion": "Try using calculate_value_counts for categorical data"
            }
        
        # Check if column has data
        if len(col_data) == 0:
            return {
                "error": f"Column '{column}' has no non-null values",
                "total_rows": len(df),
                "null_count": df[column].isnull().sum()
            }
        
        # Parse percentiles
        percentile_values = [float(x.strip()) for x in percentiles.split(',')]
        
        # Calculate percentiles
        percentile_results = {}
        for p in percentile_values:
            value = np.percentile(col_data, p)
            percentile_results[str(p)] = float(value)
        
        return {
            "percentiles": percentile_results,
            "column": column,
            "count": len(col_data),
            "min": float(col_data.min()),
            "max": float(col_data.max())
        }
        
    except Exception as e:
        return {
            "error": f"Failed to calculate percentiles: {str(e)}",
            "error_type": type(e).__name__
        }
```

---

## Testing Your Tool

### Unit Test Template

```python
import pytest
import pandas as pd
from app.services.dataset_tools import set_dataset_context, my_new_tool
from app.services.storage import StorageService

@pytest.fixture
def test_dataset():
    """Create test dataset."""
    df = pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'string_col': ['a', 'b', 'c', 'd', 'e'],
        'null_col': [1, None, 3, None, 5]
    })
    dataset_id = 'test_tool_dataset'
    StorageService.save_dataset(dataset_id, df)
    set_dataset_context(dataset_id)
    return dataset_id

def test_my_new_tool_success(test_dataset):
    """Test successful execution."""
    result = my_new_tool.invoke({
        "column": "numeric_col",
        "threshold": 0.5
    })
    
    assert "error" not in result
    assert "result" in result
    # Add specific assertions

def test_my_new_tool_missing_column(test_dataset):
    """Test with missing column."""
    result = my_new_tool.invoke({
        "column": "nonexistent",
        "threshold": 0.5
    })
    
    assert "error" in result
    assert "nonexistent" in result["error"]

def test_my_new_tool_invalid_type(test_dataset):
    """Test with wrong column type."""
    result = my_new_tool.invoke({
        "column": "string_col",
        "threshold": 0.5
    })
    
    assert "error" in result
    # Verify error message is helpful
```

---

## Debugging Tips

### Enable Verbose Logging

```python
# In langchain_llm_service.py
self.agent_executor = AgentExecutor(
    agent=self.agent,
    tools=self.tools,
    verbose=True,  # Shows tool selection and execution
    max_iterations=5,
    handle_parsing_errors=True
)
```

### Test Tool Directly

```python
# Test without agent
from app.services.dataset_tools import my_new_tool, set_dataset_context

set_dataset_context("your_dataset_id")
result = my_new_tool.invoke({"column": "price", "threshold": 0.5})
print(result)
```

### Check Tool Schema

```python
# Verify Pydantic schema is correct
from app.services.dataset_tools import my_new_tool

print(my_new_tool.name)
print(my_new_tool.description)
print(my_new_tool.args_schema.schema())
```

---

## Summary

**To add a new tool:**
1. Define input schema (Pydantic BaseModel)
2. Create tool function with `@tool` decorator
3. Add to `ALL_TOOLS` list
4. Write tests
5. Test with agent

**Key principles:**
- Clear descriptions for LLM understanding
- Type safety with Pydantic
- Comprehensive error handling
- Consistent return format
- Good documentation

Happy tool building! 🛠️
