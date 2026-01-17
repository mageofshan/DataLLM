# Quick Start Guide: Using RAG & LangChain Features

This guide shows you how to use the newly integrated RAG and LangChain features in your DataLLM application.

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Set Up Environment

Create a `.env` file in the `backend` directory:

```bash
OPENROUTER_API_KEY=your_api_key_here
```

### 3. Run Tests

```bash
cd backend
python3 test_rag_langchain.py
```

---

## 📚 Usage Examples

### Example 1: Using RAG for CSV Question Answering

```python
from app.services.vector_store import CSVVectorStore
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
import pandas as pd

# Load your CSV
df = pd.read_csv("your_data.csv")

# Initialize services
vector_store = CSVVectorStore()
llm_service = LLMService()
rag_service = RAGService(llm_service, vector_store)

# Create vector index
dataset_id = "my_dataset"
vector_store.create_collection(
    dataset_id=dataset_id,
    df=df,
    chunk_strategy="hybrid"  # or "row" or "column"
)

# Query with RAG
result = await rag_service.query_with_rag(
    dataset_id=dataset_id,
    query="What is the average sales by region?",
    top_k=5
)

print(result["response"])
print(f"Used {result['metadata']['retrieved_chunks']} chunks")
```

---

### Example 2: Using LangChain Tools Directly

```python
from app.services.dataset_tools import (
    set_dataset_context,
    get_dataset_info,
    calculate_descriptive_statistics,
    calculate_correlation
)
from app.services.storage import StorageService
import pandas as pd

# Load and save dataset
df = pd.read_csv("your_data.csv")
dataset_id = StorageService.save_dataset(df, "your_data.csv")

# Set context for tools
set_dataset_context(dataset_id)

# Use tools directly
info = get_dataset_info.invoke({})
print(f"Dataset has {info['rows']} rows and {info['columns']} columns")

stats = calculate_descriptive_statistics.invoke({
    "columns": ["price", "quantity"],
    "include_percentiles": True
})
print(f"Statistics: {stats['statistics']}")

corr = calculate_correlation.invoke({
    "columns": ["price", "quantity"],
    "method": "pearson"
})
print(f"Correlation: {corr['correlation_matrix']}")
```

---

### Example 3: Using LangChain Agent (Automatic Tool Selection)

```python
from app.services.langchain_llm_service import LangChainLLMService
from app.services.storage import StorageService
import pandas as pd

# Initialize agent
service = LangChainLLMService()

# Load dataset
df = pd.read_csv("your_data.csv")
dataset_id = StorageService.save_dataset(df, "your_data.csv")

# Ask questions - the agent will automatically select tools
result = await service.analyze_dataset(
    dataset_id=dataset_id,
    query="What's the average price and how many items are in each category?"
)

print(result["answer"])
print(f"Tools used: {[tc['tool'] for tc in result['tool_calls']]}")
```

---

### Example 4: Streaming RAG Responses

```python
from app.services.rag_service import RAGService

# ... (initialize services as in Example 1)

# Stream response for real-time chat
async for chunk in rag_service.stream_query_with_rag(
    dataset_id=dataset_id,
    query="Analyze the trends in this data",
    top_k=5
):
    print(chunk, end="", flush=True)
```

---

### Example 5: Using Different Chunking Strategies

```python
# Strategy 1: Row-based (best for small datasets)
vector_store.create_collection(
    dataset_id="dataset_rows",
    df=df,
    chunk_strategy="row"
)

# Strategy 2: Column-based (best for understanding structure)
vector_store.create_collection(
    dataset_id="dataset_cols",
    df=df,
    chunk_strategy="column"
)

# Strategy 3: Hybrid (best for most use cases)
vector_store.create_collection(
    dataset_id="dataset_hybrid",
    df=df,
    chunk_strategy="hybrid"
)
```

---

## 🛠️ Available Tools

### 1. `get_dataset_info`
**Purpose:** Get dataset structure and basic information

**Example:**
```python
result = get_dataset_info.invoke({})
# Returns: rows, columns, column_names, dtypes, sample_data
```

---

### 2. `calculate_descriptive_statistics`
**Purpose:** Calculate mean, median, std, min, max, percentiles

**Example:**
```python
result = calculate_descriptive_statistics.invoke({
    "columns": ["price", "quantity"],  # Optional, None = all numeric
    "include_percentiles": True
})
# Returns: statistics for each column
```

---

### 3. `calculate_correlation`
**Purpose:** Calculate correlation between numeric columns

**Example:**
```python
result = calculate_correlation.invoke({
    "columns": ["price", "rating"],  # Optional
    "method": "pearson",  # or "spearman", "kendall"
    "threshold": 0.5  # Optional, filter by correlation strength
})
# Returns: correlation_matrix, strong_correlations
```

---

### 4. `analyze_missing_data`
**Purpose:** Find missing values in dataset

**Example:**
```python
result = analyze_missing_data.invoke({
    "columns": None  # Optional, None = all columns
})
# Returns: missing counts and percentages per column
```

---

### 5. `detect_outliers`
**Purpose:** Detect outliers using IQR or Z-score methods

**Example:**
```python
result = detect_outliers.invoke({
    "column": "price",
    "method": "iqr",  # or "zscore"
    "threshold": 1.5  # Optional, custom threshold
})
# Returns: outlier_count, outlier_values, bounds
```

---

### 6. `group_and_aggregate`
**Purpose:** Group data and apply aggregation functions

**Example:**
```python
result = group_and_aggregate.invoke({
    "group_columns": ["category", "region"],
    "agg_column": "sales",
    "agg_function": "mean"  # or "sum", "count", "min", "max", "median", "std"
})
# Returns: grouped results
```

---

### 7. `calculate_value_counts`
**Purpose:** Count frequency of unique values

**Example:**
```python
result = calculate_value_counts.invoke({
    "column": "category",
    "top_n": 10,
    "normalize": False  # True for percentages
})
# Returns: value_counts, total_unique_values
```

---

### 8. `filter_data`
**Purpose:** Filter rows based on conditions

**Example:**
```python
result = filter_data.invoke({
    "column": "price",
    "operator": ">",  # or "<", ">=", "<=", "==", "!=", "contains", "in"
    "value": "100"
})
# Returns: matching_rows, sample data
```

---

## 🎯 Best Practices

### When to Use RAG
- ✅ Questions about specific data values
- ✅ Summarization tasks
- ✅ Exploratory data questions
- ✅ When you need context from the data

### When to Use LangChain Agent
- ✅ Complex multi-step analysis
- ✅ When you want automatic tool selection
- ✅ Statistical calculations
- ✅ Data quality checks

### When to Use Tools Directly
- ✅ Programmatic access
- ✅ Batch processing
- ✅ Custom workflows
- ✅ When you know exactly which tool you need

---

## 📊 Chunking Strategy Guide

| Dataset Size | Recommended Strategy | Reason |
|--------------|---------------------|---------|
| < 1,000 rows | `row` | Fast, detailed |
| 1,000 - 10,000 rows | `hybrid` | Balanced |
| > 10,000 rows | `hybrid` or `column` | Efficient |
| Time-series | Custom chunking | By time windows |

---

## 🔧 Configuration Options

### Vector Store Configuration

```python
# Custom persist directory
vector_store = CSVVectorStore(persist_directory="./my_vector_db")

# Query with filters
results = vector_store.query(
    dataset_id="my_data",
    query_text="sales trends",
    top_k=5,
    filter_metadata={"type": "column_summary"}
)
```

### LangChain Agent Configuration

```python
# Custom model
service = LangChainLLMService()
service.llm = ChatOpenAI(
    model="openai/gpt-4o",  # Use a different model
    temperature=0.1
)
```

---

## 🐛 Troubleshooting

### Issue: "Collection not found"
**Solution:** Create the collection first:
```python
vector_store.create_collection(dataset_id, df)
```

### Issue: "No dataset context set"
**Solution:** Set the context before using tools:
```python
set_dataset_context(dataset_id)
```

### Issue: "API key not found"
**Solution:** Set the environment variable:
```bash
export OPENROUTER_API_KEY=your_key_here
```

### Issue: Slow embedding generation
**Solutions:**
- Use smaller dataset for testing
- Use `chunk_strategy="column"` for large datasets
- Consider using GPU if available

---

## 📈 Performance Tips

1. **Cache Vector Embeddings:** Embeddings are persisted automatically in `./data/chroma_db`

2. **Optimize top_k:** Start with 3-5, increase if responses lack context

3. **Use Appropriate Chunking:** 
   - Small datasets: `row`
   - Large datasets: `hybrid`
   - Structure-focused: `column`

4. **Batch Operations:** Process multiple queries in parallel when possible

---

## 🔗 Integration with FastAPI

### Example API Endpoint

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class RAGQueryRequest(BaseModel):
    dataset_id: str
    query: str
    top_k: int = 5

@router.post("/chat/rag")
async def chat_with_rag(request: RAGQueryRequest):
    """RAG-powered chat endpoint."""
    result = await rag_service.query_with_rag(
        dataset_id=request.dataset_id,
        query=request.query,
        top_k=request.top_k
    )
    return result

@router.post("/analyze/agent")
async def analyze_with_agent(request: RAGQueryRequest):
    """LangChain agent analysis endpoint."""
    result = await langchain_service.analyze_dataset(
        dataset_id=request.dataset_id,
        query=request.query
    )
    return result
```

---

## 📚 Additional Resources

- **Main Research:** `.agent/rag_csv_integration_research.md`
- **LangChain Guide:** `README_LANGCHAIN_INTEGRATION.md`
- **Tool Creation:** `TOOL_CREATION_GUIDE.md`
- **Test Results:** `TEST_RESULTS_RAG_LANGCHAIN.md`

---

## 🎓 Example Queries

### Good RAG Queries
- "What is the highest value in the sales column?"
- "Summarize the trends in customer data"
- "What's the date range of this dataset?"

### Good Agent Queries
- "What's the average price by category?"
- "Are there any missing values in the data?"
- "Show me the correlation between price and rating"
- "How many unique products are there?"

### Complex Multi-Step Queries
- "What's the average sales for products with rating above 4?"
- "Find outliers in the price column and tell me their categories"
- "Group by region and show me the top 5 by total sales"

---

**Happy Analyzing! 🚀**

For questions or issues, refer to the documentation or run the test suite to verify your setup.
