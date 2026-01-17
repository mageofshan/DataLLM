# Migration Guide: From Manual Tools to LangChain

## Overview

This guide helps you migrate from the current manual tool implementation to the new LangChain-based system with dynamic tool selection.

---

## Current vs. New Architecture

### Current Implementation (`llm_service.py`)

**Pros:**
- ✅ Simple and straightforward
- ✅ Direct control over tool execution
- ✅ Already working with OpenRouter

**Cons:**
- ❌ Tools hardcoded in JSON format
- ❌ Limited to 2 tools (get_dataset_sample, execute_code)
- ❌ Manual tool schema definitions
- ❌ No automatic tool selection optimization
- ❌ Difficult to add new tools

### New Implementation (`langchain_llm_service.py`)

**Pros:**
- ✅ 8+ pre-built tools ready to use
- ✅ Easy to add new tools with `@tool` decorator
- ✅ Automatic schema generation from type hints
- ✅ Better LLM understanding through enhanced descriptions
- ✅ Multi-step reasoning support
- ✅ Conversation history support
- ✅ Detailed execution traces

**Cons:**
- ⚠️ Additional dependency (LangChain)
- ⚠️ Slightly more complex setup
- ⚠️ Need to test compatibility with OpenRouter

---

## Migration Strategy: Gradual Rollout (Recommended)

### Phase 1: Setup (Week 1)

1. **Install dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Test tool implementations**
   ```bash
   # Create a test script
   python -m pytest tests/test_dataset_tools.py
   ```

3. **Verify OpenRouter compatibility**
   - Test that LangChain's ChatOpenAI works with OpenRouter
   - Confirm tool calling is supported

### Phase 2: Parallel Implementation (Week 2)

1. **Add feature flag to environment**
   ```bash
   # .env
   USE_LANGCHAIN_AGENT=false  # Start with false
   ```

2. **Update API endpoint to support both**
   ```python
   # In your API route
   from app.services.llm_service import LLMService
   from app.services.langchain_llm_service import LangChainLLMService
   import os
   
   @router.post("/analyze")
   async def analyze_dataset(request: AnalyzeRequest):
       use_langchain = os.getenv("USE_LANGCHAIN_AGENT", "false").lower() == "true"
       
       if use_langchain:
           service = LangChainLLMService()
       else:
           service = LLMService()
       
       result = await service.analyze_dataset(
           dataset_id=request.dataset_id,
           query=request.query
       )
       
       return result
   ```

3. **Test both implementations side-by-side**
   - Run same queries through both services
   - Compare results for accuracy
   - Measure performance differences

### Phase 3: Gradual Migration (Week 3)

1. **Enable for internal testing**
   ```bash
   USE_LANGCHAIN_AGENT=true
   ```

2. **Test with real datasets**
   - Upload various CSV files
   - Ask diverse questions
   - Verify tool selection is appropriate

3. **Monitor for issues**
   - Check logs for errors
   - Measure response times
   - Collect user feedback

### Phase 4: Full Rollout (Week 4)

1. **Make LangChain the default**
   ```bash
   USE_LANGCHAIN_AGENT=true
   ```

2. **Remove old implementation** (optional)
   - Keep as fallback for a while
   - Eventually deprecate and remove

3. **Document the new system**
   - Update README
   - Add tool documentation
   - Create user guide

---

## Code Changes Required

### 1. Update API Endpoint

**File:** `backend/app/api/chat.py` (or wherever your analyze endpoint is)

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.langchain_llm_service import LangChainLLMService
from typing import Optional, List, Dict

router = APIRouter()

class AnalyzeRequest(BaseModel):
    dataset_id: str
    query: str
    conversation_history: Optional[List[Dict[str, str]]] = None

class AnalyzeResponse(BaseModel):
    answer: str
    tool_calls: List[Dict]
    intermediate_steps: List

@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_dataset(request: AnalyzeRequest):
    """
    Analyze a dataset using natural language query.
    
    The LLM will automatically select appropriate tools to answer the question.
    """
    try:
        service = LangChainLLMService()
        result = await service.analyze_dataset(
            dataset_id=request.dataset_id,
            query=request.query,
            conversation_history=request.conversation_history
        )
        
        return AnalyzeResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tools")
async def get_available_tools():
    """Get list of available analysis tools."""
    service = LangChainLLMService()
    tools = await service.get_available_tools()
    return {"tools": tools}
```

### 2. Update Frontend (if needed)

**File:** `frontend/src/api/chat.ts` (or similar)

```typescript
interface AnalyzeRequest {
  dataset_id: string;
  query: string;
  conversation_history?: Array<{role: string; content: string}>;
}

interface ToolCall {
  tool: string;
  input: any;
  output: any;
}

interface AnalyzeResponse {
  answer: string;
  tool_calls: ToolCall[];
  intermediate_steps: any[];
}

export async function analyzeDataset(request: AnalyzeRequest): Promise<AnalyzeResponse> {
  const response = await fetch('/api/analyze', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(request)
  });
  
  if (!response.ok) {
    throw new Error('Analysis failed');
  }
  
  return response.json();
}
```

### 3. Add Environment Variables

**File:** `backend/.env`

```bash
# Existing
OPENROUTER_API_KEY=your_key_here

# New (optional)
USE_LANGCHAIN_AGENT=true
LANGCHAIN_VERBOSE=true  # For debugging
LANGCHAIN_MAX_ITERATIONS=5
```

---

## Testing Checklist

### Unit Tests

- [ ] Test each tool independently
- [ ] Test with various data types (numeric, categorical, datetime)
- [ ] Test error handling (missing columns, invalid inputs)
- [ ] Test edge cases (empty datasets, single row, all nulls)

### Integration Tests

- [ ] Test agent with simple queries
- [ ] Test agent with complex multi-step queries
- [ ] Test conversation history
- [ ] Test with different dataset sizes

### End-to-End Tests

- [ ] Upload CSV and ask questions
- [ ] Test all tool types (stats, correlation, grouping, filtering)
- [ ] Test error messages are user-friendly
- [ ] Test response times are acceptable

### Example Test Cases

```python
# tests/test_dataset_tools.py
import pytest
import pandas as pd
from app.services.dataset_tools import (
    set_dataset_context,
    calculate_descriptive_statistics,
    calculate_correlation,
    filter_data
)
from app.services.storage import StorageService

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    df = pd.DataFrame({
        'price': [10.0, 20.0, 30.0, 40.0, 50.0],
        'rating': [4.5, 3.8, 4.2, 4.9, 3.5],
        'category': ['A', 'B', 'A', 'C', 'B']
    })
    dataset_id = 'test_dataset'
    StorageService.save_dataset(dataset_id, df)
    set_dataset_context(dataset_id)
    return dataset_id

def test_descriptive_statistics(sample_dataset):
    """Test descriptive statistics calculation."""
    result = calculate_descriptive_statistics.invoke({
        "columns": ["price"],
        "include_percentiles": True
    })
    
    assert "statistics" in result
    assert "price" in result["statistics"]
    assert result["statistics"]["price"]["mean"] == 30.0
    assert result["statistics"]["price"]["min"] == 10.0
    assert result["statistics"]["price"]["max"] == 50.0

def test_correlation(sample_dataset):
    """Test correlation calculation."""
    result = calculate_correlation.invoke({
        "columns": ["price", "rating"],
        "method": "pearson"
    })
    
    assert "correlation_matrix" in result
    assert "price" in result["correlation_matrix"]
    assert "rating" in result["correlation_matrix"]

def test_filter_data(sample_dataset):
    """Test data filtering."""
    result = filter_data.invoke({
        "column": "price",
        "operator": ">",
        "value": "25"
    })
    
    assert "matching_rows" in result
    assert result["matching_rows"] == 3  # 30, 40, 50
```

---

## Performance Considerations

### Optimization Tips

1. **Caching**
   - Cache dataset metadata (columns, types, row count)
   - Cache expensive calculations (correlation matrices)
   - Use Redis or in-memory cache

2. **Sampling**
   - For large datasets (>1M rows), use sampling for exploratory tools
   - Full dataset for aggregations and counts

3. **Async Operations**
   - All tools should support async execution
   - Use `asyncio.gather()` for parallel tool calls

4. **Rate Limiting**
   - Limit concurrent requests per user
   - Implement request queuing for expensive operations

### Expected Performance

| Operation | Small Dataset (<10K rows) | Large Dataset (>1M rows) |
|-----------|---------------------------|--------------------------|
| get_dataset_info | <100ms | <500ms |
| calculate_descriptive_statistics | <200ms | <2s |
| calculate_correlation | <500ms | <5s |
| group_and_aggregate | <300ms | <3s |
| filter_data | <200ms | <2s |

---

## Troubleshooting

### Common Issues

**Issue 1: "No dataset context set"**
```
Solution: Ensure set_dataset_context() is called before tool execution
```

**Issue 2: "Tool not found" or "Unknown tool"**
```
Solution: Verify tool is in ALL_TOOLS list and properly imported
```

**Issue 3: "OpenRouter doesn't support tool calling"**
```
Solution: Check if the model supports function calling. Use gpt-4o-mini or gpt-4o
```

**Issue 4: "Agent exceeds max iterations"**
```
Solution: Increase max_iterations or simplify the query
```

**Issue 5: "Pydantic validation error"**
```
Solution: Check tool input schema matches what LLM is providing
```

---

## Rollback Plan

If issues arise, you can quickly rollback:

1. **Set feature flag to false**
   ```bash
   USE_LANGCHAIN_AGENT=false
   ```

2. **Restart service**
   ```bash
   uvicorn app.main:app --reload
   ```

3. **Investigate issues**
   - Check logs
   - Review error messages
   - Test specific tools

4. **Fix and retry**
   - Make necessary corrections
   - Re-enable feature flag
   - Monitor closely

---

## Future Enhancements

Once the basic system is working, consider:

1. **Add more specialized tools**
   - Time series analysis
   - Statistical hypothesis testing
   - Anomaly detection
   - Predictive modeling

2. **Implement semantic routing**
   - Use embeddings to find relevant tools
   - Reduce token usage for large tool sets

3. **Add visualization tools**
   - Generate chart configurations
   - Suggest appropriate visualizations

4. **Implement caching layer**
   - Cache tool results
   - Invalidate on dataset changes

5. **Add user feedback loop**
   - Let users rate answers
   - Use feedback to improve tool selection

---

## Support & Resources

- **LangChain Documentation**: https://python.langchain.com/
- **OpenRouter API Docs**: https://openrouter.ai/docs
- **Internal Documentation**: See `LANGCHAIN_TOOL_INTEGRATION_RESEARCH.md`

---

## Summary

**Recommended Timeline:**
- Week 1: Setup and testing
- Week 2: Parallel implementation
- Week 3: Internal testing
- Week 4: Full rollout

**Key Success Metrics:**
- ✅ All existing queries work with new system
- ✅ Response accuracy >= current system
- ✅ Response time < 5 seconds for most queries
- ✅ Zero critical errors in production
- ✅ Positive user feedback

**Next Steps:**
1. Review this guide with your team
2. Install dependencies
3. Run test suite
4. Enable feature flag for testing
5. Monitor and iterate

Good luck with the migration! 🚀
