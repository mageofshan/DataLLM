# RAG Integration Quick Start Guide

## Overview
This guide will help you integrate the RAG (Retrieval-Augmented Generation) system for CSV files into your DataLLM chatbot.

## What Was Added

### New Files
1. **`backend/app/services/vector_store.py`** - ChromaDB vector store for semantic search
2. **`backend/app/services/rag_service.py`** - RAG service combining retrieval + generation
3. **`.agent/rag_csv_integration_research.md`** - Comprehensive research document

### New Dependencies
- `chromadb` - Vector database for storing embeddings
- `sentence-transformers` - Local embedding model (free, no API needed)

---

## Installation Steps

### 1. Install Dependencies

```bash
cd /Users/saishantanusivakumaran/DataLLM/backend
pip install chromadb sentence-transformers
```

### 2. Test the Vector Store

Create a test script to verify the installation:

```python
# test_rag.py
import pandas as pd
from app.services.vector_store import CSVVectorStore

# Load your test CSV
df = pd.read_csv("../test_stock_data.csv")

# Initialize vector store
vector_store = CSVVectorStore()

# Create collection
success = vector_store.create_collection("test_stock", df, chunk_strategy="hybrid")
print(f"Collection created: {success}")

# Query the vector store
results = vector_store.query(
    dataset_id="test_stock",
    query_text="What are the closing prices?",
    top_k=3
)

print("\nRetrieved documents:")
for i, doc in enumerate(results["documents"]):
    print(f"\n[{i+1}] {doc}")
```

Run it:
```bash
python test_rag.py
```

---

## Integration Options

### Option 1: Standalone RAG Endpoint (Easiest)

Add a new endpoint to your API for RAG-based queries:

```python
# In backend/app/api/chat.py (or create new file)

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.llm_service import LLMService
from app.services.vector_store import CSVVectorStore
from app.services.rag_service import RAGService

router = APIRouter()

# Initialize services
llm_service = LLMService()
vector_store = CSVVectorStore()
rag_service = RAGService(llm_service, vector_store)

class RAGQueryRequest(BaseModel):
    dataset_id: str
    query: str
    top_k: int = 5

@router.post("/chat/rag")
async def chat_with_rag(request: RAGQueryRequest):
    """Chat endpoint using RAG for context retrieval."""
    result = await rag_service.query_with_rag(
        dataset_id=request.dataset_id,
        query=request.query,
        top_k=request.top_k
    )
    return result

@router.post("/chat/rag/stream")
async def stream_chat_with_rag(request: RAGQueryRequest):
    """Streaming RAG chat endpoint."""
    from fastapi.responses import StreamingResponse
    
    async def generate():
        async for chunk in rag_service.stream_query_with_rag(
            dataset_id=request.dataset_id,
            query=request.query,
            top_k=request.top_k
        ):
            yield chunk
    
    return StreamingResponse(generate(), media_type="text/plain")
```

### Option 2: Modify Existing Storage Service

Update `storage.py` to automatically create vector indexes when datasets are uploaded:

```python
# Add to backend/app/services/storage.py

from app.services.vector_store import CSVVectorStore

class StorageService:
    # Add class variable
    vector_store = CSVVectorStore()
    
    @staticmethod
    def save_dataset(dataset_id: str, df: pd.DataFrame, filename: str):
        # ... existing save logic ...
        
        # NEW: Create vector index automatically
        try:
            success = StorageService.vector_store.create_collection(
                dataset_id=dataset_id,
                df=df,
                chunk_strategy="hybrid"
            )
            if success:
                print(f"✓ Vector index created for {dataset_id}")
        except Exception as e:
            print(f"⚠ Vector indexing failed (non-critical): {e}")
        
        return dataset_id
```

### Option 3: Hybrid Approach (Most Powerful)

Combine RAG with your existing tool calling:

```python
# Modify backend/app/services/llm_service.py

async def analyze_dataset_with_rag(
    self,
    dataset_id: str,
    query: str,
    vector_store: CSVVectorStore,
    model: str = "openai/gpt-4o-mini"
):
    """Enhanced analysis with RAG context."""
    
    # 1. Retrieve relevant context via RAG
    retrieval_results = vector_store.query(dataset_id, query, top_k=3)
    context = "\n".join(retrieval_results["documents"])
    
    # 2. Define tools (same as before)
    tools = [
        # ... your existing tools ...
    ]
    
    # 3. Create messages with RAG context
    messages = [
        {
            "role": "system",
            "content": f"""You are a data analysis assistant with access to tools and retrieved context.

RETRIEVED CONTEXT FROM DATASET:
{context}

Use this context to understand the data structure. Then use tools to perform calculations if needed."""
        },
        {"role": "user", "content": query}
    ]
    
    # 4. Continue with your existing tool calling loop
    # ... rest of analyze_dataset logic ...
```

---

## Usage Examples

### Example 1: Basic RAG Query

```python
from app.services.llm_service import LLMService
from app.services.vector_store import CSVVectorStore
from app.services.rag_service import RAGService

# Initialize
llm_service = LLMService()
vector_store = CSVVectorStore()
rag_service = RAGService(llm_service, vector_store)

# Query
result = await rag_service.query_with_rag(
    dataset_id="test_stock",
    query="What was the highest closing price?",
    top_k=5
)

print(result["response"])
print(f"\nUsed {result['metadata']['retrieved_chunks']} chunks")
```

### Example 2: Streaming Response

```python
async for chunk in rag_service.stream_query_with_rag(
    dataset_id="test_stock",
    query="Analyze the price trends",
    top_k=5
):
    print(chunk, end="", flush=True)
```

### Example 3: Check Vector Index Status

```python
info = vector_store.get_collection_info("test_stock")
if info:
    print(f"Collection: {info['name']}")
    print(f"Documents: {info['count']}")
    print(f"Metadata: {info['metadata']}")
```

---

## Testing Checklist

- [ ] Install dependencies (`chromadb`, `sentence-transformers`)
- [ ] Run test script with `test_stock_data.csv`
- [ ] Verify embeddings are generated
- [ ] Test semantic search with sample queries
- [ ] Integrate into existing API
- [ ] Test with frontend (if applicable)

---

## Performance Tips

### 1. Chunking Strategy
- **Small datasets (<1000 rows)**: Use `"row"` strategy
- **Large datasets (>10000 rows)**: Use `"hybrid"` strategy
- **Time-series data**: Consider custom chunking by time windows

### 2. Top-K Selection
- Start with `top_k=3-5`
- Increase if responses lack context
- Decrease if responses are too verbose

### 3. Embedding Model
- Current: `all-MiniLM-L6-v2` (fast, 384 dimensions)
- For better accuracy: `all-mpnet-base-v2` (slower, 768 dimensions)

### 4. Caching
- Vector embeddings are persisted in `./data/chroma_db`
- No need to re-index unless data changes

---

## Troubleshooting

### Issue: "Collection not found"
**Solution**: Ensure you've created the collection first:
```python
vector_store.create_collection(dataset_id, df)
```

### Issue: Slow embedding generation
**Solution**: 
- Use smaller embedding model
- Process in batches (already implemented)
- Consider using GPU if available

### Issue: Poor retrieval quality
**Solution**:
- Try different chunking strategies
- Increase `top_k`
- Rephrase the query to be more specific

### Issue: Out of memory
**Solution**:
- Reduce batch size in `create_collection`
- Use smaller embedding model
- Process large CSVs in chunks

---

## Next Steps

1. **Test with your data**: Try with `test_stock_data.csv`
2. **Integrate into API**: Choose Option 1, 2, or 3 above
3. **Update frontend**: Add RAG toggle or separate chat mode
4. **Monitor performance**: Track retrieval quality and response time
5. **Iterate**: Adjust chunking strategy and top_k based on results

---

## Advanced Features (Future)

- [ ] Multi-dataset RAG (query across multiple CSVs)
- [ ] Incremental updates (add new rows without full re-index)
- [ ] Custom embedding models fine-tuned on your domain
- [ ] GraphRAG for interconnected data
- [ ] Conversation memory with RAG context

---

## Resources

- **Research Document**: `.agent/rag_csv_integration_research.md`
- **ChromaDB Docs**: https://docs.trychroma.com/
- **Sentence Transformers**: https://www.sbert.net/

---

**Ready to get started?** Run the installation steps and test script above!
