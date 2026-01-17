# RAG Integration for CSV Files - Research & Implementation Guide

## Executive Summary

This document outlines how to integrate a **Retrieval-Augmented Generation (RAG)** system for CSV files into your DataLLM chatbot. Based on current best practices (2024), this guide provides multiple implementation approaches tailored to your existing architecture.

---

## 1. What is RAG and Why Use It for CSV Files?

### RAG Overview
**Retrieval-Augmented Generation (RAG)** enhances LLM responses by:
1. **Retrieving** relevant information from an external knowledge base
2. **Augmenting** the LLM prompt with this retrieved context
3. **Generating** more accurate, contextually-aware responses

### Benefits for CSV Data Analysis
- **Reduced Hallucinations**: LLM answers are grounded in actual data
- **Scalability**: Handle large CSV files that exceed LLM context windows
- **Semantic Search**: Find relevant data rows/columns using natural language
- **Multi-file Support**: Query across multiple related CSV datasets
- **Historical Context**: Maintain conversation history with data context

---

## 2. RAG Architecture for CSV Files

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG Pipeline for CSV                     │
└─────────────────────────────────────────────────────────────┘

1. DATA PREPARATION
   CSV File → Pandas DataFrame → Text Chunks
   
2. EMBEDDING GENERATION
   Text Chunks → Embedding Model → Vector Embeddings
   
3. VECTOR STORAGE
   Embeddings + Metadata → Vector Database (FAISS/Chroma)
   
4. RETRIEVAL (Query Time)
   User Query → Query Embedding → Similarity Search → Top-K Chunks
   
5. AUGMENTED GENERATION
   Retrieved Context + User Query → LLM → Enhanced Response
```

### Key Technologies

| Component | Options | Recommendation for Your Project |
|-----------|---------|----------------------------------|
| **Embedding Model** | OpenAI, Sentence-Transformers, Ollama | `sentence-transformers/all-MiniLM-L6-v2` (free, fast) or OpenAI `text-embedding-3-small` |
| **Vector Database** | FAISS, ChromaDB, Pinecone, Qdrant | **ChromaDB** (lightweight, persistent, no server needed) |
| **RAG Framework** | LangChain, LlamaIndex, Custom | **LangChain** (comprehensive, well-documented) |
| **Chunking Strategy** | Row-based, Column-based, Hybrid | **Row-based with metadata** (preserves data relationships) |

---

## 3. Implementation Approaches

### Approach A: LangChain CSV Agent (Simplest)

**Best for**: Quick integration, simple Q&A over CSV files

```python
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_csv_agent

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Create CSV agent
agent = create_csv_agent(
    llm,
    "path/to/dataset.csv",
    verbose=True,
    agent_type="openai-tools",
    allow_dangerous_code=True  # Executes LLM-generated Python code
)

# Query the agent
response = agent.invoke("What is the average closing price?")
```

**Pros**:
- Minimal code required
- Handles tool calling automatically
- Works with multiple CSV files

**Cons**:
- Executes arbitrary Python code (security risk)
- No semantic search (relies on LLM understanding)
- Limited control over retrieval process

---

### Approach B: Vector-Based RAG (Recommended)

**Best for**: Large datasets, semantic search, production systems

#### Step 1: Data Preparation & Chunking

```python
import pandas as pd
from typing import List, Dict

def csv_to_text_chunks(df: pd.DataFrame, chunk_strategy: str = "row") -> List[Dict]:
    """
    Convert CSV data into text chunks suitable for embedding.
    
    Args:
        df: Pandas DataFrame
        chunk_strategy: "row", "column", or "hybrid"
    
    Returns:
        List of dicts with 'text' and 'metadata'
    """
    chunks = []
    
    if chunk_strategy == "row":
        # Each row becomes a document
        for idx, row in df.iterrows():
            text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append({
                "text": text,
                "metadata": {
                    "row_index": idx,
                    "source": "dataset",
                    **row.to_dict()  # Store original values as metadata
                }
            })
    
    elif chunk_strategy == "column":
        # Each column becomes a document
        for col in df.columns:
            text = f"Column: {col}\nData Type: {df[col].dtype}\n"
            text += f"Sample Values: {df[col].head(10).tolist()}\n"
            text += f"Statistics: {df[col].describe().to_dict()}"
            chunks.append({
                "text": text,
                "metadata": {"column_name": col, "dtype": str(df[col].dtype)}
            })
    
    elif chunk_strategy == "hybrid":
        # Combine row and column information
        # Add column summaries
        for col in df.columns:
            text = f"Column '{col}' contains {df[col].dtype} data. "
            if df[col].dtype in ['int64', 'float64']:
                text += f"Range: {df[col].min()} to {df[col].max()}. "
                text += f"Mean: {df[col].mean():.2f}."
            chunks.append({
                "text": text,
                "metadata": {"type": "column_summary", "column": col}
            })
        
        # Add row data
        for idx, row in df.iterrows():
            text = ", ".join([f"{col}: {val}" for col, val in row.items()])
            chunks.append({
                "text": text,
                "metadata": {"type": "row_data", "row_index": idx}
            })
    
    return chunks
```

#### Step 2: Embedding & Vector Storage with ChromaDB

```python
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import uuid

class CSVVectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB vector store."""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Use sentence-transformers for embeddings (free, local)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def create_collection(self, dataset_id: str, df: pd.DataFrame):
        """Create a new collection for a dataset."""
        # Delete if exists
        try:
            self.client.delete_collection(name=dataset_id)
        except:
            pass
        
        collection = self.client.create_collection(
            name=dataset_id,
            metadata={"description": f"Vector store for dataset {dataset_id}"}
        )
        
        # Convert CSV to text chunks
        chunks = csv_to_text_chunks(df, chunk_strategy="hybrid")
        
        # Generate embeddings
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts).tolist()
        
        # Add to ChromaDB
        collection.add(
            ids=[str(uuid.uuid4()) for _ in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[chunk["metadata"] for chunk in chunks]
        )
        
        return collection
    
    def query(self, dataset_id: str, query_text: str, top_k: int = 5):
        """Retrieve top-k most relevant chunks for a query."""
        collection = self.client.get_collection(name=dataset_id)
        
        # Embed the query
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Search
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        
        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }
```

#### Step 3: RAG Service Integration

```python
from typing import Optional

class RAGService:
    def __init__(self, llm_service, vector_store: CSVVectorStore):
        self.llm_service = llm_service
        self.vector_store = vector_store
    
    async def query_with_rag(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        model: str = "openai/gpt-4o-mini"
    ) -> str:
        """
        Answer a query using RAG.
        
        1. Retrieve relevant chunks from vector store
        2. Augment prompt with retrieved context
        3. Generate response with LLM
        """
        # Retrieve relevant context
        retrieval_results = self.vector_store.query(
            dataset_id=dataset_id,
            query_text=query,
            top_k=top_k
        )
        
        # Build context from retrieved documents
        context = "\n\n".join([
            f"[Chunk {i+1}] {doc}"
            for i, doc in enumerate(retrieval_results["documents"])
        ])
        
        # Create augmented prompt
        augmented_prompt = f"""You are a data analysis assistant. Use the following retrieved information from the dataset to answer the user's question accurately.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}

Provide a clear, accurate answer based on the retrieved context. If the context doesn't contain enough information, say so."""
        
        # Generate response
        response = await self.llm_service.generate_response(
            prompt=augmented_prompt,
            system_prompt="You are a helpful data analysis assistant that answers questions based on provided context.",
            model=model
        )
        
        return response
```

---

### Approach C: Hybrid (Tool Calling + RAG)

**Best for**: Maximum flexibility, complex queries

Combine your existing tool calling approach with RAG:

```python
async def analyze_with_hybrid_rag(
    self,
    dataset_id: str,
    query: str,
    model: str = "openai/gpt-4o-mini"
) -> AnalysisResult:
    """
    Hybrid approach: Use RAG for context + tool calling for computation.
    """
    # 1. Retrieve relevant context via RAG
    retrieval_results = self.vector_store.query(dataset_id, query, top_k=3)
    context = "\n".join(retrieval_results["documents"])
    
    # 2. Define tools (same as before)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_dataset_sample",
                "description": "Get a sample of the dataset.",
                # ... (same as your current implementation)
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_code",
                "description": "Execute Python code on the dataset.",
                # ... (same as your current implementation)
            }
        }
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
    
    # 4. Continue with tool calling loop (same as your current implementation)
    # ... rest of your analyze_dataset logic
```

---

## 4. Implementation Recommendations for Your Project

### Recommended Architecture

Based on your current codebase, I recommend a **phased approach**:

#### Phase 1: Add Basic RAG (Quick Win)
1. Install dependencies: `chromadb`, `sentence-transformers`
2. Create `rag_service.py` with `CSVVectorStore` class
3. Add vector indexing when datasets are uploaded
4. Create a new endpoint `/chat/rag` that uses RAG retrieval

#### Phase 2: Hybrid Integration
1. Modify your existing `analyze_dataset` method to include RAG context
2. Use RAG for initial context, then tool calling for computations
3. Store conversation history with RAG context

#### Phase 3: Advanced Features
1. Multi-dataset RAG (query across multiple CSVs)
2. Incremental updates (add new rows without re-indexing)
3. Custom chunking strategies based on data type
4. Query rewriting for better retrieval

---

## 5. Code Integration Plan

### File Structure
```
backend/app/services/
├── llm_service.py          # Existing
├── rag_service.py          # NEW - RAG logic
├── vector_store.py         # NEW - ChromaDB wrapper
├── chunking.py             # NEW - CSV chunking strategies
└── storage.py              # MODIFY - Add vector indexing on upload
```

### Modified `storage.py`
```python
from app.services.vector_store import CSVVectorStore

class StorageService:
    vector_store = CSVVectorStore()
    
    @staticmethod
    def save_dataset(dataset_id: str, df: pd.DataFrame, filename: str):
        # ... existing save logic ...
        
        # NEW: Create vector index
        try:
            StorageService.vector_store.create_collection(dataset_id, df)
            print(f"✓ Vector index created for {dataset_id}")
        except Exception as e:
            print(f"⚠ Vector indexing failed: {e}")
        
        return dataset_id
```

### New API Endpoint
```python
# In app/api/chat.py

@router.post("/rag")
async def chat_with_rag(
    dataset_id: str,
    query: str,
    top_k: int = 5
):
    """Chat endpoint using RAG for context retrieval."""
    rag_service = RAGService(llm_service, vector_store)
    response = await rag_service.query_with_rag(
        dataset_id=dataset_id,
        query=query,
        top_k=top_k
    )
    return {"response": response}
```

---

## 6. Dependencies to Add

Add to `requirements.txt`:
```
# RAG Dependencies
chromadb>=0.4.22
sentence-transformers>=2.2.2
langchain>=0.1.0
langchain-openai>=0.0.5
langchain-experimental>=0.0.47
```

Optional (for advanced features):
```
# Advanced RAG
faiss-cpu>=1.7.4  # Alternative to ChromaDB
tiktoken>=0.5.2   # Token counting for chunking
```

---

## 7. Security Considerations

### Code Execution Risks
- Your current `execute_code` tool runs arbitrary Python code
- **Recommendation**: Sandbox execution environment or use read-only operations

### RAG-Specific Security
- **Prompt Injection**: Validate retrieved context before adding to prompt
- **Data Leakage**: Ensure vector store access is scoped to user sessions
- **Metadata Exposure**: Be careful what metadata you store with embeddings

---

## 8. Performance Optimization

### Embedding Generation
- **Batch Processing**: Generate embeddings in batches (100-1000 rows)
- **Caching**: Cache embeddings for unchanged datasets
- **Model Selection**: 
  - Fast: `all-MiniLM-L6-v2` (384 dimensions)
  - Accurate: `all-mpnet-base-v2` (768 dimensions)

### Vector Search
- **Index Optimization**: ChromaDB uses HNSW by default (fast)
- **Top-K Selection**: Start with 3-5, adjust based on context window
- **Filtering**: Use metadata filters to narrow search space

### Chunking Strategy
- **Small CSVs (<1000 rows)**: Row-based chunking
- **Large CSVs (>10,000 rows)**: Hybrid with column summaries
- **Time-series data**: Group by time windows

---

## 9. Testing Strategy

### Unit Tests
```python
def test_csv_chunking():
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    chunks = csv_to_text_chunks(df, "row")
    assert len(chunks) == 2
    assert "A: 1" in chunks[0]["text"]

def test_vector_store_query():
    store = CSVVectorStore()
    # ... test retrieval accuracy
```

### Integration Tests
```python
async def test_rag_query():
    # Upload test CSV
    # Query with RAG
    # Verify response contains expected data
```

---

## 10. Example Use Cases

### Use Case 1: Semantic Search
```
User: "Show me all records where the stock price increased significantly"
RAG: Retrieves rows with high price changes
LLM: Interprets "significantly" and formats results
```

### Use Case 2: Multi-hop Reasoning
```
User: "What's the correlation between volume and price changes?"
RAG: Retrieves volume and price columns
Tool: Executes correlation calculation
LLM: Explains the result
```

### Use Case 3: Conversational Context
```
User: "What's the average closing price?"
Assistant: "The average closing price is $159.42"
User: "And the highest?"
RAG: Retrieves previous context + new query
Assistant: "The highest closing price is $164.70"
```

---

## 11. Next Steps

### Immediate Actions
1. ✅ Review this research document
2. ⬜ Decide on implementation approach (A, B, or C)
3. ⬜ Install dependencies
4. ⬜ Implement basic RAG service
5. ⬜ Test with your `test_stock_data.csv`

### Future Enhancements
- GraphRAG for interconnected data
- Multi-modal RAG (CSV + images/charts)
- Real-time streaming RAG responses
- Fine-tuned embedding models for domain-specific data

---

## 12. References & Resources

### Documentation
- [LangChain CSV Agent](https://python.langchain.com/docs/integrations/toolkits/csv)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)

### Tutorials
- [RAG for CSV Files - Machine Learning Plus](https://machinelearningplus.com)
- [Building RAG Systems - Medium](https://medium.com)

### GitHub Examples
- [LangChain CSV Examples](https://github.com/langchain-ai/langchain)
- [ChromaDB Cookbook](https://github.com/chroma-core/chroma)

---

## Appendix: Comparison Matrix

| Feature | Current Implementation | LangChain Agent | Vector RAG | Hybrid |
|---------|------------------------|-----------------|------------|--------|
| **Semantic Search** | ❌ | ❌ | ✅ | ✅ |
| **Code Execution** | ✅ | ✅ | ❌ | ✅ |
| **Scalability** | Medium | Medium | High | High |
| **Setup Complexity** | Low | Low | Medium | High |
| **Context Awareness** | Low | Medium | High | Very High |
| **Security** | Medium | Low | High | Medium |
| **Cost** | Low | Medium | Low* | Medium |

*Using local embeddings (sentence-transformers)

---

**Document Version**: 1.0  
**Last Updated**: 2025-12-13  
**Author**: Antigravity AI Assistant
