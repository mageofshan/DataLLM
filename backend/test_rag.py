"""
Test script for RAG integration with CSV files
Run this to verify the RAG system is working correctly
"""

import asyncio
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.services.vector_store import CSVVectorStore
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


async def test_rag_system():
    """Test the complete RAG pipeline."""
    
    print("=" * 60)
    print("RAG System Test for CSV Files")
    print("=" * 60)
    
    # 1. Load test CSV
    print("\n[1/5] Loading test CSV...")
    csv_path = "../test_stock_data.csv"
    
    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # 2. Initialize vector store
    print("\n[2/5] Initializing vector store...")
    vector_store = CSVVectorStore(persist_directory="./data/chroma_db_test")
    print("✓ Vector store initialized")
    
    # 3. Create vector collection
    print("\n[3/5] Creating vector embeddings...")
    dataset_id = "test_stock_data"
    
    success = vector_store.create_collection(
        dataset_id=dataset_id,
        df=df,
        chunk_strategy="hybrid"
    )
    
    if success:
        print("✓ Vector embeddings created successfully")
        
        # Get collection info
        info = vector_store.get_collection_info(dataset_id)
        if info:
            print(f"  Collection: {info['name']}")
            print(f"  Documents: {info['count']}")
            print(f"  Metadata: {info['metadata']}")
    else:
        print("❌ Failed to create vector embeddings")
        return
    
    # 4. Test semantic search
    print("\n[4/5] Testing semantic search...")
    test_queries = [
        "What are the closing prices?",
        "Show me the highest volume",
        "What happened on January 3rd?"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = vector_store.query(
            dataset_id=dataset_id,
            query_text=query,
            top_k=2
        )
        
        print(f"  Retrieved {len(results['documents'])} chunks:")
        for i, doc in enumerate(results['documents']):
            print(f"    [{i+1}] {doc[:100]}...")
    
    # 5. Test RAG with LLM
    print("\n[5/5] Testing RAG with LLM...")
    
    llm_service = LLMService()
    
    if llm_service.mock_mode:
        print("⚠ Warning: Running in MOCK mode (OPENROUTER_API_KEY not set)")
        print("  Set OPENROUTER_API_KEY to test full RAG functionality")
    else:
        print("✓ LLM service initialized")
    
    rag_service = RAGService(llm_service, vector_store)
    
    # Test query
    test_query = "What was the highest closing price in the dataset?"
    print(f"\n  Query: '{test_query}'")
    
    result = await rag_service.query_with_rag(
        dataset_id=dataset_id,
        query=test_query,
        top_k=3
    )
    
    print(f"\n  Response:")
    print(f"  {result['response']}")
    
    if result.get('metadata'):
        print(f"\n  Metadata:")
        print(f"    Retrieved chunks: {result['metadata']['retrieved_chunks']}")
        print(f"    Relevance scores: {[f'{s:.3f}' for s in result['metadata']['relevance_scores']]}")
    
    # 6. Test streaming (if not in mock mode)
    if not llm_service.mock_mode:
        print("\n[Bonus] Testing streaming RAG...")
        stream_query = "Analyze the price trends in this dataset"
        print(f"  Query: '{stream_query}'")
        print(f"  Streaming response: ", end="", flush=True)
        
        async for chunk in rag_service.stream_query_with_rag(
            dataset_id=dataset_id,
            query=stream_query,
            top_k=3
        ):
            print(chunk, end="", flush=True)
        print()
    
    print("\n" + "=" * 60)
    print("✓ RAG System Test Complete!")
    print("=" * 60)
    
    # Cleanup
    print("\nCleaning up test collection...")
    vector_store.delete_collection(dataset_id)
    print("✓ Test collection deleted")


if __name__ == "__main__":
    asyncio.run(test_rag_system())
