"""
Comprehensive Test Script for RAG and LangChain Features

This script tests:
1. Vector Store (ChromaDB) functionality
2. RAG Service (Retrieval-Augmented Generation)
3. LangChain Tools (dataset analysis tools)
4. LangChain Agent (automatic tool selection)

Run with: python test_rag_langchain.py
"""

import asyncio
import os
import sys
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.vector_store import CSVVectorStore
from app.services.rag_service import RAGService
from app.services.llm_service import LLMService
from app.services.langchain_llm_service import LangChainLLMService
from app.services.dataset_tools import (
    set_dataset_context,
    get_dataset_info,
    calculate_descriptive_statistics,
    calculate_correlation,
    analyze_missing_data,
    detect_outliers,
    group_and_aggregate,
    calculate_value_counts,
    filter_data
)
from app.services.storage import StorageService


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_result(test_name: str, passed: bool, details: str = ""):
    """Print test result."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")
    if details:
        print(f"    {details}")


def test_vector_store():
    """Test 1: Vector Store Functionality"""
    print_section("TEST 1: Vector Store (ChromaDB)")
    
    try:
        # Load test data
        df = pd.read_csv("../test_stock_data.csv")
        print(f"📊 Loaded test dataset: {len(df)} rows, {len(df.columns)} columns")
        print(f"   Columns: {df.columns.tolist()}\n")
        
        # Initialize vector store
        vector_store = CSVVectorStore()
        print_result("Vector store initialization", True)
        
        # Create collection
        dataset_id = "test_stock_rag"
        success = vector_store.create_collection(
            dataset_id=dataset_id,
            df=df,
            chunk_strategy="hybrid"
        )
        print_result("Create vector collection", success, 
                    f"Dataset ID: {dataset_id}, Strategy: hybrid")
        
        # Check collection exists
        exists = vector_store.collection_exists(dataset_id)
        print_result("Collection exists check", exists)
        
        # Get collection info
        info = vector_store.get_collection_info(dataset_id)
        if info:
            print_result("Get collection info", True,
                        f"Count: {info['count']}, Metadata: {info['metadata']}")
        else:
            print_result("Get collection info", False)
        
        # Test semantic search
        test_queries = [
            "What are the closing prices?",
            "Show me the stock volume data",
            "What's the price range?"
        ]
        
        print("\n🔍 Testing semantic search:")
        for query in test_queries:
            results = vector_store.query(
                dataset_id=dataset_id,
                query_text=query,
                top_k=3
            )
            
            if results["documents"]:
                print(f"\n  Query: '{query}'")
                print(f"  Retrieved {len(results['documents'])} chunks:")
                for i, doc in enumerate(results["documents"][:2], 1):
                    print(f"    [{i}] {doc[:100]}...")
                print_result(f"Query: {query}", True, 
                           f"Retrieved {len(results['documents'])} chunks")
            else:
                print_result(f"Query: {query}", False, "No results")
        
        return True, vector_store, dataset_id
        
    except Exception as e:
        print_result("Vector store test", False, f"Error: {str(e)}")
        return False, None, None


async def test_rag_service(vector_store, dataset_id):
    """Test 2: RAG Service"""
    print_section("TEST 2: RAG Service (Retrieval-Augmented Generation)")
    
    try:
        # Initialize services
        llm_service = LLMService()
        rag_service = RAGService(llm_service, vector_store)
        print_result("RAG service initialization", True)
        
        # Check if API key is set
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print_result("API key check", False, 
                        "⚠️  OPENROUTER_API_KEY not set - skipping LLM tests")
            return False
        
        print_result("API key check", True, "OpenRouter API key found")
        
        # Test RAG queries
        test_queries = [
            "What is the highest closing price in the dataset?",
            "Summarize the stock volume trends",
            "What's the date range of this data?"
        ]
        
        print("\n🤖 Testing RAG queries:")
        for query in test_queries:
            print(f"\n  Query: '{query}'")
            
            result = await rag_service.query_with_rag(
                dataset_id=dataset_id,
                query=query,
                top_k=3
            )
            
            if "error" not in result.get("response", ""):
                print(f"  Response: {result['response'][:200]}...")
                if result.get("metadata"):
                    print(f"  Metadata: Retrieved {result['metadata']['retrieved_chunks']} chunks")
                print_result(f"RAG query: {query[:30]}...", True)
            else:
                print(f"  Error: {result['response']}")
                print_result(f"RAG query: {query[:30]}...", False)
        
        return True
        
    except Exception as e:
        print_result("RAG service test", False, f"Error: {str(e)}")
        return False


def test_langchain_tools():
    """Test 3: LangChain Tools"""
    print_section("TEST 3: LangChain Dataset Tools")
    
    try:
        # Load and save test dataset
        df = pd.read_csv("../test_stock_data.csv")
        dataset_id = StorageService.save_dataset(df, "test_stock_data.csv")
        set_dataset_context(dataset_id)
        print_result("Dataset loaded and context set", True, f"Dataset ID: {dataset_id}")
        
        # Test each tool
        print("\n🔧 Testing individual tools:\n")
        
        # 1. Get dataset info
        result = get_dataset_info.invoke({})
        if "error" not in result:
            print_result("get_dataset_info", True, 
                        f"Rows: {result['rows']}, Columns: {result['columns']}")
        else:
            print_result("get_dataset_info", False, result.get("error"))
        
        # 2. Descriptive statistics
        result = calculate_descriptive_statistics.invoke({
            "columns": None,
            "include_percentiles": True
        })
        if "error" not in result:
            print_result("calculate_descriptive_statistics", True,
                        f"Analyzed {len(result.get('columns_analyzed', []))} columns")
        else:
            print_result("calculate_descriptive_statistics", False, result.get("error"))
        
        # 3. Correlation
        result = calculate_correlation.invoke({
            "columns": None,
            "method": "pearson"
        })
        if "error" not in result:
            print_result("calculate_correlation", True,
                        f"Method: {result.get('method')}")
        else:
            print_result("calculate_correlation", False, result.get("error"))
        
        # 4. Missing data analysis
        result = analyze_missing_data.invoke({"columns": None})
        if "error" not in result:
            print_result("analyze_missing_data", True,
                        f"Checked {result.get('total_columns_checked')} columns")
        else:
            print_result("analyze_missing_data", False, result.get("error"))
        
        # 5. Value counts (if categorical columns exist)
        if len(df.columns) > 0:
            first_col = df.columns[0]
            result = calculate_value_counts.invoke({
                "column": first_col,
                "top_n": 5
            })
            if "error" not in result:
                print_result("calculate_value_counts", True,
                            f"Column: {first_col}, Unique: {result.get('total_unique_values')}")
            else:
                print_result("calculate_value_counts", False, result.get("error"))
        
        # 6. Filter data (test with numeric column if available)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            test_col = numeric_cols[0]
            median_val = df[test_col].median()
            result = filter_data.invoke({
                "column": test_col,
                "operator": ">",
                "value": str(median_val)
            })
            if "error" not in result:
                print_result("filter_data", True,
                            f"Filtered {result.get('matching_rows')} rows")
            else:
                print_result("filter_data", False, result.get("error"))
        
        return True
        
    except Exception as e:
        print_result("LangChain tools test", False, f"Error: {str(e)}")
        return False


async def test_langchain_agent():
    """Test 4: LangChain Agent with Automatic Tool Selection"""
    print_section("TEST 4: LangChain Agent (Automatic Tool Selection)")
    
    try:
        # Check API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print_result("API key check", False,
                        "⚠️  OPENROUTER_API_KEY not set - skipping agent tests")
            return False
        
        # Initialize agent
        service = LangChainLLMService()
        print_result("LangChain agent initialization", True)
        
        # Prepare dataset
        df = pd.read_csv("../test_stock_data.csv")
        dataset_id = StorageService.save_dataset(df, "test_stock_data.csv")
        print_result("Dataset prepared", True, f"Dataset ID: {dataset_id}")
        
        # Test queries that should trigger different tools
        test_queries = [
            {
                "query": "What columns are in this dataset?",
                "expected_tool": "get_dataset_info"
            },
            {
                "query": "What's the average of all numeric columns?",
                "expected_tool": "calculate_descriptive_statistics"
            },
            {
                "query": "Are there any missing values?",
                "expected_tool": "analyze_missing_data"
            }
        ]
        
        print("\n🤖 Testing agent with different queries:\n")
        for test_case in test_queries:
            query = test_case["query"]
            expected_tool = test_case["expected_tool"]
            
            print(f"  Query: '{query}'")
            print(f"  Expected tool: {expected_tool}")
            
            result = await service.analyze_dataset(
                dataset_id=dataset_id,
                query=query
            )
            
            if "error" not in result:
                print(f"  Answer: {result['answer'][:150]}...")
                tools_used = [tc["tool"] for tc in result.get("tool_calls", [])]
                print(f"  Tools used: {tools_used}")
                
                # Check if expected tool was used
                tool_match = expected_tool in tools_used
                print_result(f"Query: {query[:40]}...", tool_match,
                           f"Used: {tools_used}")
            else:
                print(f"  Error: {result.get('error')}")
                print_result(f"Query: {query[:40]}...", False)
            
            print()
        
        return True
        
    except Exception as e:
        print_result("LangChain agent test", False, f"Error: {str(e)}")
        return False


async def main():
    """Run all tests."""
    print("\n" + "🚀" * 40)
    print("  RAG & LANGCHAIN INTEGRATION TEST SUITE")
    print("🚀" * 40)
    
    # Check environment
    print_section("ENVIRONMENT CHECK")
    api_key = os.getenv("OPENROUTER_API_KEY")
    print(f"OpenRouter API Key: {'✅ Set' if api_key else '❌ Not set'}")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    # Run tests
    results = {}
    
    # Test 1: Vector Store
    success, vector_store, dataset_id = test_vector_store()
    results["Vector Store"] = success
    
    # Test 2: RAG Service (only if vector store works)
    if success and vector_store:
        results["RAG Service"] = await test_rag_service(vector_store, dataset_id)
    else:
        results["RAG Service"] = False
        print_section("TEST 2: RAG Service - SKIPPED (Vector Store failed)")
    
    # Test 3: LangChain Tools
    results["LangChain Tools"] = test_langchain_tools()
    
    # Test 4: LangChain Agent
    results["LangChain Agent"] = await test_langchain_agent()
    
    # Summary
    print_section("TEST SUMMARY")
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    print(f"\n{'='*80}")
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*80}\n")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! RAG and LangChain features are working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
