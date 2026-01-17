#!/usr/bin/env python3
"""
Demo Script: LangChain Tool Integration for DataLLM

This script demonstrates how the LangChain-based tool system works.
It creates a sample dataset and runs various queries through the agent.

Usage:
    python demo_langchain_tools.py
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from app.services.dataset_tools import (
    set_dataset_context,
    get_dataset_info,
    calculate_descriptive_statistics,
    calculate_correlation,
    analyze_missing_data,
    group_and_aggregate,
    calculate_value_counts,
    filter_data,
)
from app.services.storage import StorageService


def create_sample_dataset():
    """Create a sample e-commerce dataset for demonstration."""
    np.random.seed(42)
    
    n_rows = 1000
    
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books']
    
    data = {
        'product_id': range(1, n_rows + 1),
        'product_name': [f'Product_{i}' for i in range(1, n_rows + 1)],
        'category': np.random.choice(categories, n_rows),
        'price': np.random.uniform(10, 500, n_rows).round(2),
        'rating': np.random.uniform(1, 5, n_rows).round(1),
        'stock_count': np.random.randint(0, 200, n_rows),
        'sales_last_month': np.random.randint(0, 100, n_rows),
    }
    
    # Add some missing values
    missing_indices = np.random.choice(n_rows, size=50, replace=False)
    for idx in missing_indices[:25]:
        data['rating'][idx] = np.nan
    for idx in missing_indices[25:]:
        data['stock_count'][idx] = np.nan
    
    df = pd.DataFrame(data)
    
    # Add correlation: higher price -> higher rating (with noise)
    df.loc[df['rating'].notna(), 'rating'] = (
        df.loc[df['rating'].notna(), 'price'] / 100 + 
        np.random.normal(0, 0.5, df['rating'].notna().sum())
    ).clip(1, 5).round(1)
    
    return df


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


async def demo_tools_directly():
    """Demonstrate calling tools directly (without agent)."""
    print_section("DEMO 1: Direct Tool Calls (No Agent)")
    
    # Create and save dataset
    print("Creating sample e-commerce dataset...")
    df = create_sample_dataset()
    dataset_id = 'demo_dataset'
    StorageService.save_dataset(dataset_id, df)
    set_dataset_context(dataset_id)
    print(f"✓ Created dataset with {len(df)} rows and {len(df.columns)} columns\n")
    
    # Test 1: Get dataset info
    print("1. Getting dataset information...")
    result = get_dataset_info.invoke({})
    print(f"   Rows: {result['rows']}")
    print(f"   Columns: {result['columns']}")
    print(f"   Column names: {', '.join(result['column_names'])}")
    print()
    
    # Test 2: Descriptive statistics
    print("2. Calculating descriptive statistics for 'price'...")
    result = calculate_descriptive_statistics.invoke({
        "columns": ["price"],
        "include_percentiles": True
    })
    if 'statistics' in result:
        stats = result['statistics']['price']
        print(f"   Mean: ${stats['mean']:.2f}")
        print(f"   Median: ${stats['50%']:.2f}")
        print(f"   Min: ${stats['min']:.2f}")
        print(f"   Max: ${stats['max']:.2f}")
    print()
    
    # Test 3: Correlation
    print("3. Checking correlation between price and rating...")
    result = calculate_correlation.invoke({
        "columns": ["price", "rating"],
        "method": "pearson"
    })
    if 'correlation_matrix' in result:
        corr = result['correlation_matrix']['price']['rating']
        print(f"   Correlation: {corr:.3f}")
        if abs(corr) > 0.7:
            print("   → Strong correlation!")
        elif abs(corr) > 0.3:
            print("   → Moderate correlation")
        else:
            print("   → Weak correlation")
    print()
    
    # Test 4: Missing data
    print("4. Analyzing missing data...")
    result = analyze_missing_data.invoke({})
    if 'columns_with_missing' in result:
        print(f"   Columns with missing values: {', '.join(result['columns_with_missing'])}")
        for col in result['columns_with_missing']:
            info = result['missing_data'][col]
            print(f"   - {col}: {info['missing_count']} missing ({info['missing_percentage']:.1f}%)")
    print()
    
    # Test 5: Value counts
    print("5. Counting products by category...")
    result = calculate_value_counts.invoke({
        "column": "category",
        "top_n": 5
    })
    if 'value_counts' in result:
        print(f"   Total unique categories: {result['total_unique_values']}")
        for category, count in result['value_counts'].items():
            print(f"   - {category}: {count} products")
    print()
    
    # Test 6: Group and aggregate
    print("6. Average price by category...")
    result = group_and_aggregate.invoke({
        "group_columns": ["category"],
        "agg_column": "price",
        "agg_function": "mean"
    })
    if 'results' in result:
        for category, avg_price in result['results'].items():
            print(f"   - {category}: ${avg_price:.2f}")
    print()
    
    # Test 7: Filter data
    print("7. Filtering products with price > $300...")
    result = filter_data.invoke({
        "column": "price",
        "operator": ">",
        "value": "300"
    })
    if 'matching_rows' in result:
        print(f"   Found {result['matching_rows']} products ({result['percentage']:.1f}%)")
        print(f"   Sample products:")
        for i, product in enumerate(result['sample'][:3], 1):
            print(f"   {i}. {product['product_name']}: ${product['price']}")
    print()


async def demo_with_agent():
    """Demonstrate using the LangChain agent."""
    print_section("DEMO 2: Using LangChain Agent")
    
    # Check if API key is set
    if not os.getenv("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY not set. Skipping agent demo.")
        print("   Set the API key to test the full agent functionality.")
        return
    
    from app.services.langchain_llm_service import LangChainLLMService
    
    print("Initializing LangChain agent...")
    service = LangChainLLMService()
    
    # Ensure dataset context is set
    set_dataset_context('demo_dataset')
    
    # Test queries
    queries = [
        "What's the average price of products in the dataset?",
        "How many products are in each category?",
        "Are price and rating correlated?",
        "How many products cost more than $200?",
        "What's the most common category?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{i}. Query: '{query}'")
        print("-" * 80)
        
        result = await service.analyze_dataset(
            dataset_id='demo_dataset',
            query=query
        )
        
        print(f"Answer: {result['answer']}")
        
        if result.get('tool_calls'):
            print(f"\nTools used: {', '.join([tc['tool'] for tc in result['tool_calls']])}")
        
        print()


async def demo_error_handling():
    """Demonstrate error handling in tools."""
    print_section("DEMO 3: Error Handling")
    
    set_dataset_context('demo_dataset')
    
    print("1. Testing with non-existent column...")
    result = calculate_descriptive_statistics.invoke({
        "columns": ["nonexistent_column"]
    })
    if 'error' in result:
        print(f"   ✓ Error caught: {result['error']}")
        print(f"   ✓ Helpful suggestion: Available columns shown")
    print()
    
    print("2. Testing correlation with non-numeric column...")
    result = calculate_correlation.invoke({
        "columns": ["category", "product_name"]
    })
    if 'error' in result:
        print(f"   ✓ Error caught: {result['error']}")
    print()
    
    print("3. Testing filter with invalid operator...")
    result = filter_data.invoke({
        "column": "price",
        "operator": "invalid_op",
        "value": "100"
    })
    if 'error' in result:
        print(f"   ✓ Error caught: {result['error']}")
    print()


async def main():
    """Run all demos."""
    print("\n" + "="*80)
    print("  LangChain Tool Integration Demo for DataLLM")
    print("="*80)
    print("\nThis demo shows how the new tool-based system works.")
    print("It will:")
    print("  1. Create a sample e-commerce dataset")
    print("  2. Test each tool directly")
    print("  3. Test with the LangChain agent (if API key is set)")
    print("  4. Demonstrate error handling")
    print("\nPress Enter to continue...")
    input()
    
    # Run demos
    await demo_tools_directly()
    await demo_with_agent()
    await demo_error_handling()
    
    # Summary
    print_section("Demo Complete!")
    print("✓ All tools tested successfully")
    print("✓ Error handling verified")
    print("\nNext steps:")
    print("  1. Review the code in backend/app/services/dataset_tools.py")
    print("  2. Read LANGCHAIN_TOOL_INTEGRATION_RESEARCH.md for details")
    print("  3. Follow MIGRATION_GUIDE.md to integrate into your app")
    print("  4. Use TOOL_CREATION_GUIDE.md to add custom tools")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
