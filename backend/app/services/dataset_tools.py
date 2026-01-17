"""
LangChain Tools for Dataset Analysis

This module provides a comprehensive suite of tools for analyzing datasets
using LangChain's tool framework. Each tool is designed to be general-purpose
and work with any pandas DataFrame.

Tools are automatically selected by the LLM based on user queries.
"""

from langchain.tools import tool
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np
from app.services.storage import StorageService

# Thread-local storage for dataset context (for production, use contextvars)
_current_dataset_id: Optional[str] = None


def set_dataset_context(dataset_id: str):
    """Set the current dataset context for tool execution."""
    global _current_dataset_id
    _current_dataset_id = dataset_id


def get_current_dataframe() -> pd.DataFrame:
    """Retrieve the current dataset as a DataFrame."""
    if not _current_dataset_id:
        raise ValueError("No dataset context set. Call set_dataset_context first.")
    df = StorageService.load_dataset(_current_dataset_id)
    if df is None:
        raise ValueError(f"Dataset {_current_dataset_id} not found in storage.")
    return df


# ============================================================================
# BASIC INFORMATION TOOLS
# ============================================================================

@tool
def get_dataset_info() -> Dict[str, Any]:
    """
    Get basic information about the dataset structure and size.
    
    Returns:
    - Number of rows and columns
    - Column names and their data types
    - Memory usage
    - Sample of first few rows
    
    Use this tool when users ask:
    - "What does this dataset look like?"
    - "How many rows/columns are there?"
    - "What columns are available?"
    - "Show me the data structure"
    """
    try:
        df = get_current_dataframe()
        
        # Get sample data
        sample = df.head(5).to_dict(orient='records')
        
        return {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            "sample_data": sample
        }
    except Exception as e:
        return {"error": f"Failed to get dataset info: {str(e)}"}


# ============================================================================
# STATISTICAL ANALYSIS TOOLS
# ============================================================================

class DescriptiveStatsInput(BaseModel):
    """Input schema for descriptive statistics calculation."""
    columns: Optional[List[str]] = Field(
        default=None,
        description="Specific columns to analyze. If not provided, analyzes all numeric columns."
    )
    include_percentiles: bool = Field(
        default=True,
        description="Whether to include 25th, 50th, and 75th percentiles."
    )


@tool(args_schema=DescriptiveStatsInput)
def calculate_descriptive_statistics(
    columns: Optional[List[str]] = None,
    include_percentiles: bool = True
) -> Dict[str, Any]:
    """
    Calculate comprehensive descriptive statistics for numeric columns.
    
    Computes: count, mean, std, min, max, and percentiles (25%, 50%, 75%).
    
    Use this when users ask about:
    - "What's the average/mean of X?"
    - "Show me statistics for column Y"
    - "What's the distribution of Z?"
    - "Summarize the numeric data"
    
    Returns statistics for each numeric column including central tendency and spread measures.
    """
    try:
        df = get_current_dataframe()
        
        # Select columns
        if columns:
            missing_cols = set(columns) - set(df.columns)
            if missing_cols:
                return {
                    "error": f"Columns not found: {list(missing_cols)}",
                    "available_columns": df.columns.tolist()
                }
            df_subset = df[columns]
        else:
            df_subset = df.select_dtypes(include=[np.number])
        
        if df_subset.empty:
            return {
                "error": "No numeric columns found",
                "available_columns": df.columns.tolist()
            }
        
        # Calculate statistics
        percentiles = [0.25, 0.5, 0.75] if include_percentiles else []
        stats = df_subset.describe(percentiles=percentiles).to_dict()
        
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
        return {"error": f"Failed to calculate statistics: {str(e)}"}


class CorrelationInput(BaseModel):
    """Input schema for correlation analysis."""
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to include. If None, use all numeric columns."
    )
    method: str = Field(
        default="pearson",
        description="Correlation method: 'pearson', 'spearman', or 'kendall'"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Only return correlations with absolute value above this threshold (0-1)"
    )


@tool(args_schema=CorrelationInput)
def calculate_correlation(
    columns: Optional[List[str]] = None,
    method: str = "pearson",
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Calculate correlation matrix between numeric columns.
    
    Identifies relationships between variables. High correlation (>0.7 or <-0.7) 
    indicates strong relationships.
    
    Use this when users ask:
    - "What's the correlation between X and Y?"
    - "Which variables are related?"
    - "Show me correlations"
    - "Are X and Y correlated?"
    
    Methods:
    - pearson: Linear correlation (default)
    - spearman: Rank-based correlation (for non-linear relationships)
    - kendall: Rank correlation (robust to outliers)
    """
    try:
        df = get_current_dataframe()
        
        # Select numeric columns
        if columns:
            df_subset = df[columns].select_dtypes(include=[np.number])
        else:
            df_subset = df.select_dtypes(include=[np.number])
        
        if df_subset.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for correlation"}
        
        # Calculate correlation
        corr_matrix = df_subset.corr(method=method)
        
        # Convert to dict
        corr_dict = corr_matrix.to_dict()
        
        # Apply threshold if specified
        if threshold is not None:
            filtered_pairs = []
            for col1 in corr_matrix.columns:
                for col2 in corr_matrix.columns:
                    if col1 < col2:  # Avoid duplicates
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) >= threshold:
                            filtered_pairs.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": float(corr_value)
                            })
            
            return {
                "correlation_matrix": corr_dict,
                "strong_correlations": filtered_pairs,
                "threshold": threshold,
                "method": method
            }
        
        return {
            "correlation_matrix": corr_dict,
            "method": method,
            "columns_analyzed": corr_matrix.columns.tolist()
        }
    except Exception as e:
        return {"error": f"Failed to calculate correlation: {str(e)}"}


# ============================================================================
# DATA QUALITY TOOLS
# ============================================================================

class MissingDataInput(BaseModel):
    """Input schema for missing data analysis."""
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns to check. If None, check all columns."
    )


@tool(args_schema=MissingDataInput)
def analyze_missing_data(columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Analyze missing values in the dataset.
    
    Returns count and percentage of missing values per column.
    
    Use this when users ask:
    - "Are there missing values?"
    - "How much data is missing?"
    - "Which columns have null values?"
    - "Data quality check"
    
    Helps identify data quality issues and decide on imputation strategies.
    """
    try:
        df = get_current_dataframe()
        
        if columns:
            df_subset = df[columns]
        else:
            df_subset = df
        
        missing_counts = df_subset.isnull().sum()
        total_rows = len(df_subset)
        
        missing_info = {}
        for col in df_subset.columns:
            count = int(missing_counts[col])
            percentage = float(count / total_rows * 100) if total_rows > 0 else 0
            missing_info[col] = {
                "missing_count": count,
                "missing_percentage": round(percentage, 2),
                "has_missing": count > 0
            }
        
        # Summary
        columns_with_missing = [col for col, info in missing_info.items() if info["has_missing"]]
        
        return {
            "missing_data": missing_info,
            "total_rows": total_rows,
            "columns_with_missing": columns_with_missing,
            "total_columns_checked": len(df_subset.columns)
        }
    except Exception as e:
        return {"error": f"Failed to analyze missing data: {str(e)}"}


class OutlierInput(BaseModel):
    """Input schema for outlier detection."""
    column: str = Field(description="Column name to analyze for outliers")
    method: str = Field(
        default="iqr",
        description="Detection method: 'iqr' (Interquartile Range) or 'zscore' (Z-score)"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Custom threshold. For IQR: multiplier (default 1.5). For zscore: number of std devs (default 3)"
    )


@tool(args_schema=OutlierInput)
def detect_outliers(column: str, method: str = "iqr", threshold: Optional[float] = None) -> Dict[str, Any]:
    """
    Detect outliers in a specific numeric column.
    
    Methods:
    - IQR (Interquartile Range): Values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
    - Z-score: Values more than 3 standard deviations from mean
    
    Use this when users ask:
    - "Are there outliers in X?"
    - "Find unusual values"
    - "Detect anomalies"
    
    Returns outlier indices, values, and statistics.
    """
    try:
        df = get_current_dataframe()
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        series = df[column].dropna()
        
        if not pd.api.types.is_numeric_dtype(series):
            return {"error": f"Column '{column}' is not numeric"}
        
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            multiplier = threshold if threshold else 1.5
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
        elif method == "zscore":
            mean = series.mean()
            std = series.std()
            z_threshold = threshold if threshold else 3
            z_scores = np.abs((series - mean) / std)
            outliers = series[z_scores > z_threshold]
            lower_bound = mean - z_threshold * std
            upper_bound = mean + z_threshold * std
        else:
            return {"error": f"Unknown method '{method}'. Use 'iqr' or 'zscore'"}
        
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(series) * 100) if len(series) > 0 else 0
        
        return {
            "column": column,
            "method": method,
            "outlier_count": outlier_count,
            "outlier_percentage": round(outlier_percentage, 2),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_values": outliers.tolist()[:20],  # Limit to first 20
            "total_values": len(series)
        }
    except Exception as e:
        return {"error": f"Failed to detect outliers: {str(e)}"}


# ============================================================================
# AGGREGATION & GROUPING TOOLS
# ============================================================================

class GroupByInput(BaseModel):
    """Input schema for group-by operations."""
    group_columns: List[str] = Field(description="Column(s) to group by")
    agg_column: str = Field(description="Column to aggregate")
    agg_function: str = Field(
        default="mean",
        description="Aggregation: 'mean', 'sum', 'count', 'min', 'max', 'median', 'std'"
    )


@tool(args_schema=GroupByInput)
def group_and_aggregate(
    group_columns: List[str],
    agg_column: str,
    agg_function: str = "mean"
) -> Dict[str, Any]:
    """
    Group data by one or more columns and apply aggregation function.
    
    Use this when users ask:
    - "What's the average X by Y?"
    - "Sum of sales per region"
    - "Count of items in each category"
    - "Group by X and calculate Y"
    
    Example: Group by 'category' and calculate average 'price'.
    Returns grouped results with aggregated values.
    """
    try:
        df = get_current_dataframe()
        
        # Validate columns
        all_cols = group_columns + [agg_column]
        missing = set(all_cols) - set(df.columns)
        if missing:
            return {"error": f"Columns not found: {list(missing)}"}
        
        # Perform groupby
        grouped = df.groupby(group_columns)[agg_column]
        
        # Apply aggregation
        agg_map = {
            "mean": grouped.mean,
            "sum": grouped.sum,
            "count": grouped.count,
            "min": grouped.min,
            "max": grouped.max,
            "median": grouped.median,
            "std": grouped.std
        }
        
        if agg_function not in agg_map:
            return {"error": f"Unknown aggregation '{agg_function}'"}
        
        result = agg_map[agg_function]()
        
        # Convert to dict
        if isinstance(result, pd.Series):
            result_dict = result.to_dict()
        else:
            result_dict = result.to_frame().to_dict()
        
        return {
            "grouped_by": group_columns,
            "aggregated_column": agg_column,
            "aggregation_function": agg_function,
            "results": result_dict,
            "num_groups": len(result)
        }
    except Exception as e:
        return {"error": f"Failed to group and aggregate: {str(e)}"}


class ValueCountsInput(BaseModel):
    """Input schema for value counts."""
    column: str = Field(description="Column name to count values")
    top_n: int = Field(default=10, description="Number of top values to return")
    normalize: bool = Field(
        default=False,
        description="If True, return percentages instead of counts"
    )


@tool(args_schema=ValueCountsInput)
def calculate_value_counts(
    column: str,
    top_n: int = 10,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Count unique values in a column and return top N most frequent.
    
    Use this when users ask:
    - "What are the most common values in X?"
    - "How many unique categories?"
    - "Distribution of Y"
    - "Top 10 products by count"
    
    Useful for categorical data analysis and frequency distributions.
    """
    try:
        df = get_current_dataframe()
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        value_counts = df[column].value_counts(normalize=normalize).head(top_n)
        
        counts_dict = value_counts.to_dict()
        
        return {
            "column": column,
            "value_counts": counts_dict,
            "total_unique_values": int(df[column].nunique()),
            "top_n": top_n,
            "normalized": normalize,
            "total_rows": len(df)
        }
    except Exception as e:
        return {"error": f"Failed to calculate value counts: {str(e)}"}


# ============================================================================
# FILTERING & QUERYING TOOLS
# ============================================================================

class FilterDataInput(BaseModel):
    """Input schema for data filtering."""
    column: str = Field(description="Column to filter on")
    operator: str = Field(
        description="Operator: '>', '<', '>=', '<=', '==', '!=', 'contains', 'in'"
    )
    value: str = Field(description="Value to compare (will be type-converted)")


@tool(args_schema=FilterDataInput)
def filter_data(column: str, operator: str, value: str) -> Dict[str, Any]:
    """
    Filter dataset rows based on a condition.
    
    Use this when users ask:
    - "How many rows have X > Y?"
    - "Filter data where Z equals W"
    - "Show me records with price above 100"
    - "Count items in category A"
    
    Operators:
    - Comparison: >, <, >=, <=, ==, !=
    - String: contains (substring match)
    - Membership: in (comma-separated values)
    
    Returns count of matching rows and sample of filtered data.
    """
    try:
        df = get_current_dataframe()
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        col_data = df[column]
        
        # Type conversion
        try:
            if pd.api.types.is_numeric_dtype(col_data):
                compare_value = float(value)
            else:
                compare_value = value
        except:
            compare_value = value
        
        # Apply filter
        if operator == '>':
            mask = col_data > compare_value
        elif operator == '<':
            mask = col_data < compare_value
        elif operator == '>=':
            mask = col_data >= compare_value
        elif operator == '<=':
            mask = col_data <= compare_value
        elif operator == '==':
            mask = col_data == compare_value
        elif operator == '!=':
            mask = col_data != compare_value
        elif operator == 'contains':
            mask = col_data.astype(str).str.contains(str(compare_value), case=False, na=False)
        elif operator == 'in':
            values_list = [v.strip() for v in value.split(',')]
            mask = col_data.isin(values_list)
        else:
            return {"error": f"Unknown operator '{operator}'"}
        
        filtered_df = df[mask]
        count = len(filtered_df)
        
        return {
            "column": column,
            "operator": operator,
            "value": value,
            "matching_rows": count,
            "total_rows": len(df),
            "percentage": round(count / len(df) * 100, 2) if len(df) > 0 else 0,
            "sample": filtered_df.head(5).to_dict(orient='records')
        }
    except Exception as e:
        return {"error": f"Failed to filter data: {str(e)}"}


# ============================================================================
# EXPORT ALL TOOLS
# ============================================================================

ALL_TOOLS = [
    get_dataset_info,
    calculate_descriptive_statistics,
    calculate_correlation,
    analyze_missing_data,
    detect_outliers,
    group_and_aggregate,
    calculate_value_counts,
    filter_data,
]
