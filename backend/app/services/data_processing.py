import pandas as pd
import numpy as np
from fastapi import UploadFile, HTTPException
import io
from typing import Dict, Any, Union

class DataProcessingService:
    @staticmethod
    async def read_file(file: UploadFile) -> pd.DataFrame:
        """
        Reads an uploaded file into a Pandas DataFrame.
        Supports CSV, Excel, and JSON.
        """
        filename = file.filename.lower()
        content = await file.read()
        
        try:
            if filename.endswith('.csv'):
                # Try reading with default settings, then fallback to different encodings if needed
                try:
                    df = pd.read_csv(io.BytesIO(content))
                except UnicodeDecodeError:
                    df = pd.read_csv(io.BytesIO(content), encoding='latin1')
            elif filename.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(io.BytesIO(content))
            elif filename.endswith('.json'):
                df = pd.read_json(io.BytesIO(content))
            elif filename.endswith('.tsv'):
                df = pd.read_csv(io.BytesIO(content), sep='\t')
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV, Excel, JSON, or TSV.")
            
            return df
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs basic data cleaning:
        - Infers better data types
        - Handles missing values (simple strategy for now: fill numeric with mean, object with mode)
        """
        # 1. Drop entirely empty rows/cols
        df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
        
        # 2. Convert object columns to numeric if possible (e.g. "1,000" -> 1000)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                # Try to convert to numeric, coercing errors to NaN
                # This helps with columns that are mostly numbers but have some garbage
                numeric_col = pd.to_numeric(df[col], errors='coerce')
                
                # If we successfully converted a significant portion, assume it's numeric
                # (Arbitrary threshold: if > 70% are valid numbers)
                if numeric_col.notna().sum() > 0.7 * len(df):
                    df[col] = numeric_col
            except Exception:
                pass

        # 3. Handle missing values
        # For numeric columns, fill with mean
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
             df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        # For categorical/object columns, fill with mode or "Unknown"
        object_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in object_cols:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna("Unknown")

        return df

    @staticmethod
    def get_dataset_metadata(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Returns metadata about the dataset: shape, columns, datatypes, missing value counts.
        """
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": df.columns.tolist(),
            "dtypes": {k: str(v) for k, v in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "preview": df.head(5).to_dict(orient='records') # Send a small preview
        }

    @staticmethod
    def get_basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes descriptive statistics, correlation, and distribution analysis.
        """
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {
                "descriptive_statistics": {},
                "correlation_matrix": {},
                "distribution_analysis": {}
            }

        # 1. Basic Descriptive Stats
        desc = numeric_df.describe().to_dict()
        
        # 2. Correlation Matrix
        if numeric_df.shape[1] > 1:
            correlation = numeric_df.corr().to_dict()
        else:
            correlation = {}

        # 3. Extended Distribution Analysis (Skew, Kurtosis, Outliers)
        distribution = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            
            # Skewness & Kurtosis
            skew = series.skew()
            kurt = series.kurtosis()
            
            # Outlier Detection (IQR Method)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            # Histogram Data
            counts, bin_edges = np.histogram(series, bins=10)
            histogram_data = []
            for i in range(len(counts)):
                histogram_data.append({
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i+1]),
                    "count": int(counts[i]),
                    "label": f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}"
                })

            distribution[col] = {
                "skewness": float(skew) if not pd.isna(skew) else None,
                "kurtosis": float(kurt) if not pd.isna(kurt) else None,
                "outlier_count": int(len(outliers)),
                "outlier_percentage": float(len(outliers) / len(series) * 100) if len(series) > 0 else 0,
                "histogram": histogram_data
            }

        return {
            "descriptive_statistics": desc,
            "correlation_matrix": correlation,
            "distribution_analysis": distribution
        }
