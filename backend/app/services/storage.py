import pandas as pd
import uuid
import os
from typing import Optional

# Ensure storage directory exists
DATA_DIR = os.path.join(os.getcwd(), "storage", "datasets")
os.makedirs(DATA_DIR, exist_ok=True)

class StorageService:
    @staticmethod
    def save_dataset(df: pd.DataFrame, original_filename: str) -> str:
        """
        Saves a dataframe to Parquet format and returns a unique dataset ID.
        Also creates a vector index for RAG.
        """
        dataset_id = str(uuid.uuid4())
        # We use parquet for efficient storage and type preservation
        file_path = os.path.join(DATA_DIR, f"{dataset_id}.parquet")
        
        # Save metadata side-by-side if needed, but for now just the data
        df.to_parquet(file_path)
        
        return dataset_id

    @staticmethod
    def index_dataset(dataset_id: str, df: pd.DataFrame):
        """
        Creates a vector index for the dataset (CPU intensive).
        Should be run in background.
        """
        try:
            from app.services.vector_store import CSVVectorStore
            # Initialize locally to avoid import loops at module level if any
            vs = CSVVectorStore()
            print(f"Starting indexing for {dataset_id}...")
            vs.create_collection(dataset_id, df)
            print(f"Dataset {dataset_id} indexed successfully.")
        except Exception as e:
            print(f"Warning: Failed to index dataset {dataset_id}: {e}")

    @staticmethod
    def load_dataset(dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Loads a dataframe by ID.
        """
        if dataset_id == "mock-dataset-123":
            return StorageService._get_mock_dataframe()
        
        file_path = os.path.join(DATA_DIR, f"{dataset_id}.parquet")
        if os.path.exists(file_path):
            return pd.read_parquet(file_path)
        return None

    @staticmethod
    def _get_mock_dataframe() -> pd.DataFrame:
        """Creates a comprehensive mock dataframe matching the frontend mock data."""
        data = {
            "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"] * 20),
            "Product": ["Laptop", "Mouse", "Monitor", "Keyboard", "Laptop"] * 20,
            "Region": ["North", "South", "East", "West", "North"] * 20,
            "Sales": [1200.50, 25.00, 350.00, 80.00, 1150.00] * 20,
            "Profit": [300.20, 5.00, 80.50, 20.00, 280.00] * 20
        }
        return pd.DataFrame(data)

    @staticmethod
    def list_datasets():
        # Helper to debug
        return os.listdir(DATA_DIR)
