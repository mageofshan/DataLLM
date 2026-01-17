"""
Vector Store Service for CSV RAG
Handles embedding generation and vector storage using ChromaDB
"""

import os
import uuid
from typing import List, Dict, Optional
import pandas as pd
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class CSVVectorStore:
    """
    Vector store for CSV data using ChromaDB and sentence-transformers.
    Provides semantic search capabilities over CSV datasets.
    """
    
    _shared_model = None
    _shared_client = None
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        """
        Initialize the vector store.
        Uses shared static instances for client and model to avoid excessive overhead.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        os.makedirs(persist_directory, exist_ok=True)
        
        if CSVVectorStore._shared_client is None:
            CSVVectorStore._shared_client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        self.client = CSVVectorStore._shared_client
        
        # Lazy load model only when needed
        self.embedding_model = None

    def _get_model(self):
        """Lazy load the embedding model."""
        if CSVVectorStore._shared_model is None:
            print("Loading embedding model...")
            CSVVectorStore._shared_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding model loaded.")
        return CSVVectorStore._shared_model
        
    def _csv_to_text_chunks(
        self, 
        df: pd.DataFrame, 
        chunk_strategy: str = "hybrid"
    ) -> List[Dict]:
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
                        "type": "row_data",
                        "row_index": int(idx),
                        **{str(k): str(v) for k, v in row.to_dict().items()}
                    }
                })
        
        elif chunk_strategy == "column":
            # Each column becomes a document with statistics
            for col in df.columns:
                text = f"Column: {col}\nData Type: {df[col].dtype}\n"
                
                if df[col].dtype in ['int64', 'float64']:
                    stats = df[col].describe()
                    text += f"Statistics: min={stats['min']:.2f}, max={stats['max']:.2f}, "
                    text += f"mean={stats['mean']:.2f}, std={stats['std']:.2f}\n"
                
                text += f"Sample Values: {df[col].head(5).tolist()}"
                
                chunks.append({
                    "text": text,
                    "metadata": {
                        "type": "column_summary",
                        "column_name": col,
                        "dtype": str(df[col].dtype)
                    }
                })
        
        elif chunk_strategy == "hybrid":
            # Add column summaries first
            for col in df.columns:
                text = f"Column '{col}' contains {df[col].dtype} data. "
                
                if df[col].dtype in ['int64', 'float64']:
                    text += f"Range: {df[col].min()} to {df[col].max()}. "
                    text += f"Mean: {df[col].mean():.2f}. "
                    text += f"Std Dev: {df[col].std():.2f}."
                else:
                    unique_count = df[col].nunique()
                    text += f"Contains {unique_count} unique values."
                
                chunks.append({
                    "text": text,
                    "metadata": {
                        "type": "column_summary",
                        "column_name": col,
                        "dtype": str(df[col].dtype)
                    }
                })
            
            # Add row data
            for idx, row in df.iterrows():
                text = ", ".join([f"{col}: {val}" for col, val in row.items()])
                chunks.append({
                    "text": text,
                    "metadata": {
                        "type": "row_data",
                        "row_index": int(idx)
                    }
                })
        
        return chunks
    
    def create_collection(
        self, 
        dataset_id: str, 
        df: pd.DataFrame,
        chunk_strategy: str = "hybrid"
    ) -> bool:
        """
        Create a new vector collection for a dataset.
        
        Args:
            dataset_id: Unique identifier for the dataset
            df: Pandas DataFrame to index
            chunk_strategy: How to chunk the CSV data
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete existing collection if it exists
            try:
                self.client.delete_collection(name=dataset_id)
            except:
                pass
            
            # Create new collection
            collection = self.client.create_collection(
                name=dataset_id,
                metadata={
                    "description": f"Vector store for dataset {dataset_id}",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "chunk_strategy": chunk_strategy
                }
            )
            
            # Convert CSV to text chunks
            chunks = self._csv_to_text_chunks(df, chunk_strategy)
            
            if not chunks:
                return False
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in chunks]
            embeddings = self._get_model().encode(
                texts, 
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Add to ChromaDB in batches
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_embeddings = embeddings[i:i + batch_size]
                batch_texts = texts[i:i + batch_size]
                
                collection.add(
                    ids=[str(uuid.uuid4()) for _ in batch_chunks],
                    embeddings=batch_embeddings,
                    documents=batch_texts,
                    metadatas=[chunk["metadata"] for chunk in batch_chunks]
                )
            
            return True
            
        except Exception as e:
            print(f"Error creating vector collection: {e}")
            return False
    
    def query(
        self, 
        dataset_id: str, 
        query_text: str, 
        top_k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            dataset_id: Dataset identifier
            query_text: Natural language query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            Dict with 'documents', 'metadatas', and 'distances'
        """
        try:
            collection = self.client.get_collection(name=dataset_id)
            
            # Embed the query
            query_embedding = self._get_model().encode(
                [query_text],
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            
            # Search with optional filtering
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                where=filter_metadata
            )
            
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else []
            }
            
        except Exception as e:
            print(f"Error querying vector store: {e}")
            return {"documents": [], "metadatas": [], "distances": []}
    
    def delete_collection(self, dataset_id: str) -> bool:
        """Delete a vector collection."""
        try:
            self.client.delete_collection(name=dataset_id)
            return True
        except:
            return False
    
    def collection_exists(self, dataset_id: str) -> bool:
        """Check if a collection exists."""
        try:
            self.client.get_collection(name=dataset_id)
            return True
        except:
            return False
    
    def get_collection_info(self, dataset_id: str) -> Optional[Dict]:
        """Get information about a collection."""
        try:
            collection = self.client.get_collection(name=dataset_id)
            return {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
        except:
            return None
