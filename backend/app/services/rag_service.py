"""
RAG Service for CSV Data Analysis
Combines vector retrieval with LLM generation for enhanced responses
"""

from typing import Optional, Dict, Any
from app.services.vector_store import CSVVectorStore
from app.services.llm_service import LLMService


class RAGService:
    """
    Retrieval-Augmented Generation service for CSV data.
    Enhances LLM responses with relevant context from vector store.
    """
    
    def __init__(self, llm_service: LLMService, vector_store: CSVVectorStore):
        """
        Initialize RAG service.
        
        Args:
            llm_service: LLM service instance
            vector_store: Vector store instance
        """
        self.llm_service = llm_service
        self.vector_store = vector_store
    
    async def query_with_rag(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        model: str = "openai/gpt-4o-mini",
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Answer a query using RAG.
        
        Process:
        1. Retrieve relevant chunks from vector store
        2. Augment prompt with retrieved context
        3. Generate response with LLM
        
        Args:
            dataset_id: Dataset identifier
            query: User's natural language query
            top_k: Number of relevant chunks to retrieve
            model: LLM model to use
            include_metadata: Include retrieval metadata in response
        
        Returns:
            Dict with 'response', 'context', and optional 'metadata'
        """
        # Check if vector collection exists
        if not self.vector_store.collection_exists(dataset_id):
            return {
                "response": "Error: Dataset not indexed for RAG. Please upload the dataset first.",
                "context": [],
                "metadata": None
            }
        
        # Retrieve relevant context
        retrieval_results = self.vector_store.query(
            dataset_id=dataset_id,
            query_text=query,
            top_k=top_k
        )
        
        if not retrieval_results["documents"]:
            return {
                "response": "No relevant information found in the dataset.",
                "context": [],
                "metadata": None
            }
        
        # Build context from retrieved documents
        context_parts = []
        for i, (doc, metadata) in enumerate(zip(
            retrieval_results["documents"],
            retrieval_results["metadatas"]
        )):
            context_parts.append(f"[Context {i+1}] {doc}")
        
        context = "\n\n".join(context_parts)
        
        # Create augmented prompt
        augmented_prompt = f"""You are a data analysis assistant. Use the following retrieved information from the dataset to answer the user's question accurately.

RETRIEVED CONTEXT FROM DATASET:
{context}

USER QUESTION: {query}

Instructions:
- Provide a clear, accurate answer based on the retrieved context
- If the context contains specific data values, include them in your answer
- If the context doesn't contain enough information to fully answer the question, acknowledge this
- Be concise but informative
- If you see patterns or insights in the data, mention them

Answer:"""
        
        # Generate response
        response = await self.llm_service.generate_response(
            prompt=augmented_prompt,
            system_prompt="You are a helpful data analysis assistant that answers questions based on provided context from datasets.",
            model=model,
            max_tokens=1024
        )
        
        result = {
            "response": response,
            "context": retrieval_results["documents"]
        }
        
        if include_metadata:
            result["metadata"] = {
                "retrieved_chunks": len(retrieval_results["documents"]),
                "chunk_metadatas": retrieval_results["metadatas"],
                "relevance_scores": [
                    1.0 - dist for dist in retrieval_results["distances"]
                ]  # Convert distance to similarity score
            }
        
        return result
    
    async def stream_query_with_rag(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 5,
        model: str = "openai/gpt-4o-mini"
    ):
        """
        Stream a RAG response (for real-time chat).
        
        Args:
            dataset_id: Dataset identifier
            query: User's query
            top_k: Number of chunks to retrieve
            model: LLM model to use
        
        Yields:
            Response chunks as they're generated
        """
        # Retrieve context
        retrieval_results = self.vector_store.query(
            dataset_id=dataset_id,
            query_text=query,
            top_k=top_k
        )
        
        if not retrieval_results["documents"]:
            yield "No relevant information found in the dataset."
            return
        
        # Build context
        context = "\n\n".join([
            f"[Context {i+1}] {doc}"
            for i, doc in enumerate(retrieval_results["documents"])
        ])
        
        # Create augmented prompt
        augmented_prompt = f"""You are a data analysis assistant. Use the following retrieved information from the dataset to answer the user's question accurately.

RETRIEVED CONTEXT FROM DATASET:
{context}

USER QUESTION: {query}

Provide a clear, accurate answer based on the retrieved context.

Answer:"""
        
        # Stream response
        async for chunk in self.llm_service.stream_response(
            prompt=augmented_prompt,
            system_prompt="You are a helpful data analysis assistant.",
            model=model,
            max_tokens=1024
        ):
            yield chunk
    
    async def hybrid_analyze(
        self,
        dataset_id: str,
        query: str,
        top_k: int = 3,
        model: str = "openai/gpt-4o-mini"
    ) -> Dict[str, Any]:
        """
        Hybrid RAG + Tool Calling approach.
        
        Uses RAG for initial context, then allows tool calling for computations.
        
        Args:
            dataset_id: Dataset identifier
            query: User's query
            top_k: Number of chunks to retrieve
            model: LLM model to use
        
        Returns:
            Analysis result with RAG context and tool outputs
        """
        # Retrieve relevant context via RAG
        retrieval_results = self.vector_store.query(
            dataset_id=dataset_id,
            query_text=query,
            top_k=top_k
        )
        
        context = "\n".join([
            f"[Context {i+1}] {doc}"
            for i, doc in enumerate(retrieval_results["documents"])
        ])
        
        # Use the existing analyze_dataset method with RAG context
        # This would require modifying the LLMService.analyze_dataset method
        # to accept optional context parameter
        
        # For now, return RAG results
        response = await self.query_with_rag(
            dataset_id=dataset_id,
            query=query,
            top_k=top_k,
            model=model
        )
        
        return response
    
    def get_dataset_info(self, dataset_id: str) -> Optional[Dict]:
        """
        Get information about a dataset's vector index.
        
        Args:
            dataset_id: Dataset identifier
        
        Returns:
            Collection info or None if not found
        """
        return self.vector_store.get_collection_info(dataset_id)
