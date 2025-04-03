"""
Module for the RAG (Retrieval-Augmented Generation) engine
"""
import logging
from typing import List, Dict, Any

from utils.data_processor import preprocess_text, chunk_document
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from config import Config

logger = logging.getLogger(__name__)

class RAGEngine:
    """RAG (Retrieval-Augmented Generation) engine for question answering"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator, pinecone_manager: PineconeManager):
        """
        Initialize the RAG engine
        
        Args:
            embedding_generator: Instance of EmbeddingGenerator
            pinecone_manager: Instance of PineconeManager
        """
        self.embedding_generator = embedding_generator
        self.pinecone_manager = pinecone_manager
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Process and index documents
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Chunk documents for better retrieval
            chunked_docs = []
            for doc in documents:
                chunks = chunk_document(doc)
                chunked_docs.extend(chunks)
            
            logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            
            # Generate embeddings
            docs_with_embeddings = self.embedding_generator.embed_documents(chunked_docs)
            
            # Index in Pinecone
            return self.pinecone_manager.index_documents(docs_with_embeddings)
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def retrieve_relevant_context(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            top_k: Number of results to retrieve (defaults to Config.TOP_K_RESULTS)
            
        Returns:
            List of relevant document chunks
        """
        if top_k is None:
            top_k = Config.TOP_K_RESULTS
        
        # Preprocess query
        processed_query = preprocess_text(query)
        
        # Generate embedding for query
        query_embedding = self.embedding_generator.generate_embedding(processed_query)
        if query_embedding is None:
            logger.error("Failed to generate query embedding")
            return []
        
        # Search for relevant documents
        matches = self.pinecone_manager.search(query_embedding, top_k=top_k)
        
        return matches
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a user query using retrieved context
        
        Args:
            query: User query
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        # Retrieve relevant context
        relevant_docs = self.retrieve_relevant_context(query)
        
        if not relevant_docs:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try a different question.",
                "sources": []
            }
        
        # Extract content from retrieved documents
        context_texts = [doc["metadata"]["content"] for doc in relevant_docs if "content" in doc["metadata"]]
        
        # For this simple implementation, we'll just concatenate the retrieved information
        # In a more advanced implementation, you would use a language model to generate a coherent answer
        
        answer = self._generate_simple_answer(query, context_texts, relevant_docs)
        
        # Format sources
        sources = []
        for doc in relevant_docs:
            # Extract source information
            source = {
                "id": doc["id"],
                "score": doc["score"],
                "title": doc["metadata"].get("title", "Unknown"),
                "excerpt": doc["metadata"].get("content", "")[:200] + "..." if len(doc["metadata"].get("content", "")) > 200 else doc["metadata"].get("content", "")
            }
            sources.append(source)
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def _generate_simple_answer(self, query: str, context_texts: List[str], relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a simple answer based on retrieved context
        
        This is a basic implementation. In a real-world scenario, you would use a language model 
        like GPT to generate a coherent answer given the context.
        
        Args:
            query: User query
            context_texts: List of relevant text passages
            relevant_docs: Full document metadata
            
        Returns:
            Generated answer
        """
        if not context_texts:
            return "I don't have enough information to answer that question."
        
        # In this simple implementation, we'll use the most relevant document as the answer
        # Sorted by relevance score
        most_relevant = sorted(relevant_docs, key=lambda x: x.get("score", 0), reverse=True)
        
        if most_relevant:
            answer = most_relevant[0]["metadata"].get("content", "")
            
            # Add attribution
            source_title = most_relevant[0]["metadata"].get("title", "source")
            answer += f"\n\nThis information is from: {source_title}"
            
            return answer
        
        # Fallback to concatenated context if sorting fails
        return " ".join(context_texts[:2]) + "\n\n(Note: This is compiled from multiple sources)"
