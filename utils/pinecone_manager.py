"""
Module for managing vector database operations
This is a simplified in-memory implementation that mimics Pinecone's functionality
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import time

logger = logging.getLogger(__name__)

class InMemoryVectorStore:
    """An in-memory vector store to mimic Pinecone functionality"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = {}  # id -> [vector, metadata]
        self.vector_count = 0
    
    def upsert(self, vectors):
        """Insert or update vectors in the store"""
        for id, values, metadata in vectors:
            self.vectors[id] = [values, metadata]
        
        self.vector_count = len(self.vectors)
        return {'upserted_count': len(vectors)}
    
    def query(self, vector, top_k=5, include_metadata=True):
        """Query for similar vectors"""
        if not self.vectors:
            return {'matches': []}
        
        # Convert the query vector to numpy array
        query_vector = np.array(vector)
        
        # Calculate cosine similarity with all vectors
        similarities = []
        for id, (values, metadata) in self.vectors.items():
            # Convert to numpy array
            doc_vector = np.array(values)
            
            # Calculate cosine similarity
            dot_product = np.dot(query_vector, doc_vector)
            query_norm = np.linalg.norm(query_vector)
            doc_norm = np.linalg.norm(doc_vector)
            
            if query_norm > 0 and doc_norm > 0:
                similarity = dot_product / (query_norm * doc_norm)
            else:
                similarity = 0
            
            similarities.append((id, similarity, metadata if include_metadata else None))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        matches = []
        for id, score, metadata in similarities[:top_k]:
            match = {
                'id': id,
                'score': float(score)  # Convert from numpy.float to Python float
            }
            if include_metadata:
                match['metadata'] = metadata
            matches.append(match)
        
        return {'matches': matches}
    
    def describe_index_stats(self):
        """Return stats about the index"""
        return {
            'total_vector_count': self.vector_count,
            'dimensions': self.dimension
        }

class PineconeManager:
    """Class to handle interactions with a vector database (mimicking Pinecone interface)"""
    
    def __init__(self, api_key: str = None, environment: str = None, index_name: str = None, dimension: int = 384):
        """
        Initialize the vector store manager
        
        Args:
            api_key: Not used in this implementation, kept for interface compatibility
            environment: Not used in this implementation, kept for interface compatibility
            index_name: Not used in this implementation, kept for interface compatibility
            dimension: Dimension of the vectors to store
        """
        self.dimension = dimension
        self.index = None
        self._connect()
    
    def _connect(self):
        """Initialize the in-memory vector store"""
        try:
            logger.info("Initializing in-memory vector store")
            self.index = InMemoryVectorStore(dimension=self.dimension)
            logger.info("In-memory vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing in-memory vector store: {str(e)}")
            self.index = None
    
    def is_connected(self) -> bool:
        """Check if the vector store is initialized"""
        return self.index is not None
    
    def check_index_populated(self) -> bool:
        """Check if the index has data"""
        if not self.is_connected():
            return False
        
        try:
            stats = self.index.describe_index_stats()
            return stats.get('total_vector_count', 0) > 0
        except Exception as e:
            logger.error(f"Error checking index stats: {str(e)}")
            return False
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents with their embeddings in the vector store
        
        Args:
            documents: List of document dictionaries with 'id', 'embedding', and 'metadata'
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Vector store not initialized, can't index documents")
            return False
        
        if not documents:
            logger.warning("No documents to index")
            return False
        
        try:
            vectors = []
            
            for doc in documents:
                if "embedding" not in doc:
                    logger.warning(f"Document {doc.get('id', 'unknown')} has no embedding, skipping")
                    continue
                
                # Extract necessary fields
                vector_id = str(doc["id"])
                embedding = doc["embedding"]
                
                # Prepare metadata (exclude embedding)
                metadata = doc.get("metadata", {}).copy()
                metadata["content"] = doc["content"]  # Store text in metadata for retrieval
                
                vectors.append((vector_id, embedding, metadata))
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(batch)
                
            logger.info(f"Indexed {len(vectors)} documents in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the vector store
        
        Args:
            query_vector: The query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of match dictionaries with 'id', 'score', and 'metadata'
        """
        if not self.is_connected():
            logger.error("Vector store not initialized, can't search")
            return []
        
        try:
            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            # Format the results
            matches = results.get('matches', [])
            return [
                {
                    'id': match['id'],
                    'score': match['score'],
                    'metadata': match['metadata']
                }
                for match in matches
            ]
            
        except Exception as e:
            logger.error(f"Error searching in vector store: {str(e)}")
            return []
