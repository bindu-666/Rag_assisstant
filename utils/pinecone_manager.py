"""
Module for managing Pinecone vector database operations
"""
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import time
import os
import numpy as np  # Keep for fallback

logger = logging.getLogger(__name__)

class PineconeManager:
    """Class to handle interactions with Pinecone vector database"""
    
    def __init__(self, api_key: str, environment: str, index_name: str, dimension: int):
        """
        Initialize the Pinecone manager
        
        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., "gcp-starter")
            index_name: Name of the Pinecone index to use
            dimension: Dimension of the vectors to store
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.dimension = dimension
        self.pinecone = None
        self.index = None
        self.use_fallback = False
        self._connect()
        
        # Store for document metadata
        self.document_store = {}
    
    def _connect(self):
        """Connect to Pinecone and initialize the index"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            # Initialize Pinecone client
            logger.info(f"Connecting to Pinecone with index '{self.index_name}'")
            pc = Pinecone(api_key=self.api_key)
            self.pinecone = pc
            
            # Check if index exists
            available_indexes = pc.list_indexes()
            logger.info(f"Available Pinecone indexes: {available_indexes.names()}")
            
            if self.index_name not in available_indexes.names():
                logger.info(f"Creating Pinecone index '{self.index_name}'")
                pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="gcp",
                        region="us-central1"
                    )
                )
                # Wait for index to be ready
                time.sleep(10)
            
            # Connect to the index
            self.index = pc.Index(self.index_name)
            logger.info(f"Connected to Pinecone index '{self.index_name}'")
            
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {str(e)}")
            logger.warning("Falling back to in-memory vector store")
            self.pinecone = None
            self.index = None
            self.use_fallback = True
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize fallback in-memory vector store if Pinecone connection fails"""
        try:
            logger.info("Initializing in-memory vector store as fallback")
            self.index = InMemoryVectorStore(dimension=self.dimension)
            logger.info("In-memory vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing in-memory vector store: {str(e)}")
            self.index = None
    
    def is_connected(self) -> bool:
        """Check if connected to vector store"""
        return self.index is not None
    
    def check_index_populated(self) -> bool:
        """Check if the index has data"""
        if not self.is_connected():
            return False
        
        try:
            if self.use_fallback:
                stats = self.index.describe_index_stats()
                vector_count = stats.get('total_vector_count', 0)
            else:
                # For Pinecone API v2
                try:
                    stats = self.index.describe_index_stats()
                    vector_count = stats.get('total_vector_count', 0)
                except AttributeError:
                    # Newer Pinecone API
                    stats = self.index.describe_stats()
                    vector_count = stats.total_vector_count if hasattr(stats, 'total_vector_count') else 0
            
            logger.info(f"Vector count in index: {vector_count}")
            return vector_count > 0
            
        except Exception as e:
            logger.error(f"Error checking index stats: {str(e)}")
            return False
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents with their embeddings in vector store
        
        Args:
            documents: List of document dictionaries with 'id', 'embedding', and 'metadata'
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to vector store, can't index documents")
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
                
                # Prepare metadata (exclude embedding and ensure all values are strings for Pinecone)
                metadata = doc.get("metadata", {}).copy()
                metadata["content"] = doc["content"]  # Store text in metadata for retrieval
                
                if not self.use_fallback:
                    # Convert all metadata values to strings (Pinecone requirement)
                    metadata = {k: str(v) for k, v in metadata.items()}
                
                vectors.append((vector_id, embedding, metadata))
                
                # Store document in document_store for fallback retrieval
                if self.use_fallback:
                    self.document_store[vector_id] = {
                        "id": vector_id,
                        "embedding": embedding,
                        "metadata": metadata,
                        "score": 1.0  # Default score for stored documents
                    }
            
            # Upsert in batches to avoid request size limits
            batch_size = 100
            total_upserted = 0
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                
                if self.use_fallback:
                    # For in-memory store
                    result = self.index.upsert(batch)
                    total_upserted += result.get('upserted_count', 0)
                else:
                    # Format batch for Pinecone upsert
                    records = []
                    for id, values, metadata in batch:
                        records.append({
                            "id": id,
                            "values": values,
                            "metadata": metadata
                        })
                    self.index.upsert(records)
                    total_upserted += len(batch)
            
            logger.info(f"Indexed {total_upserted} documents in vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            logger.exception("Detailed error:")
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
            logger.error("Not connected to vector store, can't search")
            return []
        
        try:
            if self.use_fallback:
                # For in-memory store
                results = self.index.query(query_vector, top_k=top_k, include_metadata=True)
                matches = []
                for match in results['matches']:  # Access the 'matches' key
                    doc_id = match['id']
                    if doc_id in self.document_store:
                        doc = self.document_store[doc_id]
                        matches.append({
                            'id': doc_id,
                            'score': float(match['score']),
                            'metadata': doc['metadata']
                        })
                return matches
            else:
                # For Pinecone
                results = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
                
                matches = []
                for match in results.matches:
                    matches.append({
                        'id': match.id,
                        'score': float(match.score),
                        'metadata': match.metadata
                    })
                return matches
                
        except Exception as e:
            logger.error(f"Error searching in vector store: {str(e)}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by its ID
        
        Args:
            doc_id: The document ID
            
        Returns:
            Document dictionary if found, None otherwise
        """
        if not self.is_connected():
            logger.error("Not connected to vector store, can't retrieve document")
            return None
            
        try:
            if self.use_fallback:
                # For in-memory store, get from document_store
                return self.document_store.get(doc_id)
            else:
                # For Pinecone, fetch the vector
                try:
                    result = self.index.fetch([doc_id])
                    if doc_id in result.vectors:
                        vector_data = result.vectors[doc_id]
                        return {
                            "id": doc_id,
                            "embedding": vector_data.values,
                            "metadata": vector_data.metadata,
                            "score": 1.0  # Default score for direct fetches
                        }
                except Exception as e:
                    logger.error(f"Error fetching document from Pinecone: {str(e)}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            return None


# Fallback in-memory vector store implementation
class InMemoryVectorStore:
    """An in-memory vector store to mimic Pinecone functionality"""
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.vectors = {}  # id -> [vector, metadata]
        self.vector_count = 0
    
    def upsert(self, vectors):
        """Insert or update vectors in the store"""
        try:
            for id, values, metadata in vectors:
                # Ensure values is a list of floats
                if isinstance(values, (list, np.ndarray)):
                    values = [float(v) for v in values]
                else:
                    logger.error(f"Invalid vector values type for id {id}: {type(values)}")
                    continue
                
                # Store the vector and metadata
                self.vectors[id] = [values, metadata]
            
            self.vector_count = len(self.vectors)
            return {'upserted_count': len(vectors)}
        except Exception as e:
            logger.error(f"Error in upsert: {str(e)}")
            return {'upserted_count': 0}
    
    def query(self, vector, top_k=5, include_metadata=True):
        """Query for similar vectors"""
        try:
            if not self.vectors:
                return {'matches': []}
            
            # Convert the query vector to numpy array
            query_vector = np.array(vector, dtype=np.float32)
            
            # Calculate cosine similarity with all vectors
            similarities = []
            for id, (values, metadata) in self.vectors.items():
                try:
                    # Convert to numpy array
                    doc_vector = np.array(values, dtype=np.float32)
                    
                    # Calculate cosine similarity
                    dot_product = np.dot(query_vector, doc_vector)
                    query_norm = np.linalg.norm(query_vector)
                    doc_norm = np.linalg.norm(doc_vector)
                    
                    if query_norm > 0 and doc_norm > 0:
                        similarity = dot_product / (query_norm * doc_norm)
                    else:
                        similarity = 0
                    
                    similarities.append((id, similarity, metadata if include_metadata else None))
                except Exception as e:
                    logger.error(f"Error calculating similarity for vector {id}: {str(e)}")
                    continue
            
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
        except Exception as e:
            logger.error(f"Error in query: {str(e)}")
            return {'matches': []}
    
    def describe_index_stats(self):
        """Return stats about the index"""
        return {
            'total_vector_count': self.vector_count,
            'dimensions': self.dimension
        }
