"""
Module for generating embeddings using scikit-learn's TF-IDF and SVD
"""
import logging
from typing import List, Union, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Class to handle the generation of embeddings using TF-IDF and SVD"""
    
    def __init__(self, model_name: str = None, embedding_dim: int = 384):
        """
        Initialize the embedding generator
        
        Args:
            model_name: Not used in this implementation, kept for interface compatibility
            embedding_dim: Dimension of the embeddings to generate
        """
        self.embedding_dim = embedding_dim
        self.tfidf_vectorizer = None
        self.svd_model = None
        self.model_name = "scikit-learn-tfidf-svd"  # For consistency with interface
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the TF-IDF vectorizer and SVD model"""
        try:
            logger.info(f"Initializing TF-IDF vectorizer and SVD model")
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,  # Limit features to avoid memory issues
                stop_words='english'
            )
            
            # Use a smaller dimension for SVD (we'll pad later)
            self.svd_dim = min(100, self.embedding_dim)  # Use a smaller SVD dimension
            self.svd_model = TruncatedSVD(n_components=self.svd_dim, random_state=42)
            
            logger.info(f"TF-IDF vectorizer and SVD model initialized (SVD dim={self.svd_dim}, will pad to {self.embedding_dim})")
        except Exception as e:
            logger.error(f"Error initializing TF-IDF vectorizer and SVD model: {str(e)}")
            self.tfidf_vectorizer = None
            self.svd_model = None
    
    def is_initialized(self) -> bool:
        """Check if the model is initialized"""
        return self.tfidf_vectorizer is not None and self.svd_model is not None
    
    def fit_model(self, texts: List[str]):
        """
        Fit the TF-IDF vectorizer and SVD model on the input texts
        
        Args:
            texts: List of input texts to fit the model on
        """
        if not self.is_initialized():
            logger.error("Models not initialized, cannot fit")
            return
        
        try:
            logger.info(f"Fitting TF-IDF vectorizer on {len(texts)} documents")
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            logger.info(f"Fitting SVD model on TF-IDF matrix")
            self.svd_model.fit(tfidf_matrix)
            
            logger.info("Models fitted successfully")
        except Exception as e:
            logger.error(f"Error fitting models: {str(e)}")
    
    def generate_embedding(self, text: str) -> Union[List[float], None]:
        """
        Generate an embedding for a single text
        
        Args:
            text: The input text to embed
            
        Returns:
            List[float] or None: The embedding vector or None if an error occurred
        """
        if not self.is_initialized():
            logger.error("Models not initialized, cannot generate embedding")
            return None
        
        try:
            # Transform the text to TF-IDF
            tfidf_vector = self.tfidf_vectorizer.transform([text])
            
            # Apply SVD to get lower dimensional embedding
            embedding = self.svd_model.transform(tfidf_vector)[0]
            
            # Normalize the embedding vector
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            # Pad the embedding to match required dimension (1536 for Pinecone)
            if len(embedding) < self.embedding_dim:
                # Create a zero vector of required dimension
                padded_embedding = np.zeros(self.embedding_dim)
                
                # Fill in the actual values
                padded_embedding[:len(embedding)] = embedding
                
                # Re-normalize the padded vector
                norm = np.linalg.norm(padded_embedding)
                if norm > 0:
                    padded_embedding = padded_embedding / norm
                
                # Use the padded embedding
                embedding = padded_embedding
            
            # Convert to Python list for easier JSON serialization
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def generate_embeddings(self, texts: List[str]) -> List[Union[List[float], None]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of embedding vectors or None values for errors
        """
        if not self.is_initialized():
            logger.error("Models not initialized, cannot generate embeddings")
            return [None] * len(texts)
        
        try:
            # Transform the texts to TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            
            # Apply SVD to get lower dimensional embeddings
            svd_embeddings = self.svd_model.transform(tfidf_matrix)
            
            # Pad and normalize the embedding vectors
            final_embeddings = []
            for embedding in svd_embeddings:
                # Normalize the SVD embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                
                # Pad the embedding to match required dimension (1536 for Pinecone)
                if len(embedding) < self.embedding_dim:
                    # Create a zero vector of required dimension
                    padded_embedding = np.zeros(self.embedding_dim)
                    
                    # Fill in the actual values
                    padded_embedding[:len(embedding)] = embedding
                    
                    # Re-normalize the padded vector
                    norm = np.linalg.norm(padded_embedding)
                    if norm > 0:
                        padded_embedding = padded_embedding / norm
                    
                    # Use the padded embedding
                    embedding = padded_embedding
                
                # Convert to list and add to results
                final_embeddings.append(embedding.tolist())
            
            return final_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return [None] * len(texts)
    
    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of document dictionaries
        
        Args:
            documents: List of document dictionaries with 'content' field
            
        Returns:
            List of document dictionaries with added 'embedding' field
        """
        if not documents:
            return []
        
        texts = [doc["content"] for doc in documents]
        
        # First, fit the model on all document texts
        self.fit_model(texts)
        
        # Now generate embeddings
        embeddings = self.generate_embeddings(texts)
        
        # Create a list to store documents with embeddings
        embedded_docs = []
        
        # Iterate through documents and embeddings together
        for doc, emb in zip(documents, embeddings):
            if emb is not None:
                # Create a copy of the document to avoid modifying the original
                doc_copy = doc.copy()
                doc_copy["embedding"] = emb
                embedded_docs.append(doc_copy)
        
        return embedded_docs
