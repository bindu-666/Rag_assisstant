"""
Configuration settings for the RAG application
"""

class Config:
    # Embedding model settings
    EMBEDDING_MODEL_NAME = "scikit-learn-tfidf-svd"  # Not used with our scikit-learn implementation
    EMBEDDING_DIMENSION = 100  # Dimension for our TF-IDF + SVD embeddings
    
    # Vector store settings (In-memory implementation)
    PINECONE_ENVIRONMENT = "local"  # Not used with our in-memory implementation
    PINECONE_INDEX_NAME = "rag-documents"  # Not used with our in-memory implementation
    DEFAULT_PINECONE_API_KEY = "not-needed"  # Not used with our in-memory implementation
    
    # RAG settings
    TOP_K_RESULTS = 3  # Number of relevant documents to retrieve
    
    # Data settings
    SAMPLE_DATA_PATH = "data/sample_documents.json"
