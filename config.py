"""
Configuration settings for the RAG application
"""

class Config:
    # Embedding model settings
    EMBEDDING_MODEL_NAME = "scikit-learn-tfidf-svd"  # Using scikit-learn implementation
    EMBEDDING_DIMENSION = 1536  # Dimension for our TF-IDF + SVD embeddings to match Pinecone index
    
    # Pinecone settings
    PINECONE_ENVIRONMENT = "gcp-starter"  # Default Pinecone starter environment
    PINECONE_INDEX_NAME = "rag-index"  # Your specified index name
    DEFAULT_PINECONE_API_KEY = "pcsk_3QD2yg_ErC9WPvf686c64wNkqf7hMg8TWjFxS3vXnN2oYEdKtJF3YDFcgrZ2jw88Lbqpuw"  # Your Pinecone API key
    
    # RAG settings
    TOP_K_RESULTS = 6  # Number of relevant documents to retrieve (increased to get more potential matches)
    
    # Data settings
    SAMPLE_DATA_PATH = "data/sample_documents.json"
