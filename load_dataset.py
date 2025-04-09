"""
Script to load a dataset directly without starting the Flask application
"""
import os
import sys
import logging
import argparse
from tqdm import tqdm

from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from utils.rag_engine import RAGEngine
from utils.kaggle_loader import load_kaggle_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pinecone configuration
PINECONE_API_KEY = "pcsk_3QD2yg_ErC9WPvf686c64wNkqf7hMg8TWjFxS3vXnN2oYEdKtJF3YDFcgrZ2jw88Lbqpuw"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "rag-chatbot-index"
PINECONE_DIMENSION = 1536

def load_dataset(dataset_path, batch_size=1000):
    """
    Load a dataset directly without starting the Flask application
    
    Args:
        dataset_path: Path to the dataset
        batch_size: Number of documents to process in each batch
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Initialize components
    embedding_generator = EmbeddingGenerator()
    pinecone_manager = PineconeManager(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENVIRONMENT,
        index_name=PINECONE_INDEX_NAME,
        dimension=PINECONE_DIMENSION
    )
    rag_engine = RAGEngine(embedding_generator, pinecone_manager)
    
    # Count total documents
    total_documents = 0
    for _ in load_kaggle_dataset(dataset_path, batch_size=batch_size):
        total_documents += batch_size
    
    logger.info(f"Found approximately {total_documents} documents")
    
    # Process dataset in batches with progress bar
    processed_documents = 0
    with tqdm(total=total_documents, desc="Indexing documents") as pbar:
        for batch in load_kaggle_dataset(dataset_path, batch_size=batch_size):
            # Index batch
            rag_engine.index_documents(batch)
            
            # Update progress
            processed_documents += len(batch)
            pbar.update(len(batch))
    
    logger.info(f"Successfully indexed {processed_documents} documents")
    return True

def main():
    """Main function to parse arguments and load dataset"""
    parser = argparse.ArgumentParser(description="Load a dataset directly without starting the Flask application")
    parser.add_argument("dataset_path", help="Path to the dataset file or directory")
    parser.add_argument("--batch-size", type=int, default=1000, help="Number of documents to process in each batch")
    
    args = parser.parse_args()
    
    success = load_dataset(args.dataset_path, args.batch_size)
    
    if success:
        logger.info("Dataset loaded successfully")
        return 0
    else:
        logger.error("Failed to load dataset")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 