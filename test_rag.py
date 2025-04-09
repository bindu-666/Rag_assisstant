"""
Test script for the RAG engine
"""
import logging
import sys
from utils.rag_engine import RAGEngine
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from utils.data_processor import load_sample_data
from config import Config
from utils.kaggle_loader import load_kaggle_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    """Main function to test the RAG engine"""
    try:
        # Initialize components
        embedding_generator = EmbeddingGenerator(embedding_dim=Config.EMBEDDING_DIMENSION)
        
        # Initialize Pinecone manager with parameters from config
        pinecone_manager = PineconeManager(
            api_key=Config.DEFAULT_PINECONE_API_KEY,
            environment=Config.PINECONE_ENVIRONMENT,
            index_name=Config.PINECONE_INDEX_NAME,
            dimension=Config.EMBEDDING_DIMENSION
        )
        
        rag_engine = RAGEngine(embedding_generator, pinecone_manager)
        
        # Load dataset
        documents = load_kaggle_dataset()
        if not documents:
            logger.error("Failed to load dataset")
            return
        
        # Index documents
        success = rag_engine.index_documents(documents)
        if not success:
            logger.error("Failed to index documents")
            return
        
        logger.info("Documents indexed successfully")
        
        # Test questions
        test_questions = [
            "What is RAG?",
            "Why is chunking useful in RAG?",
            "How are you?",
            "What is React used for?",
            "What is cinema?",
            "What are the benefits of RAG?",
            "How do I implement RAG?",
            "What are the evaluation metrics for RAG?"
        ]
        
        # Process each question
        for question in test_questions:
            logger.info(f"\n\nProcessing question: {question}")
            
            # Generate response
            response = rag_engine.generate_response(question)
            
            # Print response
            logger.info(f"Answer: {response['answer']}")
            logger.info(f"Sources: {[source['title'] for source in response['sources']]}")
            
    except Exception as e:
        logger.error(f"Error in test script: {str(e)}")

if __name__ == "__main__":
    main() 