"""
Flask application for the RAG Chatbot
"""
import os
import logging
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from config import Config
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from utils.rag_engine import RAGEngine
from utils.kaggle_loader import load_kaggle_dataset
from utils.memory_monitor import MemoryMonitor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pinecone configuration
PINECONE_API_KEY = "pcsk_3QD2yg_ErC9WPvf686c64wNkqf7hMg8TWjFxS3vXnN2oYEdKtJF3YDFcgrZ2jw88Lbqpuw"
PINECONE_ENVIRONMENT = "us-east-1"
PINECONE_INDEX_NAME = "rag-chatbot-index"
PINECONE_DIMENSION = 1536

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
embedding_generator = EmbeddingGenerator()
pinecone_manager = PineconeManager(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENVIRONMENT,
    index_name=PINECONE_INDEX_NAME,
    dimension=PINECONE_DIMENSION
)
rag_engine = RAGEngine(embedding_generator, pinecone_manager)

# Initialize memory monitor
memory_monitor = MemoryMonitor(target_usage=75.0, min_batch_size=100, max_batch_size=2000)

# Global variable to track indexing progress
indexing_progress = {
    "total_documents": 0,
    "processed_documents": 0,
    "is_indexing": False,
    "current_batch_size": 1000
}

# Function to load dataset automatically
def load_dataset_automatically(dataset_path):
    """Load a dataset automatically when the application starts"""
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found: {dataset_path}")
        return False
    
    logger.info(f"Loading dataset from: {dataset_path}")
    
    # Reset indexing progress
    indexing_progress["is_indexing"] = True
    indexing_progress["processed_documents"] = 0
    indexing_progress["current_batch_size"] = 1000
    
    try:
        # Process dataset in batches
        for batch in load_kaggle_dataset(dataset_path, batch_size=indexing_progress["current_batch_size"]):
            # Check if we need to pause processing
            while memory_monitor.should_pause():
                logger.warning("High memory usage detected. Pausing for 5 seconds...")
                time.sleep(5)
            
            # Adjust batch size based on memory usage
            new_batch_size = memory_monitor.get_recommended_batch_size(indexing_progress["current_batch_size"])
            if new_batch_size != indexing_progress["current_batch_size"]:
                indexing_progress["current_batch_size"] = new_batch_size
                logger.info(f"Adjusted batch size to {new_batch_size}")
            
            # Index batch
            rag_engine.index_documents(batch)
            
            # Update progress
            indexing_progress["processed_documents"] += len(batch)
            logger.info(f"Indexed {indexing_progress['processed_documents']} documents")
            
            # Add a small delay between batches to prevent system overload
            time.sleep(0.1)
        
        # Mark indexing as complete
        indexing_progress["is_indexing"] = False
        logger.info(f"Successfully indexed {indexing_progress['processed_documents']} documents")
        return True
    
    except Exception as e:
        logger.error(f"Error indexing dataset: {str(e)}")
        indexing_progress["is_indexing"] = False
        return False

# Load dataset automatically when the application starts
# Replace this path with your actual dataset path
DATASET_PATH = "C:\\Users\\himab\\OneDrive\\Documents\\archive\\arxiv-metadata-oai-snapshot.json"  # Update this path
if os.path.exists(DATASET_PATH):
    load_dataset_automatically(DATASET_PATH)
else:
    logger.warning(f"Dataset not found at {DATASET_PATH}. Please update the path in app.py")

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({"error": "No query provided"}), 400
        
        # Get response from RAG engine
        relevant_docs = rag_engine.retrieve_relevant_context(query)
        response = rag_engine.generate_response(query, relevant_docs)
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def status():
    """Check the status of backend components"""
    try:
        return jsonify({
            "embedding_generator": embedding_generator.is_ready(),
            "pinecone_manager": pinecone_manager.is_connected(),
            "index_populated": pinecone_manager.check_index_populated(),
            "indexing_progress": indexing_progress
        })
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/index_dataset', methods=['POST'])
def index_dataset():
    """Index a Kaggle dataset"""
    try:
        data = request.get_json()
        dataset_path = data.get('dataset_path', '')
        
        if not dataset_path:
            return jsonify({"error": "No dataset path provided"}), 400
        
        if not os.path.exists(dataset_path):
            return jsonify({"error": f"Dataset not found: {dataset_path}"}), 404
        
        # Reset indexing progress
        indexing_progress["is_indexing"] = True
        indexing_progress["processed_documents"] = 0
        
        def progress_callback(processed, total):
            indexing_progress["total_documents"] = total
            indexing_progress["processed_documents"] = processed
        
        # Process dataset in batches
        for batch in load_kaggle_dataset(dataset_path, batch_size=1000):
            # Index batch
            rag_engine.index_documents(batch)
            
            # Update progress
            progress_callback(
                indexing_progress["processed_documents"] + len(batch),
                indexing_progress["total_documents"]
            )
        
        # Mark indexing as complete
        indexing_progress["is_indexing"] = False
        
        return jsonify({
            "message": "Dataset indexed successfully",
            "total_documents": indexing_progress["total_documents"]
        })
    
    except Exception as e:
        logger.error(f"Error indexing dataset: {str(e)}")
        indexing_progress["is_indexing"] = False
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 