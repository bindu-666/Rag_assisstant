import os
import logging
from flask import Flask, render_template, request, jsonify
from utils.data_processor import load_sample_data
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from utils.rag_engine import RAGEngine
from config import Config

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret")

# Initialize components
try:
    # Load sample data
    sample_data = load_sample_data()
    logger.info(f"Loaded {len(sample_data)} sample documents")
    
    # Initialize the embedding generator (using scikit-learn implementation)
    embedding_generator = EmbeddingGenerator(embedding_dim=Config.EMBEDDING_DIMENSION)
    logger.info("Initialized embedding generator with scikit-learn")
    
    # Initialize Pinecone manager with actual Pinecone API key
    pinecone_manager = PineconeManager(
        api_key=os.environ.get("PINECONE_API_KEY", Config.DEFAULT_PINECONE_API_KEY),
        environment=Config.PINECONE_ENVIRONMENT,
        index_name=Config.PINECONE_INDEX_NAME,
        dimension=Config.EMBEDDING_DIMENSION
    )
    logger.info("Initialized Pinecone manager")
    
    # Initialize RAG engine
    rag_engine = RAGEngine(embedding_generator, pinecone_manager)
    logger.info("Initialized RAG engine")
    
    # Even if vector store has data, we still need to fit our TF-IDF model
    # Extract content from sample data for fitting the model
    sample_texts = [doc["content"] for doc in sample_data]
    logger.info(f"Fitting embedding model on {len(sample_texts)} sample texts")
    embedding_generator.fit_model(sample_texts)
    
    # Check if vector store has data, if not, initialize with sample data
    if not pinecone_manager.check_index_populated():
        logger.info("Initializing vector store with sample data...")
        success = rag_engine.index_documents(sample_data)
        if success:
            logger.info("Vector store initialized successfully")
        else:
            logger.error("Failed to initialize vector store with sample data")
    
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.exception("Detailed error:")
    # We'll handle this gracefully in the routes

@app.route('/')
def home():
    """Render the home page with the chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint to handle chat requests."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Process the query using the RAG engine
        response = rag_engine.generate_response(query)
        
        return jsonify({
            'response': response['answer'],
            'sources': response['sources']
        })
    
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({'error': 'An error occurred processing your request. Please try again.'}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint to check the status of the backend components."""
    try:
        # Check if all components are initialized
        components_status = {
            'embedding_model': embedding_generator.is_initialized(),
            'pinecone': pinecone_manager.is_connected(),
            'sample_data': len(sample_data) > 0 if 'sample_data' in globals() else False
        }
        
        all_operational = all(components_status.values())
        
        return jsonify({
            'status': 'operational' if all_operational else 'degraded',
            'components': components_status
        })
    
    except Exception as e:
        logger.error(f"Error checking system status: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
