"""
Module for the RAG (Retrieval-Augmented Generation) engine
"""
import logging
from typing import List, Dict, Any, Union
import re

from utils.data_processor import preprocess_text, chunk_document
from utils.embeddings import EmbeddingGenerator
from utils.pinecone_manager import PineconeManager
from config import Config

logger = logging.getLogger(__name__)

# Common English stop words
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'were',
    'will', 'with', 'the', 'this', 'but', 'they', 'have', 'had', 'what', 'when',
    'where', 'who', 'which', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
    'more', 'most', 'other', 'some', 'such', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'can', 'just', 'should', 'now'
}

# Synonym map for common RAG-related terms
SYNONYM_MAP = {
    'rag': ['retrieval', 'augmented', 'generation', 'rag', 'retrieval-augmented'],
    'retrieval': ['retrieve', 'retrieving', 'retrieval', 'search', 'find', 'fetch'],
    'augmented': ['augment', 'augmenting', 'augmentation', 'enhance', 'enhanced', 'improvement'],
    'generation': ['generate', 'generating', 'generator', 'create', 'creating', 'creation'],
    'chunk': ['chunking', 'chunks', 'split', 'splitting', 'segment', 'segmentation'],
    'embedding': ['embeddings', 'embed', 'vector', 'vectorize', 'vectorization'],
    'benefit': ['benefits', 'advantage', 'advantages', 'value', 'values', 'useful'],
    'implement': ['implementation', 'implementing', 'setup', 'configure', 'build'],
    'evaluate': ['evaluation', 'evaluating', 'measure', 'measuring', 'assess', 'assessment']
}

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
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Add console handler if not already added
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        self.logger.info("RAGEngine initialized")
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text using the imported preprocess_text function
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        self.logger.debug(f"Preprocessing text: {text}")
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        self.logger.debug(f"Preprocessed text: {text}")
        
        return text
    
    def index_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Index documents in the vector store
        
        Args:
            documents: List of documents to index
            
        Returns:
            True if indexing was successful, False otherwise
        """
        self.logger.info(f"Indexing {len(documents)} documents")
        
        try:
            # Chunk documents
            chunked_docs = []
            for doc in documents:
                chunks = chunk_document(doc)
                chunked_docs.extend(chunks)
            
            self.logger.info(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
            
            # Preprocess all chunks
            processed_contents = [self._preprocess_text(chunk["content"]) for chunk in chunked_docs]
            
            # Fit the model on all processed contents
            self.embedding_generator.fit_model(processed_contents)
            
            # Generate embeddings for all chunks
            embeddings = self.embedding_generator.generate_embeddings(processed_contents)
            
            # Add embeddings to chunks
            for chunk, embedding in zip(chunked_docs, embeddings):
                chunk["embedding"] = embedding
            
            # Index documents in Pinecone
            success = self.pinecone_manager.index_documents(chunked_docs)
            
            if success:
                self.logger.info("Successfully indexed documents")
            else:
                self.logger.error("Failed to index documents")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error indexing documents: {str(e)}")
            return False
    
    def retrieve_relevant_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: User query
            
        Returns:
            List of relevant documents with their scores
        """
        self.logger.info(f"Retrieving context for query: {query}")
        
        # Check for conversational questions
        conversational_phrases = [
            "how are you", "how's it going", "what's up", "hi", "hello", 
            "good morning", "good afternoon", "good evening", "hey"
        ]
        
        query_lower = query.lower()
        for phrase in conversational_phrases:
            if phrase in query_lower:
                self.logger.info(f"Detected conversational query: {query}")
                return []
        
        # Check if the query is RAG-related
        rag_keywords = ["rag", "retrieval", "generation", "augmented"]
        is_rag_query = any(keyword in query_lower for keyword in rag_keywords)
        
        if not is_rag_query:
            self.logger.info(f"Query not RAG-related: {query}")
            return []
        
        # Create an enhanced query based on the question type
        enhanced_query = query
        
        if "what is" in query_lower or "what's" in query_lower:
            if "rag" in query_lower:
                enhanced_query = "introduction definition explanation what is retrieval augmented generation rag overview concept architecture components"
            elif "chunking" in query_lower:
                enhanced_query += " document chunking strategies size overlap segmentation preprocessing"
            elif "benefit" in query_lower:
                enhanced_query = "benefits advantages improvements value useful rag retrieval augmented generation hallucination accuracy"
            elif "implement" in query_lower:
                enhanced_query = "implementation steps process setup configure build rag retrieval augmented generation"
            elif "evaluate" in query_lower:
                enhanced_query = "evaluation metrics assessment measurement performance accuracy retrieval generation"
        elif "why" in query_lower:
            if "chunking" in query_lower and "rag" in query_lower:
                enhanced_query = "why is chunking used in RAG document chunking strategies purpose benefits advantages"
            elif "rag" in query_lower:
                enhanced_query = "why is RAG useful benefits advantages improvements value retrieval augmented generation"
        
        self.logger.info(f"Enhanced query: {enhanced_query}")
        
        # Preprocess the enhanced query
        processed_query = self._preprocess_text(enhanced_query)
        
        # Generate embedding for the query
        query_embedding = self.embedding_generator.generate_embedding(processed_query)
        
        # Search for similar documents
        results = self.pinecone_manager.search(query_embedding, top_k=5)
        
        # Filter out low-quality matches
        filtered_results = [doc for doc in results if doc["score"] > 0.5]
        
        self.logger.info(f"Found {len(filtered_results)} relevant documents")
        
        # For "What is RAG?" queries, prioritize documents with "introduction" or "definition" in the title
        if "what is" in query_lower and "rag" in query_lower:
            filtered_results.sort(key=lambda x: (
                "introduction" in x["metadata"].get("title", "").lower() or 
                "definition" in x["metadata"].get("title", "").lower(),
                x["score"]
            ), reverse=True)
        
        # For "Why is chunking used in RAG?" queries, prioritize documents with "chunking" in the title
        if "why" in query_lower and "chunking" in query_lower and "rag" in query_lower:
            filtered_results.sort(key=lambda x: (
                "chunking" in x["metadata"].get("title", "").lower(),
                x["score"]
            ), reverse=True)
        
        return filtered_results
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """
        Generate a response to a user query using retrieved context
        
        Args:
            query: User query
            
        Returns:
            Dictionary with 'answer' and 'sources'
        """
        self.logger.info(f"Processing query: {query}")
        
        # Check for conversational questions
        conversational_phrases = [
            "how are you", "how's it going", "what's up", "hi", "hello", 
            "good morning", "good afternoon", "good evening", "hey"
        ]
        
        query_lower = query.lower()
        for phrase in conversational_phrases:
            if phrase in query_lower:
                self.logger.info("Detected conversational query")
                return {
                    "answer": "I'm a RAG assistant focused on answering questions about RAG systems. How can I help you with information about RAG?",
                    "sources": []
                }
        
        # Check if the query is RAG-related
        rag_keywords = ["rag", "retrieval", "generation", "augmented"]
        is_rag_query = any(keyword in query_lower for keyword in rag_keywords)
        
        if not is_rag_query:
            self.logger.info("Query not RAG-related")
            return {
                "answer": "I can only answer questions about RAG (Retrieval-Augmented Generation) systems. Please ask a RAG-related question.",
                "sources": []
            }
        
        # Retrieve relevant context
        relevant_docs = self.retrieve_relevant_context(query)
        
        if not relevant_docs:
            self.logger.info("No relevant documents found")
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try a different question about RAG.",
                "sources": []
            }
        
        # Generate answer from relevant documents
        answer = self._generate_simple_answer(query, relevant_docs)
        
        # Format sources
        sources = []
        for doc in relevant_docs:
            source = {
                "id": doc["id"],
                "score": doc["score"],
                "title": doc["metadata"].get("title", "Unknown"),
                "excerpt": doc["metadata"].get("content", "")[:200] + "..." if len(doc["metadata"].get("content", "")) > 200 else doc["metadata"].get("content", "")
            }
            sources.append(source)
        
        self.logger.info(f"Generated response with {len(sources)} sources")
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def _generate_simple_answer(self, query: str, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Generate a simple answer from relevant documents
        
        Args:
            query: User query
            relevant_docs: List of relevant documents
            
        Returns:
            Generated answer
        """
        self.logger.info(f"Generating answer for query: {query}")
        
        # Extract key terms from the query
        query_terms = set(query.lower().split())
        query_terms = {term for term in query_terms if len(term) > 2}
        
        # Add synonyms for key terms
        expanded_terms = set()
        for term in query_terms:
            if term in SYNONYM_MAP:
                expanded_terms.update(SYNONYM_MAP[term])
            else:
                expanded_terms.add(term)
        
        # Score each document based on term matches
        scored_docs = []
        for doc in relevant_docs:
            content = doc["metadata"].get("content", "").lower()
            title = doc["metadata"].get("title", "").lower()
            
            # Check for exact matches in title first (higher priority)
            title_matches = sum(1 for term in expanded_terms if term in title)
            
            # Then check content matches
            content_matches = sum(1 for term in expanded_terms if term in content)
            
            # Combine scores with title matches weighted more heavily
            total_score = (title_matches * 3) + content_matches
            
            scored_docs.append((doc, total_score, doc["score"]))
        
        # Sort by number of term matches and score
        scored_docs.sort(key=lambda x: (x[1], x[2]), reverse=True)
        
        # Get the most relevant document
        most_relevant_doc = scored_docs[0][0]
        
        # Extract the answer from the most relevant document
        content = most_relevant_doc["metadata"].get("content", "")
        
        # If the content is too long, try to extract a relevant paragraph
        if len(content) > 500:
            paragraphs = content.split('\n\n')
            best_paragraph = None
            best_score = 0
            
            for paragraph in paragraphs:
                score = sum(1 for term in expanded_terms if term in paragraph.lower())
                if score > best_score:
                    best_score = score
                    best_paragraph = paragraph
            
            if best_paragraph:
                content = best_paragraph
        
        # Add attribution
        title = most_relevant_doc["metadata"].get("title", "Unknown Source")
        answer = f"{content}\n\nSource: {title}"
        
        self.logger.info(f"Generated answer from source: {title}")
        return answer
