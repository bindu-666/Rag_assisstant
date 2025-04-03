"""
Module for processing sample data for the RAG application
"""
import json
import logging
import os
from typing import List, Dict, Any

from config import Config

logger = logging.getLogger(__name__)

def load_sample_data() -> List[Dict[str, Any]]:
    """
    Load sample documents from the JSON file.
    
    Returns:
        List[Dict[str, Any]]: List of document dictionaries
    """
    try:
        file_path = Config.SAMPLE_DATA_PATH
        if not os.path.exists(file_path):
            logger.error(f"Sample data file not found at {file_path}")
            return []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate the data structure
        if not isinstance(data, list):
            logger.error("Sample data must be a list of documents")
            return []
        
        # Ensure each document has required fields
        valid_docs = []
        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                logger.warning(f"Document at index {i} is not a dictionary, skipping")
                continue
                
            if "id" not in doc:
                # Generate an ID if not present
                doc["id"] = f"doc_{i}"
                
            if "content" not in doc:
                logger.warning(f"Document at index {i} has no 'content' field, skipping")
                continue
                
            if "metadata" not in doc:
                # Add empty metadata if not present
                doc["metadata"] = {}
                
            valid_docs.append(doc)
            
        logger.info(f"Loaded {len(valid_docs)} valid documents from sample data")
        return valid_docs
        
    except Exception as e:
        logger.error(f"Error loading sample data: {str(e)}")
        return []

def preprocess_text(text: str) -> str:
    """
    Preprocess text for embedding
    
    Args:
        text: The input text
        
    Returns:
        str: Preprocessed text
    """
    # Simple preprocessing - can be expanded with NLTK or other libraries
    text = text.strip()
    
    # Remove extra whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    return text

def chunk_document(doc: Dict[str, Any], chunk_size: int = 512) -> List[Dict[str, Any]]:
    """
    Split a document into smaller chunks for better retrieval
    
    Args:
        doc: The input document
        chunk_size: Maximum chunk size in characters
        
    Returns:
        List[Dict[str, Any]]: List of document chunks
    """
    content = doc["content"]
    
    # For very short documents, don't chunk
    if len(content) <= chunk_size:
        return [doc]
    
    # Split into sentences (naive approach)
    import re
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    chunks = []
    current_chunk = ""
    chunk_id = 0
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size, save current chunk and start a new one
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append({
                "id": f"{doc['id']}_chunk_{chunk_id}",
                "content": current_chunk.strip(),
                "metadata": {
                    **doc.get("metadata", {}),
                    "parent_id": doc["id"],
                    "chunk_id": chunk_id
                }
            })
            chunk_id += 1
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append({
            "id": f"{doc['id']}_chunk_{chunk_id}",
            "content": current_chunk.strip(),
            "metadata": {
                **doc.get("metadata", {}),
                "parent_id": doc["id"],
                "chunk_id": chunk_id
            }
        })
    
    return chunks
