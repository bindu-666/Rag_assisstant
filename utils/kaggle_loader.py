"""
Module for loading and processing large datasets from Kaggle
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Generator, Optional
from tqdm import tqdm
import time

from config import Config
from utils.data_processor import preprocess_text, chunk_document

logger = logging.getLogger(__name__)

class KaggleDatasetLoader:
    """Class to handle loading and processing large datasets from Kaggle"""
    
    def __init__(self, dataset_path: str, batch_size: int = 1000):
        """
        Initialize the Kaggle dataset loader
        
        Args:
            dataset_path: Path to the Kaggle dataset
            batch_size: Number of documents to process in each batch
        """
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.total_documents = 0
        self.processed_documents = 0
        
    def count_documents(self) -> int:
        """
        Count the total number of documents in the dataset
        
        Returns:
            int: Total number of documents
        """
        try:
            # Check if the dataset is a CSV file
            if self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path, nrows=0)
                # Estimate total rows based on file size
                file_size = os.path.getsize(self.dataset_path)
                sample_size = len(pd.read_csv(self.dataset_path, nrows=1000))
                estimated_rows = int((file_size / (sample_size * 1000)) * sample_size)
                logger.info(f"Estimated {estimated_rows:,} documents in CSV dataset")
                return estimated_rows
            
            # Check if the dataset is a JSON file
            elif self.dataset_path.endswith('.json'):
                # For JSONL files, count non-empty lines
                count = 0
                with open(self.dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():  # Skip empty lines
                            count += 1
                
                logger.info(f"Found {count:,} documents in JSON dataset")
                return count
            
            # Check if the dataset is a directory of text files
            elif os.path.isdir(self.dataset_path):
                count = sum(1 for _ in self._get_text_files())
                logger.info(f"Found {count:,} text files in directory")
                return count
            
            else:
                logger.error(f"Unsupported dataset format: {self.dataset_path}")
                return 0
                
        except Exception as e:
            logger.error(f"Error counting documents: {str(e)}")
            return 0
    
    def _get_text_files(self) -> Generator[str, None, None]:
        """Get all text files in the dataset directory"""
        for root, _, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith(('.txt', '.md', '.html', '.htm')):
                    yield os.path.join(root, file)
    
    def process_documents(self, callback=None) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Process documents in batches
        
        Args:
            callback: Optional callback function to report progress
            
        Yields:
            List[Dict[str, Any]]: Batch of processed documents
        """
        try:
            # Process CSV files
            if self.dataset_path.endswith('.csv'):
                for batch in self._process_csv():
                    self.processed_documents += len(batch)
                    if callback:
                        callback(self.processed_documents, self.total_documents)
                    yield batch
            
            # Process JSON files
            elif self.dataset_path.endswith('.json'):
                for batch in self._process_json():
                    self.processed_documents += len(batch)
                    if callback:
                        callback(self.processed_documents, self.total_documents)
                    yield batch
            
            # Process text files
            elif os.path.isdir(self.dataset_path):
                for batch in self._process_text_files():
                    self.processed_documents += len(batch)
                    if callback:
                        callback(self.processed_documents, self.total_documents)
                    yield batch
            
            else:
                logger.error(f"Unsupported dataset format: {self.dataset_path}")
                yield []
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            yield []
    
    def _process_csv(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Process CSV files in batches"""
        try:
            # Read CSV in chunks to handle large files
            for chunk in pd.read_csv(self.dataset_path, chunksize=self.batch_size):
                batch = []
                for _, row in chunk.iterrows():
                    # Convert row to dictionary
                    doc = row.to_dict()
                    
                    # Ensure required fields
                    if "id" not in doc:
                        doc["id"] = f"doc_{self.processed_documents + len(batch)}"
                    
                    if "content" not in doc:
                        # Try to find a text column
                        text_cols = [col for col in doc.keys() if isinstance(doc[col], str) and len(doc[col]) > 50]
                        if text_cols:
                            doc["content"] = doc[text_cols[0]]
                        else:
                            # Use all string columns as content
                            doc["content"] = " ".join([str(doc[col]) for col in doc.keys() if isinstance(doc[col], str)])
                    
                    if "metadata" not in doc:
                        doc["metadata"] = {}
                    
                    # Add to batch
                    batch.append(doc)
                
                yield batch
                
        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            yield []
    
    def _process_json(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Process JSON files in batches"""
        try:
            batch = []
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Skip empty lines
                        if not line.strip():
                            continue
                            
                        doc = json.loads(line)
                        
                        # Extract relevant fields from arXiv metadata
                        if "title" in doc and "abstract" in doc:
                            processed_doc = {
                                "id": doc.get("id", f"doc_{self.processed_documents + len(batch)}"),
                                "content": doc.get("abstract", ""),
                                "metadata": {
                                    "title": doc.get("title", ""),
                                    "authors": doc.get("authors", []),
                                    "categories": doc.get("categories", []),
                                    "doi": doc.get("doi", ""),
                                    "journal_ref": doc.get("journal-ref", ""),
                                    "primary_category": doc.get("primary_category", ""),
                                    "published": doc.get("published", ""),
                                    "updated": doc.get("updated", "")
                                }
                            }
                        else:
                            # For non-arXiv JSON files, ensure required fields
                            processed_doc = {
                                "id": doc.get("id", f"doc_{self.processed_documents + len(batch)}"),
                                "content": doc.get("content", ""),
                                "metadata": doc.get("metadata", {})
                            }
                            
                            # If no content field, try to find a text field
                            if not processed_doc["content"]:
                                text_fields = [k for k, v in doc.items() if isinstance(v, str) and len(v) > 50]
                                if text_fields:
                                    processed_doc["content"] = doc[text_fields[0]]
                                else:
                                    logger.warning(f"Document {processed_doc['id']} has no content, skipping")
                                    continue
                        
                        # Add to batch
                        batch.append(processed_doc)
                        
                        # Yield batch if it reaches the batch size
                        if len(batch) >= self.batch_size:
                            yield batch
                            batch = []
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON line: {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing document: {str(e)}")
                        continue
                
                # Yield remaining documents
                if batch:
                    yield batch
                    
        except Exception as e:
            logger.error(f"Error processing JSON file: {str(e)}")
            yield []
    
    def _process_text_files(self) -> Generator[List[Dict[str, Any]], None, None]:
        """Process text files in batches"""
        try:
            batch = []
            for file_path in self._get_text_files():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Create document
                    doc = {
                        "id": f"doc_{self.processed_documents + len(batch)}",
                        "content": content,
                        "metadata": {
                            "source": file_path,
                            "title": os.path.basename(file_path)
                        }
                    }
                    
                    # Add to batch
                    batch.append(doc)
                    
                    # Yield batch if it reaches the batch size
                    if len(batch) >= self.batch_size:
                        yield batch
                        batch = []
                        
                except Exception as e:
                    logger.warning(f"Error processing file {file_path}: {str(e)}")
                    continue
            
            # Yield remaining documents
            if batch:
                yield batch
                
        except Exception as e:
            logger.error(f"Error processing text files: {str(e)}")
            yield []

def load_kaggle_dataset(file_path: str, batch_size: int = 1000) -> Generator[List[Dict[str, Any]], None, None]:
    """
    Load a Kaggle dataset in JSONL format and yield batches of documents
    
    Args:
        file_path: Path to the JSONL file
        batch_size: Number of documents to process in each batch
        
    Yields:
        List of documents in each batch
    """
    logger.info(f"Loading Kaggle dataset from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        return
    
    # Get total number of lines for progress tracking
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
    logger.info(f"Total lines in dataset: {total_lines}")
    
    # Initialize variables
    current_batch = []
    processed_lines = 0
    
    # Open the file and process it line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        # Use a more robust approach to handle JSONL with potential newlines
        buffer = ""
        for line in f:
            processed_lines += 1
            
            # Add the line to the buffer
            buffer += line
            
            # Try to parse the buffer as JSON
            try:
                # Try to parse the current buffer
                doc = json.loads(buffer)
                
                # Process the document to ensure required fields
                processed_doc = {
                    "id": doc.get("id", f"doc_{processed_lines}"),
                    "content": "",
                    "metadata": doc.get("metadata", {})
                }
                
                # Extract content from various possible fields
                if "content" in doc:
                    processed_doc["content"] = doc["content"]
                elif "abstract" in doc:
                    processed_doc["content"] = doc["abstract"]
                elif "text" in doc:
                    processed_doc["content"] = doc["text"]
                else:
                    # Try to find a text field
                    text_fields = [k for k, v in doc.items() if isinstance(v, str) and len(v) > 50]
                    if text_fields:
                        processed_doc["content"] = doc[text_fields[0]]
                    else:
                        logger.warning(f"Document {processed_doc['id']} has no content, skipping")
                        buffer = ""
                        continue
                
                # If successful, add to batch and reset buffer
                current_batch.append(processed_doc)
                buffer = ""
                
                # If batch is full, yield it
                if len(current_batch) >= batch_size:
                    logger.info(f"Yielding batch of {len(current_batch)} documents ({processed_lines}/{total_lines} lines processed)")
                    yield current_batch
                    current_batch = []
            
            except json.JSONDecodeError:
                # If we can't parse the buffer yet, continue adding lines
                # This handles cases where a JSON object spans multiple lines
                continue
    
    # Yield any remaining documents
    if current_batch:
        logger.info(f"Yielding final batch of {len(current_batch)} documents ({processed_lines}/{total_lines} lines processed)")
        yield current_batch
    
    logger.info(f"Finished loading dataset. Processed {processed_lines} lines.") 