"""
Module for monitoring system memory usage
"""
import psutil
import logging

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Class for monitoring system memory usage and adjusting batch sizes"""
    
    def __init__(self, target_usage=75.0, min_batch_size=100, max_batch_size=2000):
        """
        Initialize the memory monitor
        
        Args:
            target_usage: Target memory usage percentage (default: 75.0)
            min_batch_size: Minimum batch size (default: 100)
            max_batch_size: Maximum batch size (default: 2000)
        """
        self.target_usage = target_usage
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.logger = logging.getLogger(__name__)
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent
    
    def should_pause(self):
        """Check if processing should pause due to high memory usage"""
        current_usage = self.get_memory_usage()
        should_pause = current_usage > self.target_usage
        
        if should_pause:
            self.logger.warning(f"Memory usage is high: {current_usage}%")
        
        return should_pause
    
    def get_recommended_batch_size(self, current_batch_size):
        """
        Get recommended batch size based on current memory usage
        
        Args:
            current_batch_size: Current batch size
            
        Returns:
            Recommended batch size
        """
        current_usage = self.get_memory_usage()
        
        if current_usage > self.target_usage + 10:
            # Reduce batch size by 25% if memory usage is very high
            new_size = max(self.min_batch_size, int(current_batch_size * 0.75))
            self.logger.info(f"Memory usage is very high ({current_usage}%). Reducing batch size to {new_size}")
            return new_size
        elif current_usage < self.target_usage - 10:
            # Increase batch size by 25% if memory usage is very low
            new_size = min(self.max_batch_size, int(current_batch_size * 1.25))
            self.logger.info(f"Memory usage is very low ({current_usage}%). Increasing batch size to {new_size}")
            return new_size
        
        return current_batch_size 