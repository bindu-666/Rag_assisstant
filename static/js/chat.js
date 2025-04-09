// RAG Chatbot UI JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const sourcesContainer = document.getElementById('sources-container');
    const clearChatButton = document.getElementById('clearChat');
    const checkStatusButton = document.getElementById('checkStatus');
    const statusModal = new bootstrap.Modal(document.getElementById('statusModal'));
    const statusModalBody = document.getElementById('statusModalBody');
    const sendButton = document.getElementById('send-button');
    const datasetForm = document.getElementById('dataset-form');
    const indexingProgress = document.getElementById('indexing-progress');
    const progressBar = indexingProgress.querySelector('.progress-bar');
    
    // State
    let isProcessing = false;
    let statusCheckInterval = null;
    
    // Add event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    clearChatButton.addEventListener('click', clearChat);
    checkStatusButton.addEventListener('click', checkSystemStatus);
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            handleSendMessage();
        }
    });
    datasetForm.addEventListener('submit', handleDatasetLoad);
    
    // Check system status on page load
    checkSystemStatus(false);
    
    // Initialize status check
    checkStatus();
    statusCheckInterval = setInterval(checkStatus, 5000);
    
    /**
     * Handle chat form submission
     * @param {Event} e - Form submit event
     */
    async function handleChatSubmit(e) {
        e.preventDefault();
        
        const userMessage = userInput.value.trim();
        if (!userMessage || isProcessing) return;
        
        // Display user message
        appendMessage('user', userMessage);
        
        // Clear input
        userInput.value = '';
        
        // Show typing indicator
        const typingIndicator = appendTypingIndicator();
        
        try {
            // Send request to server
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: userMessage })
            });
            
            // Remove typing indicator
            typingIndicator.remove();
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Display bot response
            appendMessage('bot', data.response);
            
            // Display sources
            displaySources(data.sources);
            
        } catch (error) {
            // Remove typing indicator
            typingIndicator.remove();
            
            console.error('Error:', error);
            appendErrorMessage('Sorry, there was an error processing your request. Please try again.');
        }
    }
    
    /**
     * Append a message to the chat container
     * @param {string} type - Message type ('user', 'bot', or 'system')
     * @param {string} text - Message text
     */
    function appendMessage(type, text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        // Split text by newlines and create paragraph elements
        const paragraphs = text.split('\n').filter(p => p.trim() !== '');
        paragraphs.forEach(paragraph => {
            const p = document.createElement('p');
            p.textContent = paragraph;
            p.className = 'mb-1';
            messageContent.appendChild(p);
        });
        
        messageDiv.appendChild(messageContent);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    /**
     * Append typing indicator
     * @returns {HTMLElement} The typing indicator element
     */
    function appendTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'bot-message';
        
        const typingContent = document.createElement('div');
        typingContent.className = 'typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            typingContent.appendChild(dot);
        }
        
        typingDiv.appendChild(typingContent);
        chatMessages.appendChild(typingDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return typingDiv;
    }
    
    /**
     * Append error message
     * @param {string} text - Error message text
     */
    function appendErrorMessage(text) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = text;
        
        chatMessages.appendChild(errorDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    /**
     * Display sources in the sources container
     * @param {Array} sources - Array of source objects
     */
    function displaySources(sources) {
        sourcesContainer.innerHTML = '';
        
        if (!sources || sources.length === 0) {
            const noSources = document.createElement('p');
            noSources.className = 'text-muted';
            noSources.textContent = 'No sources available for this response.';
            sourcesContainer.appendChild(noSources);
            return;
        }
        
        sources.forEach(source => {
            const sourceCard = document.createElement('div');
            sourceCard.className = 'source-card card p-3 mb-2';
            
            // Source header with title and score
            const sourceHeader = document.createElement('div');
            sourceHeader.className = 'd-flex justify-content-between align-items-center';
            
            const sourceTitle = document.createElement('h6');
            sourceTitle.className = 'mb-0';
            sourceTitle.textContent = source.title || 'Untitled Source';
            
            const scoreSpan = document.createElement('span');
            scoreSpan.className = 'source-score badge bg-info';
            scoreSpan.textContent = `Score: ${source.score.toFixed(2)}`;
            
            sourceHeader.appendChild(sourceTitle);
            sourceHeader.appendChild(scoreSpan);
            
            // Source excerpt
            const sourceExcerpt = document.createElement('div');
            sourceExcerpt.className = 'source-excerpt mt-2';
            sourceExcerpt.textContent = source.excerpt || 'No excerpt available';
            
            // Append all elements
            sourceCard.appendChild(sourceHeader);
            sourceCard.appendChild(sourceExcerpt);
            
            sourcesContainer.appendChild(sourceCard);
        });
    }
    
    /**
     * Clear chat messages and sources
     */
    function clearChat() {
        // Clear chat messages except for the first system message
        while (chatMessages.children.length > 1) {
            chatMessages.removeChild(chatMessages.lastChild);
        }
        
        // Clear sources
        sourcesContainer.innerHTML = '<p class="text-muted">Sources for retrieved information will appear here.</p>';
    }
    
    /**
     * Check system status
     * @param {boolean} showModal - Whether to show the status modal
     */
    async function checkSystemStatus(showModal = true) {
        if (showModal) {
            statusModalBody.innerHTML = `
                <div class="d-flex justify-content-center">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </div>
                </div>
                <p class="text-center mt-3">Checking system status...</p>
            `;
            statusModal.show();
        }
        
        try {
            const response = await fetch('/api/status');
            
            if (!response.ok) {
                throw new Error(`Server responded with ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            // Update status indicator
            const statusIndicator = document.getElementById('status-indicator');
            statusIndicator.className = 'status-indicator mb-3';
            
            if (data.status === 'operational') {
                statusIndicator.innerHTML = `
                    <span class="badge bg-success">
                        <i class="fas fa-check-circle me-1"></i> System Ready
                    </span>
                `;
            } else {
                statusIndicator.innerHTML = `
                    <span class="badge bg-warning">
                        <i class="fas fa-exclamation-triangle me-1"></i> System Degraded
                    </span>
                `;
            }
            
            // Update modal if shown
            if (showModal) {
                let componentsHtml = '';
                
                for (const [component, status] of Object.entries(data.components)) {
                    const iconClass = status ? 'fa-check-circle text-success' : 'fa-times-circle text-danger';
                    const statusText = status ? 'Operational' : 'Not Available';
                    
                    componentsHtml += `
                        <div class="d-flex justify-content-between align-items-center mb-2">
                            <span>${formatComponentName(component)}</span>
                            <span>
                                <i class="fas ${iconClass} me-1"></i> ${statusText}
                            </span>
                        </div>
                    `;
                }
                
                statusModalBody.innerHTML = `
                    <div class="text-center mb-3">
                        <h5 class="mb-1">Overall Status: ${data.status === 'operational' ? 
                            '<span class="text-success">Operational</span>' : 
                            '<span class="text-warning">Degraded</span>'}</h5>
                    </div>
                    <div class="card">
                        <div class="card-header">
                            <h6 class="mb-0">Component Status</h6>
                        </div>
                        <div class="card-body">
                            ${componentsHtml}
                        </div>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('Error checking status:', error);
            
            // Update status indicator
            const statusIndicator = document.getElementById('status-indicator');
            statusIndicator.className = 'status-indicator mb-3';
            statusIndicator.innerHTML = `
                <span class="badge bg-danger">
                    <i class="fas fa-times-circle me-1"></i> System Error
                </span>
            `;
            
            // Update modal if shown
            if (showModal) {
                statusModalBody.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-circle me-2"></i>
                        Error checking system status. Please try again later.
                    </div>
                `;
            }
        }
    }
    
    /**
     * Format component name for display
     * @param {string} name - Component name
     * @returns {string} Formatted name
     */
    function formatComponentName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Handle sending messages
    async function handleSendMessage() {
        const message = userInput.value.trim();
        if (!message || isProcessing) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        userInput.value = '';
        isProcessing = true;
        
        try {
            // Send message to backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: message })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Add bot response to chat
                addMessage(data.response, 'bot', data.sources);
            } else {
                // Show error message
                addMessage(`Error: ${data.error}`, 'error');
            }
        } catch (error) {
            console.error('Error sending message:', error);
            addMessage('Error: Failed to send message. Please try again.', 'error');
        } finally {
            isProcessing = false;
        }
    }

    // Handle dataset loading
    async function handleDatasetLoad(event) {
        event.preventDefault();
        
        const datasetPath = document.getElementById('dataset-path').value.trim();
        if (!datasetPath) {
            showError('Please enter a dataset path');
            return;
        }
        
        try {
            // Show progress bar
            indexingProgress.classList.remove('d-none');
            progressBar.style.width = '0%';
            progressBar.textContent = 'Starting...';
            
            // Send request to backend
            const response = await fetch('/api/index_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ dataset_path: datasetPath })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                // Show success message
                progressBar.style.width = '100%';
                progressBar.textContent = 'Complete!';
                setTimeout(() => {
                    indexingProgress.classList.add('d-none');
                    showSuccess('Dataset loaded successfully!');
                }, 2000);
            } else {
                // Show error message
                indexingProgress.classList.add('d-none');
                showError(data.error);
            }
        } catch (error) {
            console.error('Error loading dataset:', error);
            indexingProgress.classList.add('d-none');
            showError('Failed to load dataset. Please try again.');
        }
    }

    // Check system status
    async function checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (response.ok) {
                updateStatusDisplay(data);
                updateIndexingProgress(data.indexing_progress);
            } else {
                console.error('Error checking status:', data.error);
            }
        } catch (error) {
            console.error('Error checking status:', error);
        }
    }

    // Update status display
    function updateStatusDisplay(status) {
        const statusContent = document.getElementById('status-content');
        
        let html = '<div class="list-group">';
        
        // Embedding Generator Status
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-brain me-2"></i>
                        Embedding Generator
                    </div>
                    <span class="badge ${status.embedding_generator ? 'bg-success' : 'bg-danger'}">
                        ${status.embedding_generator ? 'Ready' : 'Not Ready'}
                    </span>
                </div>
            </div>
        `;
        
        // Pinecone Manager Status
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-database me-2"></i>
                        Vector Store
                    </div>
                    <span class="badge ${status.pinecone_manager ? 'bg-success' : 'bg-warning'}">
                        ${status.pinecone_manager ? 'Connected' : 'Using Memory Store'}
                    </span>
                </div>
            </div>
        `;
        
        // Index Status
        html += `
            <div class="list-group-item">
                <div class="d-flex justify-content-between align-items-center">
                    <div>
                        <i class="fas fa-box me-2"></i>
                        Knowledge Base
                    </div>
                    <span class="badge ${status.index_populated ? 'bg-success' : 'bg-warning'}">
                        ${status.index_populated ? 'Populated' : 'Empty'}
                    </span>
                </div>
            </div>
        `;
        
        html += '</div>';
        statusContent.innerHTML = html;
    }

    // Update indexing progress
    function updateIndexingProgress(progress) {
        if (progress.is_indexing) {
            indexingProgress.classList.remove('d-none');
            const percentage = (progress.processed_documents / progress.total_documents) * 100;
            progressBar.style.width = `${percentage}%`;
            progressBar.textContent = `${progress.processed_documents.toLocaleString()} / ${progress.total_documents.toLocaleString()} documents`;
        } else if (progress.processed_documents > 0) {
            progressBar.style.width = '100%';
            progressBar.textContent = 'Complete!';
            setTimeout(() => {
                indexingProgress.classList.add('d-none');
            }, 2000);
        }
    }

    // Add message to chat
    function addMessage(message, type, sources = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}-message`;
        
        let html = `
            <div class="message-content">
                <div class="message-text">${escapeHtml(message)}</div>
        `;
        
        if (sources && sources.length > 0) {
            html += `
                <div class="sources">
                    <div class="sources-header">Sources:</div>
                    <div class="sources-list">
            `;
            
            sources.forEach(source => {
                html += `
                    <div class="source-item">
                        <div class="source-title">${escapeHtml(source.title || 'Untitled')}</div>
                        <div class="source-content">${escapeHtml(source.content)}</div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        html += '</div>';
        messageDiv.innerHTML = html;
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Show error message
    function showError(message) {
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-danger border-0';
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-exclamation-circle me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    // Show success message
    function showSuccess(message) {
        const toast = document.createElement('div');
        toast.className = 'toast align-items-center text-white bg-success border-0';
        toast.setAttribute('role', 'alert');
        toast.setAttribute('aria-live', 'assertive');
        toast.setAttribute('aria-atomic', 'true');
        
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-check-circle me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        document.body.appendChild(toast);
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    // Escape HTML to prevent XSS
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Clean up on page unload
    window.addEventListener('beforeunload', () => {
        if (statusCheckInterval) {
            clearInterval(statusCheckInterval);
        }
    });
});
