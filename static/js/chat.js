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
    
    // Add event listeners
    chatForm.addEventListener('submit', handleChatSubmit);
    clearChatButton.addEventListener('click', clearChat);
    checkStatusButton.addEventListener('click', checkSystemStatus);
    
    // Check system status on page load
    checkSystemStatus(false);
    
    /**
     * Handle chat form submission
     * @param {Event} e - Form submit event
     */
    async function handleChatSubmit(e) {
        e.preventDefault();
        
        const userMessage = userInput.value.trim();
        if (!userMessage) return;
        
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
});
