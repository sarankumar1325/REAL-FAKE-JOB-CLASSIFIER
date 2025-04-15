document.getElementById('jobForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    // Get form data
    const formData = {
        title: document.getElementById('title').value.trim(),
        company_profile: document.getElementById('company_profile').value.trim(),
        description: document.getElementById('description').value.trim(),
        requirements: document.getElementById('requirements').value.trim(),
        benefits: document.getElementById('benefits').value.trim()
    };

    // Validate input
    for (const [key, value] of Object.entries(formData)) {
        if (!value) {
            alert(`Please fill in the ${key.replace('_', ' ')} field.`);
            return;
        }
    }

    // Get UI elements
    const button = e.target.querySelector('button');
    const resultDiv = document.getElementById('result');
    const messageElement = document.getElementById('prediction-message');

    try {
        // Show loading state
        button.textContent = 'Analyzing...';
        button.disabled = true;
        resultDiv.style.display = 'none';
        resultDiv.className = ''; // Reset classes

        // Send request to backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(formData)
        });

        // Get response text first
        const responseText = await response.text();

        // Try to parse as JSON
        let result;
        try {
            result = JSON.parse(responseText);
        } catch (e) {
            console.error('Failed to parse JSON:', responseText);
            throw new Error('Invalid response from server');
        }

        // Check for error response
        if (!response.ok) {
            throw new Error(result.detail || 'Server error occurred');
        }

        // Display result
        resultDiv.className = result.is_fake ? 'fake' : 'legitimate';
        messageElement.textContent = result.message;
        
        if (result.confidence) {
            messageElement.textContent += ` (Confidence: ${(result.confidence * 100).toFixed(1)}%)`;
        }
        
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

    } catch (error) {
        console.error('Error:', error);
        
        // Display error message
        resultDiv.className = 'error';
        messageElement.textContent = `Error: ${error.message}. Please try again or contact support if the problem persists.`;
        resultDiv.style.display = 'block';
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } finally {
        // Restore button state
        button.textContent = 'Analyze Job Posting';
        button.disabled = false;
    }
}); 