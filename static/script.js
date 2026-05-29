document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const spinner = submitBtn.querySelector('.spinner');
    
    const resultCard = document.getElementById('result-card');
    const emptyState = document.getElementById('empty-state');
    const resultTime = document.getElementById('result-time');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceFill = document.getElementById('confidence-fill');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        btnText.classList.add('hidden');
        spinner.classList.remove('hidden');
        submitBtn.disabled = true;
        
        // Reset result animations if shown
        resultCard.classList.add('hidden');
        confidenceFill.style.width = '0%';
        
        const formData = new FormData(form);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok && data.success) {
                // Hide empty state, show result
                emptyState.classList.add('hidden');
                resultCard.classList.remove('hidden');
                
                // Animate numbers
                animateValue(resultTime, 0, data.predicted_time_min, 1500);
                
                // Update confidence
                setTimeout(() => {
                    resultConfidence.textContent = `${data.confidence_percent}%`;
                    confidenceFill.style.width = `${data.confidence_percent}%`;
                }, 500);
                
            } else {
                alert(data.error || 'Failed to generate prediction');
            }
        } catch (error) {
            console.error('Error:', error);
            alert('A network error occurred. Please try again.');
        } finally {
            // Reset button state
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });
    
    function animateValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            
            // Ease out cubic
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            
            obj.innerHTML = Math.floor(easeProgress * (end - start) + start);
            if (progress < 1) {
                window.requestAnimationFrame(step);
            }
        };
        window.requestAnimationFrame(step);
    }
});
