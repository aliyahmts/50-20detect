function showTab(n) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach((tab, i) => {
        tab.style.display = i === n ? 'block' : 'none';
    });
    
    // Update active tab button
    document.querySelectorAll('.tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === n);
    });

    // Auto start/stop webcam when switching tabs
    if (n === 2) {
        startWebcam();
    } else {
        stopWebcam();
    }
}

// ===================== IMAGE =====================
async function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files[0]) {
        alert("Please select an image");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const resultDiv = document.getElementById('imageResult');
    resultDiv.innerHTML = "<p>Processing...</p>";

    try {
        const response = await fetch('/detect_image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            resultDiv.innerHTML = `
                <h2>Detection Result</h2>
                <p><strong>Bill Detected:</strong> PHP ${data.total_value}</p>
                <img src="${data.annotated_image}" alt="Detection Result" style="max-width: 100%; margin-top: 15px; border: 2px solid #4CAF50;">
            `;
        } else {
            resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<p style="color:red;">Failed to connect to server.</p>`;
    }
}

// ===================== VIDEO =====================
async function uploadVideo() {
    const fileInput = document.getElementById('videoInput');
    if (!fileInput.files[0]) {
        alert("Please select a video");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const resultDiv = document.getElementById('videoResult');
    resultDiv.innerHTML = "<p>Processing video... This may take a while.</p>";

    try {
        const response = await fetch('/detect_video', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            resultDiv.innerHTML = `
                <h2>Video Processed</h2>
                <p><strong>Bill Detected:</strong> PHP ${data.total_value}</p>
                <video controls style="max-width: 100%; margin-top: 15px;">
                    <source src="${data.processed_video}" type="video/mp4">
                </video>
            `;
        } else {
            resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
        }
    } catch (err) {
        resultDiv.innerHTML = `<p style="color:red;">Failed to process video.</p>`;
    }
}

// ===================== WEBCAM =====================
function startWebcam() {
    document.getElementById('webcamStream').src = '/video_feed';
}

function stopWebcam() {
    const stream = document.getElementById('webcamStream');
    stream.src = '';
}

// Initialize: Show Image tab by default
showTab(0);