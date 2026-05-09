function showTab(n) {
    document.querySelectorAll('.tab-content').forEach((tab, i) => {
        tab.style.display = i === n ? 'block' : 'none';
    });
    
    document.querySelectorAll('.tab-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === n);
    });

    if (n === 2) {
        startWebcam();
    } else {
        stopWebcam();
    }
}

// ===================== IMAGE PREVIEW + MULTIPLE UPLOAD =====================
document.getElementById('imageInput').addEventListener('change', function() {
    const previewArea = document.getElementById('previewArea');
    previewArea.innerHTML = '<h3>Selected Images:</h3>';

    const files = this.files;
    
    Array.from(files).forEach((file, index) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            const div = document.createElement('div');
            div.className = 'preview-item';
            div.innerHTML = `
                <img src="${e.target.result}" alt="${file.name}">
                <small>${file.name}</small>
            `;
            previewArea.appendChild(div);
        };
        reader.readAsDataURL(file);
    });
});

async function uploadImages() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files.length) {
        alert("Please select at least one image");
        return;
    }

    const formData = new FormData();
    for (let file of fileInput.files) {
        formData.append('files', file);   // 'files' must match backend
    }

    const resultDiv = document.getElementById('imageResult');
    resultDiv.innerHTML = `<p>Processing ${fileInput.files.length} image(s)... Please wait.</p>`;

    try {
        const response = await fetch('/detect_image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            let html = `
                <h2>Detection Complete</h2>
                <p><strong>Total Value:</strong> PHP ${data.total_value}</p>
                <p><strong>50 Peso Bills:</strong> ${data.count_50} | 
                   <strong>20 Peso Bills:</strong> ${data.count_20}</p>
                <hr>
            `;

            if (data.images && data.images.length > 0) {
                data.images.forEach(src => {
                    html += `
                        <div style="margin: 20px 0;">
                            <img src="${src}" style="max-width: 100%; border: 3px solid #4CAF50; border-radius: 8px;">
                        </div>
                    `;
                });
            }

            resultDiv.innerHTML = html;
        } else {
            resultDiv.innerHTML = `<p style="color:red;">Error: ${data.error || 'Unknown error'}</p>`;
        }
    } catch (err) {
        console.error(err);
        resultDiv.innerHTML = `<p style="color:red;">Failed to connect to server. Check console.</p>`;
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
            const total = data.total_value || 0;
            resultDiv.innerHTML = `
                <h2>Video Processed</h2>
                <p><strong>Total Value:</strong> PHP ${total}</p>
                <p>50 Peso: ${data.count_50} | 20 Peso: ${data.count_20}</p>
                <video controls style="max-width: 100%; margin-top: 15px; border: 3px solid #4CAF50;">
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
    document.getElementById('webcamStream').src = '';
}

// Initialize
showTab(0);