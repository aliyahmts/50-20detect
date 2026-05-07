document.getElementById('uploadForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    if (!fileInput.files[0]) return;

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    const response = await fetch('/detect', {
        method: 'POST',
        body: formData
    });

    const data = await response.json();
    const resultsDiv = document.getElementById('results');

    if (data.success) {
        resultsDiv.innerHTML = `
            <h2>Results</h2>
            <p>50 Peso Bills: <strong>${data.count_50}</strong></p>
            <p>20 Peso Bills: <strong>${data.count_20}</strong></p>
            <p><strong>Total: PHP ${data.total_value}</strong></p>
            <img src="${data.annotated_image}" alt="Detection" style="max-width: 600px;">
        `;
    } else {
        resultsDiv.innerHTML = `<p style="color:red;">Error: ${data.error}</p>`;
    }
});