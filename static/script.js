document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("imgUpload");
    const detectBtn = document.getElementById("detectBtn");
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const resultList = document.getElementById("resultList");

    let selectedFile = null;

    // Preview image on canvas
    fileInput.addEventListener("change", () => {
        selectedFile = fileInput.files[0];

        if (selectedFile) {
            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };
            img.src = URL.createObjectURL(selectedFile);
        }
    });

    // Send to Flask
    detectBtn.addEventListener("click", () => {
        if (!selectedFile) {
            alert("Please upload an image first!");
            return;
        }

        const formData = new FormData();
        formData.append("image", selectedFile);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            console.log(data);

            // Show result
            resultList.innerHTML = "";
            const li = document.createElement("li");
            li.textContent = data.result;
            resultList.appendChild(li);
        })
        .catch(err => {
            console.error(err);
            alert("Error processing image");
        });
    });
});