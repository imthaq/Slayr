document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const loadingSection = document.getElementById('loading-section');
    const resultsSection = document.getElementById('results-section');
    const faceCanvas = document.getElementById('face-canvas');
    const downloadBtn = document.getElementById('download-btn');

    // Drag and Drop Events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('bg-white');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('bg-white');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('bg-white');
        if (e.dataTransfer.files.length) {
            handleFiles(e.dataTransfer.files[0]);
        }
    });

    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length) {
            handleFiles(fileInput.files[0]);
        }
    });

    function handleFiles(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        // Show Loading
        uploadSection.classList.add('hidden');
        loadingSection.classList.remove('hidden');
        loadingSection.classList.add('flex');

        const formData = new FormData();
        formData.append('image', file);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
                location.reload();
                return;
            }
            displayResults(data, file);
        })
        .catch(err => {
            console.error(err);
            alert('An error occurred during analysis.');
            location.reload();
        });
    }

    function displayResults(data, file) {
        loadingSection.classList.add('hidden');
        loadingSection.classList.remove('flex');
        resultsSection.classList.remove('hidden');

        // Populate Text Data
        document.getElementById('result-shape').innerText = data.face_shape;
        document.getElementById('result-shape-desc').innerText = data.recommendations.description || ""; // Check if description was sent
        
        document.getElementById('result-undertone').innerText = data.undertone;
        document.getElementById('result-hex').innerText = data.skin_hex;
        document.getElementById('result-color-circle').style.backgroundColor = data.skin_hex;

        document.getElementById('rec-foundation').innerText = data.recommendations.foundation;

        // Populate Lists
        const fillList = (id, items) => {
            const el = document.getElementById(id);
            el.innerHTML = '';
            items.forEach(item => {
                const li = document.createElement('li');
                li.innerText = item;
                el.appendChild(li);
            });
        };

        const fillBadges = (id, items) => {
            const el = document.getElementById(id);
            el.innerHTML = '';
            items.forEach(item => {
                const span = document.createElement('span');
                span.className = "px-3 py-1 bg-gray-200 text-gray-700 rounded-full text-xs uppercase tracking-wide";
                span.innerText = item;
                el.appendChild(span);
            });
        };
        
        const fillPalette = (id, colors) => {
             const el = document.getElementById(id);
             el.innerHTML = '';
             colors.forEach(color => {
                 const div = document.createElement('div');
                 div.className = "w-8 h-8 rounded-full border border-gray-300 shadow-sm";
                 div.style.backgroundColor = color;
                 el.appendChild(div);
             });
        }

        fillList('rec-hair', data.recommendations.hair);
        fillList('rec-beard', data.recommendations.beard);
        fillBadges('rec-glasses', data.recommendations.glasses);
        fillPalette('rec-palette', data.recommendations.palette);

        // Draw Image and Mesh
        const ctx = faceCanvas.getContext('2d');
        const img = new Image();
        img.onload = () => {
            faceCanvas.width = img.width;
            faceCanvas.height = img.height;
            ctx.drawImage(img, 0, 0);

            // Draw Mesh
            ctx.fillStyle = '#00FF00'; 
            ctx.strokeStyle = 'rgba(212, 163, 115, 0.6)'; // Muted Gold
            ctx.lineWidth = 2;

            // Draw connection lines if we had a full mesh definition, 
            // but for now simpler to just draw points or specific contours.
            // Let's just draw points for simplicity and aesthetic.
            
            data.landmarks.forEach(lm => {
                const x = lm[0] * img.width;
                const y = lm[1] * img.height;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fill();
            });
        };
        img.src = URL.createObjectURL(file);
    }

    // Download PDF
    downloadBtn.addEventListener('click', () => {
        const element = document.getElementById('results-section');
        const opt = {
            margin:       0.5,
            filename:     'Slayr_Analysis.pdf',
            image:        { type: 'jpeg', quality: 0.98 },
            html2canvas:  { scale: 2 },
            jsPDF:        { unit: 'in', format: 'letter', orientation: 'portrait' }
        };
        html2pdf().set(opt).from(element).save();
    });
});
