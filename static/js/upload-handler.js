class UploadPortal {
    constructor(config) {
        this.zoneId = config.zoneId;
        this.inputId = config.inputId;
        this.previewId = config.previewId;
        this.artElementId = config.artElementId || null;
        this.laserElementId = config.laserElementId || null;
        this.isMain = config.isMain || false;
        this.onUpload = config.onUpload || null;
        this.onFileSelected = config.onFileSelected || null;
        
        this.init();
    }
    
    init() {
        const zone = document.getElementById(this.zoneId);
        const input = document.getElementById(this.inputId);
        const preview = document.getElementById(this.previewId);
        
        if (!zone || !input || !preview) {
            console.error('UploadPortal: Missing required elements', { 
                zone: !!zone, 
                input: !!input, 
                preview: !!preview 
            });
            return;
        }
        
        // Click to upload
        zone.onclick = () => input.click();
        
        // File change handler
        input.onchange = () => {
            if (input.files.length > 0) {
                this.handleFile(input.files[0], preview);
            }
        };
        
        // Drag and drop support
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });
        
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                this.handleFile(e.dataTransfer.files[0], preview);
            }
        });
    }
    
    handleFile(file, preview) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        if (!allowedTypes.includes(file.type)) {
            if (window.showError) {
                window.showError("Format error: Please upload a valid PNG or JPG image.");
            }
            console.error('UploadPortal: Invalid file type', file.type);
            return;
        }

        // Size Check
        if (file.size === 0) {
            if (window.showError) window.showError("File is empty");
            return;
        }
        if (file.size > 5 * 1024 * 1024) {
            if (window.showError) window.showError("File size exceeds 5MB limit");
            return;
        }
        
        const reader = new FileReader();
        reader.onload = (e) => {
            const imgCheck = new Image();
            imgCheck.onload = () => {
                // Resolution Check
                if (imgCheck.naturalWidth < 200 || imgCheck.naturalHeight < 200) {
                    if (window.showError) window.showError("Image resolution too low for accurate analysis");
                    return;
                }
                if (imgCheck.naturalWidth > 4000 || imgCheck.naturalHeight > 4000) {
                    if (window.showError) window.showError("Image resolution too high. Please resize.");
                    return;
                }

                // Clear any existing errors first (both dynamic and server-flashed)
                const dynamicError = document.getElementById('dynamic-error-container');
                if (dynamicError) dynamicError.classList.add('hidden');
                
                const serverFlash = document.getElementById('server-flash-messages');
                if (serverFlash) serverFlash.classList.add('hidden');
                
                preview.src = e.target.result;
                preview.classList.remove('hidden');
                
                // Hide art element if exists and isMain
                if (this.isMain && this.artElementId) {
                    const artEl = document.getElementById(this.artElementId);
                    if (artEl) artEl.style.opacity = '0';
                }
                
                // Show laser element if exists and isMain
                if (this.isMain && this.laserElementId) {
                    const laserEl = document.getElementById(this.laserElementId);
                    if (laserEl) {
                        laserEl.style.opacity = '1';
                        laserEl.classList.add('scanning'); // Ensure animation class is active
                    }
                }
                
                // Custom callback
                if (this.onFileSelected) {
                    this.onFileSelected(file);
                }
                
                // Notify Sila
                if (typeof updateSila !== 'undefined' && this.onUpload) {
                    this.onUpload();
                }
            };
            imgCheck.src = e.target.result;
        };
        
        reader.readAsDataURL(file);
    }

    reset() {
        const preview = document.getElementById(this.previewId);
        const artEl = this.artElementId ? document.getElementById(this.artElementId) : null;
        const laserEl = this.laserElementId ? document.getElementById(this.laserElementId) : null;

        if (preview) preview.classList.add('hidden');
        if (artEl) artEl.style.opacity = '1';
        if (laserEl) {
            laserEl.style.opacity = '0';
            laserEl.classList.remove('scanning');
        }
    }
}

function initializeUploadPortal(config) {
    return new UploadPortal(config);
}

class MultiUploadPortal {
    constructor(config) {
        this.zoneId = config.zoneId;
        this.inputId = config.inputId;
        this.onFilesSelected = config.onFilesSelected;
        
        this.init();
    }
    
    init() {
        const zone = document.getElementById(this.zoneId);
        const input = document.getElementById(this.inputId);
        
        if (!zone || !input) {
            console.error('MultiUploadPortal: Missing required elements');
            return;
        }
        
        zone.onclick = () => input.click();
        
        input.onchange = (e) => {
            if (input.files.length > 0) {
                this.handleFiles(input.files);
            }
        };
        
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });
        
        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });
        
        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });
    }
    
    handleFiles(files) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/jpg'];
        const validFiles = Array.from(files).filter(f => allowedTypes.includes(f.type));
        
        if (validFiles.length === 0) {
            if (typeof updateSila !== 'undefined') {
                updateSila("⚠️ <span class='text-red-400'>Format error: Please upload a valid PNG or JPG image.</span>");
            }
            console.error('MultiUploadPortal: No valid image files');
            return;
        }

        // Size Check
        for (const file of validFiles) {
            if (file.size === 0) {
                if (window.showError) window.showError("File is empty");
                return;
            }
            if (file.size > 5 * 1024 * 1024) {
                if (window.showError) window.showError("File size exceeds 5MB limit");
                return;
            }
        }
        
        if (this.onFilesSelected) {
            this.onFilesSelected(validFiles);
        }
    }
}
