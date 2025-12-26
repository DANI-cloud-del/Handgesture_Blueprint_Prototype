function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const statusDiv = document.getElementById('status');
    const viewerContainer = document.getElementById('viewer-container');
    
    if (!fileInput.files[0]) {
        statusDiv.className = 'error show';
        statusDiv.textContent = 'Please select a file first';
        return;
    }
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    statusDiv.textContent = 'Uploading and converting...';
    statusDiv.className = 'show';
    viewerContainer.style.display = 'none';
    
    fetch('http://127.0.0.1:5000/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        if (data.error) {
            statusDiv.className = 'error show';
            statusDiv.textContent = 'Error: ' + data.error;
        } else {
            statusDiv.className = 'success show';
            statusDiv.innerHTML = '<strong>✓ Conversion Successful!</strong><br>' + 
                                 `Found ${data.mesh_data.metadata.wall_count} walls. Rendering 3D model...`;
            
            // Show 3D viewer
            viewerContainer.style.display = 'block';
            
            // Initialize Babylon.js renderer
            initBabylonScene(data.mesh_data);
        }
    })
    .catch(error => {
        statusDiv.className = 'error show';
        statusDiv.textContent = 'Upload failed: ' + error.message;
        console.error('Full error:', error);
    });
}
