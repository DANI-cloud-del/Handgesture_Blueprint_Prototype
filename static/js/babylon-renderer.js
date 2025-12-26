// Global variables
let scene, camera, engine;

// ========== UPLOAD FUNCTION (Keep your existing one) ==========
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
                                 `Found ${data.mesh_data.metadata.wall_count} walls. Loading walkthrough experience...`;
            
            viewerContainer.style.display = 'block';
            initBabylonScene(data.mesh_data);
        }
    })
    .catch(error => {
        statusDiv.className = 'error show';
        statusDiv.textContent = 'Upload failed: ' + error.message;
        console.error('Full error:', error);
    });
}

// ========== MAIN SCENE INITIALIZATION ==========
function initBabylonScene(meshData) {
    const canvas = document.getElementById('renderCanvas');
    engine = new BABYLON.Engine(canvas, true);
    scene = new BABYLON.Scene(engine);
    
    // Sky background
    scene.clearColor = new BABYLON.Color3(0.5, 0.8, 1.0);
    
    // Create building mesh
    const building = createBuildingMesh(meshData);
    
    // DISABLE wall collisions - allow flying through walls
    building.checkCollisions = false;
    
    // Get building bounds for camera placement
    const boundingBox = building.getBoundingInfo();
    const center = boundingBox.boundingBox.center;
    const size = boundingBox.boundingBox.extendSize;
    
    console.log('Building center:', center);
    console.log('Building size:', size);
    
    // FIRST PERSON CAMERA - Start at EYE LEVEL inside or near building
    camera = new BABYLON.UniversalCamera(
        "FPSCamera",
        new BABYLON.Vector3(center.x, 1.7, center.z), // Start at CENTER at eye level (1.7m)
        scene
    );
    
    // Look forward (along Z axis)
    camera.setTarget(new BABYLON.Vector3(center.x, 1.7, center.z - 10));
    
    // IMPROVED CAMERA SETTINGS
    camera.speed = 0.3; // Controlled walking speed
    camera.angularSensibility = 1500; // Less sensitive mouse
    camera.inertia = 0.7; // Smooth movement
    
    // Enable WASD + Arrow keys
    camera.keysUp.push(87);    // W
    camera.keysDown.push(83);  // S
    camera.keysLeft.push(65);  // A
    camera.keysRight.push(68); // D
    
    // DISABLE all collision detection - free exploration
    camera.checkCollisions = false;
    camera.applyGravity = false;
    
    // NO collision system
    scene.collisionsEnabled = false;
    
    camera.attachControl(canvas, true);
    
    // ========== MOUSE WHEEL ZOOM ==========
    canvas.addEventListener('wheel', (event) => {
        event.preventDefault();
        
        const direction = camera.getDirection(BABYLON.Vector3.Forward());
        const zoomSpeed = 0.5;
        
        if (event.deltaY < 0) {
            camera.position.addInPlace(direction.scale(zoomSpeed));
        } else {
            camera.position.addInPlace(direction.scale(-zoomSpeed));
        }
    });
    
    // ========== UP/DOWN MOVEMENT (Q/E keys) ==========
    window.addEventListener('keydown', (event) => {
        if (event.key === 'q' || event.key === 'Q') {
            // Move down
            camera.position.y -= 0.3;
        } else if (event.key === 'e' || event.key === 'E') {
            // Move up
            camera.position.y += 0.3;
        }
    });
    
    // Add realistic lighting
    createRealisticLighting(scene, center);
    
    // Create floor
    createFloor(scene, boundingBox);
    
    // Create skybox
    createSkybox(scene);
    
    // Render loop
    engine.runRenderLoop(() => scene.render());
    window.addEventListener('resize', () => engine.resize());
    
    console.log('✓ Free-roam mode ready!');
    console.log('Controls:');
    console.log('  - Click + Drag to look around');
    console.log('  - WASD: Move horizontally');
    console.log('  - Q/E: Move up/down');
    console.log('  - Mouse wheel: Zoom');
    console.log('  - You can fly through walls!');
}


// ========== CREATE BUILDING MESH ==========
function createBuildingMesh(meshData) {
    console.log('Creating building mesh...');
    console.log('Vertices:', meshData.vertices.length);
    console.log('Faces:', meshData.faces.length);
    
    const customMesh = new BABYLON.Mesh("building", scene);
    
    // Flatten arrays
    const positions = [];
    for (let i = 0; i < meshData.vertices.length; i++) {
        positions.push(meshData.vertices[i][0]);
        positions.push(meshData.vertices[i][1]);
        positions.push(meshData.vertices[i][2]);
    }
    
    const indices = [];
    for (let i = 0; i < meshData.faces.length; i++) {
        indices.push(meshData.faces[i][0]);
        indices.push(meshData.faces[i][1]);
        indices.push(meshData.faces[i][2]);
    }
    
    // Create vertex data
    const vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    
    // Compute normals
    vertexData.normals = [];
    BABYLON.VertexData.ComputeNormals(positions, indices, vertexData.normals);
    
    vertexData.applyToMesh(customMesh);
    
    // Create realistic wall material
    const material = new BABYLON.StandardMaterial("wallMaterial", scene);
    material.diffuseColor = new BABYLON.Color3(0.95, 0.95, 0.95); // White walls
    material.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
    material.backFaceCulling = false;
    
    customMesh.material = material;
    
    // Enable edges for better visibility
    customMesh.enableEdgesRendering();
    customMesh.edgesWidth = 2.0;
    customMesh.edgesColor = new BABYLON.Color4(0.1, 0.1, 0.1, 1);
    
    console.log('✓ Building mesh created');
    
    return customMesh;
}

// ========== REALISTIC LIGHTING ==========
function createRealisticLighting(scene, center) {
    // Sunlight
    const sun = new BABYLON.DirectionalLight(
        "sun",
        new BABYLON.Vector3(-1, -2, -1),
        scene
    );
    sun.intensity = 1.0;
    sun.position = new BABYLON.Vector3(20, 40, 20);
    
    // Ambient light
    const ambient = new BABYLON.HemisphericLight(
        "ambient",
        new BABYLON.Vector3(0, 1, 0),
        scene
    );
    ambient.intensity = 0.6;
    ambient.groundColor = new BABYLON.Color3(0.3, 0.3, 0.3);
    
    // Shadows
    const shadowGenerator = new BABYLON.ShadowGenerator(1024, sun);
    shadowGenerator.useBlurExponentialShadowMap = true;
}

// ========== CREATE FLOOR ==========
function createFloor(scene, boundingBox) {
    const size = boundingBox.boundingBox.extendSize;
    const center = boundingBox.boundingBox.center;
    
    const floor = BABYLON.MeshBuilder.CreateGround(
        "floor",
        { width: size.x * 3, height: size.z * 3 },
        scene
    );
    
    // Position floor at ground level (y=0)
    floor.position.y = 0;
    floor.checkCollisions = false; // NO collision
    
    // Floor material
    const floorMat = new BABYLON.StandardMaterial("floorMat", scene);
    floorMat.diffuseColor = new BABYLON.Color3(0.8, 0.7, 0.6);
    floorMat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
    floor.material = floorMat;
    
    floor.receiveShadows = true;
}


// ========== CREATE SKYBOX ==========
function createSkybox(scene) {
    const skybox = BABYLON.MeshBuilder.CreateBox("skyBox", { size: 1000 }, scene);
    const skyboxMaterial = new BABYLON.StandardMaterial("skyBox", scene);
    skyboxMaterial.backFaceCulling = false;
    skyboxMaterial.diffuseColor = new BABYLON.Color3(0, 0, 0);
    skyboxMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
    skyboxMaterial.emissiveColor = new BABYLON.Color3(0.5, 0.8, 1.0);
    skybox.material = skyboxMaterial;
}

// ========== WALL EDITING ==========
function enableWallEditing(scene) {
    scene.onPointerDown = (evt, pickResult) => {
        if (pickResult.hit && pickResult.pickedMesh.name === 'building') {
            showWallEditMenu(pickResult.pickedPoint);
        }
    };
}

function showWallEditMenu(position) {
    console.log('Wall clicked at:', position);
    // You can add UI here later for editing
}

// ========== VIEW CONTROL FUNCTIONS (Optional) ==========
// ========== VIEW CONTROL FUNCTIONS ==========
function resetCamera() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        
        // Reset to CENTER at eye level, looking forward
        camera.position = new BABYLON.Vector3(center.x, 1.7, center.z);
        camera.setTarget(new BABYLON.Vector3(center.x, 1.7, center.z - 10));
    }
}


function topView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Position camera directly above
        camera.position = new BABYLON.Vector3(center.x, size.y * 10, center.z);
        camera.setTarget(center);
    }
}

function sideView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Position camera to the side
        camera.position = new BABYLON.Vector3(center.x + size.x * 3, center.y, center.z);
        camera.setTarget(center);
    }
}

function perspectiveView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Nice 3/4 perspective view
        camera.position = new BABYLON.Vector3(
            center.x + size.x * 2,
            size.y * 3,
            center.z + size.z * 2
        );
        camera.setTarget(center);
    }
}

// ========== SPEED CONTROL ==========
function adjustSpeed(direction) {
    if (!camera) return;
    
    if (direction === 'slower') {
        camera.speed = Math.max(0.1, camera.speed - 0.1);
    } else if (direction === 'faster') {
        camera.speed = Math.min(2.0, camera.speed + 0.1);
    }
    
    // Update display
    const speedDisplay = document.getElementById('speed-display');
    if (speedDisplay) {
        speedDisplay.textContent = `Speed: ${camera.speed.toFixed(1)}x`;
    }
    
    console.log('Movement speed:', camera.speed);
}

// ========== VIEW CONTROL FUNCTIONS ==========
function resetCamera() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        camera.position = new BABYLON.Vector3(center.x, 1.7, center.z + size.z * 2);
        camera.setTarget(center);
    }
}

function topView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Position camera directly above
        camera.position = new BABYLON.Vector3(center.x, Math.max(size.y * 10, 50), center.z);
        camera.setTarget(center);
    }
}

function sideView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Position camera to the side
        camera.position = new BABYLON.Vector3(center.x + size.x * 3, center.y + 10, center.z);
        camera.setTarget(center);
    }
}

function perspectiveView() {
    if (!camera || !scene) return;
    
    const building = scene.getMeshByName('building');
    if (building) {
        const boundingBox = building.getBoundingInfo();
        const center = boundingBox.boundingBox.center;
        const size = boundingBox.boundingBox.extendSize;
        
        // Nice 3/4 perspective view
        camera.position = new BABYLON.Vector3(
            center.x + size.x * 2,
            Math.max(size.y * 3, 15),
            center.z + size.z * 2
        );
        camera.setTarget(center);
    }
}
