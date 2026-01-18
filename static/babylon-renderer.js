// Enhanced Babylon.js 3D Renderer with Controls

// Global variables
let scene, camera, engine;
let wallMesh, floorGrid, axesHelper;
let hemisphericLight, directionalLight;

function initBabylonScene(meshData) {
    const canvas = document.getElementById('renderCanvas');
    engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true, stencil: true });
    
    // Create scene
    scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.95, 0.95, 0.97, 1.0);
    
    // Create camera - ArcRotateCamera for better control
    camera = new BABYLON.ArcRotateCamera(
        "camera",
        Math.PI / 4,
        Math.PI / 3,
        30,
        BABYLON.Vector3.Zero(),
        scene
    );
    camera.attachControl(canvas, true);
    camera.lowerRadiusLimit = 5;
    camera.upperRadiusLimit = 100;
    camera.wheelPrecision = 50;
    camera.panningSensibility = 100;
    
    // Lighting
    hemisphericLight = new BABYLON.HemisphericLight(
        "hemiLight",
        new BABYLON.Vector3(0, 1, 0),
        scene
    );
    hemisphericLight.intensity = 1.0;
    hemisphericLight.diffuse = new BABYLON.Color3(1, 1, 1);
    hemisphericLight.specular = new BABYLON.Color3(0.3, 0.3, 0.3);
    hemisphericLight.groundColor = new BABYLON.Color3(0.5, 0.5, 0.5);
    
    // Additional directional light for better shadows
    directionalLight = new BABYLON.DirectionalLight(
        "dirLight",
        new BABYLON.Vector3(-1, -2, -1),
        scene
    );
    directionalLight.position = new BABYLON.Vector3(20, 40, 20);
    directionalLight.intensity = 0.5;
    
    // Create materials
    const wallMaterial = new BABYLON.StandardMaterial("wallMat", scene);
    wallMaterial.diffuseColor = new BABYLON.Color3(0.8, 0.8, 0.8);
    wallMaterial.specularColor = new BABYLON.Color3(0.2, 0.2, 0.2);
    wallMaterial.ambientColor = new BABYLON.Color3(0.3, 0.3, 0.3);
    wallMaterial.backFaceCulling = false;
    
    // Create wall mesh
    wallMesh = new BABYLON.Mesh("walls", scene);
    
    // Convert mesh data format
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
    
    const vertexData = new BABYLON.VertexData();
    vertexData.positions = positions;
    vertexData.indices = indices;
    
    // Compute normals
    const normals = [];
    BABYLON.VertexData.ComputeNormals(positions, indices, normals);
    vertexData.normals = normals;
    
    vertexData.applyToMesh(wallMesh);
    wallMesh.material = wallMaterial;
    
    // Create floor grid
    floorGrid = BABYLON.MeshBuilder.CreateGround(
        "floor",
        { width: 100, height: 100, subdivisions: 50 },
        scene
    );
    
    const floorMaterial = new BABYLON.StandardMaterial("floorMat", scene);
    floorMaterial.diffuseColor = new BABYLON.Color3(0.9, 0.9, 0.95);
    floorMaterial.specularColor = new BABYLON.Color3(0, 0, 0);
    floorMaterial.wireframe = true;
    floorMaterial.alpha = 0.3;
    floorGrid.material = floorMaterial;
    floorGrid.position.y = -0.01;
    
    // Create axes helper
    createAxesHelper(scene);
    
    // Add edge rendering for better visibility
    wallMesh.enableEdgesRendering();
    wallMesh.edgesWidth = 2.0;
    wallMesh.edgesColor = new BABYLON.Color4(0.2, 0.2, 0.2, 1);
    
    // Optimize camera position based on model bounds
    const boundingInfo = wallMesh.getBoundingInfo();
    const boundingBox = boundingInfo.boundingBox;
    const center = boundingBox.centerWorld;
    const extents = boundingBox.extendSizeWorld;
    const maxExtent = Math.max(extents.x, extents.y, extents.z);
    
    camera.target = center;
    camera.radius = maxExtent * 3;
    
    console.log("✓ Scene initialized");
    console.log("  - Walls: " + meshData.metadata.wall_count);
    console.log("  - Vertices: " + (positions.length / 3));
    console.log("  - Triangles: " + (indices.length / 3));
    
    // Render loop
    engine.runRenderLoop(() => {
        scene.render();
    });
    
    // Handle window resize
    window.addEventListener('resize', () => {
        engine.resize();
    });
    
    // Handle quality based on performance
    let fps = 0;
    scene.onAfterRenderObservable.add(() => {
        fps = engine.getFps();
        if (fps < 30) {
            // Reduce quality if FPS is low
            wallMesh.edgesWidth = 1.0;
        }
    });
}

function createAxesHelper(scene) {
    // Create axes helper showing X (red), Y (green), Z (blue)
    const axisSize = 5;
    
    // X axis - Red
    const axisX = BABYLON.MeshBuilder.CreateLines("axisX", {
        points: [
            BABYLON.Vector3.Zero(),
            new BABYLON.Vector3(axisSize, 0, 0)
        ]
    }, scene);
    axisX.color = new BABYLON.Color3(1, 0, 0);
    
    // Y axis - Green
    const axisY = BABYLON.MeshBuilder.CreateLines("axisY", {
        points: [
            BABYLON.Vector3.Zero(),
            new BABYLON.Vector3(0, axisSize, 0)
        ]
    }, scene);
    axisY.color = new BABYLON.Color3(0, 1, 0);
    
    // Z axis - Blue
    const axisZ = BABYLON.MeshBuilder.CreateLines("axisZ", {
        points: [
            BABYLON.Vector3.Zero(),
            new BABYLON.Vector3(0, 0, axisSize)
        ]
    }, scene);
    axisZ.color = new BABYLON.Color3(0, 0, 1);
    
    // Create labels
    const makeTextPlane = (text, color, size) => {
        const dynamicTexture = new BABYLON.DynamicTexture("DynamicTexture", 50, scene, true);
        dynamicTexture.hasAlpha = true;
        dynamicTexture.drawText(text, 5, 40, "bold 36px Arial", color, "transparent", true);
        
        const plane = BABYLON.MeshBuilder.CreatePlane("TextPlane", { size: size }, scene);
        plane.material = new BABYLON.StandardMaterial("TextPlaneMaterial", scene);
        plane.material.backFaceCulling = false;
        plane.material.specularColor = new BABYLON.Color3(0, 0, 0);
        plane.material.diffuseTexture = dynamicTexture;
        
        return plane;
    };
    
    const xChar = makeTextPlane("X", "red", 1);
    xChar.position = new BABYLON.Vector3(axisSize * 1.1, 0, 0);
    
    const yChar = makeTextPlane("Y", "green", 1);
    yChar.position = new BABYLON.Vector3(0, axisSize * 1.1, 0);
    
    const zChar = makeTextPlane("Z", "blue", 1);
    zChar.position = new BABYLON.Vector3(0, 0, axisSize * 1.1);
    
    // Group axes together
    axesHelper = new BABYLON.TransformNode("axesHelper", scene);
    axisX.parent = axesHelper;
    axisY.parent = axesHelper;
    axisZ.parent = axesHelper;
    xChar.parent = axesHelper;
    yChar.parent = axesHelper;
    zChar.parent = axesHelper;
}

// Camera view functions
function setCameraView(view) {
    if (!camera) return;
    
    const radius = camera.radius;
    
    switch(view) {
        case 'top':
            camera.alpha = 0;
            camera.beta = 0.1;
            break;
        case 'front':
            camera.alpha = 0;
            camera.beta = Math.PI / 2;
            break;
        case 'right':
            camera.alpha = Math.PI / 2;
            camera.beta = Math.PI / 2;
            break;
        case 'left':
            camera.alpha = -Math.PI / 2;
            camera.beta = Math.PI / 2;
            break;
        case 'back':
            camera.alpha = Math.PI;
            camera.beta = Math.PI / 2;
            break;
        case 'perspective':
        default:
            camera.alpha = Math.PI / 4;
            camera.beta = Math.PI / 3;
            break;
    }
}

function toggleWireframe() {
    if (!wallMesh) return;
    const btn = document.getElementById('wireframeBtn');
    wallMesh.material.wireframe = !wallMesh.material.wireframe;
    
    if (wallMesh.material.wireframe) {
        btn.classList.add('active');
        btn.querySelector('span:last-child').textContent = 'ON';
    } else {
        btn.classList.remove('active');
        btn.querySelector('span:last-child').textContent = 'OFF';
    }
}

function toggleWalls() {
    if (!wallMesh) return;
    const btn = document.getElementById('wallsBtn');
    wallMesh.isVisible = !wallMesh.isVisible;
    
    if (wallMesh.isVisible) {
        btn.classList.add('active');
        btn.querySelector('span:last-child').textContent = 'ON';
    } else {
        btn.classList.remove('active');
        btn.querySelector('span:last-child').textContent = 'OFF';
    }
}

function toggleFloor() {
    if (!floorGrid) return;
    const btn = document.getElementById('floorBtn');
    floorGrid.isVisible = !floorGrid.isVisible;
    
    if (floorGrid.isVisible) {
        btn.classList.add('active');
        btn.querySelector('span:last-child').textContent = 'ON';
    } else {
        btn.classList.remove('active');
        btn.querySelector('span:last-child').textContent = 'OFF';
    }
}

function toggleAxes() {
    if (!axesHelper) return;
    const btn = document.getElementById('axesBtn');
    axesHelper.setEnabled(!axesHelper.isEnabled());
    
    if (axesHelper.isEnabled()) {
        btn.classList.add('active');
        btn.querySelector('span:last-child').textContent = 'ON';
    } else {
        btn.classList.remove('active');
        btn.querySelector('span:last-child').textContent = 'OFF';
    }
}

function updateLighting() {
    if (!hemisphericLight) return;
    const value = document.getElementById('lightIntensity').value;
    hemisphericLight.intensity = parseFloat(value);
}

function setWallColor(color) {
    if (!wallMesh) return;
    wallMesh.material.diffuseColor = BABYLON.Color3.FromHexString(color);
}

function updateWallHeight() {
    const value = document.getElementById('wallHeight').value;
    document.getElementById('heightValue').textContent = value + 'm';
}

function updateWallThickness() {
    const value = document.getElementById('wallThickness').value;
    document.getElementById('thicknessValue').textContent = value + 'm';
}

function resetCamera() {
    if (!camera) return;
    camera.alpha = Math.PI / 4;
    camera.beta = Math.PI / 3;
    camera.radius = 30;
    camera.target = BABYLON.Vector3.Zero();
}

function downloadScreenshot() {
    if (!engine) return;
    BABYLON.Tools.CreateScreenshot(engine, camera, {precision: 2}, function(data) {
        const link = document.createElement('a');
        link.href = data;
        link.download = 'archcad-model.png';
        link.click();
    });
}
