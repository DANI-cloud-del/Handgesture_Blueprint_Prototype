import os
os.environ['DOTNET_SYSTEM_GLOBALIZATION_INVARIANT'] = '0'

from flask import Flask, request, jsonify, render_template, session, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from converters import DXFParser, PDFParser, DWGParser, MeshGenerator, ShapeClassifier, TemplateMatcher
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ezdxf
from ezdxf.addons.drawing import RenderContext, Frontend
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend
import json
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PNG_FOLDER'] = 'static/blueprints'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PNG_FOLDER'], exist_ok=True)

TEMP_DATA_FOLDER = 'temp_data'
os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

# Load ArchCAD trained model (load once at startup)
print("Loading ArchCAD detection model...")
try:
    ARCHCAD_MODEL = YOLO('models/archcad_detector/weights/best.pt')
    print("✓ ArchCAD model loaded successfully")
except Exception as e:
    print(f"⚠️  ArchCAD model not found: {e}")
    ARCHCAD_MODEL = None

# ============================================================================
# ROUTES
# ============================================================================

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/region-selector")
def region_selector():
    png_filename = session.get('png_filename', '')
    if not png_filename:
        return redirect(url_for('index'))
    return render_template('region_selector.html', png_filename=png_filename)

@app.route("/viewer")
def viewer():
    return render_template('viewer.html')

@app.route("/upload", methods=['POST'])
def upload_blueprint():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        print(f"Processing {filename}...")
        
        # Parse the file
        if file_ext == 'dxf':
            parser = DXFParser(filepath)
        elif file_ext == 'dwg':
            parser = DWGParser(filepath)
        elif file_ext == 'pdf':
            parser = PDFParser(filepath)
        else:
            return jsonify({'error': 'Unsupported format'}), 400
        
        blueprint_data = parser.parse()
        
        if not blueprint_data.get('success'):
            return jsonify({'error': blueprint_data.get('error')}), 500
        
        # Generate PNG for region selection
        png_filename = f"{filename.rsplit('.', 1)[0]}.png"
        png_path = os.path.join(app.config['PNG_FOLDER'], png_filename)
        
        # Convert to PNG
        print(f"Generating PNG: {png_path}")
        if file_ext in ['dxf', 'dwg']:
            dxf_path = filepath if file_ext == 'dxf' else f"{filepath.rsplit('.', 1)[0]}_converted.dxf"
            success = generate_png_from_dxf(dxf_path, png_path)
            
            if not success:
                return jsonify({'error': 'Failed to generate PNG preview'}), 500
        
        # Run ArchCAD model detection
        ml_detections = {}
        if ARCHCAD_MODEL is not None:
            print("Running ArchCAD ML detection...")
            ml_detections = detect_with_archcad_model(png_path)
            print(f"✓ ML Detected: {sum(len(v) for v in ml_detections.values())} objects")
        
        # Run traditional CV classification (backup/comparison)
        print("Running geometric classification...")
        classifier = ShapeClassifier()
        classified_elements = classifier.classify_from_image(png_path)
        
        # Generate visualization
        viz_filename = f"{filename.rsplit('.', 1)[0]}_classified.png"
        viz_path = os.path.join(app.config['PNG_FOLDER'], viz_filename)
        classifier.visualize_classification(png_path, viz_path)
        
        # Combine ML and CV results (prefer ML if available)
        classification_data = {
            'walls': ml_detections.get('wall', classified_elements.get('wall', [])),
            'doors': ml_detections.get('door', classified_elements.get('door', [])),
            'windows': ml_detections.get('window', classified_elements.get('window', [])),
            'furniture': {
                'tables': ml_detections.get('table', []),
                'chairs': ml_detections.get('chair', []),
                'beds': ml_detections.get('bed', []),
                'sofas': ml_detections.get('sofa', [])
            },
            'facilities': {
                'toilets': ml_detections.get('toilet', []),
                'sinks': ml_detections.get('sink', []),
                'bathtubs': ml_detections.get('bathtub', []),
                'stairs': ml_detections.get('stair', []),
                'elevators': ml_detections.get('elevator', []),
                'parking': ml_detections.get('parking', [])
            },
            'detection_method': 'ML' if ARCHCAD_MODEL else 'CV'
        }
        
        # Save to JSON
        classification_filename = f"{filename.rsplit('.', 1)[0]}_classified.json"
        classification_path = os.path.join(TEMP_DATA_FOLDER, classification_filename)
        
        with open(classification_path, 'w') as f:
            json.dump(classification_data, f)
        
        # Save blueprint data to file
        data_filename = f"{filename.rsplit('.', 1)[0]}_data.json"
        data_path = os.path.join(TEMP_DATA_FOLDER, data_filename)
        
        with open(data_path, 'w') as f:
            json.dump(blueprint_data, f)
        
        # Store filenames in session
        session['filename'] = filename
        session['png_filename'] = png_filename
        session['data_filename'] = data_filename
        session['classification_filename'] = classification_filename
        session['viz_filename'] = viz_filename
        
        print(f"✓ Classification complete")
        
        return jsonify({
            'success': True,
            'message': 'File uploaded successfully',
            'png_url': f'/static/blueprints/{png_filename}',
            'redirect': '/region-selector'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/analyze-region", methods=['POST'])
def analyze_region():
    """Analyze a specific region and return element counts"""
    try:
        data = request.json
        region = data.get('region')  # {x, y, width, height} in normalized coords
        
        if not region:
            return jsonify({'error': 'No region provided'}), 400
        
        # Load classification data
        classification_filename = session.get('classification_filename')
        if not classification_filename:
            return jsonify({'error': 'No classification data found'}), 400
        
        classification_path = os.path.join(TEMP_DATA_FOLDER, classification_filename)
        
        with open(classification_path, 'r') as f:
            classification_data = json.load(f)
        
        # Filter elements by region
        region_analysis = analyze_region_elements(classification_data, region)
        
        return jsonify({
            'success': True,
            'analysis': region_analysis
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route("/process-regions", methods=['POST'])
def process_regions():
    """Process selected regions and generate 3D model"""
    try:
        data = request.json
        selected_regions = data.get('regions', [])
        
        # Load blueprint data from file
        data_filename = session.get('data_filename')
        classification_filename = session.get('classification_filename')
        png_filename = session.get('png_filename')
        
        if not data_filename:
            return jsonify({'error': 'No blueprint data found'}), 400
        
        data_path = os.path.join(TEMP_DATA_FOLDER, data_filename)
        
        with open(data_path, 'r') as f:
            blueprint_data = json.load(f)
        
        # Load ML detections
        ml_detections = {}
        if classification_filename:
            classification_path = os.path.join(TEMP_DATA_FOLDER, classification_filename)
            with open(classification_path, 'r') as f:
                classification_data = json.load(f)
                ml_detections = {
                    'wall': classification_data.get('walls', []),
                    'door': classification_data.get('doors', []),
                    'window': classification_data.get('windows', [])
                }
        
        # Get PNG dimensions for coordinate matching
        png_path = os.path.join(app.config['PNG_FOLDER'], png_filename)
        from PIL import Image
        png_img = Image.open(png_path)
        img_width, img_height = png_img.size
        
        print(f"\n{'='*60}")
        print("PROCESSING 3D MODEL GENERATION")
        print(f"{'='*60}")
        print(f"Original walls: {len(blueprint_data.get('walls', []))}")
        print(f"ML detections: {len(ml_detections.get('wall', []))}")
        
        # OPTION 1: Use ML-filtered walls only
        if ml_detections.get('wall'):
            filtered_walls = match_dxf_walls_to_detections(
                blueprint_data.get('walls', []),
                ml_detections,
                img_width,
                img_height
            )
            blueprint_data['walls'] = filtered_walls
            print(f"✓ Using ML-filtered walls: {len(filtered_walls)}")
        
        # OPTION 2: Further filter by selected regions
        if selected_regions:
            filtered_data = filter_by_regions(blueprint_data, selected_regions)
            print(f"✓ Region-filtered walls: {len(filtered_data.get('walls', []))}")
        else:
            filtered_data = blueprint_data
        
        print(f"Final walls for 3D: {len(filtered_data.get('walls', []))}")
        print(f"{'='*60}\n")
        
        # Generate mesh
        generator = MeshGenerator(filtered_data, wall_height=3.0)
        mesh_data = generator.generate()
        
        return jsonify({
            'success': True,
            'redirect': '/viewer',
            'mesh_data': mesh_data,
            'stats': {
                'original_walls': len(blueprint_data.get('walls', [])),
                'ml_detected_walls': len(ml_detections.get('wall', [])),
                'final_walls': len(filtered_data.get('walls', []))
            }
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_with_archcad_model(image_path, conf_threshold=0.25):
    """Run ArchCAD model detection on blueprint image"""
    if ARCHCAD_MODEL is None:
        return {}
    
    try:
        results = ARCHCAD_MODEL.predict(
            image_path,
            conf=conf_threshold,
            verbose=False
        )
        
        # Organize detections by class
        detections = {}
        
        for result in results:
            img_height, img_width = result.orig_shape
            
            for box in result.boxes:
                class_name = ARCHCAD_MODEL.names[int(box.cls)]
                confidence = float(box.conf)
                
                # Get bounding box in pixel coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Calculate center and dimensions
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)
                
                detection = {
                    'bbox': [int(x1), int(y1), width, height],
                    'center': [center_x, center_y],
                    'confidence': confidence,
                    'normalized': {
                        'x': x1 / img_width,
                        'y': y1 / img_height,
                        'width': width / img_width,
                        'height': height / img_height,
                        'center_x': center_x / img_width,
                        'center_y': center_y / img_height
                    }
                }
                
                if class_name not in detections:
                    detections[class_name] = []
                
                detections[class_name].append(detection)
        
        return detections
    
    except Exception as e:
        print(f"ArchCAD detection error: {e}")
        return {}


def analyze_region_elements(classification_data, region):
    """Analyze elements within a specific region"""
    analysis = {
        'walls': 0,
        'doors': 0,
        'windows': 0,
        'tables': 0,
        'chairs': 0,
        'beds': 0,
        'sofas': 0,
        'toilets': 0,
        'sinks': 0,
        'bathtubs': 0,
        'stairs': 0,
        'elevators': 0,
        'parking': 0,
        'total': 0
    }
    
    # Check walls
    for wall in classification_data.get('walls', []):
        if is_element_in_region(wall, region):
            analysis['walls'] += 1
    
    # Check doors
    for door in classification_data.get('doors', []):
        if is_element_in_region(door, region):
            analysis['doors'] += 1
    
    # Check windows
    for window in classification_data.get('windows', []):
        if is_element_in_region(window, region):
            analysis['windows'] += 1
    
    # Check furniture
    furniture = classification_data.get('furniture', {})
    if isinstance(furniture, dict):
        for table in furniture.get('tables', []):
            if is_element_in_region(table, region):
                analysis['tables'] += 1
        
        for chair in furniture.get('chairs', []):
            if is_element_in_region(chair, region):
                analysis['chairs'] += 1
        
        for bed in furniture.get('beds', []):
            if is_element_in_region(bed, region):
                analysis['beds'] += 1
        
        for sofa in furniture.get('sofas', []):
            if is_element_in_region(sofa, region):
                analysis['sofas'] += 1
    
    # Check facilities
    facilities = classification_data.get('facilities', {})
    if isinstance(facilities, dict):
        for toilet in facilities.get('toilets', []):
            if is_element_in_region(toilet, region):
                analysis['toilets'] += 1
        
        for sink in facilities.get('sinks', []):
            if is_element_in_region(sink, region):
                analysis['sinks'] += 1
        
        for bathtub in facilities.get('bathtubs', []):
            if is_element_in_region(bathtub, region):
                analysis['bathtubs'] += 1
        
        for stair in facilities.get('stairs', []):
            if is_element_in_region(stair, region):
                analysis['stairs'] += 1
        
        for elevator in facilities.get('elevators', []):
            if is_element_in_region(elevator, region):
                analysis['elevators'] += 1
        
        for parking in facilities.get('parking', []):
            if is_element_in_region(parking, region):
                analysis['parking'] += 1
    
    analysis['total'] = sum(v for k, v in analysis.items() if k != 'total')
    
    return analysis


def is_element_in_region(element, region):
    """Check if an element's center is within a region"""
    # Get element coordinates (normalized 0-1)
    if 'normalized' in element:
        elem_x = element['normalized']['center_x']
        elem_y = element['normalized']['center_y']
    elif 'center' in element and isinstance(element['center'], list):
        # Assume already normalized if values < 1
        elem_x = element['center'][0]
        elem_y = element['center'][1]
    else:
        return False
    
    # Check if element center is within region bounds
    return (region['x'] <= elem_x <= region['x'] + region['width'] and
            region['y'] <= elem_y <= region['y'] + region['height'])


def generate_png_from_dxf(dxf_path, output_path, dpi=150):
    """Convert DXF to PNG, skipping MTEXT entities"""
    try:
        print(f"Reading DXF from: {dxf_path}")
        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()
        
        print("Filtering entities...")
        entities_to_draw = []
        for entity in msp:
            if entity.dxftype() not in ['MTEXT', 'TEXT', 'DIMENSION']:
                entities_to_draw.append(entity)
        
        print(f"Drawing {len(entities_to_draw)} entities (filtered out text)")
        
        fig = plt.figure(figsize=(15, 15), dpi=dpi)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        
        ctx = RenderContext(doc)
        out = MatplotlibBackend(ax)
        
        frontend = Frontend(ctx, out)
        for entity in entities_to_draw:
            try:
                properties = ctx.resolve_all(entity)
                frontend.draw_entity(entity, properties)
            except Exception as e:
                print(f"Skipping entity {entity.dxftype()}: {e}")
                continue
        
        out.finalize()
        
        print(f"Saving PNG to: {output_path}")
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', pad_inches=0, 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print("✓ PNG generated successfully")
        return True
    except Exception as e:
        print(f"PNG generation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def filter_by_regions(blueprint_data, regions):
    """Filter blueprint entities based on selected regions"""
    walls = blueprint_data.get('walls', [])
    
    if not walls or not regions:
        return blueprint_data
    
    all_x = []
    all_y = []
    for wall in walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    bounds = {
        'min': [min(all_x), min(all_y)],
        'max': [max(all_x), max(all_y)]
    }
    
    filtered_walls = []
    for wall in walls:
        if (is_in_selected_regions(wall['start'], regions, bounds) or 
            is_in_selected_regions(wall['end'], regions, bounds)):
            filtered_walls.append(wall)
    
    filtered_data = blueprint_data.copy()
    filtered_data['walls'] = filtered_walls
    
    return filtered_data


def is_in_selected_regions(point, regions, bounds):
    """Check if a point falls within any selected region"""
    x_norm = (point[0] - bounds['min'][0]) / (bounds['max'][0] - bounds['min'][0])
    y_norm = (point[1] - bounds['min'][1]) / (bounds['max'][1] - bounds['min'][1])
    
    for region in regions:
        if (region['x'] <= x_norm <= region['x'] + region['width'] and
            region['y'] <= y_norm <= region['y'] + region['height']):
            return True
    
    return False

def match_dxf_walls_to_detections(dxf_walls, ml_detections, image_width, image_height):
    """
    Match DXF wall segments to ML model detections.
    Only return walls that the ML model detected.
    """
    if not ml_detections or 'wall' not in ml_detections:
        print("⚠️  No wall detections from ML model, using all DXF walls")
        return dxf_walls
    
    detected_walls = ml_detections['wall']
    
    # Get DXF coordinate bounds
    all_x = []
    all_y = []
    for wall in dxf_walls:
        all_x.extend([wall['start'][0], wall['end'][0]])
        all_y.extend([wall['start'][1], wall['end'][1]])
    
    dxf_min_x, dxf_max_x = min(all_x), max(all_x)
    dxf_min_y, dxf_max_y = min(all_y), max(all_y)
    dxf_width = dxf_max_x - dxf_min_x
    dxf_height = dxf_max_y - dxf_min_y
    
    print(f"\n🔍 Matching {len(dxf_walls)} DXF walls to {len(detected_walls)} ML detections...")
    
    matched_walls = []
    
    for wall in dxf_walls:
        # Calculate wall center in DXF coordinates
        wall_center_x = (wall['start'][0] + wall['end'][0]) / 2
        wall_center_y = (wall['start'][1] + wall['end'][1]) / 2
        
        # Normalize to 0-1 space (matching ML detection coordinates)
        norm_x = (wall_center_x - dxf_min_x) / dxf_width
        norm_y = (wall_center_y - dxf_min_y) / dxf_height
        
        # Check if this wall's center falls within any ML detection bounding box
        for detection in detected_walls:
            det_bbox = detection['normalized']
            
            # Calculate detection bbox bounds
            det_x1 = det_bbox['x']
            det_y1 = det_bbox['y']
            det_x2 = det_x1 + det_bbox['width']
            det_y2 = det_y1 + det_bbox['height']
            
            # Check if wall center is inside detection bbox (with some tolerance)
            tolerance = 0.02  # 2% tolerance
            if (det_x1 - tolerance <= norm_x <= det_x2 + tolerance and
                det_y1 - tolerance <= norm_y <= det_y2 + tolerance):
                matched_walls.append(wall)
                break  # Don't match to multiple detections
    
    print(f"✓ Matched {len(matched_walls)} walls (from {len(dxf_walls)} total)")
    
    return matched_walls if matched_walls else dxf_walls

@app.route("/generate-from-ml", methods=['POST'])
def generate_from_ml():
    """Generate 3D model directly from ML detections (no region selection)"""
    try:
        # Load blueprint data
        data_filename = session.get('data_filename')
        classification_filename = session.get('classification_filename')
        png_filename = session.get('png_filename')
        
        if not data_filename or not classification_filename:
            return jsonify({'error': 'No data found'}), 400
        
        data_path = os.path.join(TEMP_DATA_FOLDER, data_filename)
        classification_path = os.path.join(TEMP_DATA_FOLDER, classification_filename)
        
        with open(data_path, 'r') as f:
            blueprint_data = json.load(f)
        
        with open(classification_path, 'r') as f:
            classification_data = json.load(f)
        
        # Get PNG dimensions
        png_path = os.path.join(app.config['PNG_FOLDER'], png_filename)
        from PIL import Image
        png_img = Image.open(png_path)
        img_width, img_height = png_img.size
        
        # Match DXF walls to ML detections
        ml_detections = {'wall': classification_data.get('walls', [])}
        
        matched_walls = match_dxf_walls_to_detections(
            blueprint_data.get('walls', []),
            ml_detections,
            img_width,
            img_height
        )
        
        # Create filtered data with only matched walls
        filtered_data = blueprint_data.copy()
        filtered_data['walls'] = matched_walls
        
        print(f"\n✓ Generating 3D from {len(matched_walls)} ML-detected walls\n")
        
        # Generate mesh
        generator = MeshGenerator(filtered_data, wall_height=3.0)
        mesh_data = generator.generate()
        
        return jsonify({
            'success': True,
            'redirect': '/viewer',
            'mesh_data': mesh_data
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route("/debug-layers")
def debug_layers():
    import ezdxf
    
    data_filename = session.get('filename')
    if not data_filename:
        return "No file uploaded"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    
    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        
        # Count entities by layer
        layer_counts = {}
        entity_types = {}
        
        for entity in msp:
            layer = entity.dxf.layer
            etype = entity.dxftype()
            
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
            entity_types[etype] = entity_types.get(etype, 0) + 1
        
        result = "<h2>DXF Debug Info</h2>"
        
        result += "<h3>Layers:</h3><ul>"
        for layer, count in sorted(layer_counts.items(), key=lambda x: -x[1]):
            result += f"<li>{layer}: {count} entities</li>"
        result += "</ul>"
        
        result += "<h3>Entity Types:</h3><ul>"
        for etype, count in sorted(entity_types.items(), key=lambda x: -x[1]):
            result += f"<li>{etype}: {count}</li>"
        result += "</ul>"
        
        return result
    
    except Exception as e:
        return f"Error: {e}"

@app.route("/debug-dxf")
def debug_dxf():
    """Debug: Show DXF file contents"""
    import ezdxf
    from collections import defaultdict
    
    data_filename = session.get('filename')
    if not data_filename:
        return "<h2>Error</h2><p>No file uploaded. Please upload a file first.</p>"
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], data_filename)
    
    # If it's a DWG, use the converted DXF
    if filepath.endswith('.dwg'):
        dxf_filepath = filepath.replace('.dwg', '_converted.dxf')
        if not os.path.exists(dxf_filepath):
            return f"<h2>Error</h2><p>Converted DXF not found. Please re-upload the file.</p>"
        filepath = dxf_filepath
    
    try:
        doc = ezdxf.readfile(filepath)
        msp = doc.modelspace()
        
        # Count by layer
        layer_stats = defaultdict(lambda: {'count': 0, 'types': defaultdict(int)})
        
        for entity in msp:
            layer = entity.dxf.layer
            etype = entity.dxftype()
            layer_stats[layer]['count'] += 1
            layer_stats[layer]['types'][etype] += 1
        
        # Build HTML report
        html = "<h2>DXF File Analysis</h2>"
        html += f"<p><strong>File:</strong> {os.path.basename(filepath)}</p>"
        html += f"<p><strong>Total entities:</strong> {len(msp)}</p>"
        
        html += "<h3>Layers Summary:</h3>"
        html += "<table border='1' style='border-collapse: collapse; margin-bottom: 30px;'>"
        html += "<tr><th>Layer</th><th>Total Entities</th></tr>"
        for layer, stats in sorted(layer_stats.items(), key=lambda x: -x[1]['count']):
            html += f"<tr><td><strong>{layer}</strong></td><td>{stats['count']}</td></tr>"
        html += "</table>"
        
        html += "<h3>Layer Details:</h3>"
        for layer, stats in sorted(layer_stats.items(), key=lambda x: -x[1]['count']):
            html += f"<h4>Layer: '{layer}' ({stats['count']} entities)</h4><ul>"
            for etype, count in sorted(stats['types'].items(), key=lambda x: -x[1]):
                html += f"<li>{etype}: {count}</li>"
            html += "</ul>"
        
        # Show LINE entities specifically
        html += "<hr><h3>LINE Entity Analysis:</h3>"
        lines_by_layer = defaultdict(list)
        
        line_query = msp.query('LINE')
        total_lines = len(line_query)
        
        for entity in line_query:
            layer = entity.dxf.layer
            start = entity.dxf.start
            end = entity.dxf.end
            length = ((end.x - start.x)**2 + (end.y - start.y)**2)**0.5
            lines_by_layer[layer].append(length)
        
        html += f"<p><strong>Total LINE entities:</strong> {total_lines}</p>"
        
        html += "<table border='1' style='border-collapse: collapse;'>"
        html += "<tr><th>Layer</th><th>Line Count</th><th>Avg Length</th><th>Min Length</th><th>Max Length</th></tr>"
        
        for layer, lengths in sorted(lines_by_layer.items(), key=lambda x: -len(x[1])):
            avg_len = sum(lengths) / len(lengths)
            html += f"<tr><td><strong>{layer}</strong></td><td>{len(lengths)}</td><td>{avg_len:.2f}</td><td>{min(lengths):.2f}</td><td>{max(lengths):.2f}</td></tr>"
        
        html += "</table>"
        
        # LWPOLYLINE analysis
        html += "<hr><h3>LWPOLYLINE Analysis:</h3>"
        polyline_query = msp.query('LWPOLYLINE')
        polyline_count = len(polyline_query)
        
        html += f"<p><strong>Total LWPOLYLINE entities:</strong> {polyline_count}</p>"
        
        if polyline_count > 0:
            poly_by_layer = defaultdict(int)
            for entity in polyline_query:
                poly_by_layer[entity.dxf.layer] += 1
            
            html += "<table border='1' style='border-collapse: collapse;'>"
            html += "<tr><th>Layer</th><th>Count</th></tr>"
            for layer, count in sorted(poly_by_layer.items(), key=lambda x: -x[1]):
                html += f"<tr><td><strong>{layer}</strong></td><td>{count}</td></tr>"
            html += "</table>"
        
        html += "<hr><p><a href='/'>Back to Upload</a></p>"
        
        return html
    
    except Exception as e:
        import traceback
        return f"<h2>Error</h2><pre>{traceback.format_exc()}</pre>"

if __name__ == "__main__":
    app.run(debug=True)
