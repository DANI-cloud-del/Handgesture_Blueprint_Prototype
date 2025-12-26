import os
# Set this BEFORE importing any other modules
os.environ['DOTNET_SYSTEM_GLOBALIZATION_INVARIANT'] = '0'

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

from converters import DXFParser, PDFParser, DWGParser, MeshGenerator

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/")
def index():
    return render_template('index.html')

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
        
        generator = MeshGenerator(blueprint_data, wall_height=3.0)
        mesh_data = generator.generate()
        
        return jsonify({
            'message': 'Conversion successful',
            'filename': filename,
            'mesh_data': mesh_data
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
