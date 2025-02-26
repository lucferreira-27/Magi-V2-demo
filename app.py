from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import uuid
import logging
import shutil
import traceback
from werkzeug.utils import secure_filename
import manga_analyzer  # Import your existing analyzer code

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['CHARACTER_FOLDER'] = 'test_assets/characters'
app.config['MANGA_FOLDER'] = 'test_assets/pages'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['CHARACTER_FOLDER'], exist_ok=True)
os.makedirs(app.config['MANGA_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy"})

@app.route('/api/characters', methods=['GET'])
def list_characters():
    characters = []
    for filename in os.listdir(app.config['CHARACTER_FOLDER']):
        if allowed_file(filename):
            character_name = os.path.splitext(filename)[0]
            characters.append({
                "name": character_name,
                "filename": filename
            })
    return jsonify({"characters": characters})

@app.route('/api/pages', methods=['GET'])
def list_pages():
    pages = []
    for filename in os.listdir(app.config['MANGA_FOLDER']):
        if allowed_file(filename):
            pages.append({
                "name": filename,
                "filename": filename
            })
    return jsonify({"pages": pages})

@app.route('/api/upload/character', methods=['POST'])
def upload_character():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['CHARACTER_FOLDER'], filename)
        file.save(file_path)
        return jsonify({
            "message": "Character uploaded successfully",
            "filename": filename
        })
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/upload/manga', methods=['POST'])
def upload_manga():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['MANGA_FOLDER'], filename)
        file.save(file_path)
        return jsonify({
            "message": "Manga page uploaded successfully",
            "filename": filename
        })
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/analyze', methods=['POST'])
def analyze_manga():
    data = request.json
    manga_page = data.get('manga_page')
    selected_characters = data.get('selected_characters', [])
    
    if not manga_page:
        return jsonify({"error": "No manga page specified"}), 400
    
    manga_path = os.path.join(app.config['MANGA_FOLDER'], manga_page)
    if not os.path.exists(manga_path):
        return jsonify({"error": "Manga page not found"}), 404
    
    # If specific characters are selected, use only those
    character_dir = app.config['CHARACTER_FOLDER']
    temp_character_dir = None
    
    try:
        if selected_characters:
            # Create a custom character folder for this analysis
            temp_character_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
            os.makedirs(temp_character_dir, exist_ok=True)
            
            # Copy selected character images to the temp directory
            for char_filename in selected_characters:
                src_path = os.path.join(character_dir, char_filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(temp_character_dir, char_filename)
                    shutil.copy2(src_path, dst_path)
            
            character_dir = temp_character_dir
        
        # Call analyzer with the specified character directory
        results = manga_analyzer.analyze_page(manga_path, character_dir=character_dir)
            
        # Clean up temp directory if it was created
        if temp_character_dir and os.path.exists(temp_character_dir):
            shutil.rmtree(temp_character_dir)
            
        return jsonify({
            "results": results,
            "manga_page": manga_page
        })
    except Exception as e:
        # Get detailed error information
        error_detail = traceback.format_exc()
        logger.error(f"Analysis error: {str(e)}\n{error_detail}")
        
        # Clean up temp directory if it was created
        if temp_character_dir and os.path.exists(temp_character_dir):
            shutil.rmtree(temp_character_dir)
            
        return jsonify({
            "error": str(e),
            "detail": error_detail
        }), 500

@app.route('/api/analyze/batch', methods=['POST'])
def analyze_batch():
    job_id = str(uuid.uuid4())
    data = request.json
    selected_pages = data.get('selected_pages', [])
    selected_characters = data.get('selected_characters', [])
    
    # If no pages are specified, use all pages
    if not selected_pages:
        selected_pages = [f for f in os.listdir(app.config['MANGA_FOLDER']) if allowed_file(f)]
    
    # Set up character directory
    character_dir = app.config['CHARACTER_FOLDER']
    temp_character_dir = None
    
    try:
        if selected_characters:
            # Create a custom character folder for this analysis
            temp_character_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
            os.makedirs(temp_character_dir, exist_ok=True)
            
            # Copy selected character images to the temp directory
            for char_filename in selected_characters:
                src_path = os.path.join(character_dir, char_filename)
                if os.path.exists(src_path):
                    dst_path = os.path.join(temp_character_dir, char_filename)
                    shutil.copy2(src_path, dst_path)
            
            character_dir = temp_character_dir
        
        # Process selected manga pages
        results = []
        errors = []
        
        for filename in selected_pages:
            if allowed_file(filename):
                manga_path = os.path.join(app.config['MANGA_FOLDER'], filename)
                try:
                    page_results = manga_analyzer.analyze_page(manga_path, character_dir=character_dir)
                    # Include the filename if not already present
                    if "manga_page" not in page_results:
                        page_results["manga_page"] = filename
                    results.append(page_results)
                except Exception as e:
                    # Log the error but continue processing other pages
                    error_msg = f"Error analyzing {filename}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        
        # Clean up temp directory if it was created
        if temp_character_dir and os.path.exists(temp_character_dir):
            shutil.rmtree(temp_character_dir)
        
        response = {
            "job_id": job_id,
            "status": "completed",
            "results": results
        }
        
        # Include errors if any occurred
        if errors:
            response["errors"] = errors
            response["status"] = "completed_with_errors"
        
        return jsonify(response)
        
    except Exception as e:
        # Get detailed error information
        error_detail = traceback.format_exc()
        logger.error(f"Batch analysis error: {str(e)}\n{error_detail}")
        
        # Clean up temp directory if it was created
        if temp_character_dir and os.path.exists(temp_character_dir):
            shutil.rmtree(temp_character_dir)
            
        return jsonify({
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "detail": error_detail
        }), 500

@app.route('/images/character/<filename>')
def character_image(filename):
    return send_from_directory(app.config['CHARACTER_FOLDER'], filename)

@app.route('/images/manga/<filename>')
def manga_image(filename):
    return send_from_directory(app.config['MANGA_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False) 