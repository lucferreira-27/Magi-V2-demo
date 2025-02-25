from PIL import Image
import numpy as np
from transformers import AutoModel
import torch
import os
import glob
import logging
import time
from datetime import datetime
import sys
import io
import json
from contextlib import redirect_stdout, redirect_stderr


class MangaAnalyzer:
    """Main class for manga page analysis and character recognition."""
    
    def __init__(self, project_name="Magi-V2-Test", test_dir=None, output_dir=None):
        """Initialize the manga analyzer with directories and setup logging."""
        # Set up paths using environment variables if provided
        self.test_dir = test_dir or os.environ.get("TEST_DIR", "./test_assets")
        self.output_dir = output_dir or os.environ.get("OUTPUT_DIR", "./output")
        self.project_name = project_name or os.path.basename(self.test_dir)
        self.pages_dir = os.path.join(self.test_dir, "pages")
        self.characters_dir = os.path.join(self.test_dir, "characters")
        
        # Set up project output directories
        self.project_dir = os.path.join(self.output_dir, self.project_name)
        self.project_pages_dir = os.path.join(self.project_dir, "pages")
        self.project_transcript_dir = os.path.join(self.project_dir, "transcript")
        self.project_profile_dir = os.path.join(self.project_dir, "profile")
        
        # Set up image formats
        self.image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp", "*.tiff"]
        
        # Initialize logger
        self.logger = self.setup_logging()
        self.logger.info(f"Starting manga analysis for project: {self.project_name}")
        
        # Initialize empty variables
        self.model = None
        self.chapter_pages = []
        self.chapter_pages_data = []
        self.character_images = []
        self.character_names = []
        self.character_bank = {}
        self.per_page_results = []
        self.transcript = []
    
    def setup_logging(self):
        """Configure and set up logging."""
        # Suppress external library logs
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("timm").setLevel(logging.ERROR)
        logging.getLogger("PIL").setLevel(logging.ERROR)
        
        # Suppress other common noisy libraries
        for logger_name in ["filelock", "huggingface_hub", "torch", "urllib3"]:
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        
        # Set root logger to warning to suppress other third-party logs
        logging.getLogger().setLevel(logging.WARNING)
        
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"manga_analysis_{timestamp}.log")
        
        # Configure our application logger
        logger = logging.getLogger("manga_analyzer")
        logger.setLevel(logging.INFO)
        
        # Create handlers
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter and add it to handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def find_images_in_directory(self, directory, extensions):
        """Find all images with specified extensions in a directory."""
        images = []
        for ext in extensions:
            images.extend(glob.glob(os.path.join(directory, ext)))
        return sorted(images)
    
    def read_image(self, path_to_image):
        """Read and process an image from a file path."""
        try:
            with open(path_to_image, "rb") as file:
                image = Image.open(file).convert("L").convert("RGB")
                image = np.array(image)
            return image
        except Exception as e:
            self.logger.error(f"Error reading image {path_to_image}: {str(e)}")
            raise
    
    def check_directories(self):
        """Check if all required directories exist and create output directories."""
        self.logger.info(f"Using test directory: {self.test_dir}")
        self.logger.info(f"Project name: {self.project_name}")
        self.logger.info(f"Output will be saved to: {self.project_dir}")
        
        # Check if directories exist
        for dir_path, dir_name in [
            (self.test_dir, "Test directory"),
            (self.pages_dir, "Pages directory"),
            (self.characters_dir, "Characters directory")
        ]:
            if not os.path.exists(dir_path):
                self.logger.error(f"{dir_name} {dir_path} does not exist")
                sys.exit(1)
        
        # Create output directories
        for dir_path, dir_name in [
            (self.project_dir, "Project directory"),
            (self.project_pages_dir, "Project pages directory"),
            (self.project_transcript_dir, "Project transcript directory"),
            (self.project_profile_dir, "Project profile directory")
        ]:
            os.makedirs(dir_path, exist_ok=True)
            self.logger.info(f"Ensured {dir_name} exists: {dir_path}")
    
    def load_files(self):
        """Load all required image files."""
        self.logger.info(f"Accepted image formats: {', '.join(self.image_extensions)}")
        
        # Load chapter pages
        self.chapter_pages = self.find_images_in_directory(self.pages_dir, self.image_extensions)
        if not self.chapter_pages:
            self.logger.error(f"No page images found in {self.pages_dir}")
            sys.exit(1)
        self.logger.info(f"Found {len(self.chapter_pages)} chapter pages")
        
        # Load character images
        self.character_images = self.find_images_in_directory(self.characters_dir, self.image_extensions)
        if not self.character_images:
            self.logger.error(f"No character images found in {self.characters_dir}")
            sys.exit(1)
        self.logger.info(f"Found {len(self.character_images)} character images")
        
        # Extract character names from filenames
        self.character_names = [os.path.splitext(os.path.basename(img))[0] for img in self.character_images]
        self.logger.info(f"Character names: {', '.join(self.character_names)}")
        
        # Set up character bank
        self.character_bank = {
            "images": self.character_images,
            "names": self.character_names
        }
    
    def load_model(self):
        """Load the AI model."""
        self.logger.info("Loading AI model...")
        start_time = time.time()
        self.model = AutoModel.from_pretrained("ragavsachdeva/magiv2", trust_remote_code=True).cuda().eval()
        self.logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    
    def load_image_data(self):
        """Load all images into memory."""
        # Load chapter page images
        self.logger.info("Loading chapter page images")
        start_time = time.time()
        self.chapter_pages_data = [self.read_image(x) for x in self.chapter_pages]
        self.logger.info(f"Loaded {len(self.chapter_pages_data)} chapter pages in {time.time() - start_time:.2f} seconds")
        
        # Load character images
        self.logger.info("Loading character images")
        start_time = time.time()
        self.character_bank["images"] = [self.read_image(x) for x in self.character_bank["images"]]
        self.logger.info(f"Loaded {len(self.character_bank['images'])} character images in {time.time() - start_time:.2f} seconds")
    
    def run_prediction(self):
        """Run the model prediction on the loaded images."""
        self.logger.info("Starting model prediction")
        start_time = time.time()
        
        # Redirect stdout/stderr to suppress direct prints from libraries
        f = io.StringIO()
        with redirect_stdout(f), redirect_stderr(f):
            with torch.no_grad():
                self.per_page_results = self.model.do_chapter_wide_prediction(
                    self.chapter_pages_data, 
                    self.character_bank, 
                    use_tqdm=True, 
                    do_ocr=True
                )
        
        self.logger.info(f"Model prediction completed in {time.time() - start_time:.2f} seconds")
    
    def save_project_metadata(self):
        """Save project metadata including character information and summary."""
        metadata = {
            "project_name": self.project_name,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pages_count": len(self.chapter_pages),
            "characters": self.character_names,
            "dialogue_count": len(self.transcript)
        }
        
        # Save metadata as JSON
        metadata_path = os.path.join(self.project_profile_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.logger.info(f"Project metadata saved to {metadata_path}")
        
        # Save character profiles
        for character_name, character_path in zip(self.character_names, self.character_images):
            char_profile_path = os.path.join(self.project_profile_dir, f"{character_name}.jpg")
            try:
                # Copy the character image to the profile directory
                with Image.open(character_path) as img:
                    img.save(char_profile_path)
                self.logger.info(f"Character profile for {character_name} saved to {char_profile_path}")
            except Exception as e:
                self.logger.error(f"Error saving character profile for {character_name}: {str(e)}")
    
    def process_results(self):
        """Process the model results and generate visualizations and transcript."""
        self.transcript = []
        self.logger.info("Processing results and generating visualizations")
        
        per_page_transcripts = []  # Store transcripts for each page
        
        for i, (image, page_result) in enumerate(zip(self.chapter_pages_data, self.per_page_results)):
            # Create page index with leading zeros for correct sorting
            page_idx = str(i).zfill(3)
            
            # Generate and save visualization
            output_image_path = os.path.join(self.project_pages_dir, f"page_{page_idx}.png")
            self.model.visualise_single_image_prediction(image, page_result, output_image_path)
            self.logger.info(f"Saved visualization for page {i} to {output_image_path}")
            
            # Process text and character associations
            speaker_name = {
                text_idx: page_result["character_names"][char_idx] 
                for text_idx, char_idx in page_result["text_character_associations"]
            }
            
            # Extract dialogue lines
            page_lines = []
            for j in range(len(page_result["ocr"])):
                if not page_result["is_essential_text"][j]:
                    continue
                name = speaker_name.get(j, "unsure") 
                line = f"<{name}>: {page_result['ocr'][j]}"
                page_lines.append(line)
                self.transcript.append(line)
            
            per_page_transcripts.append({
                "page_number": i,
                "page_file": f"page_{page_idx}.png",
                "dialogue_lines": page_lines
            })
            
            self.logger.info(f"Page {i}: Found {len(page_lines)} dialogue lines")
        
        # Save full transcript
        transcript_path = os.path.join(self.project_transcript_dir, "transcript.txt")
        with open(transcript_path, "w") as fh:
            for line in self.transcript:
                fh.write(line + "\n")
        
        # Save per-page transcripts
        per_page_transcript_path = os.path.join(self.project_transcript_dir, "per_page_transcript.json")
        with open(per_page_transcript_path, "w") as fh:
            json.dump(per_page_transcripts, fh, indent=2)
        
        # Save HTML transcript for better readability
        html_transcript_path = os.path.join(self.project_transcript_dir, "transcript.html")
        self.generate_html_transcript(html_transcript_path, per_page_transcripts)
        
        self.logger.info(f"Transcripts saved to {self.project_transcript_dir}")
        self.logger.info("Manga analysis completed successfully")
    
    def generate_html_transcript(self, output_path, per_page_transcripts):
        """Generate an HTML transcript with links to page images."""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>{self.project_name} - Manga Transcript</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        .page {{ margin-bottom: 30px; border-bottom: 1px solid #ccc; padding-bottom: 20px; }}
        .page-image {{ max-width: 300px; margin-bottom: 10px; }}
        .dialogue {{ margin-left: 20px; margin-bottom: 10px; }}
        .character {{ font-weight: bold; color: #0066cc; }}
        .navigation {{ position: fixed; top: 10px; right: 10px; background: #fff; 
                      padding: 10px; border: 1px solid #ccc; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>{self.project_name} - Manga Transcript</h1>
    <div class="navigation">
        <a href="#top">Top</a> |
        <a href="{os.path.join('..', 'profile', 'metadata.json')}">Metadata</a>
    </div>
    
    <p>Total Pages: {len(per_page_transcripts)}</p>
    <p>Total Dialogue Lines: {len(self.transcript)}</p>
    
    <h2>Characters</h2>
    <ul>
"""
        
        # Add character list
        for character in self.character_names:
            char_profile = os.path.join('..', 'profile', f"{character}.jpg")
            html_content += f'        <li><span class="character">{character}</span> - <a href="{char_profile}">profile</a></li>\n'
        
        html_content += """    </ul>
    
    <h2>Pages</h2>
"""
        
        # Add pages and dialogues
        for page_data in per_page_transcripts:
            page_num = page_data["page_number"]
            page_file = page_data["page_file"]
            page_img_path = os.path.join('..', 'pages', page_file)
            
            html_content += f"""    <div class="page" id="page{page_num}">
        <h3>Page {page_num}</h3>
        <a href="{page_img_path}">
            <img src="{page_img_path}" class="page-image" alt="Page {page_num}">
        </a>
        <div class="dialogue-container">
"""
            
            if not page_data["dialogue_lines"]:
                html_content += "            <p><em>No dialogue on this page</em></p>\n"
            else:
                for line in page_data["dialogue_lines"]:
                    # Extract character name and dialogue
                    char_start = line.find("<") + 1
                    char_end = line.find(">")
                    character = line[char_start:char_end]
                    dialogue = line[char_end+2:]
                    
                    html_content += f"""            <div class="dialogue">
                <span class="character">{character}:</span> {dialogue}
            </div>
"""
            
            html_content += """        </div>
    </div>
"""
        
        html_content += """</body>
</html>"""
        
        with open(output_path, "w") as f:
            f.write(html_content)
        
        self.logger.info(f"HTML transcript generated at {output_path}")
    
    def run(self):
        """Run the complete manga analysis pipeline."""
        try:
            self.check_directories()
            self.load_files()
            self.load_model()
            self.load_image_data()
            print(f"Processing {len(self.chapter_pages)} pages with {len(self.character_images)} characters")
            self.run_prediction()
            self.process_results()
            self.save_project_metadata()
            return True
        except Exception as e:
            self.logger.error(f"Error during manga analysis: {str(e)}")
            return False


if __name__ == "__main__":
    analyzer = MangaAnalyzer()
    analyzer.run()
