from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import base64
import io
from matplotlib import transforms
import requests
import time
from PIL import Image
import sys
import torch
import torch.nn as nn
from werkzeug.utils import secure_filename
from map_emotion_to_image import EmotionImageMapper
from filter_emotion_data import EmotionDataProcessor

# Add S2WAT model path to Python path
from S2WAT.model.configuration import TransModule_Config
from S2WAT.model.s2wat import S2WAT
from S2WAT.net import TransModule, Decoder_MVGG
from S2WAT.tools import save_transferred_img_to_memory, Sample_Test_Net, content_style_transTo_pt

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB

mapper = None

DEFAULT_EMOTION_FILE = "./dataset/WikiArt-Emotions-All.tsv"
DEFAULT_IMAGE_FILE = "./dataset/WikiArt-info.tsv"
OUTPUT_DIR = "./outputs"
STYLE_TRANSFER_DIR = "./style_transfers"  # Directory to store style transfer results
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STYLE_TRANSFER_DIR, exist_ok=True)

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_IMAGE_SIZE

# Global variables
analyzer = None

# Mapping between emotions and art styles (example)
EMOTION_TO_STYLE = {
    "joy": "Impressionism",
    "surprise": "Expressionism",
    "anger": "Surrealism",
    "fear": "Surrealism",
    "sadness": "Romanticism",
    "disgust": "Dark Romanticism"
}

# Global variables for S2WAT model
s2wat_network = None
s2wat_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_emotion_analyzer():
    """Initialize the emotion analyzer"""
    global analyzer
    if analyzer is None:
        from emotionAnalyzer import EmotionAnalyzer
        analyzer = EmotionAnalyzer()
        model_path = "models/best_model.pth"
        analyzer.load_model(model_path)


def get_dominant_emotion(emotion_probs):
    """Get the dominant emotion from emotion probabilities"""
    return max(emotion_probs.items(), key=lambda x: x[1])[0]


def process_emotion_data_automatically():
    """Automatically process emotion data at startup"""
    try:
        processor = EmotionDataProcessor(
            input_emotion_tsv=DEFAULT_EMOTION_FILE,
            input_image_tsv=DEFAULT_IMAGE_FILE,
            output_dir=OUTPUT_DIR
        )

        # Step 1: Process and map data, returning the mapped CSV path
        mapped_path = processor.process_and_map_data("mapped_emotions.csv")

        # Step 2: Normalize probabilities, returning the normalized CSV path
        normalized_path = processor.normalize_emotion_probabilities(
            input_csv_name=os.path.basename(mapped_path),
            output_csv_name="normalized.csv"
        )

        print(f"Emotion data processed automatically. Normalized file: {normalized_path}")
        return True
    except Exception as e:
        print(f"Error automatically processing emotion data: {str(e)}")
        return False


def init_image_mapper():
    """Initialize the image mapper"""
    global mapper
    if mapper is None:
        norm_csv = os.path.join(OUTPUT_DIR, "normalized.csv")
        if not os.path.exists(norm_csv):
            raise FileNotFoundError("Please call /api/process_emotion_data to generate normalized.csv first")
        mapper = EmotionImageMapper(norm_csv)


def find_best_match(emotion_probs):
    """
    Find the best matching image based on emotion probabilities

    Args:
        emotion_probs (dict): Emotion probability dictionary containing 'anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise' keys

    Returns:
        dict: Dictionary containing the best matching image URL, emotion probabilities, and similarity score
    """
    if 1:
        init_image_mapper()
        url, best_ems, score = mapper.map_emotion_to_image(emotion_probs, metric="euclidean")
        return {
            "imageUrl": url,
            "emotionProbabilities": best_ems,
            "similarityScore": score
        }
    try:
        pass
    except Exception as e:
        print(f"Error finding best match: {str(e)}")
        return None


# Helper functions for base64 image handling
def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def base64_to_image(base64_str, output_path):
    """Convert base64 string to image file"""
    img_data = base64.b64decode(base64_str)
    with open(output_path, "wb") as img_file:
        img_file.write(img_data)
    return output_path


def pil_to_base64(pil_image):
    """Convert PIL Image to base64 string"""
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_pil(base64_str):
    """Convert base64 string to PIL Image"""
    img_data = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_data))


def resize_image(image, max_dimension=512):
    """Resize image while maintaining aspect ratio, with a max dimension of 512px
    
    Args:
        image: PIL Image object
        max_dimension: Maximum size for any dimension
        
    Returns:
        Resized PIL Image
    """
    # Check if resizing is needed
    if image.width <= max_dimension and image.height <= max_dimension:
        return image
        
    # Calculate new dimensions while maintaining aspect ratio
    if image.width > image.height:
        # Landscape orientation
        new_width = max_dimension
        new_height = int((image.height * max_dimension) / image.width)
    else:
        # Portrait or square orientation
        new_height = max_dimension
        new_width = int((image.width * max_dimension) / image.height)
        
    # Resize and return
    print(f"Resizing image from {image.width}x{image.height} to {new_width}x{new_height}")
    return image.resize((new_width, new_height), Image.LANCZOS)


def init_s2wat_model():
    """Initialize the S2WAT style transfer model"""
    global s2wat_network
    if s2wat_network is None:
        try:
            print("Initializing S2WAT model...")

            # Model configuration
            transModule_config = TransModule_Config(
                nlayer=3,
                d_model=768,
                nhead=8,
                mlp_ratio=4,
                qkv_bias=False,
                attn_drop=0.,
                drop=0.,
                drop_path=0.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                norm_first=True
            )

            # Model components
            encoder = S2WAT(
                img_size=224,
                patch_size=2,
                in_chans=3,
                embed_dim=192,
                depths=[2, 2, 2],
                nhead=[3, 6, 12],
                strip_width=[2, 4, 7],
                drop_path_rate=0.,
                patch_norm=True
            )
            decoder = Decoder_MVGG(d_model=768, seq_input=True)
            transModule = TransModule(transModule_config)

            # Create network
            s2wat_network = Sample_Test_Net(encoder, decoder, transModule)

            # Load checkpoint
            checkpoint_path = "./models/checkpoint_40000_epoch.pkl"
            if not os.path.exists(checkpoint_path):
                print(f"Warning: Checkpoint not found at {checkpoint_path}")
                checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "models", "checkpoint_40000_epoch.pkl")

            if os.path.exists(checkpoint_path):
                print(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=s2wat_device)

                s2wat_network.encoder.load_state_dict(checkpoint.get('encoder', {}))
                s2wat_network.decoder.load_state_dict(checkpoint.get('decoder', {}))
                s2wat_network.transModule.load_state_dict(checkpoint.get('transModule', {}))

                # Move to device
                s2wat_network.to(s2wat_device)
                s2wat_network.eval()  # Set to evaluation mode
                print("S2WAT model initialized successfully")
            else:
                print(f"Error: Checkpoint not found at {checkpoint_path}")
                return False

            return True
        except Exception as e:
            print(f"Error initializing S2WAT model: {e}")
            return False
    return True


@torch.no_grad()
def apply_style_transfer(content_path, style_path):
    """Apply style transfer using the S2WAT model"""
    from torchvision.utils import save_image
    import torchvision.transforms as transforms

    # Ensure the model is initialized
    if not init_s2wat_model():
        return None

    # Use content_style_transTo_pt to properly process the images
    i_c, i_s = content_style_transTo_pt(content_path, style_path)
    
    # Move tensors to the appropriate device
    i_c = i_c.to(s2wat_device)
    i_s = i_s.to(s2wat_device)
    
    # Apply style transfer with arbitrary input flag set to True for handling images of any size
    i_cs = s2wat_network(i_c, i_s, arbitrary_input=True)
    
    # Save result directly to file using save_image
    result_filename = f"styled_{os.path.basename(content_path)}"
    result_path = os.path.join(STYLE_TRANSFER_DIR, result_filename)
    save_image(i_cs, result_path)
    
    # Read the saved file back as a PIL image for the response
    styled_img = Image.open(result_path)
    
    return {
        'styled_img': styled_img,
        'styled_path': result_path
    }


@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    """
    API for analyzing an image and automatically finding the best matching artwork
    """
    # Check if a file is uploaded
    if 'image' not in request.files:
        # Check if image was sent as base64
        if request.form.get('imageBase64'):
            # Extract the base64 data after the prefix (e.g., "data:image/jpeg;base64,")
            base64_data = request.form.get('imageBase64').split(",", 1)[-1]
            # Save base64 to file
            filename = f"upload_{int(time.time())}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            base64_to_image(base64_data, filepath)
        else:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
    else:
        file = request.files['image']

        # Check file name
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No selected file'
            }), 400

        # Check file type
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file format. Allowed formats: JPG, PNG, WEBP'
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

    # Open and resize the image if needed
    with Image.open(filepath) as img:
        # Resize image while maintaining aspect ratio if it exceeds max dimensions
        resized_img = resize_image(img, max_dimension=2048)
        
        # If image was resized, save the resized version
        if resized_img is not img:
            resized_img.save(filepath, quality=90)
            print(f"Image resized and saved to {filepath}")

    # Ensure the model is initialized
    init_emotion_analyzer()

    # Perform prediction
    _, emotion_probs = analyzer.predict(filepath)

    # Get the dominant emotion
    dominant_emotion = get_dominant_emotion(emotion_probs)

    # Get the matched art style
    matched_style = EMOTION_TO_STYLE.get(dominant_emotion, "Contemporary")

    # Find the best matching artwork
    best_match = find_best_match(emotion_probs)
    
    # Convert the original image to base64 for response
    original_image_base64 = image_to_base64(filepath)

    # Build the response
    response = {
        'success': True,
        'originalImageBase64': original_image_base64,
        'originalImagePath': filepath,
        'emotionAnalysis': emotion_probs,
        'dominantEmotion': dominant_emotion,
        'matchedStyle': matched_style,
        'bestMatchImage': best_match
    }

    return jsonify(response)


@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    """
    Style transfer API that accepts a content image and style image, performs style transfer, and returns the result as a base64-encoded image.
    """
    # Handle both multipart form data and JSON
    original_image_path = None
    style_path = None
    
    if request.content_type and request.content_type.startswith('multipart/form-data'):
        # Handle form data with files
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
        # Get content image
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400
            
        # Save content image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        original_image_path = filepath
        
        # Get style identifier
        style = request.form.get('style')
        if style:
            # Map style name to URL using predefined mapping
            style_path = EMOTION_TO_STYLE.get(style.lower())
            # If not in mapping, assume it's a URL
            if not style_path:
                style_path = style
    else:
        # Handle JSON with base64 or paths
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid request data'}), 400
            
        # Get content image
        if 'imageBase64' in data:
            # Handle base64 image
            base64_data = data['imageBase64']
            if base64_data.startswith('data:image'):
                base64_data = base64_data.split(',', 1)[1]
            
            # Save base64 to file
            filename = f"upload_{int(time.time())}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            base64_to_image(base64_data, filepath)
            original_image_path = filepath
        elif 'originalImagePath' in data:
            original_image_path = data['originalImagePath']
            # Check if path exists
            if not os.path.exists(original_image_path):
                # Check if in upload folder
                alternative_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(original_image_path))
                if os.path.exists(alternative_path):
                    original_image_path = alternative_path
                else:
                    return jsonify({'success': False, 'error': f'Image file not found: {original_image_path}'}), 404
        
        # Get style image source
        style_path = data.get('styleImageUrl', data.get('style'))
        
        # If there's a best match image in the request, use that URL
        if 'bestMatchImage' in data and 'imageUrl' in data['bestMatchImage']:
            style_path = data['bestMatchImage']['imageUrl']
    
    # Validate inputs
    if not original_image_path or not style_path:
        return jsonify({'success': False, 'error': 'Missing content or style information'}), 400
    
    # Convert style URL to local file path if needed
    if style_path.startswith(('http://', 'https://')):
        # Create a cache filename based on URL hash to avoid re-downloading
        import hashlib
        style_url_hash = hashlib.md5(style_path.encode('utf-8')).hexdigest()
        style_cache_path = os.path.join(app.config['UPLOAD_FOLDER'], f"style_cache_{style_url_hash}.jpg")
        
        # Download only if not cached
        if not os.path.exists(style_cache_path):
            print(f"Downloading style image from URL: {style_path}")
            response = requests.get(style_path)
            if response.status_code != 200:
                return jsonify({'success': False, 'error': f'Failed to download style image: HTTP {response.status_code}'}), 500
                
            # Save the downloaded image
            with open(style_cache_path, 'wb') as f:
                f.write(response.content)
                
            print(f"Style image cached at: {style_cache_path}")
        else:
            print(f"Using cached style image: {style_cache_path}")
            
        # Update style_path to the local file
        style_path = style_cache_path
        
    # Initialize S2WAT model
    if not init_s2wat_model():
        return jsonify({'success': False, 'error': 'Failed to initialize style transfer model'}), 500
        
    # Apply style transfer
    result = apply_style_transfer(original_image_path, style_path)
    if not result:
        return jsonify({'success': False, 'error': 'Style transfer failed'}), 500
        
    # Convert styled image to base64
    styled_image_base64 = pil_to_base64(result['styled_img'])
    
    # Return results
    return jsonify({
        'success': True,
        'styledImageBase64': styled_image_base64,
        'styledImageUrl': f"/style_transfers/{os.path.basename(result['styled_path'])}",
        'processingTime': "2.5s"  # Placeholder, could be calculated accurately
    })


if __name__ == '__main__':
    print("Starting backend server...")
    
    # Preload models at startup
    print("Preloading models...")
    
    # Process emotion data first
    print("Processing emotion data...")
    process_emotion_data_automatically()
    
    # Initialize emotion analyzer
    print("Initializing emotion analyzer...")
    init_emotion_analyzer()
    
    # Preload S2WAT model
    print("Preloading S2WAT style transfer model...")
    init_s2wat_model()
    
    print("All models loaded successfully! Server is ready.")
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=5050, debug=True)