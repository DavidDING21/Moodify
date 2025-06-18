# Moodify

Moodify is an application that analyzes the emotional content of uploaded images, matches them to appropriate artistic styles, and transforms them using style transfer. The system combines computer vision, emotion recognition, and artistic style transfer to create a unique visual experience.

<img src="./pic/Screenshot 2025-06-18 at 6.04.55 PM.png" alt="Screenshot 2025-06-18 at 6.04.55 PM" style="zoom:30%;" />

<img src="./pic/Screenshot 2025-06-18 at 6.05.18 PM.png" alt="Screenshot 2025-06-18 at 6.05.18 PM" style="zoom:30%;" />

<img src="./pic/Screenshot 2025-06-18 at 6.05.32 PM.png" alt="Screenshot 2025-06-18 at 6.05.32 PM" style="zoom:30%;" />

## Contributors

- DING Honghe (https://github.com/DavidDING21)
- CHAI Wenchang (https://github.com/CCMKCCMK)
- SHEN Zitong (https://github.com/Qween0fPandora)

## Features

- Image emotion analysis
- Artistic style matching based on detected emotions
- Style transfer preview and transformation
- Support for common image formats (.jpg, .jpeg, .png)
- Six emotion categories: anger, surprise, joy, disgust, fear, sadness

## System Requirements

- Python 3.8
- Node.js (v20.17.0)
- npm (v10.8.2)

## Installation

### 1. Python Dependencies

#### For macOS/Linux:

```bash
pip install flask
pip install torch==1.10.0 torchvision==0.11.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install Pillow
pip install flask-cors
pip install numpy
pip install matplotlib
pip install requests
pip install pandas
pip install tqdm
```

#### For Windows:

```bash
pip install flask
pip install torch==1.10.0+cpu torchvision==0.11.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install Pillow
pip install flask-cors
pip install numpy
pip install matplotlib
pip install requests
pip install pandas
pip install tqdm
```

### 2. Node.js Dependencies

```bash
cd front_end
npm install
```

### 3. Required Models

Download the following models and place them in the `back_end/models` folder:

- emotionAnalyzer: [Download from HuggingFace](https://huggingface.co/DavidDing21/emotionAnalyzer/tree/main)
- styleTransformer: [Download from Google Drive](https://drive.google.com/file/d/16Ihs_J9ULYSze2lL5cmptvMyy-ZYJ9kN/view)

## Running the Application

### 1. Start Backend Server

#### Windows:

```bash
cd back_end
python backend.py
```

#### macOS/Linux:

```bash
cd back_end
python3 backend.py  # Or using python backend.py in virtual environment
```

### 2. Start Frontend Server

```bash
cd front_end
npm run dev
```

### Port Requirements

- Backend server: Port 5050
- Frontend server: Port 5173

To check if ports are in use:

#### Windows:

```bash
netstat -ano | findstr 5050
netstat -ano | findstr 5173
```

#### macOS/Linux:

```bash
lsof -i :5050
lsof -i :5173
```

## Usage

1. Open your browser and navigate to http://localhost:5173
2. Click the "Upload Image" button
3. Select an image file (supported formats: .jpg, .jpeg, .png)
4. Wait for emotion analysis and style transfer preview
5. Download the transformed image

## Troubleshooting

### 1. Port Already in Use

#### Windows:

```bash
netstat -ano | findstr 5050
taskkill /PID <PID> /F
```

#### macOS/Linux:

```bash
lsof -i :5050
kill -9 <PID>
```

### 2. Python Dependencies Installation Issues

Try using a virtual environment:

```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Then reinstall dependencies
```

### 3. Node.js Dependencies Issues

```bash
npm cache clean --force
rm -rf node_modules
npm install
```

## Notes

- Initial loading time may be longer due to model initialization
- For optimal results, use well-lit images with clear facial expressions
- Keep both terminal windows open while using the application
- For torch installation issues, visit [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- It is recommended to use the development versions of Python, npm, and Node.js to avoid version conflicts