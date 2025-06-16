# 22COMP Emotion Style Transfer

An interactive web application that performs emotion analysis on uploaded images and transforms them with matching artistic styles.

## Features

- Image upload with drag-and-drop support
- Emotion analysis visualization (anger, surprise, joy, disgust, fear, sadness)
- Artistic style matching based on dominant emotion
- Visual transformations displaying the resulting styled image
- Responsive design for various device sizes

## Tech Stack

- Frontend: React + Vite
- UI Components: Ant Design
- Visualization: ECharts
- Animation: Framer Motion

## Backend API Requirements

The frontend requires the following API endpoints to function properly:

### 1. Image Analysis API

```
POST /api/analyze-image
```

**Purpose**: Analyze an uploaded image to detect emotions and determine the appropriate art style.

**Request:**
- Content-Type: `multipart/form-data`
- Body: 
  - `image`: The image file to analyze (JPEG, PNG, or WEBP format)

**Response:**
```json
{
  "success": true,
  "emotionAnalysis": {
    "anger": 0.05,
    "disgust": 0.02,
    "fear": 0.12,
    "joy": 0.65,
    "sadness": 0.08,
    "surprise": 0.08
  },
  "dominantEmotion": "joy",
  "matchedStyle": "Impressionism",
  "faceDetected": true,
  "confidence": 0.85
}
```

**Error Responses:**
- 400: Invalid image format or size
- 500: Server processing error

### 2. Style Transfer API

```
POST /api/style-transfer
```

**Purpose**: Apply the detected or selected artistic style to the image.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `image`: The original image file
  - `style`: The art style to apply (e.g., "Impressionism", "Expressionism", "Surrealism")
  - `intensity`: (Optional) A value between 0.0 and 1.0 indicating style transfer intensity (default: 0.7)

**Response:**
```json
{
  "success": true,
  "originalImageUrl": "/images/original/image123.jpg",
  "styledImageUrl": "/images/styled/image123_impressionism.jpg",
  "appliedStyle": "Impressionism",
  "processingTime": "3.2s"
}
```

**Error Responses:**
- 400: Invalid image, style name, or intensity value
- 500: Processing error

### 3. Style Information API

```
GET /api/art-styles
```

**Purpose**: Get information about all available art styles.

**Response:**
```json
{
  "success": true,
  "styles": [
    {
      "name": "Impressionism",
      "color": "#ffc107",
      "previewImage": "/styles/impressionism.jpg",
      "description": "Known for small, thin, visible brush strokes...",
      "artists": ["Monet", "Renoir", "Degas"],
      "suitableEmotions": ["joy", "surprise"]
    },
    {
      "name": "Expressionism",
      "color": "#f44336",
      "previewImage": "/styles/expressionism.jpg",
      "description": "Characterized by distortion, exaggeration...",
      "artists": ["Van Gogh", "Munch", "Kandinsky"],
      "suitableEmotions": ["anger", "fear"]
    }
    // Additional styles...
  ]
}
```

### 4. Individual Style Information API

```
GET /api/art-styles/:styleName
```

**Purpose**: Get detailed information about a specific art style.

**Response:**
```json
{
  "success": true,
  "name": "Impressionism",
  "color": "#ffc107",
  "previewImage": "/styles/impressionism.jpg",
  "description": "Known for small, thin, visible brush strokes...",
  "artists": ["Monet", "Renoir", "Degas"],
  "suitableEmotions": ["joy", "surprise"],
  "sampleImages": [
    "/samples/impressionism_1.jpg",
    "/samples/impressionism_2.jpg"
  ],
  "historicalContext": "Emerged in the 19th century as a departure from academic painting...",
  "techniques": ["Open composition", "Light in changing qualities", "Small brushstrokes"]
}
```

### 5. Image Gallery API

```
GET /api/gallery
```

**Purpose**: Retrieve previously processed images for the gallery section.

**Request Parameters (Query String):**
- `page`: Page number for pagination (default: 1)
- `limit`: Number of items per page (default: 12)
- `style`: (Optional) Filter by art style

**Response:**
```json
{
  "success": true,
  "images": [
    {
      "id": "img123",
      "originalUrl": "/images/original/sunset.jpg",
      "styledUrl": "/images/styled/sunset_impressionism.jpg",
      "style": "Impressionism",
      "createdAt": "2025-04-16T14:23:11Z",
      "emotions": {
        "dominant": "joy",
        "scores": {
          "joy": 0.75,
          "sadness": 0.05,
          "anger": 0.02,
          "fear": 0.06,
          "disgust": 0.02,
          "surprise": 0.10
        }
      }
    }
    // Additional images...
  ],
  "pagination": {
    "totalImages": 45,
    "totalPages": 4,
    "currentPage": 1,
    "limit": 12
  }
}
```

### 6. Webcam Capture API

```
POST /api/webcam-capture
```

**Purpose**: Process an image captured from the webcam.

**Request:**
- Content-Type: `application/json`
- Body:
  - `imageData`: Base64-encoded image data from webcam

**Response:** Same as the `/api/analyze-image` endpoint

### Emotion-Style Mapping

```json
{
  "emotionStyleMapping": {
    "joy": "Impressionism",
    "anger": "Expressionism",
    "surprise": "Surrealism",
    "disgust": "Abstract",
    "fear": "Gothic",
    "sadness": "Baroque"
  }
}
```

## Implementation Notes for Backend Developers

1. **Image Processing Library**: Use a library like OpenCV, PIL/Pillow, or TensorFlow for image manipulation and processing.

2. **Emotion Analysis Model**: Implement or integrate a pre-trained machine learning model for emotion detection from facial expressions or image context.

3. **Style Transfer Model**: Use neural style transfer techniques based on deep learning frameworks like TensorFlow or PyTorch.

4. **Data Storage**: 
   - Store original and processed images
   - Save analysis results for future reference
   - Implement user authentication if needed for personalized galleries

5. **Error Handling**: Implement comprehensive error handling for all scenarios:
   - No face detected in the image
   - Unsupported image format
   - Server resource limitations
   - Model processing errors

6. **Performance Considerations**:
   - Consider using a queue system for processing large images
   - Implement caching for style information and frequently accessed resources
   - Add progress tracking for long-running style transfer operations

7. **Security Considerations**:
   - Implement file type validation
   - Set maximum file size limits (10MB recommended)
   - Sanitize file names and other user inputs
   - Consider implementing rate limiting

## Integration Guide

To connect this front-end with a back-end:

1. Replace the mock functionality in `App.jsx` with actual API calls:
   - Replace the random emotion generation with a call to `/api/analyze-image`
   - Replace the URL assignment in `setStyledUrl` with a call to `/api/style-transfer`

2. Update the loading states and error handling as needed.

3. The back-end should handle the image processing and return results in the formats specified above.

## Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## License

This project is part of 22COMP, created by 丁泓赫.
