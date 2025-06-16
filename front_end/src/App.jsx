// App.jsx (Final Enhanced Version)
import React, { useState } from 'react';
import { Layout, Row, Col, Button, Alert, ConfigProvider, theme, message, Card, Typography } from 'antd';
import { 
  DownloadOutlined, 
  UndoOutlined, 
  ExperimentOutlined, 
  HeartOutlined,
  FireFilled,
  ArrowLeftOutlined
} from '@ant-design/icons';
import { motion } from 'framer-motion';
import UploadArea from './components/UploadArea';
import FakeProgressBar from './components/FakeProgressBar';
import EmotionRadar from './components/EmotionRadar';
import ArtStyleCard from './components/ArtStyleCard';
import StyledImage from './components/StyledImage';
import './App.css';

const { Header, Content } = Layout;
const { Title, Text } = Typography;

// Backend API URL
const API_URL = 'http://127.0.0.1:5050';

function App() {
  const [file, setFile] = useState(null);
  const [originalUrl, setOriginalUrl] = useState(null);
  const [phase, setPhase] = useState(0);
  const [emotions, setEmotions] = useState(null);
  const [artStyle, setArtStyle] = useState(null);
  const [styledUrl, setStyledUrl] = useState(null);
  const [error, setError] = useState(null);
  const [showDetails, setShowDetails] = useState(true);
  const [originalBase64, setOriginalBase64] = useState(null);
  const [styledBase64, setStyledBase64] = useState(null);
  const [bestMatchUrl, setBestMatchUrl] = useState(null);

  // Map emotion to style as fallback
  const emotionToStyle = { joy: 'Impressionism', anger: 'Expressionism', surprise: 'Surrealism', disgust: 'Abstract', fear: 'Gothic', sadness: 'Baroque' };

  // Helper function to convert file to base64
  const fileToBase64 = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = (error) => reject(error);
      reader.readAsDataURL(file);
    });
  };

  const analyzeImage = async (imageFile) => {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await fetch(`${API_URL}/api/analyze-image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  };

  const applyStyleTransfer = async (imageFile, style, bestMatchImageUrl = null) => {
    // Convert the image file to base64
    const base64Image = await fileToBase64(imageFile);
    
    // Create request data
    const requestData = {
      imageBase64: base64Image,
      style: style
    };
    
    // If we have a best match URL, use that
    if (bestMatchImageUrl) {
      requestData.styleImageUrl = bestMatchImageUrl;
    }

    // Send the request as JSON with base64 data
    const response = await fetch(`${API_URL}/api/style-transfer`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(requestData)
    });

    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }

    return await response.json();
  };

  const handleUpload = async (uploadedFile) => {
    try {
      setFile(uploadedFile);
      setOriginalUrl(URL.createObjectURL(uploadedFile));
      setPhase(1); // Analyzing Emotions phase
      
      // Analyze image in the backend - this actually does analysis and style matching together
      const analysis = await analyzeImage(uploadedFile);
      if (!analysis.success) throw new Error('Analysis failed');
      
      // Set emotion analysis results
      setEmotions(analysis.emotionAnalysis);
      setOriginalBase64(analysis.originalImageBase64);
      
      // Determine the art style
      const style = analysis.matchedStyle || emotionToStyle[analysis.dominantEmotion];
      
      // Create a fake delay to simulate the "Matching Style" phase
      // This makes the UI flow feel more natural even though both happen together on the backend
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      setArtStyle(style);
      
      // Save best match URL if available
      let bestMatchImageUrl = null;
      if (analysis.bestMatchImage && analysis.bestMatchImage.imageUrl) {
        setBestMatchUrl(analysis.bestMatchImage.imageUrl);
        bestMatchImageUrl = analysis.bestMatchImage.imageUrl;
      }
      
      // Move to style matching phase
      setPhase(2); // Matching Style phase
      
      // Create a small delay to simulate downloading the style image
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      // Enter transforming phase
      setPhase(3); // Transforming phase
      
      // Apply the style transfer
      const transfer = await applyStyleTransfer(uploadedFile, style, bestMatchImageUrl);
      
      if (!transfer.success) throw new Error('Style transfer failed');
      
      // Handle base64 image response
      if (transfer.styledImageBase64) {
        setStyledBase64(transfer.styledImageBase64);
        setStyledUrl(`data:image/jpeg;base64,${transfer.styledImageBase64}`);
      } else {
        setStyledUrl(`${API_URL}${transfer.styledImageUrl}`);
      }
      
      // The phase stays at 3 (transforming) which is now considered the complete state
      // This ensures the progress bar shows the third step as complete
      message.success('Transformation complete');
    } catch (error) {
      console.error('Error in processing:', error);
      setError(error.message || 'An unknown error occurred');
      setPhase(0);
    }
  };

  const reset = () => {
    setFile(null);
    setOriginalUrl(null);
    setPhase(0);
    setEmotions(null);
    setArtStyle(null);
    setStyledUrl(null);
    setStyledBase64(null);
    setOriginalBase64(null);
    setBestMatchUrl(null);
    setError(null);
    message.success('Reset');
  };

  // Download handler for base64 images
  const downloadResult = () => {
    if (!styledUrl) {
      message.error('No styled image available to download');
      return;
    }
    
    try {
      // Create a download link
      const link = document.createElement('a');
      link.href = styledUrl;
      link.download = `emotion-styled-${artStyle?.toLowerCase() || 'image'}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      message.success('Download started!');
    } catch (error) {
      console.error('Download error:', error);
      message.error('Failed to download image. Please try again.');
    }
  };

  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#7928CA',
          borderRadius: 8,
          fontFamily: "'Poppins', sans-serif",
        },
      }}
    >
      <Layout className="app-layout">
        <Header className="app-header">
          <div style={{ display: 'flex', alignItems: 'center', width: '100%', position: 'relative' }}>
            {file && (
              <Button
                icon={<ArrowLeftOutlined />}
                type="text"
                style={{ 
                  color: 'white', 
                  position: 'absolute',
                  left: 0,
                  border: '1px solid rgba(255,255,255,0.3)',
                  borderRadius: '4px',
                  padding: '0 10px',
                  height: '32px'
                }}
                onClick={reset}
              >
                Back
              </Button>
            )}
            <div style={{ 
              flex: 1, 
              textAlign: 'center', 
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center'
            }}>
              <FireFilled style={{ marginRight: 12, fontSize: 28 }} />
              <span>Emotion Style Transfer</span>
            </div>
          </div>
        </Header>
        <Content className="app-content">
          <div className="app-container">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <Alert 
                  message="Processing Error" 
                  description={error}
                  type="error" 
                  closable 
                  showIcon
                  onClose={() => setError(null)}
                  style={{ margin: '20px 0', borderRadius: 12 }}
                />
              </motion.div>
            )}
            {!file && <UploadArea onUpload={handleUpload} />}
            {file && phase > 0 && (
              <>
                {/* Toggle details on bar click */}
                <motion.div 
                  onClick={() => setShowDetails(!showDetails)} 
                  style={{ cursor: 'pointer' }}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3 }}
                  whileHover={{ scale: 1.01 }}
                >
                  <FakeProgressBar phase={phase} />
                  <div style={{ textAlign: 'center', fontSize: 13, color: '#666', marginTop: 5 }}>
                    {showDetails ? 'Click to hide details' : 'Click to show details'}
                  </div>
                </motion.div>
                {showDetails && (
                  <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                    <Col xs={24} md={12}>
                      {emotions && <EmotionRadar data={emotions} />}
                    </Col>
                    <Col xs={24} md={12}>
                      {artStyle && <ArtStyleCard styleName={artStyle} bestMatchUrl={bestMatchUrl} />}
                    </Col>
                  </Row>
                )}
                {/* Display images and actions in a card */}
                {styledUrl && phase === 3 && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5, delay: 0.2 }}
                  >
                    <Card 
                      style={{ marginTop: 24, overflow: 'hidden' }}
                      title={
                        <div style={{ textAlign: 'center' }}>
                          <Title level={4} className="gradient-text">
                            <ExperimentOutlined style={{ marginRight: 8 }} />
                            Transformation Complete!
                          </Title>
                        </div>
                      }
                    >
                      <Row gutter={24} align="middle">
                      <Col span={12}>
                        <div className="image-container">
                          <div className="image-label">Original</div>
                          <img 
                            src={originalBase64 ? `data:image/jpeg;base64,${originalBase64}` : originalUrl} 
                            alt="Original" 
                          />
                        </div>
                      </Col>
                      <Col span={12}>
                        <div className="image-container styled">
                          <div className="image-label">{artStyle || "Styled"}</div>
                          <img src={styledUrl} alt="Styled" />
                        </div>
                      </Col>
                    </Row>
                    <div style={{ 
                      textAlign: 'center', 
                      marginTop: 16,
                      padding: '12px 0',
                      borderTop: '1px solid #f0f0f0'
                    }}>
                      <div style={{ 
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        gap: '16px',
                        flexWrap: 'wrap'
                      }}>
                        <div>
                          <Text type="secondary">
                            <HeartOutlined style={{ color: '#FF0080' }} /> Emotion detected: 
                            <Text strong style={{ marginLeft: 5 }}>
                              {emotions && Object.keys(emotions).reduce(
                                (a, b) => emotions[a] > emotions[b] ? a : b
                              )}
                            </Text>
                          </Text>
                        </div>
                        <div style={{ display: 'flex', gap: '10px' }}>
                          <Button 
                            icon={<UndoOutlined />} 
                            onClick={reset}
                            size="middle"
                          >
                            Back
                          </Button>
                          <Button 
                            className="primary-button"
                            icon={<DownloadOutlined />} 
                            onClick={downloadResult}
                            size="middle"
                          >
                            Download Image
                          </Button>
                        </div>
                      </div>
                    </div>
                    </Card>
                  </motion.div>
                )}
              </>
            )}
          </div>
        </Content>
      </Layout>
    </ConfigProvider>
  );
}

export default App;
