// Improved UploadArea.jsx
import React, { useState } from 'react';
import { Upload, Typography, Button, Tooltip, Modal, message } from 'antd';
import { InboxOutlined, QuestionCircleOutlined } from '@ant-design/icons';
import { BsCloudUpload, BsImage, BsInfoCircle } from 'react-icons/bs';
import { motion } from 'framer-motion';

const { Dragger } = Upload;
const { Title, Text } = Typography;

const UploadArea = ({ onUpload }) => {
  const [previewVisible, setPreviewVisible] = useState(false);
  const [previewImage, setPreviewImage] = useState('');

  const props = {
    name: 'file',
    multiple: false,
    accept: 'image/*',
    showUploadList: false,
    beforeUpload: file => {
      // Check file size (limit to 10MB)
      if (file.size > 10 * 1024 * 1024) {
        message.error('Image must be smaller than 10MB!');
        return false;
      }
      
      // Check file type
      const isImage = file.type.startsWith('image/');
      if (!isImage) {
        message.error('You can only upload image files!');
        return false;
      }
      
      // Preview image before processing
      const reader = new FileReader();
      reader.onload = e => {
        setPreviewImage(e.target.result);
        setPreviewVisible(true);
      };
      reader.readAsDataURL(file);
      return false;
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  const confirmUpload = () => {
    // Convert base64 back to file and upload
    fetch(previewImage)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], "image.jpg", { type: "image/jpeg" });
        onUpload(file);
        setPreviewVisible(false);
      })
      .catch(error => {
        console.error('Error processing image:', error);
        message.error('Failed to process the image. Please try again.');
      });
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
      className="upload-container"
    >
      <Title level={2} className="gradient-text" style={{ 
        marginBottom: 30,
        textAlign: 'center'
      }}>
        <BsImage style={{ marginRight: 12, fontSize: '0.9em' }} />
        Emotion-Based Style Transfer
      </Title>
      
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: 20 }}>
        <Title level={4} style={{ 
          marginBottom: 0,
          color: '#666',
          textAlign: 'center',
          fontWeight: 'normal'
        }}>
          Upload an image to analyze emotions and transform it
        </Title>
        <Tooltip title="We use AI to analyze facial expressions and emotional context in your image, then apply a matching artistic style.">
          <BsInfoCircle style={{ marginLeft: 8, color: '#7928CA', fontSize: 18 }} />
        </Tooltip>
      </div>
      
      <Dragger {...props}>
        <motion.div
          className="upload-area-inner"
          style={{
            padding: '60px 40px',
            borderRadius: '16px',
            transition: 'all 0.3s ease',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'linear-gradient(145deg, #f6f9fc, #edf1f7)',
            border: '2px dashed rgba(121, 40, 202, 0.3)',
          }}
        >
          <div style={{ position: 'relative', marginBottom: 20 }}>
            <motion.div
              animate={{ 
                y: [0, -10, 0],
                opacity: [0.7, 1, 0.7]
              }}
              transition={{ 
                repeat: Infinity, 
                duration: 2.5, 
                ease: "easeInOut" 
              }}
              style={{ fontSize: '64px', color: '#7928CA' }}
            >
              <BsCloudUpload />
            </motion.div>
            
            <motion.div 
              style={{ 
                position: 'absolute',
                bottom: '-10px',
                left: '50%',
                width: '40px',
                height: '10px',
                marginLeft: '-20px',
                background: 'rgba(121, 40, 202, 0.15)',
                borderRadius: '50%',
                zIndex: -1
              }}
              animate={{
                width: ['40px', '60px', '40px'],
                opacity: [0.3, 0.6, 0.3]
              }}
              transition={{ 
                repeat: Infinity, 
                duration: 2.5, 
                ease: "easeInOut" 
              }}
            />
          </div>
          
          <p className="ant-upload-text" style={{ 
            fontSize: '20px', 
            fontWeight: 'bold', 
            margin: '10px 0', 
            background: 'linear-gradient(90deg, #7928CA, #FF0080)',
            WebkitBackgroundClip: 'text',
            backgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
            textFillColor: 'transparent'
          }}>
            Click or Drag Image to Upload
          </p>
          
          <p className="ant-upload-hint" style={{ color: '#888', maxWidth: '80%', textAlign: 'center' }}>
            Supports JPG, PNG, and WEBP formats. For best results, use images with clear facial expressions or emotional scenes.
          </p>
          
          <motion.div
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            style={{ marginTop: 20 }}
          >
            <Button className="primary-button">
              Browse Files
            </Button>
          </motion.div>
        </motion.div>
      </Dragger>
      
      {/* Only user uploads—no sample gallery */}

      <Modal
        open={previewVisible}
        title="Confirm Your Image"
        onCancel={() => setPreviewVisible(false)}
        footer={[
          <Button key="back" onClick={() => setPreviewVisible(false)}>
            Choose Different Image
          </Button>,
          <Button key="submit" className="primary-button" onClick={confirmUpload}>
            Confirm & Analyze
          </Button>,
        ]}
        style={{ borderRadius: '16px', overflow: 'hidden' }}
      >
        <div style={{ textAlign: 'center' }}>
          <img 
            alt="Preview" 
            style={{ 
              maxWidth: '100%', 
              maxHeight: '300px', 
              borderRadius: '12px',
              boxShadow: '0 5px 15px rgba(0,0,0,0.08)' 
            }} 
            src={previewImage} 
          />
          <Text type="secondary" style={{ display: 'block', marginTop: 16 }}>
            This image will be analyzed for emotional content and artistic style transformation.
          </Text>
        </div>
      </Modal>
    </motion.div>
  );
};

export default UploadArea;
