// ImageGallery.jsx
import React, { useState } from 'react';
import { Card, Typography, Image, Button, Tabs, Empty, Modal, message } from 'antd';
import { PictureOutlined, HistoryOutlined, StarOutlined, DownloadOutlined, ShareAltOutlined, DeleteOutlined, PlusOutlined } from '@ant-design/icons';
import { motion } from 'framer-motion';

const { Title, Text } = Typography;

// Sample data - in a real app, this would come from a database or API
const sampleGallery = [
  { 
    id: '1', 
    originalUrl: 'https://images.unsplash.com/photo-1542596768-5d1d21f1cf98?w=400&auto=format&fit=crop', 
    styledUrl: 'https://images.unsplash.com/photo-1542596768-5d1d21f1cf98?w=400&auto=format&fit=crop',
    emotion: 'Joy',
    style: 'Impressionism',
    timestamp: '2025-04-16T14:23:00Z'
  },
  { 
    id: '2', 
    originalUrl: 'https://images.unsplash.com/photo-1555955624-694ea70f8154?w=400&auto=format&fit=crop', 
    styledUrl: 'https://images.unsplash.com/photo-1555955624-694ea70f8154?w=400&auto=format&fit=crop',
    emotion: 'Anger',
    style: 'Expressionism',
    timestamp: '2025-04-15T10:17:00Z'
  },
  // Add more sample images
];

const ImageGallery = ({ onSelectImage, onCreateNew, onDeleteImage }) => {
  const [activeTab, setActiveTab] = useState('recent');
  const [selectedImage, setSelectedImage] = useState(null);
  
  const handleImageSelect = (image) => {
    setSelectedImage(image);
  };
  
  const handleUseSelectedImage = () => {
    if (selectedImage && onSelectImage) {
      onSelectImage(selectedImage);
      setSelectedImage(null);
    }
  };
  
  const handleCreateNew = () => {
    if (onCreateNew) {
      onCreateNew();
    }
  };
  
  const handleDownload = (image) => {
    if (!image || !image.styledUrl) {
      message.error('Image not available for download');
      return;
    }
    
    try {
      const link = document.createElement('a');
      link.href = image.styledUrl;
      link.download = `styled-${image.style?.toLowerCase() || 'image'}.jpg`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      message.success('Download started!');
    } catch (error) {
      console.error('Download error:', error);
      message.error('Failed to download image');
    }
  };
  
  const handleShare = (image) => {
    if (!image) return;
    
    if (navigator.share) {
      navigator.share({
        title: `My ${image.style} Styled Image`,
        text: `Check out my image transformed with ${image.emotion} emotion into ${image.style} style!`,
        url: window.location.href,
      })
      .then(() => message.success('Shared successfully!'))
      .catch(err => {
        message.error('Sharing failed');
        console.error('Error sharing:', err);
      });
    } else {
      // Fallback for browsers that don't support navigator.share
      const dummy = document.createElement('input');
      document.body.appendChild(dummy);
      dummy.value = window.location.href;
      dummy.select();
      document.execCommand('copy');
      document.body.removeChild(dummy);
      message.success('Link copied to clipboard!');
    }
  };
  
  const handleDelete = (image) => {
    if (!image) return;
    
    Modal.confirm({
      title: 'Delete Image',
      content: 'Are you sure you want to delete this image? This cannot be undone.',
      onOk() {
        if (onDeleteImage) {
          onDeleteImage(image.id);
        } else {
          message.success('Image deleted successfully');
        }
        setSelectedImage(null);
      },
      onCancel() {},
    });
  };
  
  const renderGalleryItems = (items) => {
    if (!items.length) {
      return <Empty description="No images found" />;
    }
    
    return (
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(230px, 1fr))', gap: '20px' }}>
        {items.map(item => (
          <motion.div 
            key={item.id}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Card
              hoverable
              cover={
                <div style={{ height: 200, overflow: 'hidden' }}>
                  <img 
                    alt={`Style: ${item.style}`} 
                    src={item.styledUrl} 
                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                  />
                </div>
              }
              onClick={() => handleImageSelect(item)}
              style={{ 
                overflow: 'hidden', 
                borderRadius: 8,
                border: selectedImage?.id === item.id ? '2px solid #7928CA' : 'none'
              }}
            >
              <div>
                <Text strong>Style: {item.style}</Text>
                <br />
                <Text type="secondary">Dominant emotion: {item.emotion}</Text>
              </div>
            </Card>
          </motion.div>
        ))}
      </div>
    );
  };
  
  return (
    <div>
      <Card
        title={<Title level={4}>Your Styled Images</Title>}
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={handleCreateNew}
          >
            Create New
          </Button>
        }
        style={{ marginTop: 30, borderRadius: 12 }}
      >
        <Tabs 
          activeKey={activeTab} 
          onChange={setActiveTab}
          items={[
            {
              key: 'recent',
              label: <span><HistoryOutlined /> Recent</span>,
              children: renderGalleryItems(sampleGallery)
            },
            {
              key: 'favorites',
              label: <span><StarOutlined /> Favorites</span>,
              children: renderGalleryItems(sampleGallery.filter((_, i) => i === 0))
            }
          ]}
        />
        
        {selectedImage && (
          <div style={{ marginTop: 20, display: 'flex', justifyContent: 'center' }}>
            <Button 
              type="primary" 
              size="large"
              onClick={handleUseSelectedImage}
            >
              Use Selected Image
            </Button>
          </div>
        )}
      </Card>
      
      {selectedImage && (
        <Modal
          open={!!selectedImage}
          onCancel={() => setSelectedImage(null)}
          footer={[
            <Button key="select" type="primary" onClick={handleUseSelectedImage}>
              Use This Image
            </Button>,
            <Button key="cancel" onClick={() => setSelectedImage(null)}>
              Cancel
            </Button>
          ]}
          width={800}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: '10px' }}>
              <div style={{ flex: 1 }}>
                <Title level={5}>Original Image</Title>
                <img 
                  src={selectedImage.originalUrl} 
                  alt="Original" 
                  style={{ width: '100%', borderRadius: 8 }}
                />
              </div>
              
              <div style={{ flex: 1 }}>
                <Title level={5}>{selectedImage.style} Style</Title>
                <img 
                  src={selectedImage.styledUrl} 
                  alt="Styled" 
                  style={{ width: '100%', borderRadius: 8 }}
                />
              </div>
            </div>
            
            <div>
              <Title level={4}>
                {selectedImage.emotion} → {selectedImage.style}
              </Title>
              <Text>
                Created on {new Date(selectedImage.timestamp).toLocaleString()}
              </Text>
            </div>
            
            <div style={{ display: 'flex', gap: '10px' }}>
              <Button 
                type="primary" 
                icon={<DownloadOutlined />}
                onClick={() => handleDownload(selectedImage)}
              >
                Download
              </Button>
              <Button 
                icon={<ShareAltOutlined />}
                onClick={() => handleShare(selectedImage)}
              >
                Share
              </Button>
              <Button 
                danger 
                icon={<DeleteOutlined />}
                onClick={() => handleDelete(selectedImage)}
              >
                Delete
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

export default ImageGallery;
