// ImageComparisonSlider.jsx
import React, { useState, useRef, useEffect } from 'react';
import { Card, Typography, Slider } from 'antd';
import { motion } from 'framer-motion';

const { Title } = Typography;

const ImageComparisonSlider = ({ originalSrc, styledSrc }) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [containerHeight, setContainerHeight] = useState(0);
  const imgRef = useRef(null);
  
  useEffect(() => {
    // Ensure we have valid images to work with
    if (!originalSrc || !styledSrc) return;
    
    // Setup image loading
    const img = new Image();
    img.onload = () => {
      if (imgRef.current) {
        setContainerHeight(imgRef.current.clientHeight);
      }
    };
    img.src = originalSrc;
  }, [originalSrc, styledSrc]);
  
  if (!originalSrc || !styledSrc) {
    return null; // Don't render if no images
  }
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7 }}
    >
      <Card
        title={<Title level={4}>Compare Original & Styled Image</Title>}
        style={{
          marginTop: 30,
          borderRadius: 12,
          overflow: 'hidden'
        }}
      >
        <div
          style={{
            position: 'relative',
            width: '100%',
            height: containerHeight || 'auto',
            maxWidth: '800px',
            margin: '0 auto',
            overflow: 'hidden',
            borderRadius: '8px',
            boxShadow: '0 4px 15px rgba(0,0,0,0.15)'
          }}
        >
          {/* Styled image (background) */}
          <img
            src={styledSrc}
            alt="Styled"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover'
            }}
          />
          
          {/* Original image (foreground with clip) */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              overflow: 'hidden',
              clipPath: `polygon(0 0, ${sliderPosition}% 0, ${sliderPosition}% 100%, 0 100%)`
            }}
          >
            <img
              src={originalSrc}
              alt="Original"
              style={{
                width: '100%',
                height: '100%',
                objectFit: 'cover'
              }}
            />
          </div>
          
          {/* Base image for sizing */}
          <img
            ref={imgRef}
            src={originalSrc}
            alt="Comparison base"
            style={{
              width: '100%',
              visibility: 'hidden'
            }}
          />
          
          {/* Slider divider line */}
          <div
            style={{
              position: 'absolute',
              top: 0,
              bottom: 0,
              left: `${sliderPosition}%`,
              width: '3px',
              backgroundColor: 'white',
              boxShadow: '0 0 5px rgba(0,0,0,0.5)',
              transform: 'translateX(-50%)'
            }}
          />
          
          {/* Slider handle */}
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: `${sliderPosition}%`,
              width: '36px',
              height: '36px',
              borderRadius: '50%',
              backgroundColor: 'white',
              transform: 'translate(-50%, -50%)',
              cursor: 'ew-resize',
              boxShadow: '0 0 10px rgba(0,0,0,0.5)',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center'
            }}
          >
            <div style={{
              width: '30%',
              height: '2px',
              background: '#333',
              position: 'absolute',
              transform: 'rotate(90deg)'
            }}/>
            <div style={{
              width: '30%',
              height: '2px',
              background: '#333',
              position: 'absolute'
            }}/>
          </div>
        </div>
        
        <Slider
          min={0}
          max={100}
          value={sliderPosition}
          onChange={setSliderPosition}
          style={{ marginTop: 20 }}
          trackStyle={{ backgroundColor: '#7928CA' }}
          handleStyle={{ borderColor: '#7928CA' }}
        />
        
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between',
          marginTop: '10px',
          color: '#666'
        }}>
          <span>Original</span>
          <span>Styled</span>
        </div>
      </Card>
    </motion.div>
  );
};

export default ImageComparisonSlider;
