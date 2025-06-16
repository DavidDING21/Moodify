import React from 'react';
import { motion } from 'framer-motion';
import { Card, Typography } from 'antd';

const { Title } = Typography;

const StyledImage = ({ src, styleName }) => {
  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.8 }} 
      animate={{ opacity: 1, scale: 1 }} 
      transition={{ duration: 0.8, ease: "easeOut" }}
      style={{ 
        margin: '30px 0',
        borderRadius: '12px',
        overflow: 'hidden',
      }}
    >
      <Card
        title={
          <Title level={4} style={{ margin: 0 }}>
            Styled Result: {styleName || "Artistic Transformation"}
          </Title>
        }
        style={{ 
          boxShadow: '0 10px 30px rgba(0, 0, 0, 0.2)',
          borderRadius: '12px'
        }}
        bodyStyle={{
          padding: '20px'
        }}
      >
        <img 
          src={src} 
          alt="Styled" 
          style={{ 
            width: '100%', 
            maxWidth: '100%', 
            display: 'block',
            margin: '0 auto',
            borderRadius: '8px'
          }} 
        />
      </Card>
    </motion.div>
  );
};

export default StyledImage;