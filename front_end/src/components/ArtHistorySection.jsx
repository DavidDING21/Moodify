// ArtHistorySection.jsx
import React from 'react';
import { Card, Typography, Collapse, Image, Row, Col, Tag } from 'antd';
import { motion } from 'framer-motion';

const { Title, Paragraph, Text, Link } = Typography;
const { Panel } = Collapse;

const artHistoryInfo = {
  Impressionism: {
    period: '1860s - early 1900s',
    origin: 'France',
    keyFeatures: [
      'Small, thin, visible brush strokes',
      'Emphasis on accurate depiction of light',
      'Open composition',
      'Ordinary subject matter',
      'Inclusion of movement as a crucial element'
    ],
    description: `Impressionism emerged as a radical departure from the realistic paintings of the time. 
    Artists sought to capture fleeting moments of light and color, often painting outdoors 
    (en plein air) to accurately observe and depict the effects of sunlight and atmosphere.`,
    famousWorks: [
      { title: 'Water Lilies', artist: 'Claude Monet', year: '1840-1926' },
      { title: 'Bal du moulin de la Galette', artist: 'Pierre-Auguste Renoir', year: '1876' },
      { title: 'Ballet Rehearsal', artist: 'Edgar Degas', year: '1874' }
    ],
    image: 'https://images.unsplash.com/photo-1520420097861-e4959843b520?q=80&w=500&auto=format&fit=crop'
  },
  // Add other art styles in similar format
};

const ArtHistorySection = ({ styleName }) => {
  const style = artHistoryInfo[styleName];
  
  if (!style) return null;
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7, delay: 0.3 }}
    >
      <Card
        title={<Title level={4}>About {styleName}</Title>}
        style={{
          marginTop: 30,
          borderRadius: 12,
          overflow: 'hidden'
        }}
      >
        <Row gutter={[24, 24]}>
          <Col xs={24} md={12}>
            <Image
              src={style.image}
              alt={styleName}
              style={{ borderRadius: 8 }}
            />
            <div style={{ marginTop: 16 }}>
              <Text strong>Period: </Text><Text>{style.period}</Text>
              <br />
              <Text strong>Origin: </Text><Text>{style.origin}</Text>
            </div>
          </Col>
          
          <Col xs={24} md={12}>
            <Paragraph>{style.description}</Paragraph>
            
            <Text strong>Key Features:</Text>
            <ul>
              {style.keyFeatures.map((feature, index) => (
                <li key={index}>{feature}</li>
              ))}
            </ul>
            
            <Text strong>Famous Works:</Text>
            <div style={{ marginTop: 8 }}>
              {style.famousWorks.map((work, index) => (
                <Tag color="blue" key={index} style={{ marginBottom: 8 }}>
                  "{work.title}" by {work.artist} ({work.year})
                </Tag>
              ))}
            </div>
          </Col>
        </Row>
        
        <div style={{ marginTop: 16, textAlign: 'center' }}>
          <Link href={`https://en.wikipedia.org/wiki/${styleName}`} target="_blank">
            Learn more about {styleName} on Wikipedia
          </Link>
        </div>
      </Card>
    </motion.div>
  );
};

export default ArtHistorySection;
