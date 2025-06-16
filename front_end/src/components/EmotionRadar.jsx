// Improved EmotionRadar.jsx
import React from 'react';
import ReactECharts from 'echarts-for-react';
import { Card, Typography, Row, Col, Statistic } from 'antd';
import { motion } from 'framer-motion';
import { 
  FireOutlined, 
  FrownOutlined, 
  MehOutlined, 
  SmileOutlined, 
  FrownFilled, 
  ThunderboltOutlined 
} from '@ant-design/icons';

const { Title, Text } = Typography;

const emotionIcons = {
  'Anger': <FireOutlined style={{ color: '#f5222d' }} />,
  'Disgust': <FrownFilled style={{ color: '#a0d911' }} />,
  'Fear': <MehOutlined style={{ color: '#722ed1' }} />,
  'Joy': <SmileOutlined style={{ color: '#faad14' }} />,
  'Sadness': <FrownOutlined style={{ color: '#1890ff' }} />,
  'Surprise': <ThunderboltOutlined style={{ color: '#eb2f96' }} />
};

const emotionColors = {
  'Anger': '#f5222d',
  'Disgust': '#a0d911',
  'Fear': '#722ed1',
  'Joy': '#faad14',
  'Sadness': '#1890ff',
  'Surprise': '#eb2f96'
};

const EmotionRadar = ({ data }) => {
  const emotions = ['Anger', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise'];
  const values = emotions.map(e => +(data[e.toLowerCase()] * 100).toFixed(0));
  
  const option = {
    backgroundColor: 'transparent',
    radar: {
      indicator: emotions.map(e => ({ 
        name: e, 
        max: 100,
        color: emotionColors[e]
      })),
      radius: '60%', // Slightly reduced radius for better fit
      splitNumber: 4,
      axisName: {
        color: '#555',
        fontSize: 12, // Reduced font size for better fit
        fontWeight: 'bold'
      },
      splitArea: {
        areaStyle: {
          color: ['rgba(255, 255, 255, 0.8)', 'rgba(250, 250, 250, 0.8)']
        }
      },
      splitLine: {
        lineStyle: {
          color: 'rgba(200, 200, 200, 0.3)'
        }
      },
      axisLine: {
        lineStyle: {
          color: 'rgba(200, 200, 200, 0.3)'
        }
      }
    },
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(255, 255, 255, 0.9)',
      borderColor: '#e6e6e6',
      borderWidth: 1,
      textStyle: {
        color: '#333'
      },
      formatter: (params) => {
        const value = params.value;
        let result = `<div style="font-weight: bold; margin-bottom: 4px;">${params.name}</div>`;
        
        emotions.forEach((emotion, index) => {
          result += `<div style="display: flex; justify-content: space-between; margin: 5px 0;">
            <span>${emotion}:</span>
            <span style="font-weight: bold; color: ${emotionColors[emotion]}">${value[index]}%</span>
          </div>`;
        });
        
        return result;
      }
    },
    series: [{
      name: 'Emotion Scores',
      type: 'radar',
      lineStyle: {
        width: 2,
        color: '#7928CA'
      },
      data: [{
        value: values,
        name: 'Emotion Intensity',
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 1,
            y2: 1,
            colorStops: [{
              offset: 0, color: 'rgba(121, 40, 202, 0.6)'
            }, {
              offset: 1, color: 'rgba(255, 0, 128, 0.6)'
            }]
          }
        },
        symbol: 'circle',
        symbolSize: 8,
        itemStyle: {
          color: '#FF0080'
        }
      }]
    }]
  };

  // Determine dominant emotion
  const dominantIndex = values.indexOf(Math.max(...values));
  const dominantEmotion = emotions[dominantIndex];
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.7 }}
    >
      <Card 
        title={
          <Title level={4}>
            Emotion Analysis
          </Title>
        }
        style={{ 
          borderRadius: 12,
          overflow: 'hidden',
          boxShadow: '0 5px 15px rgba(0, 0, 0, 0.1)',
          maxHeight: '600px', // Set maximum height
          display: 'flex',
          flexDirection: 'column'
        }}
        styles={{
          body: {
            padding: '16px',
            flex: 1
          }
        }}
      >
        <Row gutter={16}>
          <Col xs={20} md={11}>
            <Card variant="borderless" style={{ background: 'rgba(250, 250, 250, 0.8)' }}>
              <Statistic 
                title="Dominant Emotion"
                value={dominantEmotion}
                valueStyle={{ 
                  color: emotionColors[dominantEmotion],
                  fontSize: '24px' // Reduced font size for better fit
                }}
                prefix={emotionIcons[dominantEmotion]}
                suffix={<Text type="secondary" style={{ fontSize: '0.5em' }}>{values[dominantIndex]}%</Text>}
              />
            </Card>
          </Col>
          <Col xs={20} md={13}>
            <Text type="secondary">
              The emotional analysis of your image reveals a predominant {dominantEmotion.toLowerCase()} sentiment, 
              which will be matched with an appropriate artistic style for transformation.
            </Text>
          </Col>
        </Row>
        
        <div style={{ height: '300px', width: '100%' }}> {/* Fixed height container */}
          <ReactECharts 
            option={option} 
            style={{ height: '100%', width: '100%' }} 
            className="emotion-radar"
            opts={{ renderer: 'svg' }} // SVG renderer scales better
          />
        </div>
      </Card>
    </motion.div>
  );
};

export default EmotionRadar;
