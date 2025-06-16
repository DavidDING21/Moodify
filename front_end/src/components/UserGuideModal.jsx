// UserGuideModal.jsx
import React from 'react';
import { Modal, Steps, Typography, Button, Image } from 'antd';
import { 
  UploadOutlined, 
  ScanOutlined, 
  FormatPainterOutlined, 
  CheckCircleOutlined 
} from '@ant-design/icons';

const { Step } = Steps;
const { Title, Paragraph, Text } = Typography;

const UserGuideModal = ({ visible, onClose }) => {
  return (
    <Modal
      open={visible}
      onCancel={onClose}
      footer={[
        <Button key="close" onClick={onClose}>
          Close
        </Button>
      ]}
      width={800}
      title="How Emotion Style Transfer Works"
    >
      <Steps direction="vertical" current={-1}>
        <Step 
          title="Upload Your Image" 
          icon={<UploadOutlined />}
          description={
            <div className="step-content">
              <Paragraph>
                Upload any image containing faces or scenes with emotional content. 
                Our system works best with clear images showing distinct emotions.
              </Paragraph>
              <div className="example-images">
                <Image.PreviewGroup>
                  <Image 
                    width={150} 
                    src="https://images.unsplash.com/photo-1531747118685-ca8fa6e08806?q=80&w=400&auto=format&fit=crop" 
                    alt="Example upload" 
                  />
                  <Image 
                    width={150} 
                    src="https://images.unsplash.com/photo-1545167622-3a6ac756afa4?q=80&w=400&auto=format&fit=crop" 
                    alt="Example upload" 
                  />
                </Image.PreviewGroup>
              </div>
            </div>
          }
        />
        
        <Step 
          title="Emotion Analysis" 
          icon={<ScanOutlined />}
          description={
            <div className="step-content">
              <Paragraph>
                Our AI model analyzes the image to detect emotional content across six basic emotions: 
                joy, sadness, anger, fear, surprise, and disgust. The system creates a detailed 
                emotion profile represented as a radar chart.
              </Paragraph>
              <Text type="secondary">
                The analysis considers facial expressions, postures, colors, and compositional elements.
              </Text>
            </div>
          }
        />
        
        <Step 
          title="Style Matching & Transfer" 
          icon={<FormatPainterOutlined />}
          description={
            <div className="step-content">
              <Paragraph>
                Based on the dominant emotion, the system selects an artistic style that historically 
                or culturally corresponds to that emotion. For example, joyful images might be matched 
                with Impressionism, while fearful images might be paired with Gothic art styles.
              </Paragraph>
              <Paragraph>
                Our neural style transfer algorithm then transforms your image using the characteristics 
                of the matched artistic style, creating a unique artistic interpretation.
              </Paragraph>
            </div>
          }
        />
        
        <Step 
          title="Explore & Share" 
          icon={<CheckCircleOutlined />}
          description={
            <div className="step-content">
              <Paragraph>
                Once the transformation is complete, you can:
              </Paragraph>
              <ul>
                <li>Compare the original and styled images with our comparison slider</li>
                <li>Learn about the art style and its historical context</li>
                <li>Download your transformed image or share it on social media</li>
                <li>Try different images to explore other emotion-style pairings</li>
              </ul>
            </div>
          }
        />
      </Steps>
      
      <div style={{ marginTop: 24, textAlign: 'center' }}>
        <Title level={5} className="gradient-text">Ready to transform your emotions into art?</Title>
      </div>
    </Modal>
  );
};

export default UserGuideModal;
