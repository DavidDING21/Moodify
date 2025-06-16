import React from 'react';
import { Card, Typography, Image } from 'antd';
import { motion } from 'framer-motion';
import { FormatPainterOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

// Style descriptions for different art styles
const styleDescriptions = {
  'Impressionism': {
    description: "Characterized by small, thin brushstrokes that capture the essence of the subject rather than details, emphasizing the play of natural light and vibrant colors. Associated with joyful emotions.",
    examples: "Monet, Renoir, Degas",
    image: "https://images.metmuseum.org/CRDImages/ep/original/DP346474.jpg"
  },
  'Expressionism': {
    description: "A style that emphasizes the artist's emotional experience rather than an impression of the external world, often using distorted forms and bold colors. Associated with powerful emotions like anger.",
    examples: "Munch, Kandinsky, Schiele",
    image: "https://uploads6.wikiart.org/images/edvard-munch/the-scream-1893(2).jpg!Large.jpg"
  },
  'Surrealism': {
    description: "A 20th-century avant-garde movement in art exploring the subconscious and dreams, juxtaposing unexpected imagery. Associated with surprise and sometimes fear.",
    examples: "Dalí, Magritte, Ernst",
    image: "https://uploads7.wikiart.org/images/salvador-dali/the-persistence-of-memory-1931.jpg!Large.jpg"
  },
  'Abstract': {
    description: "Art that does not attempt to represent an accurate depiction of visual reality, instead using shapes, colors, forms to achieve its effect. Can be associated with disgust or chaos.",
    examples: "Kandinsky, Mondrian, Pollock",
    image: "https://www.moma.org/media/W1siZiIsIjMxODI0MiJdLFsicCIsImNvbnZlcnQiLCItcXVhbGl0eSA5MCAtcmVzaXplIDIwMDB4MjAwMFx1MDAzZSJdXQ.jpg?sha=f6522ef85554762b"
  },
  'Gothic': {
    description: "Characterized by dark, mysterious, and sometimes supernatural themes. It often features dramatic lighting and an atmosphere of fear or dread.",
    examples: "Friedrich, Blake, Fuseli",
    image: "https://uploads3.wikiart.org/images/henry-fuseli/the-nightmare-1781.jpg!Large.jpg"
  },
  'Baroque': {
    description: "Known for dramatic use of light, rich colors, and intense emotions. Often depicts scenes of grandeur, tragedy, or melancholy. Associated with deep feelings like sadness.",
    examples: "Caravaggio, Rembrandt, Rubens",
    image: "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Rembrandt_-_The_Return_of_the_Prodigal_Son.jpg/1920px-Rembrandt_-_The_Return_of_the_Prodigal_Son.jpg"
  },
  'Romanticism': {
    description: "A 19th-century movement emphasizing emotion, individualism, and glorification of nature. Often depicts melancholy and introspective scenes.",
    examples: "Friedrich, Turner, Delacroix",
    image: "https://upload.wikimedia.org/wikipedia/commons/b/b9/Caspar_David_Friedrich_-_Wanderer_above_the_sea_of_fog.jpg"
  },
  'Dark Romanticism': {
    description: "A subgenre of Romanticism exploring the darker side of human nature, featuring themes of decay, madness, and the grotesque.",
    examples: "Fuseli, Goya, Blake",
    image: "https://uploads0.wikiart.org/00340/images/francisco-goya/francisco-de-goya-y-lucientes-saturn-devouring-one-of-his-sons-1821-22.jpg!Large.jpg"
  },
  'Contemporary': {
    description: "Artwork created in the present era, reflecting modern themes and techniques. Highly varied in style and approach.",
    examples: "Banksy, Kusama, Koons",
    image: "https://www.moma.org/media/W1siZiIsIjIwODY0MSJdLFsicCIsImNvbnZlcnQiLCItcXVhbGl0eSA5MCAtcmVzaXplIDIwMDB4MjAwMFx1MDAzZSJdXQ.jpg?sha=fdc1aaf606089a44"
  }
};

const ArtStyleCard = ({ styleName, bestMatchUrl }) => {
  const style = styleDescriptions[styleName] || {
    description: "A unique artistic style that blends various techniques and approaches.",
    examples: "Various artists",
    image: "https://via.placeholder.com/400x300?text=Art+Style"
  };

  // Use best match URL if provided
  const imageUrl = bestMatchUrl || style.image;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.7 }}
    >
      <Card 
        title={
          <Title level={4}>
            <FormatPainterOutlined style={{ marginRight: 10 }} />
            Matched Art Style: {styleName}
          </Title>
        } 
        style={{ borderRadius: 12 }}
      >
        <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
          <Image
            src={imageUrl}
            alt={styleName}
            style={{ 
              width: '100%', 
              maxHeight: 300, 
              objectFit: 'cover',
              borderRadius: 8
            }}
            fallback="data:image/png;base64,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"
          />
          
          {bestMatchUrl && (
            <div className="match-indicator" style={{ 
              position: 'absolute', 
              top: 10, 
              right: 10, 
              background: 'rgba(16, 142, 233, 0.85)', 
              color: 'white',
              padding: '4px 10px',
              borderRadius: 4,
              fontSize: '12px',
              fontWeight: 600
            }}>
              Best Emotion Match
            </div>
          )}
          
          <div>
            <Paragraph style={{ fontSize: '16px' }}>
              {style.description}
            </Paragraph>
            <Paragraph type="secondary">
              <b>Notable Artists:</b> {style.examples}
            </Paragraph>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default ArtStyleCard;