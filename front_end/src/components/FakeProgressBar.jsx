// Enhanced FakeProgressBar.jsx
import React, { useEffect, useState } from 'react';
import { Progress, Typography, Card } from 'antd';
import { motion } from 'framer-motion';
import { BsBarChartFill, BsBrushFill, BsMagic } from 'react-icons/bs';

const { Text } = Typography;

const phases = [
  { name: 'Analyzing Emotions', icon: <BsBarChartFill />, color: '#7928CA', description: 'Detecting facial expressions and context' },
  { name: 'Matching Style', icon: <BsBrushFill />, color: '#9F30BE', description: 'Finding the perfect art style for your emotions' },
  { name: 'Transforming', icon: <BsMagic />, color: '#52c41a', description: 'Transforming your image with neural style transfer' },
];

const FakeProgressBar = ({ phase }) => {
  const [percent, setPercent] = useState(0);
  
  // Adjust target calculation: phase 3 should reach 100%
  const target = phase === 3 ? 100 : (phase * 33.33);
  
  useEffect(() => {
    // Reset percent when phase changes to avoid jumps
    setPercent(phase > 0 ? Math.min((phase - 1) * 33.33, 0) : 0);
    
    const interval = setInterval(() => {
      setPercent(prev => {
        if (prev >= target) {
          clearInterval(interval);
          return target;
        }
        return prev + 1;
      });
    }, 20);
    
    return () => clearInterval(interval);
  }, [phase, target]);
  
  // Add a variable to determine if we're in the final complete state
  const isComplete = phase === 3;
  
  // Calculate progress percentage for the connecting line
  const lineProgress = ((phase - 1) / 2) * 100;
  
  return (
    <motion.div 
      initial={{ opacity: 0, y: 20 }} 
      animate={{ opacity: 1, y: 0 }} 
      transition={{ duration: 0.5 }}
    >
      <Card 
        style={{ 
          margin: '20px 0', 
          borderRadius: '16px',
          overflow: 'hidden',
          background: 'white',
          boxShadow: '0 6px 20px rgba(0, 0, 0, 0.06)'
        }}
        bodyStyle={{ padding: '16px 24px' }}
        bordered={false}
      >
        <div>
          {/* Step indicators */}
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            position: 'relative',
            margin: '0 15px 15px',
            padding: '0 10px'
          }}>
            {phases.map((p, i) => {
              const isActive = i + 1 <= phase;
              const isCurrentPhase = i + 1 === phase;

              return (
                <div
                  key={i}
                  style={{
                    position: 'relative',
                    zIndex: 2,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    width: '33%', // Equal width for each phase (3 phases now)
                  }}
                >
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ 
                      scale: isCurrentPhase ? [1, 1.1, 1] : 1, 
                      opacity: isActive ? 1 : 0.4 
                    }}
                    transition={{ 
                      duration: isCurrentPhase ? 1.5 : 0.3,
                      repeat: isCurrentPhase ? Infinity : 0,
                      repeatType: 'loop'
                    }}
                    style={{
                      width: 36,
                      height: 36,
                      borderRadius: '50%',
                      background: isActive ? p.color : '#f0f0f0',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      color: isActive ? 'white' : '#999',
                      fontSize: '18px',
                      marginBottom: 8,
                      boxShadow: isCurrentPhase 
                        ? `0 0 0 4px ${p.color}33` 
                        : isActive 
                          ? `0 2px 8px ${p.color}66`
                          : 'none'
                    }}
                  >
                    {p.icon}
                  </motion.div>
                  <div style={{
                    fontSize: '12px',
                    fontWeight: isActive ? 'bold' : 'normal',
                    color: isActive ? p.color : '#999',
                    textAlign: 'center',
                    width: '100%'
                  }}>
                    {p.name}
                  </div>
                </div>
              );
            })}
            
            {/* Connecting line */}
            <div style={{
              position: 'absolute',
              top: 18,
              left: 30,
              right: 30,
              height: 2,
              background: '#f0f0f0',
              zIndex: 1
            }} />
            
            {/* Progress line overlay */}
            <motion.div
              initial={{ width: '0%' }}
              animate={{ width: `${lineProgress}%` }}
              transition={{ duration: 0.5 }}
              style={{
                position: 'absolute',
                top: 18,
                left: 30,
                height: 2,
                background: 'linear-gradient(90deg, #7928CA, #52c41a)',
                zIndex: 1,
                borderRadius: 2
              }}
            />
          </div>
          
          {/* Progress bar */}
          <div style={{ marginTop: 20, position: 'relative' }}>
            <Text strong style={{ display: 'block', marginBottom: 8 }}>
              {phase > 0 && phase <= phases.length ? phases[phase - 1]?.name : 'Ready'}
              {isComplete && ' Complete!'}
            </Text>
            
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <Progress 
                percent={Math.round(percent)} 
                status={isComplete ? 'success' : 'active'} 
                strokeColor={{ 
                  '0%': phases[Math.min(phase - 1, phases.length - 1)]?.color || '#7928CA', 
                  '100%': isComplete ? '#52c41a' : '#FF0080' 
                }} 
                strokeWidth={8}
                showInfo={true}
              />
            </motion.div>
          </div>
        </div>
      </Card>
    </motion.div>
  );
};

export default FakeProgressBar;
