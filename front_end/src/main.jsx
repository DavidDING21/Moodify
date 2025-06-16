import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import 'antd/dist/reset.css'

// Clean up any potential existing styles that might conflict
document.documentElement.style.height = '100%'
document.body.style.height = '100%'
document.body.style.margin = '0'
document.body.style.padding = '0'
document.getElementById('root').style.height = '100%'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
