# 🎨 DressCode AI

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![AI Powered](https://img.shields.io/badge/AI-Powered-purple.svg)](https://github.com/yourusername/dresscode-ai)

> **Your Personal AI Fashion Stylist** - Upload any clothing item and get instant AI-powered recommendations for complementary pieces to complete your perfect outfit.

## 🎥 Demo

![DressCode AI Demo](media/demo.gif)



## ✨ Features

- **🔮 AI-Powered Recommendations**: Advanced machine learning algorithms analyze your uploaded fashion items
- **📱 Intuitive Interface**: Clean, modern UI with drag-and-drop image upload
- **🎯 Smart Matching**: Get complementary clothing recommendations based on color, style, and season
- **📊 Item Detection**: Automatic detection and analysis of uploaded fashion items
- **💎 Beautiful Design**: Gradient backgrounds, glassmorphism effects, and smooth animations
- **⚡ Real-time Processing**: Fast AI inference for instant recommendations

## 🚀 Quick Start

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn
- Python backend server (for AI processing)

### Frontend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/dresscode-ai.git
   cd dresscode-ai
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Start the development server**
   ```bash
   npm start
   # or
   yarn start
   ```

4. **Open your browser**
   Navigate to `http://localhost:3000`

### Backend Setup

The frontend expects a backend API running on `http://localhost:8000`. Make sure your Python/FastAPI backend is running with the following endpoint: The model is too big for uploading to github so you have to train it the code for model development is given in v2.ipynb also the dataset and recommendation dataset was too big to upload so you can find it here : https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data

```
POST /predict
- Accepts: multipart/form-data
- Parameters: file (image), num_recommendations (integer)
- Returns: JSON with recommendations and metadata
```

## 🏗️ Project Structure

```
dresscode-ai/
├── src/
│   ├── components/
│   │   └── FashionRecommendationApp.js   # Main React component
│   ├── styles/
│   │   └── globals.css                   # Global styles
│   └── App.js                           # App entry point
├── public/
│   ├── index.html
│   └── favicon.ico
├── package.json
└── README.md
```

## 🎨 Technologies Used

### Frontend
- **React** - UI framework
- **Lucide React** - Beautiful icons
- **CSS-in-JS** - Styled components with hover effects
- **Modern JavaScript** - ES6+ features

### Backend (Not included in this repo)
- **Python** - Backend language
- **FastAPI** - Web framework
- **Machine Learning** - Fashion recommendation algorithms CNN Model
- **Computer Vision** - Image processing and analysis

## 📖 How It Works

1. **Upload**: Users drag and drop or click to upload a fashion item image
2. **Analysis**: The AI backend processes the image to detect:
   - Item category (shirts, pants, shoes, etc.)
   - Color analysis
   - Style attributes
   - Seasonal compatibility
3. **Recommendation**: The system finds complementary items based on:
   - Color harmony
   - Style matching
   - Seasonal appropriateness
   - Fashion rules and trends
4. **Display**: Results are beautifully presented with item details and images

## 🎯 API Response Format

```json
{
  "input_metadata": {
    "baseColour": "Blue",
    "subCategory": "Jeans",
    "season": "Fall",
    "usage": "Casual"
  },
  "recommendations": {
    "Topwear": [
      {
        "productDisplayName": "Casual Shirt",
        "baseColour": "White",
        "season": "Fall",
        "usage": "Casual",
        "image_url": "https://example.com/image.jpg"
      }
    ],
    "Footwear": [
      {
        "productDisplayName": "Sneakers",
        "baseColour": "White",
        "season": "Fall",
        "usage": "Casual",
        "image_url": "https://example.com/shoes.jpg"
      }
    ]
  }
}
```

## 🎨 UI Features

- **Glassmorphism Design**: Modern frosted glass effects
- **Gradient Backgrounds**: Beautiful purple-to-pink gradients
- **Hover Animations**: Interactive elements with smooth transitions
- **Responsive Layout**: Works on desktop, tablet, and mobile
- **Loading States**: Elegant loading animations
- **Error Handling**: User-friendly error messages

## 🔧 Configuration

Update the API base URL in the component:

```javascript
const API_BASE_URL = 'http://localhost:8000'; // Change this to your API URL
```

## 🤝 Contributing

We love contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

## 📝 Development Roadmap

- [ ] **Mobile App**: React Native version
- [ ] **Social Features**: Share outfits with friends
- [ ] **Wardrobe Management**: Upload and organize your entire wardrobe
- [ ] **Style Profiles**: Personalized recommendations based on your style
- [ ] **Shopping Integration**: Direct links to purchase recommended items
- [ ] **Weather Integration**: Weather-appropriate recommendations

## 🐛 Known Issues

- Image upload size limit (handled gracefully)
- Backend dependency for recommendations
- Limited to fashion items (other objects may not work well)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Lucide React** for beautiful icons
- **React Team** for the amazing framework
- **Fashion AI Community** for inspiration
- **Open Source Contributors** who make projects like this possible

## 📧 Contact

- **Project Link**: [https://github.com/yourusername/dresscode-ai](https://github.com/shivanksi42/DressCode-AI)
- **Issues**: [Bug Reports & Feature Requests](https://github.com/shivanksi42/DressCode-AI/issues)

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

Made with ❤️ by [Shivam Kumar](https://github.com/shivanksi42)

</div>
