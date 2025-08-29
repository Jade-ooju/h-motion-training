# H-Motion Training: Motion Classification System

A machine learning-based motion classification system for robotics applications, designed to classify hand movements into Pick, Hold, and Place actions using joint position and gaze data.

## 🎯 Project Overview

This project implements an ensemble machine learning pipeline for real-time motion classification, specifically designed for Unity integration. The system processes hand joint data and gaze information to classify human hand movements with high accuracy.

## 🚀 Key Features

- **Multi-class Classification**: Pick, Hold, and Place motion detection
- **Ensemble Learning**: Combines Random Forest, Gradient Boosting, and SVM
- **Real-time Processing**: Unity integration bridge for live motion classification
- **Feature Engineering**: Advanced feature extraction and selection pipeline
- **High Performance**: 90.91% accuracy on validation data
- **H2O Dataset Integration**: Convert and integrate H2O motion dataset for enhanced training

## 🏗️ System Architecture

### Data Processing Pipeline
```
Raw JSON Data → Preprocessing → Feature Extraction → Feature Selection → Model Training → Prediction
```

### Model Architecture
- **Input**: Hand joint positions (26 joints × 7 values) + Gaze data
- **Feature Compression**: 300 frames × 185 features → 740 statistical features
- **Feature Selection**: 740 → 100 most important features (86.5% reduction)
- **Ensemble Models**: Random Forest, Gradient Boosting, SVM
- **Output**: Motion classification with confidence scores

### Technical Specifications
- **Data Format**: JSON with joint positions, rotations, and gaze data
- **Input Dimensions**: (samples, 300, 185) → (samples, 100)
- **Model Performance**: 90.91% accuracy
- **Processing Time**: Real-time capable

## 📁 Project Structure

```
h-motion-training/
├── Preprocess/
│   ├── Core/
│   │   ├── data_preprocessor.py      # Data preprocessing pipeline
│   │   └── motion_classifier_bridge.py # Unity integration bridge
│   ├── Models/
│   │   └── ensemble_motion_classifier.py # Ensemble model training
│   ├── Data/                          # Training data (JSON files)
│   └── Tools/                         # Utility scripts
│       └── h2o_converter.py          # H2O dataset converter
├── H2O_Dataset/                       # H2O motion dataset
├── Outputs/                           # Generated models and results
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- scikit-learn
- numpy
- pandas
- matplotlib
- joblib

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd h-motion-training

# Install dependencies
pip install -r requirements.txt

# Navigate to preprocessing directory
cd Preprocess
```

## 📊 Data Format

### Input JSON Structure
```json
{
  "sessionId": "session-001",
  "startTime": "2025-01-01 00:00:00",
  "interactionTargetObject": "ObjectName",
  "interactionTargetPosition": {"x": 0.1, "y": 0.2, "z": 0.3},
  "frames": [
    {
      "timestamp": 0.0,
      "isRightHand": true,
      "joints": [
        {
          "jointName": "Wrist",
          "position": {"x": 0.1, "y": 0.1, "z": 0.1},
          "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
          "confidence": 1.0
        }
      ]
    }
  ]
}
```

### Supported Data Sources
- **Custom JSON**: Manually created motion data
- **H2O Dataset**: Converted from H2O motion dataset (270 files)
  - Pick actions: 84 files
  - Hold actions: 114 files  
  - Place actions: 72 files

### Supported Joint Names
- Wrist, ForearmWrist, Palm
- Thumb: Metacarpal, Proximal, Distal, Tip
- Index: Metacarpal, Proximal, Intermediate, Distal, Tip
- Middle: Metacarpal, Proximal, Intermediate, Distal, Tip
- Ring: Metacarpal, Proximal, Intermediate, Distal, Tip
- Pinky: Metacarpal, Proximal, Intermediate, Distal, Tip

## 🚀 Usage

### 1. Data Preprocessing
```bash
cd Preprocess
python Core/data_preprocessor.py
```

### 2. H2O Dataset Conversion (Optional)
```bash
cd Preprocess/Tools
python h2o_converter.py
```

### 3. Model Training
```bash
python Models/ensemble_motion_classifier.py
```

### 4. Unity Integration
```bash
python Core/motion_classifier_bridge.py
```

## 🎯 Model Performance

### Overall Accuracy: 90.91%

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|---------|-----------|---------|
| Pick  | 0.90      | 0.90    | 0.90      | 10      |
| Hold  | 1.00      | 1.00    | 1.00      | 2       |
| Place | 0.90      | 0.90    | 0.90      | 10      |

### Model Comparison
- **Random Forest**: 81.82%
- **Gradient Boosting**: 90.91% ⭐
- **SVM**: 90.91%
- **Ensemble**: 90.91%

## 🔍 Feature Engineering

### Feature Compression
- **Input**: 300 frames × 185 features = 55,500 dimensions
- **Compressed**: 740 statistical features
- **Compression Ratio**: 98.7%

### Feature Selection
- **Selected Features**: 100 out of 740 (13.5%)
- **Selection Method**: Univariate feature selection (F-score)
- **Top Features**: Statistical measures of joint movements

## 🐛 Problem Solving History

### Issue 1: Single Class Problem
- **Problem**: All samples classified as single class
- **Root Cause**: UTF-8 BOM encoding issues in JSON files
- **Solution**: Changed file reading from `utf-8` to `utf-8-sig`

### Issue 2: File Processing Errors
- **Problem**: 60 out of 110 files failed to process
- **Root Cause**: BOM characters causing JSON parsing failures
- **Solution**: Implemented proper encoding handling

### Issue 3: Model Loading Errors
- **Problem**: Bridge looking for wrong model files
- **Root Cause**: Hardcoded file paths
- **Solution**: Updated paths to match generated files

## 🔮 Future Improvements

### Short-term
- [x] Add more training data for better generalization (H2O dataset integration)
- [ ] Implement cross-validation for more reliable performance metrics
- [ ] Add data augmentation techniques
- [ ] Test model performance with expanded H2O dataset

### Long-term
- [ ] Integrate LSTM/RNN for temporal pattern learning
- [ ] Implement online learning for continuous improvement
- [ ] Add support for more motion types
- [ ] Optimize for edge devices
- [ ] Explore other motion datasets for further expansion

## 📝 Technical Notes

### Why Not RNN/LSTM?
- **Data Size**: 110 samples insufficient for deep learning
- **Real-time Requirements**: Traditional ML faster for inference
- **Interpretability**: Feature importance analysis possible
- **Stability**: Ensemble methods more robust with small datasets

### Performance Considerations
- **Feature Selection**: Critical for preventing overfitting
- **Data Quality**: UTF-8 BOM handling essential
- **Model Persistence**: Version compatibility important

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Unity for the robotics simulation platform
- scikit-learn for the machine learning framework
- The robotics research community for motion classification insights

---

**Last Updated**: August 28, 2025  
**Version**: 1.0.0  
**Status**: Production Ready
