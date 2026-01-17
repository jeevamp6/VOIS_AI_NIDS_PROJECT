# 🔒 AI-Based Network Intrusion Detection System (AI-NIDS)

An intelligent Network Intrusion Detection System that uses Machine Learning to identify potential security threats in network traffic. This project demonstrates the application of Random Forest classifiers for binary classification of network traffic as either **Normal** or **Intrusion**.

## 📋 Project Overview

AI-NIDS is designed for academic demonstration and final-year evaluation, featuring:

- 🤖 **Random Forest Classifier** for binary classification
- 📊 **Interactive Streamlit Dashboard** with real-time monitoring
- 🎯 **Live Traffic Simulation** with customizable parameters
- 📁 **Flexible Dataset Support** (CSV files or synthetic data generation)
- 📈 **Comprehensive Model Evaluation** with metrics and visualizations

## 🛠️ Technology Stack

- **Python 3.8+**
- **scikit-learn** - Machine Learning algorithms
- **Streamlit** - Interactive web dashboard
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Seaborn & Matplotlib** - Data visualization

## 📁 Project Structure

```
AI_NIDS_Project/
│
├── data/
│   └── raw/
│       └── sample_dataset.csv          # Sample network traffic dataset
│
├── nids_main.py                        # Main application with ML pipeline
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone or Download the Project

```bash
# If using git
git clone <repository-url>
cd AI_NIDS_Project

# Or download and extract the project folder
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv nids_env

# Activate virtual environment
# On Windows:
nids_env\Scripts\activate
# On macOS/Linux:
source nids_env/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Dataset Placement

Ensure your dataset is placed in the correct location:

```
AI_NIDS_Project/
└── data/
    └── raw/
        └── sample_dataset.csv
```

**Note**: The system will automatically generate synthetic data if no CSV file is found.

## 🎮 Running the Application

### Method 1: Using Streamlit (Recommended)

```bash
streamlit run nids_main.py
```

The application will automatically open in your web browser at:
**http://localhost:8501**

### Method 2: Direct Python Execution

```bash
python nids_main.py
```

## 📊 Dataset Information

### Default Dataset Features

The system expects network traffic data with the following features:

- **duration**: Connection duration (seconds)
- **protocol_type**: Protocol (tcp, udp, icmp)
- **service**: Network service (http, ftp, smtp, etc.)
- **flag**: Connection status flag
- **src_bytes**: Bytes sent from source to destination
- **dst_bytes**: Bytes sent from destination to source
- **Additional features**: Error rates, connection counts, host-based statistics
- **label**: Target variable (normal/intrusion)

### Using Custom Datasets

1. Place your CSV file in `data/raw/`
2. Ensure it contains the same feature columns as the sample dataset
3. The system will automatically detect and load your dataset
4. Supported format: CIC-IDS2017 compatible

## 🎛️ Application Features

### 1. Project Overview
- System description and capabilities
- Key features and technology stack

### 2. Dataset Information
- Real-time dataset statistics
- Sample data preview
- Class distribution metrics

### 3. Model Training
- **One-click training** with Random Forest classifier
- Automatic data preprocessing and encoding
- Real-time training progress

### 4. Model Performance
- Accuracy, precision, and recall metrics
- Confusion matrix visualization
- Detailed classification report

### 5. Live Traffic Simulation
- **Manual traffic simulation** with customizable parameters
- **Random traffic generation** for testing
- Real-time intrusion detection
- Confidence scores for predictions

### 6. Interactive Controls
- Protocol and service selection
- Traffic volume sliders
- Error rate adjustments
- Connection count parameters

## 🔧 Troubleshooting

### Common Issues

#### 1. Module Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

#### 2. Dataset Loading Issues
- Ensure CSV file is in `data/raw/` directory
- Check file permissions
- Verify CSV format matches expected structure

#### 3. Streamlit Not Opening
```bash
# Try specifying port
streamlit run nids_main.py --server.port 8501

# Check if port is in use
netstat -ano | findstr :8501
```

#### 4. Model Training Errors
- Check dataset for missing values
- Ensure sufficient data samples (minimum 50 records)
- Verify label column exists and contains 'normal'/'intrusion'

#### 5. Virtual Environment Issues
```bash
# Deactivate and recreate environment
deactivate
python -m venv nids_env
nids_env\Scripts\activate
pip install -r requirements.txt
```

### Performance Optimization

- For large datasets (>10,000 records), consider reducing sample size
- Use GPU acceleration if available (install cupy)
- Close unnecessary browser tabs for better UI performance

## 📚 Academic Usage

### For Final Year Projects

This project is suitable for:
- **Computer Science** - Machine Learning applications
- **Cybersecurity** - Intrusion detection systems
- **Data Science** - Classification algorithms
- **Information Technology** - Network security

### Extensions and Enhancements

1. **Advanced ML Models**: Implement deep learning (LSTM, CNN)
2. **Real-time Integration**: Connect to live network feeds
3. **Multi-class Classification**: Detect specific attack types
4. **Feature Engineering**: Add network behavior analysis
5. **Alert System**: Email/SMS notifications for intrusions

### Research Opportunities

- Compare different ML algorithms
- Analyze feature importance
- Study adversarial attacks on IDS
- Optimize for real-time performance
- Implement federated learning for privacy

## 🤝 Contributing

### Code Style
- Follow PEP 8 guidelines
- Add comprehensive comments
- Use descriptive variable names
- Include error handling

### Testing
- Test with different datasets
- Verify edge cases
- Check UI responsiveness
- Validate model performance

## 📄 License

This project is for educational purposes. Please ensure proper attribution if used in academic work.

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Verify your installation steps
3. Review dataset format requirements
4. Test with the provided sample dataset

---

**🔒 AI-NIDS** - Protecting Networks with Intelligence

*Academic Project for Machine Learning in Cybersecurity*
