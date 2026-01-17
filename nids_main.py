import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="AI-Based Network Intrusion Detection System",
    page_icon="🔒",
    layout="wide"
)

# Global variables
model = None
label_encoders = {}
feature_columns = []

def load_data():
    """
    Load network traffic data from CSV or generate synthetic data
    Returns: DataFrame with network traffic features and labels
    """
    try:
        # Try to load from data/raw directory
        data_path = os.path.join("data", "raw", "sample_dataset.csv")
        
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            st.success(f"✅ Dataset loaded from {data_path}")
            return df
        else:
            # Generate synthetic data if CSV not found
            st.warning("⚠️ Dataset not found. Generating synthetic data...")
            return generate_synthetic_data()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return generate_synthetic_data()

def generate_synthetic_data():
    """
    Generate synthetic network traffic data for demonstration
    Returns: DataFrame with synthetic network traffic features
    """
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'duration': np.random.exponential(1, n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'private'], n_samples),
        'flag': np.random.choice(['SF', 'REJ', 'RSTR'], n_samples),
        'src_bytes': np.random.exponential(1000, n_samples),
        'dst_bytes': np.random.exponential(500, n_samples),
        'land': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'wrong_fragment': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'urgent': np.random.choice([0, 1], n_samples, p=[0.98, 0.02]),
        'hot': np.random.randint(0, 10, n_samples),
        'num_failed_logins': np.random.randint(0, 5, n_samples),
        'logged_in': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'num_compromised': np.random.randint(0, 10, n_samples),
        'root_shell': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
        'su_attempted': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'num_root': np.random.randint(0, 5, n_samples),
        'num_file_creations': np.random.randint(0, 10, n_samples),
        'num_shells': np.random.randint(0, 5, n_samples),
        'num_access_files': np.random.randint(0, 20, n_samples),
        'num_outbound_cmds': np.zeros(n_samples),
        'is_host_login': np.random.choice([0, 1], n_samples, p=[0.99, 0.01]),
        'is_guest_login': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'count': np.random.randint(1, 100, n_samples),
        'srv_count': np.random.randint(1, 50, n_samples),
        'serror_rate': np.random.uniform(0, 1, n_samples),
        'srv_serror_rate': np.random.uniform(0, 1, n_samples),
        'rerror_rate': np.random.uniform(0, 1, n_samples),
        'srv_rerror_rate': np.random.uniform(0, 1, n_samples),
        'same_srv_rate': np.random.uniform(0, 1, n_samples),
        'diff_srv_rate': np.random.uniform(0, 1, n_samples),
        'srv_diff_host_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_count': np.random.randint(1, 255, n_samples),
        'dst_host_srv_count': np.random.randint(1, 255, n_samples),
        'dst_host_same_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_diff_srv_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_same_src_port_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_serror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_serror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_rerror_rate': np.random.uniform(0, 1, n_samples),
        'dst_host_srv_rerror_rate': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate labels (70% normal, 30% intrusion)
    labels = np.random.choice(['normal', 'intrusion'], n_samples, p=[0.7, 0.3])
    df['label'] = labels
    
    return df

def preprocess_data(df):
    """
    Preprocess the data: handle categorical variables and prepare features
    Returns: X (features), y (labels), processed DataFrame
    """
    global label_encoders, feature_columns
    
    # Make a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle categorical variables
    categorical_columns = ['protocol_type', 'service', 'flag']
    
    for col in categorical_columns:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Select feature columns (exclude original categorical and label)
    exclude_columns = categorical_columns + ['label']
    feature_columns = [col for col in df_processed.columns if col not in exclude_columns]
    
    # Prepare features and labels
    X = df_processed[feature_columns]
    y = df_processed['label']
    
    return X, y, df_processed

def train_model(X, y):
    """
    Train Random Forest classifier
    Returns: trained model, accuracy score, test predictions
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=10,
        min_samples_split=5
    )
    
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf_model, accuracy, y_test, y_pred, X_test

def simulate_live_traffic():
    """
    Simulate live network traffic for real-time prediction
    Returns: DataFrame with simulated traffic data
    """
    np.random.seed(int(np.random.random() * 1000))
    
    traffic_data = {
        'duration': np.random.exponential(1),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp']),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'private']),
        'flag': np.random.choice(['SF', 'REJ', 'RSTR']),
        'src_bytes': np.random.exponential(1000),
        'dst_bytes': np.random.exponential(500),
        'land': np.random.choice([0, 1], p=[0.99, 0.01]),
        'wrong_fragment': np.random.choice([0, 1], p=[0.95, 0.05]),
        'urgent': np.random.choice([0, 1], p=[0.98, 0.02]),
        'hot': np.random.randint(0, 10),
        'num_failed_logins': np.random.randint(0, 5),
        'logged_in': np.random.choice([0, 1], p=[0.3, 0.7]),
        'num_compromised': np.random.randint(0, 10),
        'root_shell': np.random.choice([0, 1], p=[0.95, 0.05]),
        'su_attempted': np.random.choice([0, 1], p=[0.9, 0.1]),
        'num_root': np.random.randint(0, 5),
        'num_file_creations': np.random.randint(0, 10),
        'num_shells': np.random.randint(0, 5),
        'num_access_files': np.random.randint(0, 20),
        'num_outbound_cmds': 0,
        'is_host_login': np.random.choice([0, 1], p=[0.99, 0.01]),
        'is_guest_login': np.random.choice([0, 1], p=[0.8, 0.2]),
        'count': np.random.randint(1, 100),
        'srv_count': np.random.randint(1, 50),
        'serror_rate': np.random.uniform(0, 1),
        'srv_serror_rate': np.random.uniform(0, 1),
        'rerror_rate': np.random.uniform(0, 1),
        'srv_rerror_rate': np.random.uniform(0, 1),
        'same_srv_rate': np.random.uniform(0, 1),
        'diff_srv_rate': np.random.uniform(0, 1),
        'srv_diff_host_rate': np.random.uniform(0, 1),
        'dst_host_count': np.random.randint(1, 255),
        'dst_host_srv_count': np.random.randint(1, 255),
        'dst_host_same_srv_rate': np.random.uniform(0, 1),
        'dst_host_diff_srv_rate': np.random.uniform(0, 1),
        'dst_host_same_src_port_rate': np.random.uniform(0, 1),
        'dst_host_srv_diff_host_rate': np.random.uniform(0, 1),
        'dst_host_serror_rate': np.random.uniform(0, 1),
        'dst_host_srv_serror_rate': np.random.uniform(0, 1),
        'dst_host_rerror_rate': np.random.uniform(0, 1),
        'dst_host_srv_rerror_rate': np.random.uniform(0, 1)
    }
    
    return pd.DataFrame([traffic_data])

def predict_traffic(traffic_df, trained_model):
    """
    Make predictions on live traffic data
    Returns: prediction and confidence score
    """
    global label_encoders, feature_columns
    
    # Preprocess the traffic data
    traffic_processed = traffic_df.copy()
    
    # Encode categorical variables
    categorical_columns = ['protocol_type', 'service', 'flag']
    for col in categorical_columns:
        if col in traffic_processed.columns and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen labels
            unique_values = set(traffic_processed[col].astype(str))
            known_values = set(le.classes_)
            
            # Replace unknown values with most common known value
            unknown_values = unique_values - known_values
            if unknown_values:
                most_common = le.classes_[0]
                traffic_processed.loc[traffic_processed[col].astype(str).isin(unknown_values), col] = most_common
            
            traffic_processed[col + '_encoded'] = le.transform(traffic_processed[col].astype(str))
    
    # Select features
    X_traffic = traffic_processed[feature_columns]
    
    # Make prediction
    prediction = trained_model.predict(X_traffic)[0]
    prediction_proba = trained_model.predict_proba(X_traffic)[0]
    confidence = max(prediction_proba)
    
    return prediction, confidence

def main():
    """
    Main Streamlit application
    """
    global model
    
    st.title("🔒 AI-Based Network Intrusion Detection System")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("🎛️ Control Panel")
    
    # Project Overview Section
    st.header("📊 Project Overview")
    st.markdown("""
    **AI-NIDS** is an intelligent Network Intrusion Detection System that uses Machine Learning 
    to identify potential security threats in network traffic.
    
    **Key Features:**
    - 🤖 Random Forest Classifier for binary classification
    - 📊 Real-time traffic simulation and prediction
    - 🎯 Interactive dashboard with live monitoring
    - 📁 Support for both synthetic and real datasets
    """)
    
    # Load and display data
    st.header("📁 Dataset Information")
    df = load_data()
    
    if df is not None:
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📈 Total Records", len(df))
            st.metric("🔢 Features", len(df.columns) - 1)
        
        with col2:
            label_counts = df['label'].value_counts()
            st.metric("✅ Normal Traffic", label_counts.get('normal', 0))
            st.metric("⚠️ Intrusions", label_counts.get('intrusion', 0))
        
        # Display sample data
        with st.expander("👀 View Sample Data"):
            st.dataframe(df.head(10))
        
        # Train Model Button
        st.header("🤖 Model Training")
        
        if st.button("🚀 Train Model Now", type="primary", use_container_width=True):
            with st.spinner("🔄 Training Random Forest model..."):
                try:
                    # Preprocess data
                    X, y, df_processed = preprocess_data(df)
                    
                    # Train model
                    model, accuracy, y_test, y_pred, X_test = train_model(X, y)
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.accuracy = accuracy
                    st.session_state.y_test = y_test
                    st.session_state.y_pred = y_pred
                    
                    st.success(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")
                    
                except Exception as e:
                    st.error(f"❌ Error during training: {e}")
        
        # Display results if model is trained
        if 'model' in st.session_state:
            model = st.session_state.model
            accuracy = st.session_state.accuracy
            y_test = st.session_state.y_test
            y_pred = st.session_state.y_pred
            
            # Display metrics
            st.subheader("📊 Model Performance")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Accuracy", f"{accuracy:.4f}")
            
            with col2:
                precision = accuracy_score(y_test, y_pred, pos_label='intrusion')
                st.metric("🔍 Precision (Intrusion)", f"{precision:.4f}")
            
            with col3:
                recall = accuracy_score(y_test, y_pred, pos_label='normal')
                st.metric("📈 Recall (Normal)", f"{recall:.4f}")
            
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=['normal', 'intrusion'])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Intrusion'],
                       yticklabels=['Normal', 'Intrusion'], ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Classification Report
            st.subheader("📋 Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(4))
    
    # Live Traffic Simulation
    st.header("🌐 Live Traffic Simulation")
    
    if model is None:
        st.warning("⚠️ Please train the model first to enable live traffic prediction.")
    else:
        # Traffic simulation controls
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎛️ Simulation Controls")
            
            # Protocol selection
            protocol = st.selectbox("Protocol Type", ['tcp', 'udp', 'icmp'])
            
            # Service selection
            service = st.selectbox("Service", ['http', 'ftp', 'smtp', 'private'])
            
            # Traffic volume sliders
            src_bytes = st.slider("Source Bytes", 0, 10000, 1000)
            dst_bytes = st.slider("Destination Bytes", 0, 5000, 500)
            
            duration = st.slider("Duration (seconds)", 0, 60, 5)
        
        with col2:
            st.subheader("📊 Traffic Characteristics")
            
            # Connection flags
            flag = st.selectbox("Connection Flag", ['SF', 'REJ', 'RSTR'])
            
            # Error rates
            serror_rate = st.slider("SYN Error Rate", 0.0, 1.0, 0.1)
            rerror_rate = st.slider("REJ Error Rate", 0.0, 1.0, 0.1)
            
            # Connection counts
            count = st.slider("Connection Count", 1, 100, 10)
            srv_count = st.slider("Service Count", 1, 50, 5)
        
        # Predict button
        if st.button("🔍 Analyze Traffic", type="secondary", use_container_width=True):
            # Create custom traffic data
            traffic_data = {
                'duration': duration,
                'protocol_type': protocol,
                'service': service,
                'flag': flag,
                'src_bytes': src_bytes,
                'dst_bytes': dst_bytes,
                'land': 0,
                'wrong_fragment': 0,
                'urgent': 0,
                'hot': 0,
                'num_failed_logins': 0,
                'logged_in': 1,
                'num_compromised': 0,
                'root_shell': 0,
                'su_attempted': 0,
                'num_root': 0,
                'num_file_creations': 0,
                'num_shells': 0,
                'num_access_files': 0,
                'num_outbound_cmds': 0,
                'is_host_login': 0,
                'is_guest_login': 0,
                'count': count,
                'srv_count': srv_count,
                'serror_rate': serror_rate,
                'srv_serror_rate': serror_rate,
                'rerror_rate': rerror_rate,
                'srv_rerror_rate': rerror_rate,
                'same_srv_rate': 0.8,
                'diff_srv_rate': 0.2,
                'srv_diff_host_rate': 0.1,
                'dst_host_count': 25,
                'dst_host_srv_count': 20,
                'dst_host_same_srv_rate': 0.8,
                'dst_host_diff_srv_rate': 0.2,
                'dst_host_same_src_port_rate': 0.7,
                'dst_host_srv_diff_host_rate': 0.1,
                'dst_host_serror_rate': serror_rate,
                'dst_host_srv_serror_rate': serror_rate,
                'dst_host_rerror_rate': rerror_rate,
                'dst_host_srv_rerror_rate': rerror_rate
            }
            
            traffic_df = pd.DataFrame([traffic_data])
            
            # Make prediction
            prediction, confidence = predict_traffic(traffic_df, model)
            
            # Display results
            st.subheader("🎯 Prediction Results")
            
            if prediction == 'intrusion':
                st.error(f"🚨 **INTRUSION DETECTED!**")
                st.error(f"Confidence: {confidence:.4f}")
                st.error("⚠️ Immediate attention required!")
            else:
                st.success(f"✅ **NORMAL TRAFFIC**")
                st.success(f"Confidence: {confidence:.4f}")
                st.success("🛡️ No suspicious activity detected")
            
            # Display traffic details
            with st.expander("📋 Traffic Details"):
                st.json(traffic_data)
    
    # Auto-simulation section
    st.header("🔄 Automatic Traffic Simulation")
    
    if model is not None:
        if st.button("🎲 Generate Random Traffic", use_container_width=True):
            # Simulate random traffic
            random_traffic = simulate_live_traffic()
            prediction, confidence = predict_traffic(random_traffic, model)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 'intrusion':
                    st.error("🚨 Intrusion Detected")
                else:
                    st.success("✅ Normal Traffic")
                
                st.metric("Confidence", f"{confidence:.4f}")
            
            with col2:
                st.subheader("Traffic Summary")
                st.write(f"Protocol: {random_traffic['protocol_type'].iloc[0]}")
                st.write(f"Service: {random_traffic['service'].iloc[0]}")
                st.write(f"Source Bytes: {random_traffic['src_bytes'].iloc[0]:.0f}")
                st.write(f"Duration: {random_traffic['duration'].iloc[0]:.2f}s")
    
    # Footer
    st.markdown("---")
    st.markdown("🔒 **AI-NIDS** - AI-Based Network Intrusion Detection System")
    st.markdown("📚 Academic Project for Machine Learning in Cybersecurity")

if __name__ == "__main__":
    main()
