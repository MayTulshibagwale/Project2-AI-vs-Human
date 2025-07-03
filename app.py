# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================
# Core Libraries for Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Text Preprocessing & NLP
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Document Processing
import docx
import PyPDF2
import pdfplumber
from io import BytesIO

# Deep Learning
import torch
import torch.nn as nn
import torch.nn.functional as F

#=========================================================================================================
#=========================================================================================================

class CNN_TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim,
                 dropout_rate, embedding_matrix, freeze_embeddings=True):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim,
                      out_channels=n_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)             
        embedded = embedded.permute(0, 2, 1)         
        conved = [F.relu(conv(embedded)) for conv in self.convs]  
        pooled = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in conved] 
        cat = self.dropout(torch.cat(pooled, dim=1)) 
        return self.fc(cat)  

class LSTM_TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, freeze_embeddings=True):
        super(LSTM_TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        final_feature_map = lstm_out[:, -1, :]
        return self.fc(final_feature_map)

class RNN_TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim, output_dim, n_layers, dropout_rate, freeze_embeddings=True):
        super(RNN_TextClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(embedding_matrix)
        self.embedding.weight.requires_grad = not freeze_embeddings

        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        rnn_out, _ = self.rnn(embedded)
        final_feature_map = rnn_out[:, -1, :]
        return self.fc(final_feature_map)

#=========================================================================================================
#=========================================================================================================
   
# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud

#=========================================================================================================
#=========================================================================================================

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""  # Avoid None
    except Exception as e:
        st.error(f"Failed to extract text from PDF: {e}")
    return text

def extract_text_from_docx(uploaded_file):
    text = ""
    try:
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Failed to extract text from DOCX: {e}")
    return text

#=========================================================================================================
#=========================================================================================================
# Page Configuration
st.set_page_config(
    page_title="AI vs Human Text Detection",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# MODEL LOADING SECTION (FINAL CORRECTED VERSION)
# ============================================================================

@st.cache_resource
def load_models():
    """
    Loads all trained ML models (SVM, Decision Tree, AdaBoost) and the TF-IDF vectorizer.
    Models are loaded from the 'models/' directory.
    """
    models = {}
    # Define the directory where your .pkl files are stored.
    # This variable should be a string, e.g., 'models'.
    model_dir = 'models' 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try: # This is the main try block for the entire function
        #st.info(f"Attempting to load models from the '{model_dir}' directory...", icon="📦")

        # Load TF-IDF Vectorizer
        try:
            # Correct usage of os.path.join: combine the directory variable with the filename
            models['vectorizer'] = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
            models['vectorizer_available'] = True
            #st.success("TF-IDF Vectorizer loaded successfully.", icon="✅")
        except FileNotFoundError:
            st.error(f"TF-IDF Vectorizer not found at '{os.path.join(model_dir, 'tfidf_vectorizer.pkl')}'. "
                     "Please ensure the file exists in the 'models/' directory.", icon="❌")
            models['vectorizer_available'] = False
        except Exception as e:
            st.error(f"Error loading TF-IDF Vectorizer: {e}", icon="❌")
            models['vectorizer_available'] = False

        # Load SVM Model (your 'svm_model.pkl' is expected to be a pipeline)
        try:
            # Correct usage of os.path.join
            models['svm'] = joblib.load(os.path.join(model_dir, 'svm_model.pkl'))
            models['svm_available'] = True
            #st.success("SVM Model (Pipeline) loaded successfully.", icon="✅")
        except FileNotFoundError:
            st.error(f"SVM Model not found at '{os.path.join(model_dir, 'svm_model.pkl')}'. "
                     "Please ensure the file exists in the 'models/' directory.", icon="❌")
            models['svm_available'] = False
        except Exception as e:
            st.error(f"Error loading SVM Model: {e}", icon="❌")
            models['svm_available'] = False

        # Load Decision Tree Model (your 'decision_tree_model.pkl' is expected to be a pipeline)
        try:
            # Correct usage of os.path.join
            models['dt'] = joblib.load(os.path.join(model_dir, 'decision_tree_model.pkl'))
            models['dt_available'] = True
            #st.success("Decision Tree Model (Pipeline) loaded successfully.", icon="✅")
        except FileNotFoundError:
            st.error(f"Decision Tree Model not found at '{os.path.join(model_dir, 'decision_tree_model.pkl')}'. "
                     "Please ensure the file exists in the 'models/' directory.", icon="❌")
            models['dt_available'] = False
        except Exception as e:
            st.error(f"Error loading Decision Tree Model: {e}", icon="❌")
            models['dt_available'] = False

        # Load AdaBoost Model (your 'adaboost_model.pkl' is expected to be a pipeline)
        try:
            # Correct usage of os.path.join
            models['ab'] = joblib.load(os.path.join(model_dir, 'adaboost_model.pkl'))
            models['ab_available'] = True
            #st.success("AdaBoost Model (Pipeline) loaded successfully.", icon="✅")
        except FileNotFoundError:
            st.error(f"AdaBoost Model not found at '{os.path.join(model_dir, 'adaboost_model.pkl')}'. "
                     "Please ensure the file exists in the 'models/' directory.", icon="❌")
            models['ab_available'] = False
        except Exception as e: # This 'except' needs to be correctly aligned with its 'try'
            st.error(f"Error loading AdaBoost Model: {e}", icon="❌")
            models['ab_available'] = False

        # --- ADDED: Load DL Specific Resources (vocab, embedding_matrix, MAX_SEQUENCE_LENGTH) ---
        try:
            models['word_to_idx'] = joblib.load(os.path.join(model_dir, 'word_to_idx.pkl')) 
            models['MAX_SEQUENCE_LENGTH'] = joblib.load(os.path.join(model_dir, 'MAX_SEQUENCE_LENGTH.pkl'))
            # For embedding_matrix, ensure you saved it with torch.save() (.pt or .pth)
            models['embedding_matrix'] = torch.load(os.path.join(model_dir, 'embedding_matrix.pt'), map_location=device) 
            
            models['vocab_size'] = len(models['word_to_idx'])
            models['embedding_dim'] = models['embedding_matrix'].shape[1] # Infer from matrix
            models['dl_resources_available'] = True
            #st.success("Deep Learning resources (vocab, embeddings, seq_len) loaded.", icon="✅")
        except FileNotFoundError as e:
            st.warning(f"DL resources not found: {e}. Ensure word_to_idx.pkl, MAX_SEQUENCE_LENGTH.pkl, embedding_matrix.pt are saved in 'models/'. Skipping DL models.", icon="⚠️")
            models['dl_resources_available'] = False
        except Exception as e:
            st.error(f"Error loading DL resources: {e}. Skipping DL models.", icon="❌")
            models['dl_resources_available'] = False

        # --- ADDED: Load DL Models (CNN, LSTM, RNN) ---
        if models.get('dl_resources_available'):
            # IMPORTANT: CNN_TextClassifier, LSTM_TextClassifier, RNN_TextClassifier classes MUST be defined at top of app.py.
            dl_models_config = {
                'cnn': {'class': CNN_TextClassifier, 'file': 'CNN.pkl', 'params': {'n_filters': 100, 'filter_sizes': [3, 4, 5], 'output_dim': 1, 'dropout_rate': 0.5, 'freeze_embeddings': True}},
                'lstm': {'class': LSTM_TextClassifier, 'file': 'LSTM.pkl', 'params': {'hidden_dim': 128, 'output_dim': 1, 'n_layers': 2, 'bidirectional': True, 'dropout_rate': 0.5, 'freeze_embeddings': True}},
                'rnn': {'class': RNN_TextClassifier, 'file': 'RNN.pkl', 'params': {'hidden_dim': 128, 'output_dim': 1, 'n_layers': 2, 'dropout_rate': 0.5, 'freeze_embeddings': True}},
            }
            for key, config in dl_models_config.items():
                try:
                    # Instantiate the model with correct parameters from training
                    dl_model_instance = config['class'](
                        vocab_size=models['vocab_size'],
                        embedding_dim=models['embedding_dim'],
                        embedding_matrix=models['embedding_matrix'],
                        **config['params'] # Unpack other model-specific parameters
                    )
                    # Load the state dictionary
                    dl_model_instance.load_state_dict(torch.load(os.path.join(model_dir, config['file']), map_location=device))
                    dl_model_instance.eval() # Set to evaluation mode
                    dl_model_instance.to(device) # Move to device
                    models[key] = dl_model_instance
                    models[f'{key}_available'] = True
                    #st.success(f"{key.upper()} Model loaded.", icon="✅")
                except FileNotFoundError:
                    st.warning(f"{key.upper()} Model not found at '{os.path.join(model_dir, config['file'])}'. Skipping.", icon="⚠️")
                    models[f'{key}_available'] = False
                except Exception as e:
                    st.error(f"Error loading {key.upper()} Model: {e}. Ensure class is defined and parameters match.", icon="❌")
                    models[f'{key}_available'] = False
        else:
            st.info("Skipping Deep Learning model loading as core DL resources are unavailable.", icon="ℹ️")

        # --- Final check for critical components ---
        # Ensure TF-IDF vectorizer is loaded AND at least one classifier is available.
        has_vectorizer = models.get('vectorizer_available', False)
        has_any_classifier = models.get('svm_available', False) or \
                             models.get('dt_available', False) or \
                             models.get('ab_available', False) or \
                             models.get('cnn_available', False) or \
                             models.get('lstm_available', False) or \
                             models.get('rnn_available', False) 

        if not has_vectorizer:
            st.error(" Critical Error: TF-IDF Vectorizer is missing. Cannot proceed without feature extraction.", icon="🛑")
            return None # Return None to indicate a critical failure, stopping further app execution
        if not has_any_classifier:
            st.error(" Critical Error: No classification models (SVM, DT, AdaBoost) were loaded. Cannot proceed with predictions.", icon="🛑")
            return None # Return None to indicate a critical failure

        st.success("All essential models and vectorizer are ready for use!", icon="🎉")
        return models # This is the ONLY 'return models' that should be here, at the end of the main 'try' block

    except Exception as e: # This is the outer except block for any general loading errors
        st.error(f"An unexpected error occurred during model loading: {e}", icon="❌")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def text_cleaning(text):
    # Remove special characters, digits, and extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)      # Remove digits
    text = text.lower().strip()
    return text

def stop_word_removal(text):
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def word_normalization(text):
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    normalized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(normalized_words)

def make_prediction(text, model_choice, models):
    """Make prediction using the selected model"""
    if models is None:
        st.error("Models not loaded. Cannot make prediction.", icon="❌")
        return None, None

    try:
        prediction = None
        probabilities = None
        current_device = models.get('device', 'cpu')

        # 1. Preprocess the raw input text (REQUIRED for all models)
        # These functions (text_cleaning, stop_word_removal, word_normalization)
        # must be defined at the top of app.py.
        processed_text = text_cleaning(text)
        processed_text = stop_word_removal(processed_text)
        processed_text = word_normalization(processed_text)
        
        # --- ML Model Predictions ---
        if model_choice == "svm_model" and models.get('svm_available'):
            # Use the complete pipeline (SVM Model)
            prediction = models['svm'].predict([processed_text])[0]
            probabilities = models['svm'].predict_proba([processed_text])[0]
            
        elif model_choice == "decision_tree_model" and models.get('dt_available'):
            prediction = models['dt'].predict([processed_text])[0]
            probabilities = models['dt'].predict_proba([processed_text])[0]

        elif model_choice == "adaboost_model" and models.get('ab_available'): # FIXED: Changed 'models.get' to 'model_choice'
            prediction = models['ab'].predict([processed_text])[0] # FIXED: Input to predict should be [processed_text]
            probabilities = models['ab'].predict_proba([processed_text])[0] # FIXED: Input to predict_proba should be [processed_text]

        # --- Deep Learning Model Predictions ---
        elif models.get('dl_resources_available') and (model_choice in ["cnn_model", "lstm_model", "rnn_model"]):
            # Prepare text for DL model (tokenization, indexing, padding)
            word_to_idx = models['word_to_idx']
            MAX_SEQUENCE_LENGTH = models['MAX_SEQUENCE_LENGTH']

            # Convert processed_text to indexed sequence
            indexed_text = [word_to_idx.get(word, word_to_idx["<UNK>"]) for word in processed_text.split()]
            if len(indexed_text) > MAX_SEQUENCE_LENGTH:
                indexed_text = indexed_text[:MAX_SEQUENCE_LENGTH]
            else:
                indexed_text = indexed_text + [word_to_idx["<PAD>"]] * (MAX_SEQUENCE_LENGTH - len(indexed_text))
            
            # Convert to PyTorch tensor and move to device
            input_tensor = torch.tensor([indexed_text], dtype=torch.long).to(current_device)

            model_to_use = None
            if model_choice == "cnn_model" and models.get('cnn_available'):
                model_to_use = models['cnn']
            elif model_choice == "lstm_model" and models.get('lstm_available'):
                model_to_use = models['lstm']
            elif model_choice == "rnn_model" and models.get('rnn_available'):
                model_to_use = models['rnn']
            
            if model_to_use:
                model_to_use.eval()
                with torch.no_grad():
                    outputs = model_to_use(input_tensor)
                    probabilities_tensor = torch.sigmoid(outputs)
                    prediction_tensor = (probabilities_tensor >= 0.5).float()

                    prediction = prediction_tensor.cpu().numpy()[0]
                    probabilities = probabilities_tensor.cpu().numpy()[0]
            else:
                st.warning(f"Selected Deep Learning model '{model_choice}' is not available.", icon="⚠️")
                return None, None
            
        if prediction is not None and probabilities is not None:
            class_names = ['Human-written', 'AI-written']
            prediction_label = class_names[int(prediction)] # Ensure prediction is int for indexing
            return prediction_label, probabilities
        else:
            st.error(f"Prediction logic for model '{model_choice}' failed or model is not available.", icon="❌")
            return None, None
            
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}", icon="❌")
        # For debugging, you might temporarily uncomment this:
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}")
        return None, None

def get_available_models(models):
    """Get list of available models for selection"""
    available = []
    
    if models is None:
        return available
    
    if models.get('svm_available'):
        available.append(("svm_model", "SVM Model")) # FIXED: Removed extra space
    if models.get('dt_available'):
        available.append(("decision_tree_model", "Decision Tree")) # FIXED: Removed extra space
    if models.get('ab_available'):
        available.append(("adaboost_model", "AdaBoost Model")) # FIXED: Removed extra space
    
    # ADDED: Deep Learning Models
    if models.get('cnn_available'):
        available.append(("cnn_model", "CNN Model"))
    if models.get('lstm_available'):
        available.append(("lstm_model", "LSTM Model"))
    if models.get('rnn_available'):
        available.append(("rnn_model", "RNN Model"))

    return available
# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Document Analysis", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()

# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Detection App</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to your **AI vs Human Text Detection** web application! This app analyzes text from various sources
    and predicts whether it was written by an **AI** or a **human**, providing confidence scores.
    You can upload documents, paste text, or compare different machine learning (**SVM, Decision Tree, AdaBoost**)
    and deep learning (**CNN, LSTM, RNN**) models.
    """)
    
    # App overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter text manually
        - Choose between models
        - Get instant predictions
        - See confidence scores
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Document Analysis
        - Upload MS Word (.docx) or PDF (.pdf) files
        - Extract text for analysis
        - Process single documents or multiple texts
        - Download analysis reports
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare multiple models on a single text
        - Side-by-side performance comparison
        - Analyze model agreement
        - Feature importance insights
        """)
    
    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            if models.get('svm_available'):
                st.info("** SVM Model**\n✅ Available")
            else:
                st.warning("**SVM Model**\n❌ Not Available")
        
        with col2:
            if models.get('dt_available'):
                st.info("**🎯 Decision Tree**\n✅ Available")
            else:
                st.warning("**🎯 Decision Tree**\n❌ Not Available")
        
        with col3:
            if models.get('ab_available'):
                st.info("**🔤 AdaBoost**\n✅ Available")
            else:
                st.warning("**🔤 AdaBoost**\n❌ Not Available")
        
        # ADDED: CNN Model Status
        with col4:
            if models.get('cnn_available'):
                st.info("**CNN Model**\n✅ Available")
            else:
                st.warning("**CNN Model**\n❌ Not Available")
        
        # ADDED: LSTM Model Status
        with col5:
            if models.get('lstm_available'):
                st.info("**LSTM Model**\n✅ Available")
            else:
                st.warning("**LSTM Model**\n❌ Not Available")
        
        # ADDED: RNN Model Status
        with col6:
            if models.get('rnn_available'):
                st.info("**RNN Model**\n✅ Available")
            else:
                st.warning("**RNN Model**\n❌ Not Available")
        
    else:
        st.error("❌ Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================

elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to determine if it was AI or human-written.")
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )
            
            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here to analyze for AI vs Human authorship...",
                height=150
            )
            
            # Character count
            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")
            
            # Example texts
            with st.expander("📝 Try these example texts"):
                examples = [
                    "The rapid advancement of artificial intelligence is transforming various industries, leading to enhanced efficiency and innovative solutions across diverse sectors.",
                    "As an AI language model, I do not have personal experiences or emotions. My responses are generated based on the data I have been trained on.",
                    "I woke up this morning feeling a bit groggy, but a strong cup of coffee and the anticipation of a productive day quickly brightened my mood. I plan to tackle my coding project first.",
                    "While artificial intelligence has made significant strides in replicating human cognitive functions, it still lacks the nuanced understanding of context and emotional intelligence inherent in human communication.",
                    "It was a really long day, and honestly, I'm just looking forward to ordering some pizza and binging my favorite show. Maybe I'll even finish that book I started last week, who knows!"
                ]
                
                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state.user_input = example
                            st.rerun()
            
            # Use session state for user input
            if 'user_input' in st.session_state:
                user_input = st.session_state.user_input
            
            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text for authorship...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)
                        
                        if prediction and probabilities is not None:
                            # Display prediction
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                if prediction == "Human-written":
                                    st.success(f"🎯 Prediction: **{prediction}**")
                                else:
                                    st.error(f"🎯 Prediction: **{prediction}**")
                            
                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")
                            
                            # Create probability chart
                            st.subheader("📊 Prediction Probabilities")
                            
                            # Detailed probabilities
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("😞 Human-written", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("😊 AI-written", f"{probabilities[1]:.1%}")
                            
                            # Bar chart
                            class_names = ['Human-written', 'AI-written']
                            prob_df = pd.DataFrame({
                                'Authorship': class_names,
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Authorship'), height=300)
                            
                        else:
                            st.error("Failed to make prediction. Please check the console for errors.", icon="❌")
                else:
                    st.warning("Please enter some text to classify!", icon="⚠️")
        else:
            st.error("No models available for prediction. Please ensure models are loaded correctly in the 'models/' directory.", icon="❌")
    else:
        st.warning("Models not loaded. Please check the model files.", icon="⚠️")

# ============================================================================
# DOCUMENT ANALYSIS PAGE
# ============================================================================

elif page == "📁 Document Analysis":
    st.header("📁 Upload File for Document Analyst")
    st.markdown("Upload MS Word (.docx) or PDF (.pdf) files to extract and analyze text for AI vs Human authorship.")
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # File upload
            uploaded_file = st.file_uploader(
                "Choose a document (or text file/CSV)",
                type=['txt', 'csv','pdf', 'docx'],
                help="Upload a .txt file (one text per line) or .csv file (text in first column), .pdf, or .docx file." 
            )
            
            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for analysis:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )
                
                # Process file
                if st.button("📊 Analyze Document"):
                    try:

                        texts_to_process = []

                        # Read file content
                        if uploaded_file.type == "text/plain":
                            content = str(uploaded_file.read(), "utf-8")
                            texts_to_process = [line.strip() for line in content.split('\n') if line.strip()]
                        elif uploaded_file.type == "text/csv":
                            df = pd.read_csv(uploaded_file)
                            texts_to_process = df.iloc[:, 0].astype(str).tolist()
                        elif uploaded_file.type == "application/pdf":
                            with st.spinner("Extracting text from PDF..."):
                                extracted_text = extract_text_from_pdf(uploaded_file) # Use your function
                            if extracted_text.strip():
                                texts_to_process = [extracted_text] # Treat entire PDF as one text entry
                            else:
                                st.warning("Could not extract readable text from PDF. File might be scanned or corrupted.", icon="⚠️")
                        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            with st.spinner("Extracting text from DOCX..."):
                                extracted_text = extract_text_from_docx(uploaded_file) # Use your function
                            if extracted_text.strip():
                                texts_to_process = [extracted_text] # Treat entire DOCX as one text entry
                            else:
                                st.warning("Could not extract readable text from DOCX. File might be empty or corrupted.", icon="⚠️")
                        else:
                            st.error("Unsupported file type. Please upload a .txt, .csv, .pdf, or .docx file.", icon="❌")

                        if not texts_to_process:
                            st.error("No valid text found in the uploaded file to process.", icon="❌")
                        else:
                            st.info(f"Processing {len(texts_to_process)} text(s) from the document...", icon="📄")
                            
                            # Process all texts
                            results = []
                            progress_bar = st.progress(0)
                            
                            for i, text_content in enumerate(texts_to_process):
                                if text_content.strip():
                                    prediction, probabilities = make_prediction(text_content, model_choice, models)
                                    
                                    if prediction and probabilities is not None:
                                        results.append({
                                            'Text_Snippet': text_content[:200] + "..." if len(text_content) > 200 else text_content, # Show longer snippet
                                            'Full_Text': text_content,
                                            'Prediction': prediction,
                                            'Confidence': f"{max(probabilities):.1%}",
                                            # CHANGED: Probability labels
                                            'Human_Prob': f"{probabilities[0]:.1%}",
                                            'AI_Prob': f"{probabilities[1]:.1%}"
                                        })
                                progress_bar.progress((i + 1) / len(texts_to_process))
                            
                            if results:
                                st.success(f"✅ Analysis complete for {len(results)} text(s)!", icon="🎉")
                                
                                results_df = pd.DataFrame(results)
                                
                                # Summary statistics
                                st.subheader("📊 Analysis Summary") # CHANGED: Header text
                                col1, col2, col3, col4 = st.columns(4)
                                
                                human_count = sum(1 for r in results if r['Prediction'] == 'Human-written')
                                ai_count = len(results) - human_count
                                avg_confidence = np.mean([float(r['Confidence'].strip('%')) for r in results])
                                
                                with col1:
                                    st.metric("Total Texts Analyzed", len(results))
                                with col2:
                                    st.metric("👤 Human-written", human_count)
                                with col3:
                                    st.metric("🤖 AI-written", ai_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                                
                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(
                                    results_df[['Text_Snippet', 'Prediction', 'Confidence']],
                                    use_container_width=True
                                )
                                
                                # Download option
                                csv = results_df.to_csv(index=False).encode('utf-8') # Ensure CSV is encoded for download
                                st.download_button(
                                    label="📥 Download Full Analysis Report (.csv)", # CHANGED: Label
                                    data=csv,
                                    file_name=f"authorship_analysis_report_{model_choice}_{uploaded_file.name}.csv", # CHANGED: File name
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed from the document.", icon="❌")
                                
                    except Exception as e:
                        st.error(f"An error occurred during document analysis: {e}", icon="❌") # CHANGED: Message
            else:
                st.info("Please upload a document to get started.", icon="ℹ️")
                
                # Show example file formats
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):** (Each line treated as a separate text)
                    ```
                    The author's unique voice and nuanced storytelling suggest human creativity.
                    As a large language model, I generate responses based on complex algorithms.
                    ```
                    
                    **CSV File (.csv):** (Text in the first column)
                    ```
                    text,id
                    "The unexpected twist in the narrative truly captivated my imagination.",1
                    "AI systems are capable of processing vast amounts of data efficiently.",2
                    ```

                    **PDF File (.pdf) & Word File (.docx):**
                    - The app will extract all readable text from the document.
                    - The entire extracted text will be analyzed as a single input.
                    """)
        else:
            st.error("No models available for document analysis. Please ensure models are loaded correctly.", icon="❌") # CHANGED: Message
    else:
        st.warning("Models not loaded. Please check the model files.", icon="⚠️")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")
    
    if models:
        available_models = get_available_models(models)
        
        if len(available_models) >= 2:
            # Text input for comparison
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Enter text to see how different models perform...",
                height=100
            )
            
            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")
                
                # Get predictions from all available models
                comparison_results = []
                
                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)
                    
                    if prediction and probabilities is not None:
                        comparison_results.append({
                            'Model': model_name,
                            'Prediction': prediction,
                            'Confidence': f"{max(probabilities):.1%}",
                            'Negative %': f"{probabilities[0]:.1%}",
                            'Positive %': f"{probabilities[1]:.1%}",
                            'Raw_Probs': probabilities
                        })
                
                if comparison_results:
                    # Comparison table
                    comparison_df = pd.DataFrame(comparison_results)
                    st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Negative %', 'Positive %']])
                    
                    # Agreement analysis
                    predictions = [r['Prediction'] for r in comparison_results]
                    if len(set(predictions)) == 1:
                        st.success(f"✅ All models agree: **{predictions[0]} Sentiment**")
                    else:
                        st.warning("⚠️ Models disagree on prediction")
                        for result in comparison_results:
                            model_name = result['Model'].split(' ')[1] if ' ' in result['Model'] else result['Model']
                            st.write(f"- {model_name}: {result['Prediction']}")
                    
                    # Side-by-side probability charts
                    st.subheader("📊 Detailed Probability Comparison")
                    
                    cols = st.columns(len(comparison_results))
                    
                    for i, result in enumerate(comparison_results):
                        with cols[i]:
                            model_name = result['Model']
                            st.write(f"**{model_name}**")
                            
                            chart_data = pd.DataFrame({
                                'Sentiment': ['Negative', 'Positive'],
                                'Probability': result['Raw_Probs']
                            })
                            st.bar_chart(chart_data.set_index('Sentiment'))
                    
                else:
                    st.error("Failed to get predictions from models")
        
        elif len(available_models) == 1:
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
            
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")
    
    if models:
        st.success("✅ Models are loaded and ready!")
        
        # Model details
        st.subheader("🔧 Machine Learning Models") # CHANGED: Header to specifically mention ML models
        st.markdown("Your app supports the following traditional machine learning models:")
        
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 📈 SVM Model
            **Type:** Support Vector Machine (SVM)
            **Algorithm:** SVC (Support Vector Classifier)
            **Features:** TF-IDF vectors
            
            **Strengths:**
            - Highly effective in high-dimensional spaces.
            - Uses a kernel trick to handle non-linear relationships.
            - Robust against overfitting with proper regularization (C parameter).
            """)
            
        with col2:
            st.markdown("""
            ### 🎯 Decision Tree
            **Type:** Tree-based Classification Model
            **Algorithm:** DecisionTreeClassifier
            **Features:** TF-IDF vectors
            
            **Strengths:**
            - Easy to understand and interpret (can be visualized).
            - Requires less data preprocessing (no need for feature scaling).
            - Can handle both numerical and categorical data.
            """)
        
        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features:** 5,000 most important terms
        - **N-grams:** Unigrams (1-word) and Bigrams (2-word phrases)
        - **Min Document Frequency:** 2 (terms must appear in at least 2 documents)
        - **Stop Words:** English stop words removed
        """)
        
        with col3:
            # CHANGED: Description for AdaBoost
            st.markdown("""
            ### 🚀 AdaBoost Model
            **Type:** Ensemble Classification Model (Boosting)
            **Algorithm:** AdaBoostClassifier (Adaptive Boosting)
            **Features:** TF-IDF vectors
            
            **Strengths:**
            - Improves accuracy by combining multiple "weak" learners.
            - Focuses on misclassified samples in successive iterations, improving overall performance.
            - Less prone to overfitting than a single decision tree.
            """)

        # ADDED: Deep Learning Models Section
        st.subheader("🧠 Deep Learning Models") # NEW Header for DL models
        st.markdown("app also supports the following deep learning models, leveraging word embeddings:")
        
        col_dl1, col_dl2, col_dl3 = st.columns(3) # New columns for DL models
        
        with col_dl1:
            # ADDED: Description for CNN Model
            st.markdown("""
            ### 🧠 CNN Model
            **Type:** Convolutional Neural Network (CNN)
            **Features:** Word Embeddings (e.g., GloVe, Word2Vec, fastText)
            
            **Strengths:**
            - Excellent at capturing local patterns (like n-grams) in text.
            - Learns hierarchical features from word embeddings.
            - Effective for text classification tasks.
            """)
            
        with col_dl2:
            # ADDED: Description for LSTM Model
            st.markdown("""
            ### 🧠 LSTM Model
            **Type:** Long Short-Term Memory (LSTM) - a type of Recurrent Neural Network (RNN)
            **Features:** Word Embeddings
            
            **Strengths:**
            - Designed to learn long-term dependencies in sequential data (text).
            - Mitigates vanishing/exploding gradients problems of traditional RNNs.
            - Highly effective for tasks requiring understanding of context over long sentences.
            """)
            
        with col_dl3:
            # ADDED: Description for RNN Model
            st.markdown("""
            ### 🧠 RNN Model
            **Type:** Recurrent Neural Network (RNN)
            **Features:** Word Embeddings
            
            **Strengths:**
            - Processes sequential data by maintaining an internal state.
            - Basic building block for understanding text sequences.
            - Captures temporal dependencies between words.
            """)

        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        # CHANGED: Description to cover both TF-IDF and Word Embeddings
        st.markdown("""
        **For Machine Learning Models (SVM, Decision Tree, AdaBoost):**
        - **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
            - **Purpose:** Converts text into numerical vectors, reflecting word importance by weighting terms that are frequent in a document but rare across the corpus.
            - **Parameters:** Configured during training (e.g., `ngram_range`, `max_features`, `min_df`, `max_df`) to capture single words and multi-word phrases, and to filter very common or very rare terms.
        
        **For Deep Learning Models (CNN, LSTM, RNN):**
        - **Representation:** Word Embeddings (e.g., GloVe, Word2Vec, fastText)
            - **Purpose:** Represents words as dense vectors in a continuous vector space. These vectors capture semantic relationships (words with similar meanings have similar vectors).
            - **Implementation:** Pre-trained GloVe embeddings are used to initialize the embedding layer of the neural networks, providing a rich, pre-learned representation of words.
        """)

        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("svm_model.pkl", "SVM Model", models.get('svm_available', False)),
            ("decision_tree_model.pkl", "Decision Tree Model", models.get('dt_available', False)),
            ("adaboost_model.pkl", "AdaBoost Model", models.get('ab_available', False)),
            ("CNN.pkl", "CNN Model", models.get('cnn_available', False)),
            ("LSTM.pkl", "LSTM Model", models.get('lstm_available', False)),
            ("RNN.pkl", "RNN Model", models.get('rnn_available', False)),
            ("word_to_idx.pkl", "DL Vocabulary (word_to_idx)", models.get('dl_resources_available', False) and 'word_to_idx' in models),
            ("MAX_SEQUENCE_LENGTH.pkl", "DL Max Sequence Length", models.get('dl_resources_available', False) and 'MAX_SEQUENCE_LENGTH' in models),
            ("embedding_matrix.pt", "DL Embedding Matrix", models.get('dl_resources_available', False) and 'embedding_matrix' in models)
        ]
        
        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })
        
        st.table(pd.DataFrame(file_status))
        
        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Dataset:** AI vs Human Text Datasets
        - **Classes:** Human-written and AI-written text.
        - **Preprocessing:** Text cleaning, stop word removal, lemmatization (for all models).
        - **Feature Representation:** TF-IDF (for ML models), GloVe Word Embeddings (for DL models).
        - **Training:** All models were trained and hyperparameter-tuned using cross-validation techniques for optimal performance.
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (e.g., SVM, Decision Tree, CNN, LSTM, RNN, AdaBoost).
        2. **Enter text** in the text area (you can type, paste, or use examples).
        3. **Click 'Predict Authorship'** to get AI vs Human classification results.
        4. **View results:** See the predicted author type, confidence score, and probability breakdown.
        5. **Try examples:** Use the provided example texts to test the models.
        """)
    
    with st.expander("📁 Document Analysis"): # CHANGED: Page name from "Batch Processing"
        st.write("""
        1. **Prepare your document/file:**
           - **.txt file:** Each line can be treated as a separate text input.
           - **.csv file:** Ensure the text you want to analyze is in the first column.
           - **.pdf file:** The app will attempt to extract all readable text.
           - **.docx file:** The app will attempt to extract all readable text.
        2. **Upload the file** using the file uploader.
        3. **Select a model** for processing.
        4. **Click 'Analyze Document'** to analyze the text.
        5. **Download results** as a CSV file with predictions and probabilities.
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to compare models on.
        2. **Click 'Compare All Models'** to get predictions from all available models.
        3. **View comparison table** showing each model's prediction and confidence.
        4. **Analyze agreement:** See if models agree or disagree on the authorship.
        5. **Compare probabilities:** Side-by-side probability charts for deeper insight.
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure all trained model files (`.pkl` for ML, `.pkl` or `.pt` for DL) are saved in the `models/` directory.
        - Check that required files exist:
          - `tfidf_vectorizer.pkl` (required for ML models)
          - `svm_model.pkl`, `decision_tree_model.pkl`, `adaboost_model.pkl` (for ML Classifiers)
          - `CNN.pkl`, `LSTM.pkl`, `RNN.pkl` (for Deep Learning Classifiers)
          - `word_to_idx.pkl`, `MAX_SEQUENCE_LENGTH.pkl`, `embedding_matrix.pt` (for DL resources)
        - For Deep Learning models, ensure their Python classes are defined and imported correctly in `app.py` before loading their saved weights.
        
        **Prediction errors:**
        - Make sure input text is not empty.
        - Try shorter texts if encountering performance issues with very long documents.
        - Check that text contains readable characters.
        
        **File upload issues:**
        - Ensure file format is `.txt`, `.csv`, `.pdf`, or `.docx`.
        - Check file encoding (UTF-8 is recommended for text/CSV).
        - Verify CSV has text in the first column.
        - For PDFs, text extraction might fail on scanned images (non-selectable text).
        """)
    
    # System information
    st.subheader("💻 Your Project Structure")
    st.code("""
    ai_human_detection_project/
    ├── app.py                      # Main Streamlit application
    ├── requirements.txt            # Project dependencies
    ├── models/                     # Your trained models and resources
    │   ├── svm_model.pkl
    │   ├── decision_tree_model.pkl
    │   ├── adaboost_model.pkl
    │   ├── CNN.pkl
    │   ├── LSTM.pkl
    │   ├── RNN.pkl
    │   ├── tfidf_vectorizer.pkl    # Feature extraction for ML
    │   ├── word_to_idx.pkl         # DL Vocabulary
    │   ├── MAX_SEQUENCE_LENGTH.pkl # DL sequence length
    ├── data/                       # Training and test data
    │   ├── training_data/
    │   └── test_data/
    ├── notebooks/                  # Development notebooks
    │   └── # Your code.ipynb       # E.g., Mrunmayee_Tulshibagwale_project2_R11815197.ipynb
    └── README.md                   # Project documentation
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detection App**
Built with Streamlit

**Models:** 
- SVM Model
- Decision Model
- AdaBoost Model
- Deep Learning:
    - CNN Model
    - LSTM Model
    - RNN Model

**Framework:** scikit-learn, PyTorch
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | AI vs Human Text Detection Demo | By Mrunmayee Tulshibagwale<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
    <small>This app demonstrates AI vs Human text classification using trained ML & DL models</small>
</div>
""", unsafe_allow_html=True)