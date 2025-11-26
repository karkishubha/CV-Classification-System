"""
Streamlit app for Resume Classification System
"""
import streamlit as st
import pickle
import os
from pathlib import Path
import PyPDF2

# Set page configuration
st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTitle {
        color: #1f77b4;
        text-align: center;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #000;
    }
    .result-box h3 {
        color: #000;
        margin-top: 0;
    }
    .result-box p {
        color: #000;
        font-size: 1.1rem;
    }
    .ai-box {
        background-color: #e3f2fd;
        border-left: 5px solid #1976d2;
    }
    .web-box {
        background-color: #f3e5f5;
        border-left: 5px solid #7b1fa2;
    }
    .data-box {
        background-color: #e8f5e9;
        border-left: 5px solid #388e3c;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 Resume Classification System")
st.markdown("""
Classify resumes into three categories:
- **AI**: Artificial Intelligence, Machine Learning, Deep Learning
- **Web**: Web Development, Frontend, Backend, Full Stack
- **Data**: Data Analysis, Data Engineering, Business Intelligence
""")

# Load the model
@st.cache_resource
def load_model():
    model_path = Path('models/resume_classifier.pkl')
    if not model_path.exists():
        st.info("⏳ Training model for the first time. This may take a moment...")
        import subprocess
        try:
            subprocess.run(['python', 'train_model.py'], check=True, capture_output=True)
            st.success("✅ Model trained successfully!")
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            st.stop()
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

# Initialize the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Initialize session state
if "resume_content" not in st.session_state:
    st.session_state.resume_content = ""
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Create tabs for main functionality and samples
tab1, tab2 = st.tabs(["📥 Classify Resume", "📚 Sample Resumes"])

with tab1:
    # Main input section
    st.markdown("### 📥 Upload or Paste Your Resume")
    
    # Create two columns for input methods
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📤 Upload File")
        uploaded_file = st.file_uploader(
            "Choose a file (.txt or .pdf)",
            type=['txt', 'pdf'],
            key="upload_resume"
        )
        
        if uploaded_file is not None:
            resume_content = None
            
            if uploaded_file.type == 'text/plain':
                resume_content = uploaded_file.read().decode('utf-8')
            elif uploaded_file.type == 'application/pdf':
                try:
                    pdf_reader = PyPDF2.PdfReader(uploaded_file)
                    resume_content = ""
                    for page in pdf_reader.pages:
                        resume_content += page.extract_text()
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")
                    resume_content = None
            
            if resume_content:
                st.session_state.resume_content = resume_content
                st.success("✅ File uploaded successfully!")
    
    with col2:
        st.markdown("#### 📝 Paste Text")
        pasted_text = st.text_area(
            "Or paste your resume text here:",
            height=200,
            placeholder="Paste your resume content...",
            key="paste_resume"
        )
        
        if pasted_text:
            st.session_state.resume_content = pasted_text
    
    # Display extracted/pasted content
    if st.session_state.resume_content:
        st.markdown("### 📋 Resume Content")
        st.text_area(
            "Your resume content:",
            value=st.session_state.resume_content,
            height=200,
            disabled=True,
            key="content_preview"
        )
        
        # Classification button
        if st.button("🚀 Classify Resume", key="classify_btn", use_container_width=True):
            with st.spinner("Classifying resume..."):
                prediction = model.predict([st.session_state.resume_content])[0]
                confidence = max(model.predict_proba([st.session_state.resume_content])[0])
                probs = model.predict_proba([st.session_state.resume_content])[0]
                
                st.session_state.show_results = True
                st.session_state.prediction = prediction
                st.session_state.confidence = confidence
                st.session_state.probs = probs
    
    # Display classification results
    if st.session_state.show_results:
        st.divider()
        st.markdown("### ✨ Classification Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Primary Classification")
            prediction = st.session_state.prediction
            confidence = st.session_state.confidence
            
            if prediction == 'AI':
                st.markdown(f"""
                <div class="result-box ai-box">
                <h3>🤖 AI/Machine Learning</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == 'Web':
                st.markdown(f"""
                <div class="result-box web-box">
                <h3>🌐 Web Development</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box data-box">
                <h3>📊 Data Science/Analytics</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Category Probabilities")
            probs = st.session_state.probs
            for i, category in enumerate(model.classes_):
                st.progress(probs[i], text=f"{category}: {probs[i]:.2%}")

with tab2:
    st.markdown("### Sample Resume Templates")
    
    sample_resumes = {
        "AI Engineer": """JOHN DOE
    john@example.com | linkedin.com/in/johndoe
    
    EXPERIENCE
    Senior Machine Learning Engineer - Tech Corp (2021-Present)
    - Built deep learning models for NLP tasks using TensorFlow and PyTorch
    - Implemented computer vision pipeline for image classification
    - Optimized neural networks achieving 95% accuracy
    
    Machine Learning Engineer - AI Startup (2019-2021)
    - Developed reinforcement learning algorithms for autonomous systems
    - Worked with TensorFlow, scikit-learn, and Keras
    - Published research on neural network optimization
    
    SKILLS
    Python, TensorFlow, PyTorch, Deep Learning, NLP, Computer Vision, scikit-learn, Pandas, NumPy, CUDA, Git
    
    EDUCATION
    MS Computer Science - Stanford University (2019)
    BS Mathematics - MIT (2017)""",
        
        "Web Developer": """JANE SMITH
    jane@example.com | linkedin.com/in/janesmith
    
    EXPERIENCE
    Senior Full Stack Developer - Web Solutions Inc (2020-Present)
    - Developed responsive web applications using React and Node.js
    - Designed and implemented REST APIs with Express.js
    - Managed databases using PostgreSQL and MongoDB
    - Led team of 5 frontend developers
    
    Frontend Developer - Digital Agency (2018-2020)
    - Built interactive UIs using React, Vue.js, and Angular
    - Optimized website performance reducing load time by 40%
    - Implemented responsive designs with HTML5, CSS3, JavaScript
    
    SKILLS
    React, Vue.js, Angular, Node.js, Express.js, JavaScript, TypeScript, HTML5, CSS3, PostgreSQL, MongoDB, Docker, Git
    
    EDUCATION
    BS Information Technology - UC Berkeley (2018)""",
        
        "Data Analyst": """ALEX JOHNSON
    alex@example.com | linkedin.com/in/alexjohnson
    
    EXPERIENCE
    Senior Data Analyst - Analytics Corp (2020-Present)
    - Analyzed large datasets using Python and SQL to drive business insights
    - Created dashboards and visualizations using Tableau and Power BI
    - Built ETL pipelines to process 10M+ records daily
    - Improved reporting accuracy by 35% through data validation
    
    Data Scientist - Finance Tech (2018-2020)
    - Performed statistical analysis on financial data
    - Developed predictive models using scikit-learn and XGBoost
    - Automated reporting reducing manual effort by 80%
    
    SKILLS
    Python, R, SQL, Tableau, Power BI, Excel, Pandas, NumPy, scikit-learn, Spark, Statistics, ETL, Git
    
    EDUCATION
    MS Data Science - Carnegie Mellon University (2018)
    BS Statistics - UC Davis (2016)"""
    }
    
    sample_col1, sample_col2 = st.columns([1, 1])
    
    with sample_col1:
        sample_choice = st.selectbox("Choose a sample resume:", list(sample_resumes.keys()))
        st.text_area(f"{sample_choice} Resume:", value=sample_resumes[sample_choice], height=300, disabled=True)
    
    with sample_col2:
        st.markdown("### Classify Sample")
        
        if st.button(f"🚀 Classify {sample_choice}", use_container_width=True, key="classify_sample"):
            with st.spinner("Classifying sample resume..."):
                prediction = model.predict([sample_resumes[sample_choice]])[0]
                confidence = max(model.predict_proba([sample_resumes[sample_choice]])[0])
                probs = model.predict_proba([sample_resumes[sample_choice]])[0]
                
                st.session_state.sample_prediction = prediction
                st.session_state.sample_confidence = confidence
                st.session_state.sample_probs = probs
                st.session_state.show_sample_results = True
    
    # Display sample classification results
    if st.session_state.get("show_sample_results", False):
        st.divider()
        st.markdown("### ✨ Classification Result")
        
        res_col1, res_col2 = st.columns(2)
        
        with res_col1:
            st.subheader("Primary Classification")
            prediction = st.session_state.sample_prediction
            confidence = st.session_state.sample_confidence
            
            if prediction == 'AI':
                st.markdown(f"""
                <div class="result-box ai-box">
                <h3>🤖 AI/Machine Learning</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            elif prediction == 'Web':
                st.markdown(f"""
                <div class="result-box web-box">
                <h3>🌐 Web Development</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-box data-box">
                <h3>📊 Data Science/Analytics</h3>
                <p><b>Category:</b> {prediction}</p>
                <p><b>Confidence:</b> {confidence:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with res_col2:
            st.subheader("Category Probabilities")
            probs = st.session_state.sample_probs
            for i, category in enumerate(model.classes_):
                st.progress(probs[i], text=f"{category}: {probs[i]:.2%}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Resume Classification System v1.0")
