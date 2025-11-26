"""
Script to train and save the resume classification model
"""
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import os

# Sample training data for each category
training_data = {
    'AI': [
        "Experienced AI Engineer with expertise in machine learning, deep learning, neural networks, TensorFlow, PyTorch, NLP, computer vision, and AI model development.",
        "Data Scientist specializing in artificial intelligence, predictive modeling, clustering algorithms, and intelligent systems development.",
        "ML Engineer skilled in Python, scikit-learn, Keras, reinforcement learning, and large language models deployment.",
        "AI Research Scientist with publications on neural networks, transformers, attention mechanisms, and advanced deep learning architectures.",
        "Developed chatbots using NLP techniques, implemented computer vision systems, and deployed AI models in production.",
        "Experience with TensorFlow, PyTorch, Hugging Face transformers, and building intelligent recommendation systems.",
        "Strong background in machine learning algorithms, model optimization, feature engineering, and AI pipeline development.",
        "Specialized in deep learning frameworks, CNN, RNN, LSTM networks, and AI solution architecture.",
        "Expert in natural language processing, sentiment analysis, text classification, and language model fine-tuning.",
        "Proficient in implementing machine learning models using scikit-learn, XGBoost, and advanced statistical methods."
    ],
    'Web': [
        "Full Stack Web Developer proficient in React, Angular, Vue.js, HTML5, CSS3, JavaScript, and modern web frameworks.",
        "Backend Developer with expertise in Node.js, Django, Flask, PHP, ASP.NET, and RESTful API development.",
        "Frontend Engineer specializing in responsive design, UI/UX implementation, React hooks, state management with Redux.",
        "Web Application Developer skilled in React, TypeScript, Webpack, and building scalable single page applications.",
        "Experienced in developing web applications using HTML, CSS, JavaScript, jQuery, and AJAX technologies.",
        "Expert in Node.js, Express.js, MongoDB, PostgreSQL, and building full-stack web applications.",
        "Proficient in Vue.js, Vuex, REST APIs, and creating interactive user interfaces for web platforms.",
        "Frontend Developer with expertise in Angular, TypeScript, RxJS, and component-based architecture.",
        "Full Stack Developer using Python Django, React, Docker, AWS, and continuous deployment practices.",
        "Web Developer experienced in PHP, Laravel, MySQL, HTML/CSS, and traditional server-side web development."
    ],
    'Data': [
        "Data Analyst with expertise in SQL, Python, R, data visualization, Tableau, Power BI, and business intelligence.",
        "Business Analyst specializing in data analysis, statistical modeling, A/B testing, and data-driven decision making.",
        "Data Engineer skilled in Apache Spark, Hadoop, data warehousing, ETL pipelines, and big data processing.",
        "Analytics Manager with experience in Google Analytics, cohort analysis, funnel analysis, and user behavior metrics.",
        "Database Administrator experienced in SQL Server, PostgreSQL, MongoDB, data backup, and performance optimization.",
        "Statistical Analyst with strong background in hypothesis testing, regression analysis, and quantitative research.",
        "BI Developer experienced in building dashboards using Tableau, Power BI, Looker, and reporting frameworks.",
        "Data Professional skilled in Python pandas, NumPy, data cleaning, transformation, and exploratory data analysis.",
        "ETL Developer specializing in data pipeline design, data integration, and data quality assurance.",
        "Analytics Engineer with expertise in SQL, dbt, building analytical models, and data infrastructure."
    ]
}

# Flatten the data
X_train = []
y_train = []

for category, resumes in training_data.items():
    X_train.extend(resumes)
    y_train.extend([category] * len(resumes))

# Create and train the model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=2000, min_df=1, stop_words='english')),
    ('classifier', MultinomialNB())
])

print("Training resume classification model...")
model.fit(X_train, y_train)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model_path = 'models/resume_classifier.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {model_path}")

# Test the model
test_resumes = [
    "Python developer with TensorFlow and neural network experience",
    "React and JavaScript web developer with 5 years experience",
    "SQL expert in data analysis and business intelligence"
]

predictions = model.predict(test_resumes)
print("\nTest predictions:")
for resume, prediction in zip(test_resumes, predictions):
    print(f"Resume: {resume[:50]}... -> Category: {prediction}")
