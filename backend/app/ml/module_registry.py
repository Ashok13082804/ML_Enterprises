"""
MLVerse X — Module Registry
Defines all 100 ML modules with their metadata and ML configuration.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class TaskType(str, Enum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RECOMMENDATION = "recommendation"
    TIME_SERIES = "time_series"
    NLP_CLASSIFICATION = "nlp_classification"
    NLP_GENERATION = "nlp_generation"
    COMPUTER_VISION = "computer_vision"
    ANOMALY_DETECTION = "anomaly_detection"


class ModuleCategory(str, Enum):
    BEGINNER_ML = "beginner_ml"
    NLP = "nlp"
    COMPUTER_VISION = "computer_vision"
    TIME_SERIES = "time_series"
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    INDUSTRIAL = "industrial"


@dataclass
class ModuleConfig:
    id: str                          # unique slug, e.g. "house-price-prediction"
    name: str                        # display name
    category: ModuleCategory
    task_type: TaskType
    description: str
    icon: str                        # emoji or icon name
    tags: List[str]
    input_type: str                  # "tabular", "text", "image", "audio", "video", "time_series"
    default_algorithms: List[str]    # list of sklearn/xgb/lgbm/etc algorithm names
    target_description: Optional[str] = None
    feature_hints: List[str] = field(default_factory=list)
    sample_dataset_url: Optional[str] = None
    supports_batch_predict: bool = True
    supports_realtime: bool = False  # for CV/NLP streaming
    color: str = "#6366f1"


# ─── The 100 Module Registry ───────────────────────────────────────────────────
MODULE_REGISTRY: Dict[str, ModuleConfig] = {}


def register(cfg: ModuleConfig):
    MODULE_REGISTRY[cfg.id] = cfg
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 1 — BEGINNER ML (20)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="house-price-prediction",
    name="House Price Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict house prices based on features like area, bedrooms, location, and amenities.",
    icon="🏠",
    tags=["regression", "real estate", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "xgboost", "lightgbm", "linear_regression"],
    target_description="House price in USD",
    feature_hints=["area_sqft", "bedrooms", "bathrooms", "location", "year_built", "garage"],
    color="#10b981",
))

register(ModuleConfig(
    id="student-performance",
    name="Student Performance Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict student academic performance based on study habits, attendance, and demographics.",
    icon="📚",
    tags=["regression", "education", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "linear_regression", "ridge"],
    target_description="Final exam score (0-100)",
    feature_hints=["study_hours", "attendance_rate", "previous_scores", "extracurricular"],
    color="#3b82f6",
))

register(ModuleConfig(
    id="salary-prediction",
    name="Salary Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict employee salary based on experience, skills, education, and location.",
    icon="💰",
    tags=["regression", "hr", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "linear_regression"],
    target_description="Annual salary in USD",
    feature_hints=["years_experience", "education_level", "job_title", "location", "skills"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="employee-attrition",
    name="Employee Attrition Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether an employee is likely to leave the company.",
    icon="👤",
    tags=["classification", "hr", "churn", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm", "logistic_regression"],
    target_description="Attrition (Yes/No)",
    feature_hints=["age", "department", "job_satisfaction", "work_life_balance", "years_at_company"],
    color="#ef4444",
))

register(ModuleConfig(
    id="loan-approval",
    name="Loan Approval Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict loan approval based on applicant's financial profile.",
    icon="🏦",
    tags=["classification", "finance", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "logistic_regression", "svm"],
    target_description="Loan Approved (Yes/No)",
    feature_hints=["income", "credit_score", "loan_amount", "employment_status", "debt_to_income"],
    color="#8b5cf6",
))

register(ModuleConfig(
    id="credit-risk",
    name="Credit Risk Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Classify borrowers into credit risk categories.",
    icon="📊",
    tags=["classification", "finance", "risk", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm", "gradient_boosting"],
    target_description="Credit risk level (Low/Medium/High)",
    feature_hints=["credit_score", "income", "existing_debt", "payment_history"],
    color="#ec4899",
))

register(ModuleConfig(
    id="customer-churn",
    name="Customer Churn Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether a customer will churn from your service.",
    icon="📉",
    tags=["classification", "churn", "crm", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm", "logistic_regression"],
    target_description="Churn (Yes/No)",
    feature_hints=["tenure", "monthly_charges", "contract_type", "support_calls"],
    color="#f97316",
))

register(ModuleConfig(
    id="insurance-premium",
    name="Insurance Premium Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict insurance premium cost based on individual risk factors.",
    icon="🛡️",
    tags=["regression", "insurance", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "xgboost", "linear_regression"],
    target_description="Annual premium in USD",
    feature_hints=["age", "bmi", "smoker", "region", "children"],
    color="#06b6d4",
))

register(ModuleConfig(
    id="car-price",
    name="Car Price Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict the price of a car based on its features.",
    icon="🚗",
    tags=["regression", "automotive", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "ridge"],
    target_description="Car price in USD",
    feature_hints=["year", "make", "model", "mileage", "engine_size", "fuel_type"],
    color="#84cc16",
))

register(ModuleConfig(
    id="used-bike-price",
    name="Used Bike Price Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict the resale price of used motorcycles.",
    icon="🏍️",
    tags=["regression", "automotive", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "linear_regression"],
    target_description="Bike price in USD",
    feature_hints=["brand", "model", "year", "km_driven", "engine_cc", "owner"],
    color="#a78bfa",
))

register(ModuleConfig(
    id="medical-insurance",
    name="Medical Insurance Cost Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict individual medical insurance costs based on health and demographic factors.",
    icon="🏥",
    tags=["regression", "healthcare", "insurance", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "xgboost", "linear_regression"],
    target_description="Medical cost in USD",
    feature_hints=["age", "sex", "bmi", "children", "smoker", "region"],
    color="#14b8a6",
))

register(ModuleConfig(
    id="flight-fare",
    name="Flight Fare Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict flight ticket prices based on route, time, and airline.",
    icon="✈️",
    tags=["regression", "travel", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "lightgbm"],
    target_description="Ticket price in USD",
    feature_hints=["airline", "source", "destination", "departure_time", "stops", "duration"],
    color="#6366f1",
))

register(ModuleConfig(
    id="weather-prediction",
    name="Weather Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Predict weather conditions based on atmospheric measurements.",
    icon="🌤️",
    tags=["classification", "weather", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "xgboost", "decision_tree"],
    target_description="Weather type (Sunny/Cloudy/Rainy/Snowy/Stormy)",
    feature_hints=["temperature", "humidity", "pressure", "wind_speed", "visibility"],
    color="#0ea5e9",
))

register(ModuleConfig(
    id="rainfall-prediction",
    name="Rainfall Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether it will rain tomorrow.",
    icon="🌧️",
    tags=["classification", "weather", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "logistic_regression", "gradient_boosting"],
    target_description="Rain tomorrow (Yes/No)",
    feature_hints=["MinTemp", "MaxTemp", "Rainfall", "Humidity9am", "Humidity3pm", "Pressure"],
    color="#3b82f6",
))

register(ModuleConfig(
    id="electricity-consumption",
    name="Electricity Consumption Prediction",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Predict household electricity consumption in kWh.",
    icon="⚡",
    tags=["regression", "energy", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "xgboost", "linear_regression"],
    target_description="Electricity consumption (kWh)",
    feature_hints=["area_sqft", "occupants", "appliances", "season", "temperature"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="energy-demand-forecasting",
    name="Energy Demand Forecasting",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.REGRESSION,
    description="Forecast energy demand for grid management and planning.",
    icon="🔋",
    tags=["regression", "energy", "forecasting", "tabular"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "lstm"],
    target_description="Energy demand in MW",
    feature_hints=["hour", "day_of_week", "month", "temperature", "holiday"],
    color="#10b981",
))

register(ModuleConfig(
    id="movie-recommendation",
    name="Movie Recommendation",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.RECOMMENDATION,
    description="Recommend movies based on user preferences and collaborative filtering.",
    icon="🎬",
    tags=["recommendation", "collaborative-filtering", "tabular"],
    input_type="tabular",
    default_algorithms=["svd", "knn_collaborative", "nmf"],
    target_description="Movie ratings / recommendations",
    feature_hints=["user_id", "movie_id", "rating", "genre", "year"],
    color="#f97316",
))

register(ModuleConfig(
    id="book-recommendation",
    name="Book Recommendation",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.RECOMMENDATION,
    description="Recommend books based on reading history and preferences.",
    icon="📖",
    tags=["recommendation", "collaborative-filtering", "tabular"],
    input_type="tabular",
    default_algorithms=["svd", "knn_collaborative", "nmf"],
    target_description="Book ratings / recommendations",
    feature_hints=["user_id", "book_id", "rating", "genre", "author"],
    color="#8b5cf6",
))

register(ModuleConfig(
    id="music-recommendation",
    name="Music Recommendation",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.RECOMMENDATION,
    description="Recommend music tracks based on listening history and features.",
    icon="🎵",
    tags=["recommendation", "music", "tabular"],
    input_type="tabular",
    default_algorithms=["svd", "knn_collaborative", "content_based"],
    target_description="Track recommendations",
    feature_hints=["user_id", "track_id", "play_count", "genre", "tempo"],
    color="#ec4899",
))

register(ModuleConfig(
    id="product-recommendation",
    name="Product Recommendation",
    category=ModuleCategory.BEGINNER_ML,
    task_type=TaskType.RECOMMENDATION,
    description="Recommend products based on purchase history (e-commerce).",
    icon="🛍️",
    tags=["recommendation", "e-commerce", "tabular"],
    input_type="tabular",
    default_algorithms=["svd", "als", "knn_collaborative"],
    target_description="Product recommendations",
    feature_hints=["user_id", "product_id", "purchase_count", "category", "price"],
    color="#14b8a6",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 2 — NLP (20)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="fake-news-detection",
    name="Fake News Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Detect fake/misinformation news articles using NLP.",
    icon="📰",
    tags=["nlp", "classification", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "tfidf_random_forest", "bert_classifier"],
    target_description="Fake or Real",
    feature_hints=["title", "text", "subject", "author"],
    color="#ef4444",
))

register(ModuleConfig(
    id="spam-email-detection",
    name="Spam Email Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Classify emails as spam or legitimate.",
    icon="📧",
    tags=["nlp", "classification", "text", "email"],
    input_type="text",
    default_algorithms=["naive_bayes", "tfidf_logistic", "tfidf_svm"],
    target_description="Spam or Ham",
    feature_hints=["subject", "body", "sender"],
    color="#f97316",
))

register(ModuleConfig(
    id="sms-spam-detection",
    name="SMS Spam Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Detect spam SMS messages.",
    icon="📱",
    tags=["nlp", "classification", "text", "sms"],
    input_type="text",
    default_algorithms=["naive_bayes", "tfidf_logistic", "tfidf_random_forest"],
    target_description="Spam or Ham",
    feature_hints=["message"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="sentiment-analysis",
    name="Sentiment Analysis",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Analyze sentiment of text as positive, negative, or neutral.",
    icon="😊",
    tags=["nlp", "sentiment", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_sentiment", "vader"],
    target_description="Sentiment (Positive/Negative/Neutral)",
    feature_hints=["text", "review"],
    color="#10b981",
))

register(ModuleConfig(
    id="emotion-detection",
    name="Emotion Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Detect emotions in text: joy, sadness, anger, fear, surprise, disgust.",
    icon="🎭",
    tags=["nlp", "emotion", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_emotion", "tfidf_random_forest"],
    target_description="Emotion label",
    feature_hints=["text"],
    color="#8b5cf6",
))

register(ModuleConfig(
    id="language-detection",
    name="Language Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Detect the language of a given text.",
    icon="🌍",
    tags=["nlp", "language", "text"],
    input_type="text",
    default_algorithms=["langdetect", "fasttext_lid"],
    target_description="Language code (en, fr, es, ...)",
    feature_hints=["text"],
    color="#06b6d4",
))

register(ModuleConfig(
    id="news-classification",
    name="News Classification",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Classify news articles into categories: sports, politics, tech, etc.",
    icon="📋",
    tags=["nlp", "classification", "news", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "tfidf_random_forest", "bert_classifier"],
    target_description="News category",
    feature_hints=["headline", "body", "source"],
    color="#3b82f6",
))

register(ModuleConfig(
    id="document-categorization",
    name="Document Categorization",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Automatically categorize documents by topic or type.",
    icon="📁",
    tags=["nlp", "classification", "documents", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "lda_topic_model", "bert_classifier"],
    target_description="Document category",
    feature_hints=["title", "content", "keywords"],
    color="#84cc16",
))

register(ModuleConfig(
    id="resume-screening",
    name="Resume Screening",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Screen and classify resumes by job role suitability.",
    icon="📄",
    tags=["nlp", "hr", "recruitment", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "tfidf_random_forest", "bert_classifier"],
    target_description="Job role category / screening result",
    feature_hints=["resume_text", "skills", "experience"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="interview-assistant",
    name="AI Interview Assistant",
    category=ModuleCategory.NLP,
    task_type=TaskType.NLP_GENERATION,
    description="Generate interview questions and evaluate candidate answers using local LLM.",
    icon="🎤",
    tags=["nlp", "generation", "hr", "ollama"],
    input_type="text",
    default_algorithms=["ollama_llm"],
    target_description="Interview questions and evaluation",
    feature_hints=["job_description", "skills", "experience_level"],
    supports_realtime=True,
    color="#a78bfa",
))

register(ModuleConfig(
    id="intent-detection",
    name="Intent Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Detect user intent from natural language utterances.",
    icon="🎯",
    tags=["nlp", "intent", "chatbot", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_classifier", "tfidf_svm"],
    target_description="Intent class",
    feature_hints=["utterance"],
    color="#14b8a6",
))

register(ModuleConfig(
    id="hate-speech-detection",
    name="Hate Speech Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Detect hate speech and offensive content in text.",
    icon="🚫",
    tags=["nlp", "classification", "moderation", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_classifier", "tfidf_random_forest"],
    target_description="Hate speech (Yes/No)",
    feature_hints=["text"],
    color="#ef4444",
))

register(ModuleConfig(
    id="toxic-comment-detection",
    name="Toxic Comment Detection",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Detect multiple toxicity types in online comments.",
    icon="☣️",
    tags=["nlp", "classification", "moderation", "text"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_classifier"],
    target_description="Toxicity types (toxic, severe_toxic, obscene, ...)",
    feature_hints=["comment_text"],
    color="#dc2626",
))

register(ModuleConfig(
    id="grammar-correction",
    name="Grammar Correction",
    category=ModuleCategory.NLP,
    task_type=TaskType.NLP_GENERATION,
    description="Automatically correct grammar and spelling errors using local LLM.",
    icon="✏️",
    tags=["nlp", "generation", "grammar", "text"],
    input_type="text",
    default_algorithms=["ollama_llm", "t5_correction"],
    target_description="Corrected text",
    feature_hints=["text"],
    supports_realtime=True,
    color="#10b981",
))

register(ModuleConfig(
    id="essay-scoring",
    name="Essay Scoring",
    category=ModuleCategory.NLP,
    task_type=TaskType.REGRESSION,
    description="Automatically score essays based on content quality, grammar, and coherence.",
    icon="📝",
    tags=["nlp", "regression", "education", "text"],
    input_type="text",
    default_algorithms=["tfidf_regression", "bert_regression"],
    target_description="Essay score (0-100)",
    feature_hints=["essay_text", "prompt"],
    color="#6366f1",
))

register(ModuleConfig(
    id="keyword-extraction",
    name="Keyword Extraction",
    category=ModuleCategory.NLP,
    task_type=TaskType.NLP_GENERATION,
    description="Extract key terms and phrases from documents using NLP.",
    icon="🔑",
    tags=["nlp", "extraction", "text"],
    input_type="text",
    default_algorithms=["tfidf_keywords", "rake", "yake", "keybert"],
    target_description="Extracted keywords",
    feature_hints=["text", "max_keywords"],
    color="#f97316",
))

register(ModuleConfig(
    id="text-summarization",
    name="Text Summarization",
    category=ModuleCategory.NLP,
    task_type=TaskType.NLP_GENERATION,
    description="Summarize long documents into concise summaries.",
    icon="📜",
    tags=["nlp", "generation", "summarization", "text"],
    input_type="text",
    default_algorithms=["ollama_llm", "extractive_lsa", "t5_summarization"],
    target_description="Summary",
    feature_hints=["text", "max_length"],
    supports_realtime=True,
    color="#8b5cf6",
))

register(ModuleConfig(
    id="machine-translation",
    name="Machine Translation",
    category=ModuleCategory.NLP,
    task_type=TaskType.NLP_GENERATION,
    description="Translate text between multiple languages using local models.",
    icon="🌐",
    tags=["nlp", "translation", "text"],
    input_type="text",
    default_algorithms=["ollama_llm", "helsinki_nlp"],
    target_description="Translated text",
    feature_hints=["text", "source_lang", "target_lang"],
    supports_realtime=True,
    color="#06b6d4",
))

register(ModuleConfig(
    id="speech-emotion-recognition",
    name="Speech Emotion Recognition",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Recognize emotions from audio speech recordings.",
    icon="🎙️",
    tags=["nlp", "speech", "emotion", "audio"],
    input_type="audio",
    default_algorithms=["librosa_rf", "cnn_audio"],
    target_description="Emotion from speech",
    feature_hints=["audio_file"],
    color="#ec4899",
))

register(ModuleConfig(
    id="voice-command-recognition",
    name="Voice Command Recognition",
    category=ModuleCategory.NLP,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Recognize voice commands from audio using Whisper and NLP.",
    icon="🎤",
    tags=["nlp", "speech", "voice", "audio"],
    input_type="audio",
    default_algorithms=["whisper_vosk", "whisper_nlp"],
    target_description="Recognized command",
    feature_hints=["audio_file"],
    supports_realtime=True,
    color="#a78bfa",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 3 — COMPUTER VISION (20)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="face-attendance",
    name="Face Attendance System",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Automated attendance system using face recognition.",
    icon="👤",
    tags=["cv", "face", "attendance", "recognition"],
    input_type="image",
    default_algorithms=["face_recognition_dlib", "deepface"],
    target_description="Recognized person / attendance marked",
    supports_realtime=True,
    color="#3b82f6",
))

register(ModuleConfig(
    id="face-mask-detection",
    name="Face Mask Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect whether people are wearing face masks.",
    icon="😷",
    tags=["cv", "detection", "healthcare", "safety"],
    input_type="image",
    default_algorithms=["yolov8", "mobilenet_classifier"],
    target_description="Mask / No Mask",
    supports_realtime=True,
    color="#10b981",
))

register(ModuleConfig(
    id="driver-drowsiness",
    name="Driver Drowsiness Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect driver fatigue and drowsiness from facial analysis.",
    icon="😴",
    tags=["cv", "safety", "detection", "automotive"],
    input_type="image",
    default_algorithms=["mediapipe_landmark", "dlib_facial"],
    target_description="Alert / Drowsy",
    supports_realtime=True,
    color="#ef4444",
))

register(ModuleConfig(
    id="helmet-detection",
    name="Helmet Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect whether workers/riders are wearing helmets.",
    icon="⛑️",
    tags=["cv", "safety", "detection", "ppe"],
    input_type="image",
    default_algorithms=["yolov8"],
    target_description="Helmet / No Helmet",
    supports_realtime=True,
    color="#f97316",
))

register(ModuleConfig(
    id="vehicle-detection",
    name="Vehicle Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect and count vehicles in images and video.",
    icon="🚗",
    tags=["cv", "detection", "traffic", "yolo"],
    input_type="image",
    default_algorithms=["yolov8"],
    target_description="Vehicle count and locations",
    supports_realtime=True,
    color="#6366f1",
))

register(ModuleConfig(
    id="license-plate-recognition",
    name="License Plate Recognition",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect and read license plates from vehicle images.",
    icon="🔢",
    tags=["cv", "ocr", "traffic", "recognition"],
    input_type="image",
    default_algorithms=["yolov8_ocr", "pytesseract"],
    target_description="License plate text",
    color="#8b5cf6",
))

register(ModuleConfig(
    id="traffic-sign-detection",
    name="Traffic Sign Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect and classify traffic signs in images.",
    icon="🚦",
    tags=["cv", "detection", "traffic", "autonomous"],
    input_type="image",
    default_algorithms=["yolov8", "cnn_classifier"],
    target_description="Traffic sign type",
    color="#f59e0b",
))

register(ModuleConfig(
    id="yolo-object-detection",
    name="YOLO Object Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="General purpose object detection using YOLOv8 (80 COCO classes).",
    icon="🔍",
    tags=["cv", "detection", "yolo", "general"],
    input_type="image",
    default_algorithms=["yolov8n", "yolov8s", "yolov8m"],
    target_description="Detected objects with bounding boxes",
    supports_realtime=True,
    color="#ec4899",
))

register(ModuleConfig(
    id="pose-estimation",
    name="Human Pose Estimation",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Estimate human body pose and keypoints from images.",
    icon="🏃",
    tags=["cv", "pose", "mediapipe", "keypoints"],
    input_type="image",
    default_algorithms=["mediapipe_pose", "yolov8_pose"],
    target_description="Body keypoints",
    supports_realtime=True,
    color="#14b8a6",
))

register(ModuleConfig(
    id="crowd-counting",
    name="Crowd Counting",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Count people in crowds from images or video.",
    icon="👥",
    tags=["cv", "counting", "crowd", "surveillance"],
    input_type="image",
    default_algorithms=["csrnet", "yolov8"],
    target_description="People count",
    color="#0ea5e9",
))

register(ModuleConfig(
    id="fire-detection",
    name="Fire Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect fire and flames in images and video streams.",
    icon="🔥",
    tags=["cv", "detection", "safety", "emergency"],
    input_type="image",
    default_algorithms=["yolov8", "cnn_classifier"],
    target_description="Fire detected (Yes/No)",
    supports_realtime=True,
    color="#ef4444",
))

register(ModuleConfig(
    id="smoke-detection",
    name="Smoke Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect smoke in images for early fire warning.",
    icon="💨",
    tags=["cv", "detection", "safety", "emergency"],
    input_type="image",
    default_algorithms=["yolov8", "cnn_classifier"],
    target_description="Smoke detected (Yes/No)",
    supports_realtime=True,
    color="#6b7280",
))

register(ModuleConfig(
    id="ppe-detection",
    name="PPE Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect Personal Protective Equipment compliance in workplaces.",
    icon="🦺",
    tags=["cv", "safety", "industrial", "ppe"],
    input_type="image",
    default_algorithms=["yolov8"],
    target_description="PPE compliance status",
    supports_realtime=True,
    color="#f97316",
))

register(ModuleConfig(
    id="animal-detection",
    name="Animal Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect and classify animal species in images.",
    icon="🦁",
    tags=["cv", "detection", "wildlife", "classification"],
    input_type="image",
    default_algorithms=["yolov8", "cnn_classifier"],
    target_description="Animal species",
    color="#84cc16",
))

register(ModuleConfig(
    id="crop-disease-detection",
    name="Crop Disease Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect diseases in crop leaves from images.",
    icon="🌿",
    tags=["cv", "agriculture", "disease", "classification"],
    input_type="image",
    default_algorithms=["cnn_classifier", "efficientnet", "resnet"],
    target_description="Crop disease type",
    color="#22c55e",
))

register(ModuleConfig(
    id="plant-species-identification",
    name="Plant Species Identification",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Identify plant species from leaf/flower images.",
    icon="🌸",
    tags=["cv", "biology", "classification", "nature"],
    input_type="image",
    default_algorithms=["efficientnet", "resnet", "mobilenet"],
    target_description="Plant species name",
    color="#10b981",
))

register(ModuleConfig(
    id="skin-disease-detection",
    name="Skin Disease Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect and classify skin conditions from dermatoscopy images.",
    icon="🔬",
    tags=["cv", "healthcare", "medical", "dermatology"],
    input_type="image",
    default_algorithms=["efficientnet", "resnet", "densenet"],
    target_description="Skin condition type",
    color="#f59e0b",
))

register(ModuleConfig(
    id="brain-tumor-detection",
    name="Brain Tumor Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect brain tumors in MRI scans.",
    icon="🧠",
    tags=["cv", "healthcare", "medical", "mri"],
    input_type="image",
    default_algorithms=["unet", "resnet", "cnn_classifier"],
    target_description="Tumor type or absence",
    color="#8b5cf6",
))

register(ModuleConfig(
    id="pneumonia-detection",
    name="Pneumonia Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect pneumonia from chest X-ray images.",
    icon="🫁",
    tags=["cv", "healthcare", "medical", "xray"],
    input_type="image",
    default_algorithms=["resnet", "densenet", "efficientnet"],
    target_description="Pneumonia (Normal/Pneumonia)",
    color="#3b82f6",
))

register(ModuleConfig(
    id="diabetic-retinopathy",
    name="Diabetic Retinopathy Detection",
    category=ModuleCategory.COMPUTER_VISION,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Grade diabetic retinopathy severity from retinal fundus images.",
    icon="👁️",
    tags=["cv", "healthcare", "medical", "ophthalmology"],
    input_type="image",
    default_algorithms=["efficientnet", "resnet", "densenet"],
    target_description="Retinopathy grade (0-4)",
    color="#ec4899",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 4 — TIME SERIES (10)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="stock-prediction",
    name="Stock Price Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict stock price movements using time series models.",
    icon="📈",
    tags=["time-series", "finance", "forecasting"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "arima", "xgboost_ts"],
    target_description="Future stock price",
    feature_hints=["date", "open", "high", "low", "close", "volume"],
    color="#22c55e",
))

register(ModuleConfig(
    id="cryptocurrency-forecasting",
    name="Cryptocurrency Forecasting",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Forecast cryptocurrency prices and trends.",
    icon="₿",
    tags=["time-series", "crypto", "forecasting"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "gru", "xgboost_ts"],
    target_description="Future crypto price",
    feature_hints=["timestamp", "open", "high", "low", "close", "volume"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="sales-forecasting",
    name="Sales Forecasting",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Forecast future sales for business planning.",
    icon="🛒",
    tags=["time-series", "business", "forecasting"],
    input_type="time_series",
    default_algorithms=["prophet", "arima", "lstm", "xgboost_ts"],
    target_description="Forecasted sales",
    feature_hints=["date", "sales", "store_id", "product_id", "promotion"],
    color="#10b981",
))

register(ModuleConfig(
    id="demand-forecasting",
    name="Demand Forecasting",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Forecast product demand for inventory management.",
    icon="📦",
    tags=["time-series", "supply-chain", "forecasting"],
    input_type="time_series",
    default_algorithms=["prophet", "arima", "lstm", "xgboost_ts"],
    target_description="Forecasted demand",
    feature_hints=["date", "demand", "product_id", "store_id"],
    color="#6366f1",
))

register(ModuleConfig(
    id="weather-forecasting",
    name="Weather Forecasting",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Forecast temperature, humidity, and weather conditions.",
    icon="🌡️",
    tags=["time-series", "weather", "forecasting"],
    input_type="time_series",
    default_algorithms=["prophet", "lstm", "arima"],
    target_description="Forecasted weather parameters",
    feature_hints=["date", "temperature", "humidity", "pressure", "wind_speed"],
    color="#0ea5e9",
))

register(ModuleConfig(
    id="air-pollution-prediction",
    name="Air Pollution Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict air quality index and pollutant levels.",
    icon="🌫️",
    tags=["time-series", "environment", "forecasting"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "xgboost_ts"],
    target_description="AQI / pollutant concentration",
    feature_hints=["date", "pm25", "pm10", "no2", "so2", "co"],
    color="#6b7280",
))

register(ModuleConfig(
    id="traffic-prediction",
    name="Traffic Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict traffic volume and congestion patterns.",
    icon="🚦",
    tags=["time-series", "traffic", "urban", "forecasting"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "xgboost_ts"],
    target_description="Traffic volume / speed",
    feature_hints=["timestamp", "volume", "speed", "location_id", "day_type"],
    color="#f97316",
))

register(ModuleConfig(
    id="water-quality-prediction",
    name="Water Quality Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict water quality parameters over time.",
    icon="💧",
    tags=["time-series", "environment", "water"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "arima"],
    target_description="Water quality index",
    feature_hints=["date", "ph", "turbidity", "dissolved_oxygen", "temperature"],
    color="#06b6d4",
))

register(ModuleConfig(
    id="solar-energy-prediction",
    name="Solar Energy Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict solar energy generation from weather and panel data.",
    icon="☀️",
    tags=["time-series", "energy", "renewables"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "xgboost_ts"],
    target_description="Energy generation in kWh",
    feature_hints=["date", "irradiance", "temperature", "cloud_cover", "panel_area"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="wind-energy-prediction",
    name="Wind Energy Prediction",
    category=ModuleCategory.TIME_SERIES,
    task_type=TaskType.TIME_SERIES,
    description="Predict wind energy generation from meteorological data.",
    icon="💨",
    tags=["time-series", "energy", "renewables"],
    input_type="time_series",
    default_algorithms=["lstm", "prophet", "xgboost_ts"],
    target_description="Energy generation in kWh",
    feature_hints=["date", "wind_speed", "wind_direction", "temperature", "turbine_id"],
    color="#84cc16",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 5 — FINANCE AI (10)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="credit-card-fraud",
    name="Credit Card Fraud Detection",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Detect fraudulent credit card transactions using ML.",
    icon="💳",
    tags=["classification", "fraud", "finance", "anomaly"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "isolation_forest", "lightgbm"],
    target_description="Fraud (0/1)",
    feature_hints=["amount", "time", "v1_v28_pca_features"],
    color="#ef4444",
))

register(ModuleConfig(
    id="transaction-fraud",
    name="Transaction Fraud Detection",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Detect fraudulent financial transactions.",
    icon="🔐",
    tags=["classification", "fraud", "banking"],
    input_type="tabular",
    default_algorithms=["xgboost", "lightgbm", "isolation_forest", "autoencoder"],
    target_description="Fraud (Yes/No)",
    feature_hints=["amount", "merchant_category", "location", "time", "device"],
    color="#dc2626",
))

register(ModuleConfig(
    id="loan-default-prediction",
    name="Loan Default Prediction",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether a borrower will default on a loan.",
    icon="⚠️",
    tags=["classification", "finance", "risk"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm", "logistic_regression"],
    target_description="Default (Yes/No)",
    feature_hints=["loan_amount", "interest_rate", "income", "dti", "credit_history"],
    color="#f97316",
))

register(ModuleConfig(
    id="portfolio-risk-analysis",
    name="Portfolio Risk Analysis",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.REGRESSION,
    description="Analyze and predict portfolio risk metrics (VaR, Sharpe ratio).",
    icon="📊",
    tags=["regression", "finance", "risk", "portfolio"],
    input_type="tabular",
    default_algorithms=["regression_ensemble", "xgboost"],
    target_description="Risk metrics",
    feature_hints=["asset_weights", "returns_history", "volatility"],
    color="#6366f1",
))

register(ModuleConfig(
    id="customer-lifetime-value",
    name="Customer Lifetime Value",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.REGRESSION,
    description="Predict the future revenue a customer will generate.",
    icon="💎",
    tags=["regression", "crm", "finance"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Customer LTV in USD",
    feature_hints=["purchase_history", "recency", "frequency", "monetary"],
    color="#a78bfa",
))

register(ModuleConfig(
    id="dynamic-pricing",
    name="Dynamic Pricing",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.REGRESSION,
    description="Predict optimal pricing based on demand, competition, and market conditions.",
    icon="🏷️",
    tags=["regression", "pricing", "optimization"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Optimal price",
    feature_hints=["demand", "competition_price", "time", "inventory", "season"],
    color="#10b981",
))

register(ModuleConfig(
    id="invoice-classification",
    name="Invoice Classification",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Automatically classify invoices by category and vendor type.",
    icon="🧾",
    tags=["classification", "finance", "nlp", "documents"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "bert_classifier"],
    target_description="Invoice category",
    feature_hints=["invoice_text", "vendor", "amount"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="expense-categorization",
    name="Expense Categorization",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Automatically categorize expenses from transaction descriptions.",
    icon="💸",
    tags=["classification", "finance", "nlp"],
    input_type="text",
    default_algorithms=["tfidf_logistic", "naive_bayes", "bert_classifier"],
    target_description="Expense category",
    feature_hints=["description", "amount", "merchant"],
    color="#06b6d4",
))

register(ModuleConfig(
    id="stock-trend-prediction",
    name="Stock Trend Prediction",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether stock price will go up or down.",
    icon="📉",
    tags=["classification", "finance", "time-series"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lstm", "gradient_boosting"],
    target_description="Up / Down",
    feature_hints=["open", "high", "low", "close", "volume", "ma_20", "rsi"],
    color="#22c55e",
))

register(ModuleConfig(
    id="financial-sentiment-analysis",
    name="Financial Sentiment Analysis",
    category=ModuleCategory.FINANCE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Analyze sentiment of financial news and reports.",
    icon="📰",
    tags=["nlp", "sentiment", "finance"],
    input_type="text",
    default_algorithms=["finbert", "tfidf_logistic", "vader"],
    target_description="Bullish / Neutral / Bearish",
    feature_hints=["headline", "article_text"],
    color="#8b5cf6",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 6 — HEALTHCARE AI (10)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="disease-prediction",
    name="Disease Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Predict likely diseases based on symptoms.",
    icon="🩺",
    tags=["classification", "healthcare", "diagnosis"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Disease name",
    feature_hints=["symptoms", "age", "gender", "medical_history"],
    color="#ef4444",
))

register(ModuleConfig(
    id="diabetes-prediction",
    name="Diabetes Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict diabetes risk based on clinical measurements.",
    icon="💉",
    tags=["classification", "healthcare", "diabetes"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "logistic_regression", "svm"],
    target_description="Diabetic (Yes/No)",
    feature_hints=["glucose", "bmi", "age", "blood_pressure", "insulin"],
    color="#f97316",
))

register(ModuleConfig(
    id="heart-disease-prediction",
    name="Heart Disease Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict heart disease risk from patient data.",
    icon="❤️",
    tags=["classification", "healthcare", "cardiology"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "logistic_regression", "neural_network"],
    target_description="Heart disease (Yes/No)",
    feature_hints=["age", "sex", "chest_pain", "cholesterol", "resting_bp", "max_hr"],
    color="#dc2626",
))

register(ModuleConfig(
    id="kidney-disease-prediction",
    name="Kidney Disease Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict chronic kidney disease from lab results.",
    icon="🫘",
    tags=["classification", "healthcare", "nephrology"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm"],
    target_description="CKD (Yes/No)",
    feature_hints=["blood_urea", "creatinine", "hemoglobin", "potassium", "sodium"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="cancer-prediction",
    name="Cancer Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict malignant vs benign tumors from biopsy features.",
    icon="🔬",
    tags=["classification", "healthcare", "oncology"],
    input_type="tabular",
    default_algorithms=["random_forest", "svm", "neural_network", "xgboost"],
    target_description="Malignant (Yes/No)",
    feature_hints=["radius_mean", "texture_mean", "perimeter_mean", "area_mean"],
    color="#8b5cf6",
))

register(ModuleConfig(
    id="medical-report-summarization",
    name="Medical Report Summarization",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.NLP_GENERATION,
    description="Summarize lengthy medical reports into clear summaries using local LLM.",
    icon="📋",
    tags=["nlp", "healthcare", "summarization", "ollama"],
    input_type="text",
    default_algorithms=["ollama_llm"],
    target_description="Medical summary",
    feature_hints=["report_text"],
    supports_realtime=True,
    color="#06b6d4",
))

register(ModuleConfig(
    id="drug-recommendation",
    name="Drug Recommendation",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Recommend appropriate drugs based on patient conditions.",
    icon="💊",
    tags=["classification", "healthcare", "pharmacology"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Recommended drug",
    feature_hints=["age", "sex", "blood_pressure", "cholesterol", "na_k_ratio"],
    color="#10b981",
))

register(ModuleConfig(
    id="hospital-readmission",
    name="Hospital Readmission Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict whether a patient will be readmitted within 30 days.",
    icon="🏥",
    tags=["classification", "healthcare", "hospital"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "lightgbm"],
    target_description="Readmitted (Yes/No)",
    feature_hints=["age", "diagnosis", "num_procedures", "discharge_type", "length_of_stay"],
    color="#3b82f6",
))

register(ModuleConfig(
    id="icu-mortality-prediction",
    name="ICU Mortality Prediction",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict ICU patient mortality risk using clinical data.",
    icon="📟",
    tags=["classification", "healthcare", "icu", "critical-care"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting", "neural_network"],
    target_description="Mortality risk (Yes/No)",
    feature_hints=["age", "sapsii_score", "sofa_score", "heart_rate", "blood_pressure"],
    color="#ef4444",
))

register(ModuleConfig(
    id="patient-risk-stratification",
    name="Patient Risk Stratification",
    category=ModuleCategory.HEALTHCARE,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Stratify patients into risk categories for care prioritization.",
    icon="📊",
    tags=["classification", "healthcare", "risk"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "clustering_gmm"],
    target_description="Risk level (Low/Medium/High/Critical)",
    feature_hints=["age", "comorbidities", "lab_values", "vitals"],
    color="#a78bfa",
))

# ═══════════════════════════════════════════════════════════════════════════════
# CATEGORY 7 — INDUSTRIAL AI (10)
# ═══════════════════════════════════════════════════════════════════════════════

register(ModuleConfig(
    id="predictive-maintenance",
    name="Predictive Maintenance",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.BINARY_CLASSIFICATION,
    description="Predict equipment failures before they occur.",
    icon="🔧",
    tags=["classification", "industrial", "iot", "maintenance"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "isolation_forest", "lstm"],
    target_description="Failure (Yes/No)",
    feature_hints=["temperature", "vibration", "pressure", "runtime_hours", "current"],
    color="#f97316",
))

register(ModuleConfig(
    id="machine-failure-prediction",
    name="Machine Failure Prediction",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Predict failure type in industrial machines.",
    icon="⚙️",
    tags=["classification", "industrial", "manufacturing"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Failure type",
    feature_hints=["air_temp", "process_temp", "speed", "torque", "tool_wear"],
    color="#ef4444",
))

register(ModuleConfig(
    id="quality-inspection",
    name="Quality Inspection",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.COMPUTER_VISION,
    description="Automated visual quality inspection using computer vision.",
    icon="🔍",
    tags=["cv", "industrial", "quality", "manufacturing"],
    input_type="image",
    default_algorithms=["yolov8", "cnn_classifier", "anomaly_detection"],
    target_description="Pass / Fail / Defect type",
    color="#22c55e",
))

register(ModuleConfig(
    id="defect-detection",
    name="Defect Detection",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.COMPUTER_VISION,
    description="Detect manufacturing defects in product images.",
    icon="⛔",
    tags=["cv", "industrial", "defect", "manufacturing"],
    input_type="image",
    default_algorithms=["yolov8", "autoencoder_anomaly"],
    target_description="Defect type and location",
    color="#dc2626",
))

register(ModuleConfig(
    id="supply-chain-optimization",
    name="Supply Chain Optimization",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.REGRESSION,
    description="Optimize supply chain costs and lead times using ML.",
    icon="🔗",
    tags=["regression", "industrial", "supply-chain", "optimization"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="Optimal reorder quantities / lead times",
    feature_hints=["lead_time", "demand", "cost", "supplier_reliability"],
    color="#6366f1",
))

register(ModuleConfig(
    id="inventory-prediction",
    name="Inventory Prediction",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.REGRESSION,
    description="Predict optimal inventory levels to minimize waste and stockouts.",
    icon="📦",
    tags=["regression", "industrial", "inventory", "supply-chain"],
    input_type="tabular",
    default_algorithms=["prophet", "xgboost", "lstm"],
    target_description="Optimal inventory level",
    feature_hints=["historical_demand", "lead_time", "safety_stock", "season"],
    color="#10b981",
))

register(ModuleConfig(
    id="warehouse-analytics",
    name="Warehouse Analytics",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.MULTICLASS_CLASSIFICATION,
    description="Analyze warehouse operations and predict bottlenecks.",
    icon="🏭",
    tags=["classification", "industrial", "warehouse", "logistics"],
    input_type="tabular",
    default_algorithms=["random_forest", "gradient_boosting", "clustering"],
    target_description="Efficiency score / bottleneck type",
    feature_hints=["order_volume", "pick_rate", "staff_count", "location_id"],
    color="#f59e0b",
))

register(ModuleConfig(
    id="smart-manufacturing",
    name="Smart Manufacturing Dashboard",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.REGRESSION,
    description="Predict OEE and manufacturing KPIs from sensor data.",
    icon="🏗️",
    tags=["regression", "industrial", "manufacturing", "oee"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "gradient_boosting"],
    target_description="OEE score and production metrics",
    feature_hints=["availability", "performance", "quality", "shifts"],
    color="#a78bfa",
))

register(ModuleConfig(
    id="production-yield-prediction",
    name="Production Yield Prediction",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.REGRESSION,
    description="Predict manufacturing yield from process parameters.",
    icon="📊",
    tags=["regression", "industrial", "manufacturing", "yield"],
    input_type="tabular",
    default_algorithms=["random_forest", "xgboost", "neural_network"],
    target_description="Yield percentage",
    feature_hints=["temperature", "pressure", "speed", "material_grade", "machine_age"],
    color="#84cc16",
))

register(ModuleConfig(
    id="equipment-health-monitoring",
    name="Equipment Health Monitoring",
    category=ModuleCategory.INDUSTRIAL,
    task_type=TaskType.REGRESSION,
    description="Monitor and score equipment health from sensor data.",
    icon="📡",
    tags=["regression", "industrial", "iot", "health"],
    input_type="tabular",
    default_algorithms=["isolation_forest", "autoencoder", "xgboost"],
    target_description="Health score (0-100)",
    feature_hints=["vibration", "temperature", "current", "noise_level", "runtime"],
    color="#06b6d4",
))


def get_module(module_id: str) -> Optional[ModuleConfig]:
    return MODULE_REGISTRY.get(module_id)


def get_modules_by_category(category: ModuleCategory) -> List[ModuleConfig]:
    return [m for m in MODULE_REGISTRY.values() if m.category == category]


def get_all_modules() -> List[ModuleConfig]:
    return list(MODULE_REGISTRY.values())
