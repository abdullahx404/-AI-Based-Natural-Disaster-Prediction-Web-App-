#!/bin/bash
# Quick Setup & Run Guide
# Save this as setup_guide.sh and run: bash setup_guide.sh

PROJECT_DIR="/Users/hussain/Documents/Projects/-AI-Based-Natural-Disaster-Prediction-Web-App-"

echo "🚀 AI-Based Disaster Prediction - Quick Setup Guide"
echo "=================================================="
echo ""

# Function to activate venv
activate_env() {
    echo "📦 Activating virtual environment..."
    cd "$PROJECT_DIR"
    source .venv/bin/activate
    echo "✅ Virtual environment activated"
    echo ""
}

# Function to run preprocessing
run_preprocessing() {
    echo "🔧 Running data preprocessing..."
    activate_env
    python3 code/preprocessing.py
}

# Function to train models
train_models() {
    echo "🤖 Training baseline models..."
    activate_env
    python3 code/baseline_models.py
}

# Function to evaluate models
evaluate_models() {
    echo "📊 Evaluating model performance..."
    activate_env
    python3 code/model_evaluation.py
}

# Function to run full pipeline
run_full_pipeline() {
    echo "🚀 Running complete ML pipeline..."
    activate_env
    echo ""
    echo "Step 1/3: Data Preprocessing..."
    python3 code/preprocessing.py
    echo ""
    echo "Step 2/3: Model Training..."
    python3 code/baseline_models.py
    echo ""
    echo "Step 3/3: Model Evaluation..."
    python3 code/model_evaluation.py
    echo ""
    echo "✅ Pipeline complete! Check results/ directory"
}

# Function to start Jupyter notebook
start_notebook() {
    echo "📓 Starting Jupyter Notebook..."
    activate_env
    jupyter notebook notebooks/ml_pipeline.ipynb
}

# Function to start Streamlit app
start_streamlit() {
    echo "🌐 Starting Streamlit app..."
    activate_env
    streamlit run app.py
}

# Show menu
echo "📋 Select an option:"
echo ""
echo "1) Activate virtual environment"
echo "2) Run data preprocessing"
echo "3) Train ML models"
echo "4) Evaluate models"
echo "5) Run complete pipeline (all steps)"
echo "6) Start Jupyter Notebook"
echo "7) Start Streamlit app"
echo "8) Show environment info"
echo ""
echo "Enter your choice (1-8): "
read choice

case $choice in
    1) activate_env ;;
    2) run_preprocessing ;;
    3) train_models ;;
    4) evaluate_models ;;
    5) run_full_pipeline ;;
    6) start_notebook ;;
    7) start_streamlit ;;
    8) 
        echo "🔍 Environment Information:"
        echo "=========================="
        echo "Python version:"
        python3 --version
        echo ""
        echo "Virtual environment path:"
        echo "$PROJECT_DIR/.venv"
        echo ""
        echo "Installed packages (main):"
        source .venv/bin/activate
        pip list | grep -E "pandas|numpy|scikit-learn|matplotlib|seaborn|xgboost|flask|streamlit"
        ;;
    *) echo "❌ Invalid choice" ;;
esac

echo ""
echo "Done! 🎉"
