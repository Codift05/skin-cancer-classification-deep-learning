#!/bin/bash

# Skin Cancer Classification - Project Setup & Run Script
# Linux/Mac shell script

echo ""
echo "=========================================="
echo "  Skin Cancer Classification Project"
echo "=========================================="
echo ""

show_menu() {
    echo ""
    echo "Select option:"
    echo "1. Install dependencies"
    echo "2. Run training notebook (Jupyter)"
    echo "3. Run web app (Streamlit)"
    echo "4. Install & setup (first time)"
    echo "5. Exit"
    echo ""
}

while true; do
    show_menu
    read -p "Enter choice (1-5): " choice
    
    case $choice in
        1)
            echo ""
            echo "Installing dependencies..."
            pip install -r requirements.txt
            ;;
        2)
            echo ""
            echo "Starting Jupyter Lab with training notebook..."
            cd notebook
            jupyter lab training.ipynb
            cd ..
            ;;
        3)
            echo ""
            echo "Starting Streamlit web app..."
            streamlit run app/app.py
            ;;
        4)
            echo ""
            echo "=========================================="
            echo "  First Time Setup"
            echo "=========================================="
            echo ""
            echo "This will:"
            echo "- Create virtual environment"
            echo "- Install all dependencies"
            echo "- Check dataset structure"
            echo ""
            read -p "Continue? (y/n): " confirm
            
            if [[ $confirm == "y" || $confirm == "Y" ]]; then
                # Create virtual environment
                echo "Creating virtual environment..."
                python3 -m venv venv
                
                # Activate virtual environment
                source venv/bin/activate
                
                # Install dependencies
                echo "Installing dependencies..."
                pip install --upgrade pip
                pip install -r requirements.txt
                
                echo ""
                echo "=========================================="
                echo "  Setup Complete!"
                echo "=========================================="
                echo ""
                echo "Next steps:"
                echo "1. Place dataset in data/ folder (benign/ and malignant/ subdirectories)"
                echo "2. Run: jupyter lab notebook/training.ipynb  (to train model)"
                echo "3. Run: streamlit run app/app.py  (to start web app)"
                echo ""
            fi
            ;;
        5)
            echo ""
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done
