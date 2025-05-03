📱 Mobile Price Range Prediction

This project uses machine learning to predict the price range of mobile phones based on their technical specifications such as RAM, battery power, processor, and more.
🧠 Model

We use a Logistic Regression model with the following pipeline:

    StandardScaler for feature normalization

    Multinomial Logistic Regression (with lbfgs solver)

    Cross-Validation for model evaluation

    GridSearchCV for hyperparameter tuning

🎯 Accuracy

✅ Training Accuracy: 97%
✅ Cross-Validation Accuracy: ~97% (5-fold)
📊 Features Used

    Battery power

    RAM

    Mobile weight

    Camera specs (front and primary)

    Number of cores

    Screen resolution

    Presence of 4G/3G/WiFi/Touchscreen/etc.

📁 Files
File	Description
train.csv	Training data with features and price_range label
test.csv	Test data without labels (used for submission)
submission.csv	Final predictions for test set
main.py	Complete training and prediction pipeline
README.md	Project overview
🚀 How to Run

# Install dependencies
pip install -r requirements.txt

# Run the script
python main.py

This will:

    Train the model

    Evaluate accuracy

    Generate predictions for the test dataset

    Save predictions to submission.csv

📦 Requirements

    pandas

    matplotlib

    scikit-learn

Install via:

pip install pandas matplotlib scikit-learn

📌 Future Improvements

    Try more models (Random Forest, SVM, etc.)

    Feature engineering (combine or remove less informative features)

    UI interface for inputting phone specs and predicting price

👨‍💻 Author

Made with ❤️ by Wayne
(Feel free to add links to your GitHub, YouTube, Instagram, etc.)
