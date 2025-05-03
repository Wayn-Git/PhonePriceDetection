# Mobile Phone Price Classification

This project implements a machine learning model to predict mobile phone price ranges based on various specifications. The model classifies phones into different price categories using logistic regression.

## Overview

The goal of this project is to accurately classify mobile phones into price ranges (0, 1, 2, 3) based on features such as RAM, battery power, camera quality, and other specifications. This can be useful for manufacturers to determine optimal pricing strategies or for consumers to understand if a phone is reasonably priced.

## Dataset

The dataset contains mobile phone specifications and their corresponding price ranges:

- `train.csv`: Training data with features and price range labels
- `test.csv`: Test data with features only (price range to be predicted)

### Features

The dataset includes the following features:
- `battery_power`: Battery power in mAh
- `blue`: Bluetooth availability (0/1)
- `clock_speed`: Processor clock speed
- `dual_sim`: Dual SIM support (0/1)
- `fc`: Front camera megapixels
- `four_g`: 4G support (0/1)
- `int_memory`: Internal memory in GB
- `m_dep`: Mobile depth in cm
- `mobile_wt`: Weight of mobile phone
- `n_cores`: Number of processor cores
- `pc`: Primary camera megapixels
- `px_height`: Pixel resolution height
- `px_width`: Pixel resolution width
- `ram`: Random Access Memory in MB
- `sc_h`: Screen height of mobile in cm
- `sc_w`: Screen width of mobile in cm
- `talk_time`: Longest time a single battery charge will last
- `three_g`: 3G support (0/1)
- `touch_screen`: Touch screen support (0/1)
- `wifi`: WiFi support (0/1)

### Target Variable

- `price_range`: Price category of the mobile phone (0: Low cost, 1: Medium cost, 2: High cost, 3: Very high cost)

## Implementation

The project uses a simple but effective machine learning pipeline:

1. Data preprocessing with `StandardScaler` to normalize the features
2. Multinomial Logistic Regression as the classification algorithm
3. Training and evaluation on a split of the training data
4. Final model training on the full training dataset
5. Prediction on the test set and submission generation

## Model Performance

The model achieves good accuracy on both the validation and training datasets:
- Validation accuracy is calculated on a 20% holdout from the training data
- The model is then retrained on the full training dataset for final predictions

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- matplotlib
- numpy

## Usage

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main script: `python mobile_price_classification.py`

## Files

- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `mobile_price_classification.py`: Main script for model training and prediction
- `submission.csv`: Output file with predictions

## Future Improvements

Potential enhancements to the model:
- Feature engineering to create more informative features
- Hyperparameter tuning to optimize model performance
- Trying more advanced algorithms like Random Forest or XGBoost
- Implementing cross-validation for more robust model evaluation
