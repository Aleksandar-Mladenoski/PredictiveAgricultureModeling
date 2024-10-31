# Predictive Modeling for Agriculture

This repository contains my solution to a DataCamp project on predictive modeling for agriculture. The project focuses on developing machine learning models to predict various agricultural factors based on environmental and soil data.

---

## Project Overview

This project applies predictive modeling to agricultural data, aiming to improve insights into crop yield, soil quality, and related agricultural variables. Using supervised learning, this project evaluates several models to identify the most effective in providing reliable predictions for agricultural data analysis.

## Dataset

The dataset used in this project includes various soil and environmental attributes that are key to agricultural productivity. Features include:
- **Soil Properties**: pH, organic matter content, soil texture, etc.
- **Environmental Conditions**: Temperature, rainfall, humidity, etc.
- **Target Variable**: Crop yield or another relevant agricultural metric.

The dataset is sourced from [DataCamp Datalab](https://www.datacamp.com/datalab).

---

## Methods

### 1. Data Preprocessing
- **Standardization**: Scaled features for improved model accuracy.
- **Handling Missing Values**: Imputed missing data with mean values.

### 2. Machine Learning Models
- **Random Forests**: Evaluated for feature importance and predictive accuracy.
- **Gradient Boosting**: Assessed as an alternative with boosted performance.
- **Model Selection and Tuning**: Hyperparameter tuning to optimize each model’s performance.

### 3. Model Evaluation
- **Mean Absolute Error (MAE)**: Used as the primary metric for model evaluation.
- **Cross-Validation**: Ensured robustness across multiple data splits.

---

## Code Structure

- `main.py`: Contains the code for data loading, preprocessing, model training, and evaluation.
- `data_processing.py`: Scripts for cleaning and preprocessing the dataset.

---

## Results and Analysis

The predictive models demonstrate promising accuracy, with the Random Forest model showing high reliability for agricultural predictions. Fine-tuning and feature selection further improved results, making this project a valuable resource for data-driven agricultural insights.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Required Libraries: Install dependencies via `pip install -r requirements.txt`.

### Usage

```bash
# Clone the repository
git clone https://github.com/Aleksandar-Mladenoski/predictive-agriculture-modeling.git

# Navigate to the project directory
cd predictive-agriculture-modeling

# Run the main script
python main.py
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## Acknowledgments

Thanks to DataCamp for providing the dataset through [DataCamp Datalab](https://www.datacamp.com/datalab) as part of their instructional resources.
```
