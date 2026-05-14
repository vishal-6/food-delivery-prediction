# Food Delivery Time Prediction

## Overview
Machine learning project to predict food delivery status (On Time, Delayed, Very Late) across 10,000+ delivery records using classification models.

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Models & Results
| Model         | Accuracy |
|---------------|----------|
| Naive Bayes   | 72.70%   |
| KNN           | 74.45%   |
| Decision Tree | 81.20%   |

## Features
- Synthetic food delivery dataset (10,000 samples, 16 features)
- Custom Geo Distance feature using Haversine formula on GPS coordinates
- Peak hour and weekend feature engineering
- Hyperparameter tuning with GridSearchCV
- EDA plots: delivery status, peak hours, correlation matrix, geo distance

## Project Structure

git add .
git commit -m "Add README"
git push
