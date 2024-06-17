# NBA-Game-Predictor

Project: Predicting NBA Game Outcomes Using Machine Learning

Table of Contents
Introduction
Data Collection and Cleaning
Setting up and Training Models
Results
Next Steps

Introduction
As a big fan of the NBA, I find it fascinating to use freely available data to predict game outcomes. With advanced feature engineering and data cleaning using Pandas, I aim to test various supervised learning methods on the resulting data. In the future, I plan to explore training a neural network and using player-level data to capture matchup-specific dependencies, although initial attempts were not successful.

Data Collection and Cleaning
Data for this project was collected from two open-source APIs. Player-level stats were retrieved from the NBA Stats API using a Python package, while team-level data came from Sports-Reference using another package.

Player Level Data
Initially, I collected data for each player in every game, including stats like points, rebounds, assists, blocks, and steals. Collecting this data involved scraping it from the NBA Stats API, and to avoid overloading the API, I implemented a 2-second pause between requests using time.sleep(2). Due to the large volume of data, it was stored locally in an HDF5 file for faster retrieval during training and inference.

Team Level Data
When player-level data did not yield good prediction results, I switched to team-level data. The collected team statistics are listed in the appendix. Many of these features were highly correlated, so a feature correlation matrix was created to identify uncorrelated features for better predictive power.

Feature Engineering
To ensure the model uses only information available before each game's tip-off, I engineered features representing team averages over the previous 5, 10, and 15 games. This increased the number of features to 228 per game. However, due to high correlations among these features, dimensionality reduction became necessary.

Feature Scaling
Before applying dimensionality reduction, I performed mean/standard deviation feature scaling to ensure each feature had a mean of 0 and a standard deviation of 1.

Dimensionality Reduction
To reduce dimensionality, I used Principal Component Analysis (PCA), which finds principal components—linear combinations of features that are orthogonal to each other. By taking the top k principal components, I preserved as much variance as possible. The number of retained principal components was treated as a hyperparameter for the predictive model.

Setting up and Training Models
I considered several "out of the box" machine learning classifiers from Scikit-Learn, using a grid search to determine the best parameters. The models tested included Random Forest and Logistic Regression.

Custom Evaluation Functions
Two custom evaluation functions were used: one targeting the best true positive rate for a fixed tolerable false positive rate, and another focusing on the best prediction accuracy above a fixed confidence threshold. These functions were used to evaluate model performance during grid search and on new data, respectively.

Choosing Best Parameters
Best parameters were chosen using Scikit-Learn's GridSearchCV with 5-fold cross-validation. The best performing model was selected based on the custom evaluation function, and its generalizability was estimated using the 2018-2019 NBA season as a holdout dataset.

Results
Random Forest
The Random Forest classifier performed well, with the best model retaining 80 principal components. Despite the high computational expense, the results were promising, as shown in the ROC curve and accuracy plots.

Logistic Regression
Logistic Regression did not perform as well as Random Forest, likely due to non-separable classes in the feature space. However, it still outperformed the baseline, as demonstrated by its ROC curve and accuracy plots.

Next Steps
Moving forward, I plan to try different models such as Support Vector Machine (SVM), Naive Bayes with kernel density estimation, and Gradient Boosting Classifier. A simple Multi-Layer Perceptron (MLP) could also be explored, though caution is needed to avoid overfitting.

Additionally, I aim to create a module for continuous data collection, enabling real-time predictions. While this project is primarily academic, it offers valuable insights into NBA games and machine learning. However, I do not recommend using these models for sports betting.

This project leverages Python, Pandas, Scikit-Learn, and open-source APIs for data collection and machine learning, making it a robust and engaging exploration of NBA game prediction.
