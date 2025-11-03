# customer_behavior_modelling_for_purchase_intent_prediction
The objective of this project is to understand and predict customer purchasing behavior on an e-commerce platform using session-level interaction data. Each data point represents a unique user session containing features such as time spent on site, pages viewed, cart value, and ad engagement. The goal is to classify whether a user made a purchase (1) or not (0) based on these behavioral signals.

The analysis was conducted using Python, leveraging the following libraries and frameworks:
* Pandas and NumPy for data preprocessing and feature manipulation
* Matplotlib and Seaborn for exploratory data analysis and visualization
* Scikit-learn for implementing machine learning models and evaluation metrics

An initial exploratory data analysis was performed to examine the dataset structure, identify outliers, and study feature distributions. Outliers were handled using the IQR method, and log transformation was applied to normalize skewed numerical attributes. Categorical variables were one-hot encoded to facilitate model training, while redundant or weakly correlated features were removed to mitigate multicollinearity.

Two supervised learning algorithms: Logistic Regression and Random Forest Classifier, were employed for classification. Models were trained and evaluated using accuracy and weighted F1-score metrics to address the imbalance in the target variable. Furthermore, hyperparameter tuning was carried out using GridSearchCV to explore optimal configurations for both models.
