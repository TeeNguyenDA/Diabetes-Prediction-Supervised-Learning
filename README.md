# machine_learning_project-supervised-learning

In this Project, we are going to perform a full supervised learning machine learning project on a "Diabetes" dataset. This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The data source of this dataset is from [Kaggle](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset).

**Project objective**: To diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset.

**Constraints of the dataset**: "Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. From the data set in the (.csv) file, we can find several variables, some of them are independent (several medical predictor variables) and only one target dependent variable (Outcome)."*

**Context of the dataset**: The context of the medical predictor variables and the target variables are from females at least 21 years old from the Pima Indian heritage.

## Repository Structure:

Repository Link: [Diabetes-Prediction-Supervised-Learning](https://github.com/TeeNguyenDA/Diabetes-Prediction-Supervised-Learning)

* [Supervised Learning - Project](Supervised Learning - Project.ipynb)
* [diabetes.csv](diabetes.csv)
* [README.md](README.md)
* [Final Project - Description.docx](Final Project - Description.docx)

## Project Outcomes

- Supervised Learning: Use supervised learning techniques to build a machine learning model that can predict whether a patient has diabetes or not, based on certain diagnostic measurements.The project involves three main parts: exploratory data analysis, preprocessing and feature engineering, and training a machine learning model.

## Duration:
Approximately 3 hours and 20 minutes.

## Project Description:
In this projects, we will apply supervised learning techniques to a real-world data set and use data visualization tools to communicate the insights gained from the analysis.

**Project High-level Overview**:
At a high level, the project will involve the following tasks:

- Exploratory data analysis and pre-processing: We will import and clean the data sets, analyze and visualize the relationships between the different variables, handle missing values and outliers, and perform feature engineering as needed.

- Supervised learning: We will use the Diabetes dataset to build a machine learning model that can predict whether a patient has diabetes or not, using appropriate evaluation metrics such as accuracy, precision, recall, F1-score, and ROC-AUC. We will select at least two models, including one ensemble model, and compare their performance.

The ultimate goal of the project is to gain insights from the data sets and communicate these insights to stakeholders using appropriate visualizations and metrics to make informed decisions based on the business questions asked."

## Actual Project Flow:

- **Part I : EDA - Exploratory Data Analysis**

Answers those questions:

1. Are there any missing values in the dataset?
2. What is the distribution of each predictor variable?
3. Are there any outliers in the predictor variables?
4. How are the predictor variables related to the outcome variable?
5. How are the predictor variables related to each other?
6. Is there any interaction effect between the predictor variables?
7. What is the average age of the individuals in the dataset?
8. What is the average glucose level for individuals with diabetes and without diabetes?
9.  Are there any differences in the predictor variables between males and females (if gender information is available)?

- **Part II : Preprocessing & Feature Engineering**

Include these steps:

1. Handling missing values
2. Handling outliers
3. Dropping duplicates (if any)
4. Feature Engineering/Selection
5. Features transformation
6. Splitting the data and Scaling

- **Part III : Training ML Model**

Include:

0. Apply 5-fold cross-validation, compute average scores to find the best 2 models: Logistic Regression and Random Forest Classifier Models
1. Logistic Regression Model
2. Random Forest Classifier Model

- **Part IV : Conclusion**

Include:

- Conclusion from the exploratory data analysis (EDA) conducted
- Conclusion from the the machine learning models developed

## Final Project Findings:

- **Conclusion from the exploratory data analysis (EDA) conducted**:

* Note that all patients here are females at least 21 years old of Pima Indian heritage. The average age of the individuals in the dataset is 33 years old.

* There is a moderate postive linear relationship between 'Glucose' and 'Outcome' (0.47). Higher glucose levels might be associated with a higher likelihood of the individual having diabetes (the outcome being 1). There is also a weak positive correlations between 'BMI, Age, Pregnancies' and 'Outcome'(repsectively 0.29, 0.24, 0.2). As BMI increases or older ages or having more pregnancies, the likelihood of having diabetes (the outcome being 1) might also increase. 

* Some medical measurements in this dataset are not independent and there are correlations between them. There's a moderate positive linear correlation between 'Age' and 'Pregnancies' (0.54); indicating that as age increases, the number of pregnancies might also increase. A moderate positive linear correlation between 'Insulin' and 'SkinThickness' (0.44) indicates as the Insulin level increases, the thickness of the skin might also increase. Despite being a slight likelihood: people with higher BMI might tend to have thicker skin (correlation between 'SkinThickness' and 'BMI' is 0.39), while higher glucose levels might be associated with higher insulin levels (correlation between 'Insulin' and 'Glucose' is 0.33.

* Glucose is a rather important determiner in predicting whether the patient is having diabetes or not (feature_rank #4 by using RFE with LogisticRegression as the base). The average glucose level for individuals without diabetes is 109.980, whereas the Average glucose level for individuals with diabetes is 141.257.

- **Conclusion from the the machine learning models developed**:

* Our task is to diagnostically predict whether a patient has diabetes, based on certain diagnostic measurements included in the dataset which us a binary classification task. We tried to apply 5-fold cross-validation and compute average scores to find the better performing machine learning algorithms among: Logistic Regression, Decision Trees and Random Forest. The result indicates that Logistic regression and Random Forest seem to perform much better.

* This is an imbalanced dataset wherein the target variable to predict diabetes has more patients samples not having diabetes (much more 0 values compared to 1 values in the 'Outcome' target variable). To combat that, we know Accuracy score won't work well in evaluating the model performance, and therefore prefer to use Precision and Recall to perform model evaluation comparison. At the end, the Logistic Regression Model is the best performing with 72.91% of the correctly predicted diabete cases turned out to be truly women over 21 from the Pima Indian tribes having diabetes. 63.64% of those positive predictions were successfully predicted by our model.