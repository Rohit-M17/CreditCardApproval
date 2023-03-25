# CreditCardApprovalAnalysis
# D8TA Final Project - Credit Card Approval

The goal of this project is to ensure data privacy and fairness for the credit card dataset. We aim to remove any unfair discrimination in the credit card approval process and promote Fair Equality of Opportunity (FEO).


## Authors

- [@RohitManimaran](https://www.github.com/octokatherine)
- [@Shreya Balaji](https://github.com/sbala025)
- [@Vishal Menon](https://github.com/vmeno0020)

## Data

The data used for this project is the Credit Card Approval dataset from Kaggle https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction. It contains 438558 credit card user's entries and 14 attributes, including personal and financial information about credit card applicants.
## Exploratory Data Analysis
Exploratory data analysis (EDA) is a pivotal step in data analysis that helps prevent downstream issues. We conducted EDA on the Credit Card Approval dataset to identify patterns and relationships between variables, detect outliers, and check for missing data. This allowed us to gain a better understanding of the data and make informed decisions about how to preprocess and engineer features.
![4](https://user-images.githubusercontent.com/59380765/227697845-45c29464-2768-4d77-8d8f-493acd133e28.png)
![1](https://user-images.githubusercontent.com/59380765/227697847-a3fea62d-31a1-4b53-b27f-89e17a5a764c.png)
![2](https://user-images.githubusercontent.com/59380765/227697848-96757b47-c442-4858-ae18-c1d87110717c.png)
![3](https://user-images.githubusercontent.com/59380765/227697849-80a2e9b4-d31d-4d52-8ce4-590875d86665.png)

## Feature Engineering
We developed a method to classify applicants as "good" or "bad" based on a formula that calculates available funds after accounting for expenses such as children, car, phone, and housing. Dividing this value by total income yields the percentage of funds available for other expenses, with a higher percentage indicating a better potential credit card owner. We also used a Pearson correlation analysis to determine which factors to include in the formula.
## Differential Privacy
Differential privacy is a privacy-preserving technique that adds random noise to sensitive data before sharing it with others. The goal is to prevent individual data points from being linked to specific individuals, while still allowing useful statistical analyses to be performed on the data.

In this project, Laplace noise was added from a Laplace distribution centered around zero. The amount of noise added was determined by the sensitivity of the data, which is a measure of how much the output of a statistical query can change if a single individual's data is removed or changed. Epsilon was used to control the amount of noise added to the original data to protect against privacy breaches.
## Neural Network Model Devlopment

The project involved developing a three-layer neural network with 32, 16, and 1 output layer nodes to predict the credit quality of an applicant based on the following inputs:

- AMT_INCOME_TOTAL: Applicant's total income
- CNT_CHILDREN: Number of children the applicant has
- FLAG_OWN_CAR: Whether the applicant owns a car or not (binary: 0 or 1)
- FLAG_PHONE: Whether the applicant provided a mobile phone number (binary: 0 or 1)
The model was optimized using the Adam optimizer, which combines momentum and learning rate decay to minimize the loss function, and L2 regularization, which adds a penalty to the loss function to lower the weight of certain nodes. The loss graph was used to track the performance of the model during training.

![output](https://user-images.githubusercontent.com/59380765/227697747-6e7352c3-a4c1-4cf0-b76c-12139becc197.png)

## Results
The addition of differential privacy to the credit card approval model helped to promote fairness and privacy in the credit application process. As epsilon increased, the accuracy and loss values remained stable, indicating that the DP added did not significantly affect the model's performance.

- An epsilon of 0.001 resulted in the model misclassifying 3218 negative samples as positive but correctly classified all positive samples. The model was more prone to false positives without DP.
- An epsilon of 0.01 resulted in fewer false positives than 0.001, and increasing epsilon helped reduce the number of false positives.
- An epsilon of 10 resulted in significantly higher accuracy and validation accuracy, lower loss and validation loss than without differential privacy, and improved performance indicated by some true positive predictions in the confusion matrix.
![EpsilonTable](https://user-images.githubusercontent.com/59380765/227697708-368bf67b-57b3-4aee-842c-8345eb64f4fc.png)

The choice of epsilon value can depend on the type of credit plan the person is applying for.
## Additional Detail
Additional details about the project include:

- A method was developed to classify applicants as "good" or "bad" based on available funds after accounting for expenses such as children, car, phone, and housing. This was compared to the status attribute to provide a baseline on each applicant.
- A Pearson correlation analysis was run against the status attribute to determine what factors were needed to create an equation for calculating available funds.
- The income was calculated using the formula:

        Income - ((Avg cost of child * # of children) + Avg cost of car ownership (flag) + Avg cost of owning a phone (flag) + Avg cost of housing).

### Overall, the project demonstrates the use of machine learning and differential privacy techniques to create a fair and private credit card approval model.

