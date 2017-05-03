#Fraud detection case study

Team members:

 - Rita Golovonevsky
 - Eliot Logan-
 - Catherine Chen
 - Mark Llorente
 - Dmytro Kovalchuk
EDA 

- Notebook
- visualizations

Feature Engineering
-
- Created a column containing the string 'Party' in the description.
- Created a column containing the string 'pass' in the description.
- New column indicating ALL CAPS in event description.
- created a new column of common webmail email domains (gmail, hotmail, yahoo).
- Dummify non-numeric features.

Description classifier

- build model adapted
- builds confusion matrix
- accuracy:  .92
- precision: .89
- recall:    .2
- dumps into pickle

Pipeline Class:

- Run Baseline scores (precision, recall)
- Logistic Regression
- SVM
- Random Forest
- Parameter tuning

Database connection:

- Mongo
- for each 

Todos:

- confusion matrix & build cost function

Web App:
