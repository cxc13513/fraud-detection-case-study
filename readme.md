#Fraud detection case study

Presentation link: https://docs.google.com/presentation/d/1ysho24cQ_N6M8hqZh_97WahK1PdsEzYAo-8e6rZMrOo/edit#slide=id.p

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
- F1 = .33
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
