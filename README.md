# Customer-Churn-Prediction
Predict the probability of churn:

classifier.py : preprocessing(Missing values treatment, Label encoding,Standardization) , choose classifier and get accuracy, enter new data about customer and get result if he will churn or not. 

api.py : using Flask and HTML i made UI for user can use it easily to know if customer will churn or not.

classifier_underSampling.py : update classifier.py after undersampling. after using it the accurcy increased but the model trained on 1000 form 7000 row from data.

api_underSampling.py : update api.py after undersampling.

classifier_overSampling.py : update classifier.py after undersampling. after using it the accurcy decreased and the data get balance but the model trained on 1000 form 7000 row from data.

api_overSampling.py : update api.py after oversampling.

Assignment no-2 LDA.ipynb : used LDA get accurcy more than using pca

Assignment no-2 PCA.ipynb : used PCA

Assignment no.2.ipynb : work on all data.

Assignment no.2 work on 8 features.ipynb : work on 8 features and used GradientBoostingClassifier.

Assignment no.2 OVER SAMPLING.ipynb : after using it the accurcy decreased and the data get balance but the model trained on 2000 form 7000 row from data.

Assignment no.2 Under SAMPLING.ipynb : after using it the accurcy increased and the data get balance but the model trained on 1000 form 7000 row from data.

Data set: customer churn occurs when customers or subscribers stop doing business with a company or service, also known as customer attrition. It is also referred as loss of clients or customers. One industry in which churn rates are particularly useful is the telecommunications industry, because most customers have multiple options from which to choose within a geographic location.
