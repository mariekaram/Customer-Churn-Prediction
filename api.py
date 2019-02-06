from flask import Flask
from flask import jsonify, request, render_template
from classifier import Model
clsf = Model()
app = Flask(__name__)


@app.route("/")
def preprocessing():
	clsf.read_df()
	clsf.fill_na()
	clsf.scaling()
	clsf.label_df()
	clsf.split_df()
	clsf.train_test()
	return  "<h2> Preprocessing dataset of Customer Churn :-</h2>\
	<form method='GET' action='http://127.0.0.1:5050/chooseclas'>\
            <input type='submit' value='preprocessing'>\
            </form>"

@app.route("/chooseclas")
def chooseclas():
	return "<h2> Choose a classifier :-</h2>\
	<form method = 'GET' action ='http://127.0.0.1:5050/evaluate'>\
    </br>\
    <select name='name'>\
      <option value='LinearRegression'>LinearRegression</option>\
      <option value='KNN'>KNN</option>\
      <option value='SVM'>SVM</option>\
      <option value='NaiveBayes'>NaiveBayes</option>\
      <option value='DecisionTree'>DecisionTree</option>\
      <option value='RandomForest'>RandomForest</option>\
    </select>\
  </br></br>\
  <input type='submit' value='Accuracy'>\
  </form>"

@app.route("/evaluate",methods=['GET','POST'])
def evaluate():
	classifier = request.args.get('name')
	clsf.classifier(classifier)
	score_train = clsf.evl_train()
	score_test = clsf.evl_test()    
	con1,con2 = clsf.conf()

        
	return "<h2> Accuracy for train :-</h2>\
			"+str(round(score_train, 2)*100)+"%\
			</br>\
			<h2> Accuracy for test :-</h2>\
			"+str(round(score_test, 2)*100)+"%\
			</br>\
			<h3> Confusion Matrix :-</h3>\
            "+str(con1)+"<br>\
            "+str(con2)+"</br></br>\
            \
            <form action='http://127.0.0.1:5050/predict_page'></br>\
            <input type='submit' value='Enter your data'>\
            </form>\
            "
@app.route("/predict_page")
def predict_page():
    return"<form method='GET' action='http://127.0.0.1:5050/predict'>\
            <h2> Please enter your data :- </h2>\
            <table><tr>\
            <td>Gender</td>\
            <td><select name='gender'>\
            <option>Male</option>\
            <option>Female</option>\
            </select></td></tr>\
            <tr><td>SeniorCitizen</td>\
            <td><select name='SeniorCitizen'>\
            <option>0</option>\
            <option>1</option>\
            </select></td></tr>\
            <tr><td>Tenure</td>\
            <td><input type='text' name=tenure></td></tr>\
            <tr><td>OnlineSecurity</td>\
            <td><select name='OnlineSecurity'>\
            <option>Yes</option>\
            <option>No</option>\
            <option>No internet service</option>\
            </select></td></tr>\
            <tr><td>Contract</td>\
            <td><select name='Contract'>\
            <option>Month-to-month</option>\
            <option>One year</option>\
            <option>Two year</option>\
            </select></td></tr>\
            <tr><td>PaperlessBilling</td>\
            <td><select name='PaperlessBilling'>\
            <option>Yes</option>\
            <option>No</option>\
            </select></td></tr>\
            <tr><td>PaymentMethod</td>\
            <td><select name='PaymentMethod'>\
            <option>Electronic check</option>\
            <option>Mailed check</option>\
            <option>Bank transfer (automatic)</option>\
            <option>Credit card (automatic)</option>\
            </select></td></tr>\
            <tr><td>MonthlyCharges</td>\
            <td><input type='text' name=MonthlyCharges></td></tr></table>\
            <input type='submit' value='predict'>\
            </form>\
    "
@app.route("/predict",methods=['GET','POST'])
def predict():
    gender = request.args.get('gender')
    SeniorCitizen = request.args.get('SeniorCitizen')
    tenure = request.args.get('tenure')
    OnlineSecurity = request.args.get('OnlineSecurity')
    Contract = request.args.get('Contract')
    PaperlessBilling = request.args.get('PaperlessBilling')
    PaymentMethod = request.args.get('PaymentMethod')
    MonthlyCharges = request.args.get('MonthlyCharges')
    #Predict the probability of churn 
    Chu = clsf.predict(gender, SeniorCitizen, tenure, OnlineSecurity,Contract, PaperlessBilling, PaymentMethod, MonthlyCharges)
    return "<h2>Will the customer be churned??!</h2>\
    <h3>"+str(Chu)+" </h3>"


if __name__ == '__main__':
	try:
		app.run(port='5050',host='0.0.0.0')
	except Exception as e:
		print("Error")

