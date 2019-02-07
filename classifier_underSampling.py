
# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
class Model:
    def __init__(self):
        self.sc = StandardScaler()
        

    def read_df(self):
        self.df = pd.read_csv("dataset.csv")
# Missing values treatment
    def fill_na(self):
        # fill null values with the mean values of that feature
        self.df["tenure"].fillna(self.df["tenure"].mean(), inplace=True)
        # fill null values with the mode values of that feature is repeated more often than any other 
        self.df["SeniorCitizen"].fillna(self.df["SeniorCitizen"].mode()[0], inplace=True)

# Standardization
    def scaling(self):
        self.df['tenure']=self.sc.fit_transform(self.df['tenure'].values.reshape(-1,1))
        self.df['MonthlyCharges']=self.sc.fit_transform(self.df['MonthlyCharges'].values.reshape(-1,1))

# Label encoding 

    def label_df(self):
        self.le = {}
        self.le_name_mapping = {}

        for i in self.df.columns:
            if self.df[i].dtype == 'O':
                self.le[i] = LabelEncoder()
                self.df[i] = self.le[i].fit_transform(self.df[i])
                self.le_name_mapping[i] = dict(zip(self.le[i].classes_, self.le[i].transform(self.le[i].classes_)))  
                #print(i,":-",le_name_mapping[i])

#Feature engineering
    def split_df(self):
        self.X = self.df[['gender', 'SeniorCitizen', 'tenure', 'OnlineSecurity', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']].values
        self.y = self.df['Churn'].values

    def under_samplin(self):
        from sklearn.linear_model import LogisticRegression
        from imblearn.under_sampling import InstanceHardnessThreshold
        iht = InstanceHardnessThreshold(random_state=0,
                                         estimator=LogisticRegression(
                                             solver='lbfgs', multi_class='auto'))
        self.X_resampled, self.y_resampled = iht.fit_resample(self.X, self.y)


                    
# * The data is biased to 0
# Splitting the dataset into the Training set and Test set
    def train_test(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.X_resampled, self.y_resampled, test_size=0.2, random_state=42)
    def classifier(self, classifier):
        if classifier == "LinearRegression":
            self.cl = LogisticRegression()
        elif classifier == "KNN":
            #KNeighborsClassifier()
             import sklearn.metrics as metrics
             score=[]
             for k in range(1,100):
                 knn=KNeighborsClassifier(n_neighbors=k,weights='uniform')
                 knn.fit(self.x_train,self.y_train)
                 predKNN=knn.predict(self.x_test)
                 accuracy=metrics.accuracy_score(predKNN,self.y_test)
                 score.append(accuracy*100)
             self.n  = score.index(max(score))+1
             self.cl = KNeighborsClassifier(n_neighbors=self.n)
             #print(self.n)
             #self.cl = KNeighborsClassifier()
        elif classifier == "SVM":
            self.cl = SVC(kernel='linear')
        elif classifier == "NaiveBayes":
            self.cl = GaussianNB()
        elif classifier == "DecisionTree":
            self.cl = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        elif classifier == "RandomForest":
            self.cl = RandomForestClassifier(n_estimators=10,random_state=45,criterion='gini')
        self.cl.fit(self.x_train,self.y_train)
    def evl_train(self):
        Accuracy_train = self.cl.score(self.x_train,self.y_train)
        return Accuracy_train

    def evl_test(self):
        y_pred = self.cl.predict(self.x_test)
        Accuracy_test = accuracy_score(self.y_test,y_pred)
        return Accuracy_test

    def conf(self):
        y_pred = self.cl.predict(self.x_test)
        # Making the Confusion Matrix will contain the correct and incorrect prediction on the dataset.
        cm = confusion_matrix(self.y_test, y_pred)
        return cm
 # accuracy

    def predict(self,gender,SeniorCitizen,tenure,OnlineSecurity,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges):
        li=[]
        li.append(self.le['gender'].transform([gender])[0])
        li.append(int(SeniorCitizen))
        li.append(float(tenure))
        li.append(self.le['OnlineSecurity'].transform([OnlineSecurity])[0])
        li.append(self.le['Contract'].transform([Contract])[0])
        li.append(self.le['PaperlessBilling'].transform([PaperlessBilling])[0])
        li.append(self.le['PaymentMethod'].transform([PaymentMethod])[0])
        li.append(float(MonthlyCharges))
        
        dic={}
        c=0
        for i in ['gender', 'SeniorCitizen', 'tenure', 'OnlineSecurity', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']:
            dic[i]=[li[c]]
            c+=1
        data=pd.DataFrame(data=dic)

        return self.le["Churn"].inverse_transform(self.cl.predict(data))

    def pick(self):
        pickle_save = open("cl.pickle","wb")
        pickle.dump(self.cl,pickle_save)
        pickle_save.close()
        
if __name__=='__main__':
    clsf = Model()
    clsf.preprocessing()
    clsf.chooseclas()
    clsf.classifier()
    clsf.evaluate()
    clsf.predict_page()
    clsf.predict(gender=0,SeniorCitizen=0.0,tenure=0.039,OnlineSecurity=0,Contract=1, PaperlessBilling=1,PaymentMethod=3,MonthlyCharges= -0.26)
    clsf.pick()





