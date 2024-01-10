from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn
import pickle
import numpy as np
import sklearn
import lime
from lime import lime_tabular, lime_image
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv(r'E:\mypython\ML_Deployment\module\train.csv')
#train_data.head(100)
X_train = train_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

label_encoder = LabelEncoder()
X_train['Sex'] = label_encoder.fit_transform(X_train['Sex'])
X_train['Embarked'] = label_encoder.fit_transform(X_train['Embarked'])
X_train = X_train.dropna()

X_train.head(10)

explainer = LimeTabularExplainer(X_train.values, 
                                 feature_names=X_train.columns.values.tolist(), 
                                 class_names=['0','1'], 
                                 verbose=True, 
                                 mode='classification')

exp = explainer.explain_instance(X_train.values[0], model.predict_roba)


explainer = LimeTabularExplainer(train_data.values, 
                                 feature_names=train_data.columns.values.tolist(), 
                                 class_names=[0,1], 
                                 verbose=True, 
                                 mode='classification')

model = pickle.load(open('module\pipe.pkl', 'rb'))

exp = explainer.explain_instance(input1, model.predict)

prediction = model.predict(input)

print(prediction)


input = np.array([2, 1, 31, 0, 0, 10,1],dtype=object).reshape(1,7)

input1 = np.array([2, 1, 31, 0, 0, 10,1])

np.isnan(input1)

X_train.iloc[0].values

input



print(X_train.values[0])

with open(r'E:\mypython\ML_Deployment\module\model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

feature_names = loaded_model.feature_names
print("Feature Names:", feature_names)

feature_names = model.feature_names_in_
print("Feature Names:", feature_names)




print("Feature Names:", feature_names)