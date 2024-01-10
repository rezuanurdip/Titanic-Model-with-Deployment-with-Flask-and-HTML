from flask import Flask, request, render_template
import pickle
import numpy as np
import sklearn
import lime
from lime import lime_tabular, lime_image
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import jinja2


model = pickle.load(open('E:\mypython\ML_Deployment\Titanic Mode\Titanic Mode.pkl', 'rb'))
X_train = pd.read_csv(r'E:\mypython\ML_Deployment\Titanic Mode\X_train.csv')

explainer = LimeTabularExplainer(X_train.values, 
                                 feature_names=X_train.columns.values.tolist(), 
                                 class_names=['0','1'], 
                                 verbose=True, 
                                 mode='classification')
    

app = Flask(__name__)
app.jinja_env.globals.update(abs=abs)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    if output == 0:
        out = 'Dead'
    else:
        out = 'Alive'
        
    final_features1 = np.array(features)
    exp = explainer.explain_instance(final_features1, model.predict_proba)
    lime_explanation = exp.as_list()
        
    return render_template('index3.html', prediction_text='Survival Prediction: You were {}'.format(out), lime_explanation=lime_explanation)


if __name__ == "__main__":
    app.run(debug=True,host='0.0.0.0')
    

