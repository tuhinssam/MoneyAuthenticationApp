from flask import Flask, request, render_template
import numpy
import pandas
import pickle
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)
pickle_in = open('classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def predict():
    """
    Lets authenticate bank note
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
    """
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    return "Predicted val = "+ str(prediction)

@app.route('/predict_file', methods=['POST'])
def predict_from_file():
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df=pandas.read_csv(request.files.get("file"))
    prediction = classifier.predict(df)
    return "predicted value from file = "+str(list(prediction))

@app.route('/predictui', methods=['POST'])
def predict_ui():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [numpy.array(int_features)]
    prediction = classifier.predict(final_features)
    if prediction[0]==0:
        return render_template('index.html', prediction_text='Bank note is not valid')
    else:
        return render_template('index.html', prediction_text='Given note is authorized bank note')
    
if __name__ == '__main__':
    app.run(debug = True)