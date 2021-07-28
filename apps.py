# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


# %%
apps = Flask(__name__)
model = pickle.load(open('optimalvehiclepricing.pkl', 'rb'))

# %%
@apps.route('/')
def home():
    return render_template('index.html')


# %%
@apps.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    return render_template('index.html', prediction_text='The optimal price of vehicle would be {}'.format(output))


# %%
if __name__ == "__main__":
    apps.run(debug=True)


