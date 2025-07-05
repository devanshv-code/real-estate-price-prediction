#flask file by me
from flask import Flask, render_template,request
import numpy as np
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')  #This will load home.html from templates folder
@app.route('/prediction',methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        # Load model and scaler
        model = pickle.load(open('ridge.pkl', 'rb'))
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        # Get form inputs
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        basement = 1 if request.form['basement'] == 'yes' else 0
        hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        parking = int(request.form['parking'])
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0

        # Furnishing status: One-hot encode (semi-furnished, unfurnished)
        furnishingstatus_input = request.form['furnishingstatus']
        if furnishingstatus_input == 'unfurnished':
            furnishingstatus = [0, 1]
        elif furnishingstatus_input == 'semi-furnished':
            furnishingstatus = [1, 0]
        else:  # furnished
            furnishingstatus = [0, 0]

        # Feature Engineering
        total_rooms = bedrooms + bathrooms
        is_big_house = 1 if area > 3000 else 0

        # Final feature vector (match model training order!)
        raw_features = np.array([[area, bedrooms, bathrooms, stories, mainroad,
                                  guestroom, basement, hotwaterheating, airconditioning,
                                  parking, prefarea] + furnishingstatus +[total_rooms, is_big_house]])

        # Scale input
        scaled_features = scaler.transform(raw_features)

        # Predict and convert from log(price) to price
        log_price = model.predict(scaled_features)[0]
        actual_price = np.exp(log_price)

        return render_template('result.html', prediction=round(actual_price, 2))

    return render_template('pred.html')
if __name__ == '__main__':
    app.run(debug=True)

   
