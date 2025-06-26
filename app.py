
import os
import cv2
import json
import io
import pickle
import base64
import requests
import numpy as np
import pandas as pd
from PIL import Image        
import tensorflow as tf
from tensorflow import keras
from lime import lime_image
from keras.models import load_model
from skimage.segmentation import mark_boundaries
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import Flask, render_template, request, redirect, url_for


app = Flask(__name__)


CropIdentificationModel = load_model('models\CropIdentificationModel.h5')

class_name = {0: 'Apple', 1: 'Corn', 2: 'Grape', 3: 'Other', 4: 'Potato', 5: 'Strawberry', 6: 'Tomato', 7: 'Wheat'}

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


print("Model Lodded http://127.0.0.1:5000/")


@app.route('/Crop-rcmd', methods=['GET','POST'])
def Crop_rcmd():
    return render_template('CropRecommendation.html')


@app.route('/CIDD_input', methods=['GET','POST'])
def CIDD_input():
    return render_template('CIDD_input.html')


@app.route('/CIDDL_input', methods=['GET','POST'])
def CIDDL_input():
    return render_template('CIDDL_input.html')


# ---------------------------------  Crop Recommendation  -------------------------------------------------


def weather_fetch(city_name):
    
    api_key = "c127a9db9412175ae1101104ce0ffe05"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    weather_info = response.json()

    if weather_info["cod"] != "404":
        y = weather_info["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None

@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            if final_prediction == "rice":
                img1 = "static/Images/Crop suggestion/Rice/1.jpg"
                img2 = "static/Images/Crop suggestion/Rice/2.jpg"
                img3 = "static/Images/Crop suggestion/Rice/3.jpg"


            elif final_prediction == "maize":
                img1 = "static/Images/Crop suggestion/maize/1.jpg"
                img2 = "static/Images/Crop suggestion/maize/2.jpg"
                img3 = "static/Images/Crop suggestion/maize/3.jpg"
            
            elif final_prediction == "chickpea":
                img1 = "static/Images/Crop suggestion/chickpea/1.jpg"
                img2 = "static/Images/Crop suggestion/chickpea/2.jpg"
                img3 = "static/Images/Crop suggestion/chickpea/3.jpg"
            
            elif final_prediction == "kidneybeans":
                img1 = "static/Images/Crop suggestion/kidneybeans/1.png"
                img2 = "static/Images/Crop suggestion/kidneybeans/2.jpg"
                img3 = "static/Images/Crop suggestion/kidneybeans/3.jpg"
            
            elif final_prediction == "pigeonpeas":
                img1 = "static/Images/Crop suggestion/pigeonpeas/1.JPEG"
                img2 = "static/Images/Crop suggestion/pigeonpeas/2.jpg"
                img3 = "static/Images/Crop suggestion/pigeonpeas/3.jpg"
            
            elif final_prediction == "mothbeans":
                img1 = "static/Images/Crop suggestion/mothbeans/1.jpg"
                img2 = "static/Images/Crop suggestion/mothbeans/2.jpg"
                img3 = "static/Images/Crop suggestion/mothbeans/3.JPEG"
            
            elif final_prediction == "mungbean":
                img1 = "static/Images/Crop suggestion/mungbean/1.jpg"
                img2 = "static/Images/Crop suggestion/mungbean/2.jpg"
                img3 = "static/Images/Crop suggestion/mungbean/3.jpg"
            
            elif final_prediction == "blackgram":
                img1 = "static/Images/Crop suggestion/blackgram/1.jpg"
                img2 = "static/Images/Crop suggestion/blackgram/2.jpg"
                img3 = "static/Images/Crop suggestion/blackgram/3.jpg"
            
            elif final_prediction == "lentil":
                img1 = "static/Images/Crop suggestion/lentil/1.jpg"
                img2 = "static/Images/Crop suggestion/lentil/2.JPEG"
                img3 = "static/Images/Crop suggestion/lentil/3.jpg"
            
            elif final_prediction == "pomegranate":
                img1 = "static/Images/Crop suggestion/pomegranate/1.jpg"
                img2 = "static/Images/Crop suggestion/pomegranate/2.jpg"
                img3 = "static/Images/Crop suggestion/pomegranate/3.jpg"
            
            elif final_prediction == "banana":
                img1 = "static/Images/Crop suggestion/banana/1.jpg"
                img2 = "static/Images/Crop suggestion/banana/2.jpg"
                img3 = "static/Images/Crop suggestion/banana/3.jpg"
            
            elif final_prediction == "mango":
                img1 = "static/Images/Crop suggestion/mango/1.jpg"
                img2 = "static/Images/Crop suggestion/mango/2.jpg"
                img3 = "static/Images/Crop suggestion/mango/3.jpg"
            
            elif final_prediction == "grapes":
                img1 = "static/Images/Crop suggestion/grapes/1.jpg"
                img2 = "static/Images/Crop suggestion/grapes/2.jpg"
                img3 = "static/Images/Crop suggestion/grapes/3.jpg"
            
            elif final_prediction == "watermelon":
                img1 = "static/Images/Crop suggestion/watermelon/1.jpg"
                img2 = "static/Images/Crop suggestion/watermelon/2.jpg"
                img3 = "static/Images/Crop suggestion/watermelon/3.jpg"
            
            elif final_prediction == "muskmelon":
                img1 = "static/Images/Crop suggestion/muskmelon/1.JPEG"
                img2 = "static/Images/Crop suggestion/muskmelon/2.JPEG"
                img3 = "static/Images/Crop suggestion/muskmelon/3.JPEG"
            
            elif final_prediction == "apple":
                img1 = "static/Images/Crop suggestion/apple/1.jpg"
                img2 = "static/Images/Crop suggestion/apple/2.jpg"
                img3 = "static/Images/Crop suggestion/apple/3.jpg"
            
            elif final_prediction == "orange":
                img1 = "static/Images/Crop suggestion/orange/1.jpg"
                img2 = "static/Images/Crop suggestion/orange/2.jpg"
                img3 = "static/Images/Crop suggestion/orange/3.jpg"
            
            elif final_prediction == "papaya":
                img1 = "static/Images/Crop suggestion/papaya/1.jpg"
                img2 = "static/Images/Crop suggestion/papaya/2.jpg"
                img3 = "static/Images/Crop suggestion/papaya/3.jpg"
            
            elif final_prediction == "coconut":
                img1 = "static/Images/Crop suggestion/coconut/1.jpg"
                img2 = "static/Images/Crop suggestion/coconut/2.jpg"
                img3 = "static/Images/Crop suggestion/coconut/3.jpg"
            
            elif final_prediction == "cotton":
                img1 = "static/Images/Crop suggestion/cotton/1.jpg"
                img2 = "static/Images/Crop suggestion/cotton/2.jpg"
                img3 = "static/Images/Crop suggestion/cotton/3.jpeg"
            
            elif final_prediction == "jute":
                img1 = "static/Images/Crop suggestion/jute/1.jpg"
                img2 = "static/Images/Crop suggestion/jute/2.JPEG"
                img3 = "static/Images/Crop suggestion/jute/3.jpg"
            
            elif final_prediction == "coffee":
                img1 = "static/Images/Crop suggestion/coffee/1.jpeg"
                img2 = "static/Images/Crop suggestion/coffee/2.jpg"
                img3 = "static/Images/Crop suggestion/coffee/3.JPEG"
            

            return render_template('CR_result.html', prediction=final_prediction, image_1 = img1, image_2= img2, image_3=img3)

        else:

            return render_template('try_again.html')




# ---------------------------------  Crop Identification and Diseases Detection  -------------------------------------------------

@app.route('/uploadandanalysis', methods=['POST' ,'GET'])
def uploadandanalysis():
    global file
    global file_path
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return crop_identification_model()

    
def crop_identification_model():


    global img
    global crop_name
    global confidence
    global cropidentificatonmodel_output_class
    global diseasedetectionmodel_output_class

    image_path = file_path
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img,(224,224))    
    img = np.expand_dims(img_resized,axis=0)


    resnetcropidentificationmodel_prediction=CropIdentificationModel.predict(img)
    
    cropidentificatonmodel_output_class = class_name[np.argmax(resnetcropidentificationmodel_prediction)]
    print("Crop identification = ",cropidentificatonmodel_output_class)
    crop_name =cropidentificatonmodel_output_class



    if cropidentificatonmodel_output_class == 'Apple':

        return apple()
    
    elif cropidentificatonmodel_output_class == 'Corn':

        return corn()
    
    elif cropidentificatonmodel_output_class == 'Grape':

        return grape()
       
    elif cropidentificatonmodel_output_class == 'Potato':
        
        return potato()
    
    elif cropidentificatonmodel_output_class == 'Strawberry':

        return strawberry()
    
    elif cropidentificatonmodel_output_class == 'Tomato':

        return tomato()
    
    elif cropidentificatonmodel_output_class == 'Wheat':

        return wheat()
    
    elif cropidentificatonmodel_output_class == 'Other':

        return render_template('Other.html')
    


def apple():

    applediseasedetectionmodel = load_model('models\AppleDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust', 3: 'Apple___healthy'}
    print(class_name)

    applediseasedetectionmodel_prediction=applediseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(applediseasedetectionmodel_prediction)]

    apple_diseases = pd.read_excel("Datasets\Apple\Apple Diseases.xlsx")

    if diseasedetectionmodel_output_class == 'Apple___Apple_scab':

        diseasedetectionmodel_output_class = "Apple Scab"
        disease_result = diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Apple\Apple___Apple_scab\img3.jpg"
        img2 = "static\Images\Siteimages\Apple\Apple___Apple_scab\Img1.jpg"
        img3 = "static\Images\Siteimages\Apple\Apple___Apple_scab\img2.jpg"

        dt= apple_diseases.iloc [0,2]
        Symptoms= apple_diseases.iloc [0,3]
        Causes = apple_diseases.iloc [0,4]
        Prevention_Methods = apple_diseases.iloc [0,7]
        oc = apple_diseases.iloc [0,5]
        cc = apple_diseases.iloc [0,6]
    
    elif diseasedetectionmodel_output_class == 'Apple___Black_rot':

        diseasedetectionmodel_output_class = "Black Rot"
        disease_result = diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Apple\Apple___Black_rot\img1.jpg"
        img2 = "static\Images\Siteimages\Apple\Apple___Black_rot\img2.jpg"
        img3 = "static\Images\Siteimages\Apple\Apple___Black_rot\img3.jpg"

        dt= apple_diseases.iloc [1,2]
        Symptoms= apple_diseases.iloc [1,3]
        Causes = apple_diseases.iloc [1,4]
        Prevention_Methods = apple_diseases.iloc [1,7]
        oc = apple_diseases.iloc [1,5]
        cc = apple_diseases.iloc [1,6]
    
    
    elif diseasedetectionmodel_output_class == 'Apple___Cedar_apple_rust':

        diseasedetectionmodel_output_class = "Cendar Apple Rust"
        disease_result = diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Apple\Apple___Cedar_apple_rust\img1.jpg"
        img2 = "static\Images\Siteimages\Apple\Apple___Cedar_apple_rust\img2.jpg"
        img3 = "static\Images\Siteimages\Apple\Apple___Cedar_apple_rust\img3.jpg"

        dt= apple_diseases.iloc [2,2]
        Symptoms= apple_diseases.iloc [2,3]
        Causes = apple_diseases.iloc [2,4]
        Prevention_Methods = apple_diseases.iloc [2,7]
        oc = apple_diseases.iloc [2,5]
        cc = apple_diseases.iloc [2,6]


    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def corn():


    corndiseasedetectionmodel = load_model('models\CornDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 1: 'Corn_(maize)___Common_rust_', 2: 'Corn_(maize)___healthy', 3: 'Corn_(maize)___Northern_Leaf_Blight'}
    print(class_name)

    corndiseasedetectionmodel_prediction=corndiseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(corndiseasedetectionmodel_prediction)]

    
    corn_diseases = pd.read_excel("Datasets\Corn\Corn Diseases.xlsx")
    
    if diseasedetectionmodel_output_class == 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot':

        diseasedetectionmodel_output_class = "Gray Leaf Spot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Corn\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot\img1.jpg"
        img2 = "static\Images\Siteimages\Corn\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot\img2.jpg"
        img3 = "static\Images\Siteimages\Corn\Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot\img3.jpg"

        Symptoms= corn_diseases.iloc [0,3]
        Causes = corn_diseases.iloc [0,4]
        Prevention_Methods = corn_diseases.iloc [0,7]
        oc = corn_diseases.iloc [0,5]
        cc = corn_diseases.iloc [0,6]
        dt= corn_diseases.iloc [0,2]
    
    elif diseasedetectionmodel_output_class == 'Corn_(maize)___Common_rust_':

        diseasedetectionmodel_output_class = "Common Rust"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Corn\Corn_(maize)___Common_rust_\img1.png"
        img2 = "static\Images\Siteimages\Corn\Corn_(maize)___Common_rust_\img2.jpg"
        img3 = "static\Images\Siteimages\Corn\Corn_(maize)___Common_rust_\img3.jpg"

        Symptoms= corn_diseases.iloc [1,3]
        Causes = corn_diseases.iloc [1,4]
        Prevention_Methods = corn_diseases.iloc [1,7]
        oc = corn_diseases.iloc [1,5]
        cc = corn_diseases.iloc [1,6]
        dt= corn_diseases.iloc [1,2]
    
    
    elif diseasedetectionmodel_output_class == 'Corn_(maize)___Northern_Leaf_Blight':

        diseasedetectionmodel_output_class = "Northern Leaf Blight"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Corn\Corn_(maize)___Northern_Leaf_Blight\img1.jpg"
        img2 = "static\Images\Siteimages\Corn\Corn_(maize)___Northern_Leaf_Blight\img2.jpg"
        img3 = "static\Images\Siteimages\Corn\Corn_(maize)___Northern_Leaf_Blight\img3.jpeg"

        Symptoms= corn_diseases.iloc [2,3]
        Causes = corn_diseases.iloc [2,4]
        oc = corn_diseases.iloc [2,5]
        cc = corn_diseases.iloc [2,6]
        Prevention_Methods = corn_diseases.iloc [2,7]
        dt= corn_diseases.iloc [2,2]


    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def grape():


    grapediseasedetectionmodel = load_model('models\GrapeDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Grape___Black_rot', 1: 'Grape___Esca_(Black_Measles)', 2: 'Grape___healthy', 3: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)'}
    print(class_name)

    grapediseasedetectionmodel_prediction=grapediseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(grapediseasedetectionmodel_prediction)]

    grape_diseases = pd.read_excel("Datasets\Grape\Grape Disease.xlsx")


    if diseasedetectionmodel_output_class == 'Grape___Black_rot':

        diseasedetectionmodel_output_class = "Black Rot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Grape\Grape___Black_rot\img1.jpg"
        img2 = "static\Images\Siteimages\Grape\Grape___Black_rot\img2.jpg"
        img3 = "static\Images\Siteimages\Grape\Grape___Black_rot\img3.jpg"

        Symptoms= grape_diseases.iloc [0,3]
        Causes = grape_diseases.iloc [0,4]
        oc = grape_diseases.iloc [0,5]
        cc = grape_diseases.iloc [0,6]
        Prevention_Methods = grape_diseases.iloc [0,7]
        dt= grape_diseases.iloc [0,2]


    
    elif diseasedetectionmodel_output_class == 'Grape___Esca_(Black_Measles)':

        diseasedetectionmodel_output_class = "Black Measles"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img1.jpg"
        img2 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img2.jpg"
        img3 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img3.jpg"

        Symptoms= grape_diseases.iloc [1,3]
        Causes = grape_diseases.iloc [1,4]
        oc = grape_diseases.iloc [1,5]
        cc = grape_diseases.iloc [1,6]
        Prevention_Methods = grape_diseases.iloc [1,7]
        dt= grape_diseases.iloc [1,2]

    
    
    elif diseasedetectionmodel_output_class == 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)':

        diseasedetectionmodel_output_class = "Isariopsis Leaf Spot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img1.jpg"
        img2 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img2.jpg"
        img3 = "static\Images\Siteimages\Grape\Grape___Esca_(Black_Measles)\img3.jpg"

        Symptoms= grape_diseases.iloc [2,3]
        Causes = grape_diseases.iloc [2,4]
        oc = grape_diseases.iloc [2,5]
        cc = grape_diseases.iloc [2,6]
        Prevention_Methods = grape_diseases.iloc [2,7]
        dt= grape_diseases.iloc [2,2]

    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def potato():

    potatodiseasedetectionmodel = load_model('models\PotatoDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Potato___Early_blight', 1: 'Potato___healthy', 2: 'Potato___Late_blight'}
    print(class_name)

    potatodiseasedetectionmodel_prediction=potatodiseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(potatodiseasedetectionmodel_prediction)]

    potato_diseases = pd.read_excel("Datasets\Potato\Potato Disease.xlsx")
    
    if diseasedetectionmodel_output_class == 'Potato___Early_blight':

        diseasedetectionmodel_output_class = "Early Bligh"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Potato\Potato___Early_blight\img1.jpg"
        img2 = "static\Images\Siteimages\Potato\Potato___Early_blight\img2.jpg"
        img3 = "static\Images\Siteimages\Potato\Potato___Early_blight\img3.jpg"

        Symptoms= potato_diseases.iloc [0,3]
        Causes = potato_diseases.iloc [0,4]
        oc = potato_diseases.iloc [0,5]
        cc = potato_diseases.iloc [0,6]
        Prevention_Methods = potato_diseases.iloc [0,7]
        dt= potato_diseases.iloc [0,2]
    
    elif diseasedetectionmodel_output_class == 'Potato___Late_blight':

        diseasedetectionmodel_output_class = "Late Blight"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Potato\Potato___Late_blight\img1.jpg"
        img2 = "static\Images\Siteimages\Potato\Potato___Late_blight\img2.jpg"
        img3 = "static\Images\Siteimages\Potato\Potato___Late_blight\img3.jpg"

        Symptoms= potato_diseases.iloc [1,3]
        Causes = potato_diseases.iloc [1,4]
        oc = potato_diseases.iloc [1,5]
        cc = potato_diseases.iloc [1,6]
        Prevention_Methods = potato_diseases.iloc [1,7]
        dt= potato_diseases.iloc [1,2]
        


    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def strawberry():

    
    strawberyydiseasedetectionmodel = load_model('models\StrawberryDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Strawberry___healthy', 1: 'Strawberry___Leaf_scorch'}
    print(class_name)

    strawberyydiseasedetectionmodel_prediction=strawberyydiseasedetectionmodel.predict(img)  
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(strawberyydiseasedetectionmodel_prediction)]

    strawberry_diseases = pd.read_excel("Datasets\Strawberry\Strawberry Disease.xlsx")
    
    if diseasedetectionmodel_output_class == 'Strawberry___Leaf_scorch':

        diseasedetectionmodel_output_class = "Leaf Scorch"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Strawberry\Strawberry___Leaf_scorch\img1.jpg"
        img2 = "static\Images\Siteimages\Strawberry\Strawberry___Leaf_scorch\img2.jpg"
        img3 = "static\Images\Siteimages\Strawberry\Strawberry___Leaf_scorch\img3.jpg"

        Symptoms= strawberry_diseases.iloc [0,3]
        Causes = strawberry_diseases.iloc [0,4]
        oc = strawberry_diseases.iloc [0,5]
        cc = strawberry_diseases.iloc [0,6]
        Prevention_Methods = strawberry_diseases.iloc [0,7]
        dt= strawberry_diseases.iloc [0,2]
    
    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def tomato():

    
    tomatodiseasedetectionmodel = load_model('models\TomatoDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Tomato___Bacterial_spot', 1: 'Tomato___Early_blight', 2:'Tomato___healthy', 3:'Tomato___Late_blight', 4:'Tomato___Leaf_Mold', 5:'Tomato___Septoria_leaf_spot', 
                          6:'Tomato___Spider_mites Two-spotted_spider_mite', 7:'Tomato___Target_Spot', 8:'Tomato___Tomato_mosaic_virus', 9:'Tomato___Tomato_Yellow_Leaf_Curl_Virus' }
    print(class_name)

    tomatodiseasedetectionmodel_prediction=tomatodiseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(tomatodiseasedetectionmodel_prediction)]

    tomato_diseases = pd.read_excel("D:/Final Year Project/Datasets/Tomato/tomato Disease.xlsx")
    
    if diseasedetectionmodel_output_class == 'Tomato___Bacterial_spot':

        diseasedetectionmodel_output_class = "Bacterial Spot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Bacterial_spot\img1.jpeg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Bacterial_spot\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Bacterial_spot\img3.jpg"

        Symptoms= tomato_diseases.iloc [0,3]
        Causes = tomato_diseases.iloc [0,4]
        oc = tomato_diseases.iloc [0,5]
        cc = tomato_diseases.iloc [0,6]
        Prevention_Methods = tomato_diseases.iloc [0,7]
        dt= tomato_diseases.iloc [0,2]
    
    elif diseasedetectionmodel_output_class == 'Tomato___Early_blight':

        diseasedetectionmodel_output_class = "Early Blight"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Early_blight\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Early_blight\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Early_blight\img3.jpg"

        Symptoms= tomato_diseases.iloc [1,3]
        Causes = tomato_diseases.iloc [1,4]
        oc = tomato_diseases.iloc [1,5]
        cc = tomato_diseases.iloc [1,6]
        Prevention_Methods = tomato_diseases.iloc [1,7]
        dt= tomato_diseases.iloc [1,2]
    
    
    elif diseasedetectionmodel_output_class == 'Tomato___Late_blight':

        diseasedetectionmodel_output_class = "Late Blight"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Late_blight\img1.png"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Late_blight\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Late_blight\img3.png"

        Symptoms= tomato_diseases.iloc [2,3]
        Causes = tomato_diseases.iloc [2,4]
        oc = tomato_diseases.iloc [2,5]
        cc = tomato_diseases.iloc [2,6]
        Prevention_Methods = tomato_diseases.iloc [2,7]
        dt= tomato_diseases.iloc [2,2]

    elif diseasedetectionmodel_output_class == 'Tomato___Leaf_Mold':

        diseasedetectionmodel_output_class = "Leaf Mold"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Leaf_Mold\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Leaf_Mold\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Leaf_Mold\img1.jpg"

        Symptoms= tomato_diseases.iloc [3,3]
        Causes = tomato_diseases.iloc [3,4]
        oc = tomato_diseases.iloc [3,5]
        cc = tomato_diseases.iloc [3,6]
        Prevention_Methods = tomato_diseases.iloc [3,7]
        dt= tomato_diseases.iloc [3,2]

    elif diseasedetectionmodel_output_class == 'Tomato___Septoria_leaf_spot':

        diseasedetectionmodel_output_class = "Septoria Leaf Spot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Septoria_leaf_spot\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Septoria_leaf_spot\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Septoria_leaf_spot\img1.jpg"   

        Symptoms= tomato_diseases.iloc [4,3]
        Causes = tomato_diseases.iloc [4,4]
        oc = tomato_diseases.iloc [4,5]
        cc = tomato_diseases.iloc [4,6]
        Prevention_Methods = tomato_diseases.iloc [4,7]
        dt= tomato_diseases.iloc [4,2]


    elif diseasedetectionmodel_output_class == 'Tomato___Spider_mites Two-spotted_spider_mite':

        diseasedetectionmodel_output_class = "Spider Mites"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Spider_mites Two-spotted_spider_mite\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Spider_mites Two-spotted_spider_mite\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Spider_mites Two-spotted_spider_mite\img1.jpg"

        Symptoms= tomato_diseases.iloc [5,3]
        Causes = tomato_diseases.iloc [5,4]
        oc = tomato_diseases.iloc [5,5]
        cc = tomato_diseases.iloc [5,6]
        Prevention_Methods = tomato_diseases.iloc [5,7]
        dt= tomato_diseases.iloc [5,2]


    elif diseasedetectionmodel_output_class == 'Tomato___Target_Spot':

        diseasedetectionmodel_output_class = "Target Spot"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Target_Spot\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Target_Spot\img2.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Target_Spot\img1.jpg"

        Symptoms= tomato_diseases.iloc [6,3]
        Causes = tomato_diseases.iloc [6,4]
        oc = tomato_diseases.iloc [6,5]
        cc = tomato_diseases.iloc [6,6]
        Prevention_Methods = tomato_diseases.iloc [6,7]
        dt= tomato_diseases.iloc [6,2]


    elif diseasedetectionmodel_output_class == 'Tomato___Tomato_mosaic_virus':

        diseasedetectionmodel_output_class = "Mosaic Virus"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Tomato_mosaic_virus\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Tomato_mosaic_virus\img2.JPEG"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Tomato_mosaic_virus\img3.jpg"

        Symptoms= tomato_diseases.iloc [7,3]
        Causes = tomato_diseases.iloc [7,4]
        oc = tomato_diseases.iloc [7,5]
        cc = tomato_diseases.iloc [7,6]
        Prevention_Methods = tomato_diseases.iloc [7,7]
        dt= tomato_diseases.iloc [7,2]


    elif diseasedetectionmodel_output_class == 'Tomato___Tomato_Yellow_Leaf_Curl_Virus':

        diseasedetectionmodel_output_class = "Yellow Leaf Curl Virus"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Tomato\Tomato___Tomato_Yellow_Leaf_Curl_Virus\img1.jpg"
        img2 = "static\Images\Siteimages\Tomato\Tomato___Tomato_Yellow_Leaf_Curl_Virus\img3.jpg"
        img3 = "static\Images\Siteimages\Tomato\Tomato___Tomato_Yellow_Leaf_Curl_Virus\img3.jpg"

        Symptoms= tomato_diseases.iloc [8,3]
        Causes = tomato_diseases.iloc [8,4]
        oc = tomato_diseases.iloc [8,5]
        cc = tomato_diseases.iloc [8,6]
        Prevention_Methods = tomato_diseases.iloc [8,7]
        dt= tomato_diseases.iloc [8,2]

    return render_template('CIDD_result.html', prediction=crop_name, disease= diseasedetectionmodel_output_class ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)

def wheat():

    wheatdiseasedetectionmodel = load_model('models\WheatDiseaseDetectionModel.h5')

    disease_class_name = {0: 'Wheat___Brown_Rust', 1: 'Wheat___Healthy', 2:'Wheat___Yellow_Rust'}

    print(class_name)

    wheatdiseasedetectionmodel_prediction=wheatdiseasedetectionmodel.predict(img)
    diseasedetectionmodel_output_class = disease_class_name[np.argmax(wheatdiseasedetectionmodel_prediction)]

    wheat_diseases = pd.read_excel("Datasets\Wheat\Wheat Disease.xlsx")
    
    if diseasedetectionmodel_output_class == 'Wheat___Brown_Rust':

        diseasedetectionmodel_output_class = "Brown Rust"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Wheat\Wheat___Brown_Rust\img1.jpg"
        img2 = "static\Images\Siteimages\Wheat\Wheat___Brown_Rust\img1.jpg"
        img3 = "static\Images\Siteimages\Wheat\Wheat___Brown_Rust\img1.jpg"

        Symptoms= wheat_diseases.iloc [0,3]
        Causes = wheat_diseases.iloc [0,4]
        oc = wheat_diseases.iloc [0,5]
        cc = wheat_diseases.iloc [0,6]
        Prevention_Methods = wheat_diseases.iloc [0,7]
        dt= wheat_diseases.iloc [0,2]
        
    
    elif diseasedetectionmodel_output_class == 'Wheat___Yellow_Rust':

        diseasedetectionmodel_output_class = "Yellow Rust"
        disease_result =  diseasedetectionmodel_output_class

        img1 = "static\Images\Siteimages\Wheat\Wheat___Yellow_Rust\img1.png"
        img2 = "static\Images\Siteimages\Wheat\Wheat___Yellow_Rust\img2.jpg"
        img3 = "static\Images\Siteimages\Wheat\Wheat___Yellow_Rust\img3.jpg"
    
        Symptoms= wheat_diseases.iloc [1,3]
        Causes = wheat_diseases.iloc [1,4]
        oc = wheat_diseases.iloc [1,5]
        cc = wheat_diseases.iloc [1,6]
        Prevention_Methods = wheat_diseases.iloc [1,7]
        dt= wheat_diseases.iloc [1,2]
        

    return render_template('CIDD_result.html', prediction=crop_name, disease= disease_result ,image_file=file.filename, image_1 = img1, image_2= img2, image_3=img3, dt=dt, s=Symptoms, c=Causes, p_m= Prevention_Methods, ocinfo=oc, ccinfo=cc)



@app.route('/uploadandanalysiswithlime', methods=['POST' ,'GET'])
def uploadandanalysiswithlime():
    global file
    global file_path
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return explain_prediction_with_lime(file_path)


def explain_prediction_with_lime(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)[0]

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array,
        classifier_fn=lambda images: CropIdentificationModel.predict(preprocess_input(images)),
        top_labels=1,
        hide_color=0,
        num_samples=500
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )

    Lime_img = mark_boundaries(temp, mask)

    pil_img = Image.fromarray((Lime_img * 255).astype(np.uint8))
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG', quality=100)
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
    img_data = f'data:image/png;base64,{img_base64}'

    return render_template('CIDDL_result.html', image_1=img_data)




@app.route('/diagnosis', methods=['POST' ,'GET'])
def diagnosis():

    global img
    global crop_name
    global confidence
    global cropidentificatonmodel_output_class
    global diseasedetectionmodel_output_class

    image_path = file_path
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img,(224,224))    
    img = np.expand_dims(img_resized,axis=0)


    resnetcropidentificationmodel_prediction=CropIdentificationModel.predict(img)
    
    cropidentificatonmodel_output_class = class_name[np.argmax(resnetcropidentificationmodel_prediction)]
    print("Crop identification = ",cropidentificatonmodel_output_class)
    crop_name =cropidentificatonmodel_output_class



    if cropidentificatonmodel_output_class == 'Apple':

        return apple()
    
    elif cropidentificatonmodel_output_class == 'Corn':

        return corn()
    
    elif cropidentificatonmodel_output_class == 'Grape':

        return grape()
       
    elif cropidentificatonmodel_output_class == 'Potato':
        
        return potato()
    
    elif cropidentificatonmodel_output_class == 'Strawberry':

        return strawberry()
    
    elif cropidentificatonmodel_output_class == 'Tomato':

        return tomato()
    
    elif cropidentificatonmodel_output_class == 'Wheat':

        return wheat()
    
    elif cropidentificatonmodel_output_class == 'Other':

        return render_template('Other.html')
    


# ---------------------------------  Feedbacks  -------------------------------------------------

FEEDBACK_FILE = 'feedbacks.csv'

def load_feedbacks():
    try:
        df = pd.read_csv(FEEDBACK_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Name', 'Feedback'])  
    return df

def save_feedback(name, feedback):
    df = load_feedbacks()
    new_feedback = pd.DataFrame({'Name': [name], 'Feedback': [feedback]})
    df = pd.concat([df, new_feedback], ignore_index=True)  
    df.to_csv(FEEDBACK_FILE, index=False)  

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        name = request.form['name']
        feedback = request.form['feedback']
        save_feedback(name, feedback)
        return redirect(url_for('index'))  

    feedbacks = load_feedbacks().tail(6).to_dict('records') 
    return render_template('index.html', feedbacks=feedbacks)



if __name__ == '__main__':
    app.run(debug=True)