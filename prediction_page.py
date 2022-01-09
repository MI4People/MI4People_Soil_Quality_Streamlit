# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:10:31 2021

@author: paul springer
"""

import streamlit as st
import tensorflow.keras as K
import pandas as pd
import numpy as np
import requests
from sentinelhub import SHConfig
import math

from sentinelhub import MimeType, CRS, BBox, SentinelHubRequest, \
    DataCollection, bbox_to_dimensions

favicon = 'Logo.JPG'

model0 = K.models.load_model(
    'model_as_in_paper_with_sentinel_regression.h5')
model1 = K.models.load_model(
    'model_without_images_regression_seed_3.h5')
model2 = K.models.load_model(
    'model_without_images_regression_seed_5.h5')
model3 = K.models.load_model(
    'model_without_images_regression_seed_7.h5')
model4 = K.models.load_model(
    'model_without_images_regression_seed_10.h5')
model5 = K.models.load_model(
    'model_without_images_regression_seed_12.h5')
model6 = K.models.load_model(
    'model_without_images_regression_seed_13.h5')
model7 = K.models.load_model(
    'model_without_images_regression_seed_24.h5')
model8 = K.models.load_model(
    'model_without_images_regression_seed_123.h5')
model9 = K.models.load_model(
    'model_without_images_regression_seed_666.h5')
model10 = K.models.load_model(
    'model_without_images_regression_seed_1993.h5')

layers = pd.read_csv('Global_covariate_layers.csv')
prefixes = layers["prefix"].unique()

max_values = pd.read_csv("max_values.csv")
min_values = pd.read_csv("min_values.csv")

def show_predict_page():
    
    st.set_page_config(page_title='Soil Quality', page_icon = favicon,
                            layout = 'wide', initial_sidebar_state = 'auto')
    
    st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 40rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 40rem;}}
    </style>
    ''',unsafe_allow_html=True)
    
    st.sidebar.title("What is this app about?")
    st.sidebar.write("This small Web App is an demonstrative MVP for the future\
                     MI4People’s Soil Quality Prediction System.")
    st.sidebar.write("The Soil Quality Prediction System will be able to take\
                     GPS coordinates from a mobile device and return most\
                     important soil quality factors that are predicted by means\
                     of satellite data and AI. This system will be open-source\
                     and should be applied to enable small farmers in developing\
                     countries to better understand their soil, make\
                     intelligent choice about which crops to plant, and to\
                     fertilize and protect soil from pests in a more suitable,\
                     sustainable, and environmental-friendly manner. It will\
                     lead to better crop yields with eco-friendly farming and,\
                     as consequences, to more stable food supply chains,\
                     less famine and undernutrition. The system will be\
                     free of charge.")
    st.sidebar.write("The presented MVP mimics this system. Currently it\
                     predicts only one soil quality factor: organic carbon.\
                     You should specify the coordinates (as decimal degrees)\
                     as it would be an input from a mobile device. The current\
                     MVP is focused on Africa, so, please use only coordinates\
                     within African continent. If you do not do so, you’ll get\
                     an error message. Please, also use only meaningful\
                     locations for prediction of organic carbon in the soil,\
                     i.e., no oceans, concreted areas in cities or similar.\
                     Otherwise the prediciton will be meaningless.")
    st.sidebar.write("You also should specify depth at which prediction should\
                     be made (in cm)")
    st.sidebar.write("If you have quastions or observe a bug or an error message\
                     please contact us via:")
    st.sidebar.markdown('<a href="mailto:info@mi4people.org">info@mi4people.org</a>',
                        unsafe_allow_html=True)
    

    col1, mid, col2 = st.columns([20,1,5])
    with col1:
        st.title("Organic Carbon Prediction")
    with col2:
        st.image('Logo.JPG', width=100)

    
    st.write("Please specify the location and the soil depth you want to\
             have prediction for. Please read the description to the left\
             before you start.")
    
    lon = st.number_input('Longitude', min_value = -17.520278,
                          max_value = 51.281214,
                          value = 36.435938,
                          step = 1.0,
                          format = "%.6f")
    lat = st.number_input('Latitude', min_value = -34.833333,
                          max_value = 37.347222,
                          value = -6.088688,
                          step = 1.0,
                          format = "%.6f")
    depth = st.slider("Depth at which prediction should be made (in cm)",
                      0, 50, 10, 1)
    
    resolution = 20 # resolution fro SentinelHub data

    
    ok = st.button("Predict")
    if ok:
        input_user = pd.DataFrame([lon, lat, depth])
        
        #get data from OpenLandMap. Pay attention to otder of variables.
        #It is important for input for neural network model
        out_OpenLandMap = {} 
        for i in range(0,len(prefixes)):
            request = "http://api.openlandmap.org/query/point?lat=" + str(lat) + "&lon=" + str(lon) + "&coll=" + prefixes[i] + "&regex=(" + '|'.join(layers[layers["prefix"] == prefixes[i]]["global_covariate_layer"]) + ")" 
            
            #first call does not work sometimes. If so, do further calls
            #But break after 5 calls
            response = requests.get(request)
            i = 1
            while response.status_code != 200:
                if i > 5:
                    st.subheader("OpenLandMap API for gathering satellite data seems to be down. Pls contact info@mi4people.ord")
                    break
                response = requests.get(request)
                i = i + 1

            response = response.json()
            response_without_coord = response['response'][0]
            response_without_coord.pop("lon")
            response_without_coord.pop("lat")
            out_OpenLandMap.update(response_without_coord)
        
        out_OpenLandMap = pd.DataFrame([out_OpenLandMap]).T
        out_OpenLandMap = out_OpenLandMap.reindex(
            index=layers['global_covariate_layer'])
        
        input_user_and_out_OpenLandMap = input_user.append(
            out_OpenLandMap, ignore_index=True)
        
        #get data from SentinelHub
        config = SHConfig()

        #account mi4people (already not working)
        #config.instance_id = '0c5d4483-7a81-48da-9be5-b74d3ca7c996'
        #config.sh_client_id = '69692447-b415-461d-a73d-9ad4841f013e'
        #config.sh_client_secret = '5f?[xZBfbcB~Vy;JIboqWq9*rE}TnT4Pju5,eLIF'
        #config.save()

        #acount gmx (Paul's private account)
        config.instance_id = '4df1188c-af0b-4f5d-9803-597915205fb8'
        config.sh_client_id = 'fe7ca65a-8de6-406a-8407-6a747478d635'
        config.sh_client_secret = 'xQkVu:za1Y156N/E6E3^j-KBE5E)sZ,Zzn6obAEs'
        config.save()
        
        evalscript_all_bands = """
            //VERSION=3

            function setup() {
                return {
                    input: [{
                        bands: ["B01","B02","B03","B04","B05","B06","B07",
                                "B08", "B8A","B09","B10","B11","B12"],
                        units: "DN"
                    }],
                    output: {
                        bands: 13,
                        sampleType: "INT16"
                    }
                };
            }
            
                function evaluatePixel(sample) {
                    return [sample.B01,
                            sample.B02,
                            sample.B03,
                            sample.B04,
                            sample.B05,
                            sample.B06,
                            sample.B07,
                            sample.B08,
                            sample.B8A,
                            sample.B09,
                            sample.B10,
                            sample.B11,
                            sample.B12];
                }
        """
        
        out_SentinelHub = []

        coord = [lon - 0.0001, lat - 0.0001, lon + 0.0001, lat + 0.0001]
    
        bbox = BBox(bbox=coord, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)
    
        request_all_bands = SentinelHubRequest(
            evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=('2021-10-16', '2021-12-16'), # we take a picture with less possible clouds in the last two months
                    mosaicking_order='leastCC'
                    )
                ],
            responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
            bbox=bbox,
            size=size,
            config=config
            )
    
        all_bands_response = request_all_bands.get_data()
    
    
        pixels_raw = all_bands_response[0] # get the pixalse
    
        # just to be sure that we indeed get only one pixel. Otherwise take the one from the center
        xStart = math.floor((pixels_raw.shape[0] - 1) / 2)
        yStart = math.floor((pixels_raw.shape[1] - 1) / 2)
        xEnd = xStart + 1
        yEnd = yStart + 1
        pixels = pixels_raw[xStart:xEnd,yStart:yEnd,0:13]
        
        out_SentinelHub.append(pixels[0][0].tolist())
        out_SentinelHub = pd.DataFrame(out_SentinelHub,
                                       columns=["B01","B02","B03","B04","B05",
                                                "B06","B07","B08","B8A","B09",
                                                "B10","B11","B12"])
    
        final_response = input_user_and_out_OpenLandMap.append(out_SentinelHub.T,
                                                           ignore_index=True)
    
        norm_input = (final_response[0] - 
                      min_values["0"])/(max_values["0"] - min_values["0"]) 
        norm_input = pd.DataFrame(norm_input).T 
        norm_input = norm_input.to_numpy()
        
        prediction0 = pd.DataFrame(model0.predict(norm_input))
        prediction0.clip(lower=0, inplace=True)
        prediction0 = np.exp(prediction0)-1
        
        prediction1 = pd.DataFrame(model1.predict(norm_input))
        prediction1.clip(lower=0, inplace=True)
        prediction1 = np.exp(prediction1)-1
        
        prediction2 = pd.DataFrame(model2.predict(norm_input))
        prediction2.clip(lower=0, inplace=True)
        prediction2 = np.exp(prediction2)-1
        
        prediction3 = pd.DataFrame(model3.predict(norm_input))
        prediction3.clip(lower=0, inplace=True)
        prediction3 = np.exp(prediction3)-1
        
        prediction4 = pd.DataFrame(model4.predict(norm_input))
        prediction4.clip(lower=0, inplace=True)
        prediction4 = np.exp(prediction4)-1
        
        prediction5 = pd.DataFrame(model5.predict(norm_input))
        prediction5.clip(lower=0, inplace=True)
        prediction5 = np.exp(prediction5)-1
        
        prediction6 = pd.DataFrame(model6.predict(norm_input))
        prediction6.clip(lower=0, inplace=True)
        prediction6 = np.exp(prediction6)-1
        
        prediction7 = pd.DataFrame(model7.predict(norm_input))
        prediction7.clip(lower=0, inplace=True)
        prediction7 = np.exp(prediction7)-1
        
        prediction8 = pd.DataFrame(model8.predict(norm_input))
        prediction8.clip(lower=0, inplace=True)
        prediction8 = np.exp(prediction8)-1
        
        prediction9 = pd.DataFrame(model9.predict(norm_input))
        prediction9.clip(lower=0, inplace=True)
        prediction9 = np.exp(prediction9)-1
        
        prediction10 = pd.DataFrame(model10.predict(norm_input))
        prediction10.clip(lower=0, inplace=True)
        prediction10 = np.exp(prediction10)-1
        
        
        if prediction0[0][0] > 120:
            prediction0[0][0] = 120
            
        if prediction1[0][0] > 120:
            prediction1[0][0] = 120
            
        if prediction2[0][0] > 120:
            prediction2[0][0] = 120
            
        if prediction3[0][0] > 120:
            prediction3[0][0] = 120
            
        if prediction4[0][0] > 120:
            prediction4[0][0] = 120
            
        if prediction5[0][0] > 120:
            prediction5[0][0] = 120
        
        if prediction6[0][0] > 120:
            prediction6[0][0] = 120
            
        if prediction7[0][0] > 120:
            prediction7[0][0] = 120
            
        if prediction8[0][0] > 120:
            prediction8[0][0] = 120
            
        if prediction9[0][0] > 120:
            prediction9[0][0] = 120
            
        if prediction10[0][0] > 120:
            prediction10[0][0] = 120
            
        prediction_tab = pd.DataFrame({"results" : [prediction0[0][0], 
                                                    prediction1[0][0],
                                                    prediction2[0][0],
                                                    prediction3[0][0],
                                                    prediction4[0][0],
                                                    prediction5[0][0],
                                                    prediction6[0][0],
                                                    prediction8[0][0],
                                                    prediction10[0][0]]})
        prediction = prediction_tab.median() 
        #prediction_max = prediction_tab.max()
        #prediction_min = prediction_tab.min()
        
        #st.subheader(
        #    f"The estimated value of organic carbon is  between \
        #        {prediction_min[0]:.2f}"+" and "+f"{prediction_max[0]:.2f}" +
        #    " g/kg")
            
        st.subheader(
            f"The estimated value of organic carbon is {prediction[0]:.2f} g/kg")
        
        for_plot = pd.DataFrame({"lat" : [lat], "lon" : [lon]})
        st.map(for_plot, zoom = 2)