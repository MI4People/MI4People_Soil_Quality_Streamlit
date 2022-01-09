# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 11:10:31 2021

@author: paul springer

this script determines how the web page of Soil Quality Evaluation MVP
Web App works. It contains both front-end element and calculations. The Web App
is built with streamlit.

Rough Structure:

- User's Input
- Call to OpenLandMap API for input data
- Call to Sentinel Hub for input data
- Model Prediction
- Output
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

# Phot uses as Favicon in the browser
favicon = 'Logo.JPG'

# Load a neural networks that predicts organic carbon. For how this network
# was trained, see "As in Paper v1.1.ipynb".
model = K.models.load_model(
    'model_as_in_paper_with_sentinel_regression.h5')

# Below are names/prefixes for global layers from OpenLandMap.org that are used
# as inputs for our models. We need these names to make a call to OpenLandMap's
# Rest API
layers = pd.read_csv('Global_covariate_layers.csv')

# a call to OpenLandMap's API includes a prefix and a name of the required layer.
# One and the same prefix contains several layers. To speed up calculations,
# we do one call per prefix (and not per layer name). I.e., each call get values
# for several layers simultaneously. Here we just define unique prefixes. There
# are 3 unique prefixes.
prefixes = layers["prefix"].unique()

# Input values must be normalized as during the training of the model. We
# use min-max-normalization. The velues below were obtained during the training
# process on training data set. 
max_values = pd.read_csv("max_values.csv")
min_values = pd.read_csv("min_values.csv")


# The function below defines the actual web page. it is called in app.py.
def show_predict_page():
    
    # these lines are to define favicon and name of the web page in the tab of
    # the browser
    st.set_page_config(page_title='Soil Quality', page_icon = favicon,
                            layout = 'wide', initial_sidebar_state = 'auto')
    
    # these lines are to define the width of the sidebar "What is this app about?"
    st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 40rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 40rem;}}
    </style>
    ''',unsafe_allow_html=True)
    
    # Define side bar "What is this app about?". It is used only as information
    # for the user and can be closed by pressing x.
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
    
    
    # Here you see two columns that must be defined to place page title and
    # MI4People logo side by side
    col1, mid, col2 = st.columns([20,1,5])
    with col1:
        st.title("Organic Carbon Prediction")
    with col2:
        st.image('Logo.JPG', width=100) 

    
    # some instructions for the user
    st.write("Please specify the location and the soil depth you want to\
             have prediction for. Please read the description to the left\
             before you start.")
    
    # Here is the user's input for longitude, latitude, and depth. This is
    # required to say the model were to make predictions. "min_value" and "max_value"
    # correspond to boundaries of African continent. "value" is the default number.
    # Note: streamlit's function for numeric user's input is a bit strange.
    # It allows not only to type a number but also to define the number by clicking
    # "+" and "-" buttons. It looks like one cannot switch it off. The "step"
    # defines how much "+" and "-" buttons change the input.
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
    # depth input is defined as a slider.
    depth = st.slider("Depth at which prediction should be made (in cm)",
                      0, 50, 10, 1)
    
    # Defines a "Predict" button that is also store as a binary variable.
    # Clicking on this button changes the value to True
    ok = st.button("Predict")
    if ok:
        # Store user's input for location as a data frame
        input_user = pd.DataFrame([lon, lat, depth])
        
        # Get data from OpenLandMap. More on this data, you can find in 
        # "Soil_Data_Preparation_v0.2.R".
        # Pay attention to the otder of the variables.
        # Order is important for input for neural network model
        # As mentioned above, we make a call per prefix
        out_OpenLandMap = {} 
        for i in range(0,len(prefixes)):
            # define call url. it contains location, and prefix and name of the
            # required global layer. To make call for several layers simultaneously,
            # we need to separate layer names by | and put result into brackets.
            request = "http://api.openlandmap.org/query/point?lat=" + str(lat) + "&lon=" + str(lon) + "&coll=" + prefixes[i] + "&regex=(" + '|'.join(layers[layers["prefix"] == prefixes[i]]["global_covariate_layer"]) + ")" 
            
            # First call does not work sometimes. If so, repeat the call
            # But break after 5 calls with information for the user that
            # OpenLandMap API seems to have problems
            response = requests.get(request)
            i = 1
            while response.status_code != 200: # Status code = 200 means successful call
                if i > 5:
                    st.subheader("An API for gathering satellite data seems to be down. Pls contact info@mi4people.ord")
                    break
                response = requests.get(request)
                i = i + 1

            response = response.json()
            response_without_coord = response['response'][0]
            
            # delete coordinates from response. Otherwise we would have
            # (the same) coordinates for repsonses for each prefix.
            response_without_coord.pop("lon")
            response_without_coord.pop("lat")
            out_OpenLandMap.update(response_without_coord)
        
        # Order the final combined response in the order as used by the model
        # It is the same order as used in 'Global_covariate_layers.csv'
        out_OpenLandMap = pd.DataFrame([out_OpenLandMap]).T
        out_OpenLandMap = out_OpenLandMap.reindex(
            index=layers['global_covariate_layer'])
        
        # combine users input with data from OpenLandMap
        input_user_and_out_OpenLandMap = input_user.append(
            out_OpenLandMap, ignore_index=True)
        
        
        
        # Now, get data from SentinelHub. You can find more on this data in
        # Download_Sentinel_2_Pixels.ipynb.
        # Sentinel Hub requires credentials to make calls. Free access is
        # granted only for one month. Until, we pay for the service, we need
        # to create new account and credentials avery month.
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
        
        # resolution (in meters) of satellite data when making a call to 
        # SentinelHub data.
        resolution = 20
        
        # The code below is based on tutorial from Sentinel Hub
        # https://sentinelhub-py.readthedocs.io/en/latest/examples/process_request.html
        # important here is that we use all 13 spectral bands of Sentinel-2
        # satellite (notation B*)
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
        
        # Sentinel Hub is built to request pictures from Sentinel-2. For current
        # model, however, we need only the values of the pixel exactly at the
        # location of interest. So, we request a very small picture (bbox) with
        # location of interest in the middle. If the response  is a not a 1x1
        # pixel "picture", we take the most central pixel.
        coord = [lon - 0.0001, lat - 0.0001, lon + 0.0001, lat + 0.0001]
    
        bbox = BBox(bbox=coord, crs=CRS.WGS84)
        size = bbox_to_dimensions(bbox, resolution=resolution)
    
        request_all_bands = SentinelHubRequest(
            evalscript=evalscript_all_bands,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=('2021-10-16', '2021-12-16'), # we take a picture with less possible clouds in the last two months
                    mosaicking_order='leastCC' # leastCC means least cloud coverage
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
    
    
        pixels_raw = all_bands_response[0] # get the pixals
    
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
    
        
        # Add response from Sentinel Hub to user's input and response from
        # OpenLandMap. The result is the input for our model
        final_response = input_user_and_out_OpenLandMap.append(out_SentinelHub.T,
                                                           ignore_index=True)
        
        #Normalize the input using min-max-normalization.
        norm_input = (final_response[0] - 
                      min_values["0"])/(max_values["0"] - min_values["0"]) 
        norm_input = pd.DataFrame(norm_input).T 
        norm_input = norm_input.to_numpy()
        
        # Make prediction. Our current model can only predict oc values between
        # 0 and 120 (by predicting ln(oc+1)). See also "As in Paper v1.1.ipynb"
        # Therefore, we need to make some transformations.
        prediction = pd.DataFrame(model.predict(norm_input))
        prediction.clip(lower=0, inplace=True)
        prediction = np.exp(prediction)-1
        if prediction[0][0] > 120:
            prediction[0][0] = 120
        
        # Show the prediction to the user    
        st.subheader(
            f"The estimated value of organic carbon is {prediction[0][0]:.2f} g/kg")
        
        # Plot the map showing the location of the prediction
        for_plot = pd.DataFrame({"lat" : [lat], "lon" : [lon]})
        st.map(for_plot, zoom = 2)