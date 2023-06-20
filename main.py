
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np # linear algebra
import pandas as pd

# loading the saved models
House_model = pickle.load(open("C:/Users/HP/OneDrive/Desktop/InternShip/Bharat_Intern_Project/House prices prediction/Housing_Prices_Prediction_model.sav", 'rb'))

Iris_model = pickle.load(open("C:/Users/HP/OneDrive/Desktop/InternShip/Bharat_Intern_Project/Iris flowers classifition/Iris_dataset_model.sav",'rb'))

# parkinsons_model = pickle.load(open('C:/Users/HP/OneDrive/Desktop/Multiple_Disease/Multiple Disease Prediction System/saved models/parkinsons_model.sav', 'rb'))



# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Bharat Intern Project System',
                          
                          ['Housing Prices Prediction',
                           'Iris Dataset Classification',
                        #    'Parkinsons Prediction'
                           ],
                          icons=['activity','activity'
                                #  ,'person'
                                 ],
                          default_index=0)
    
    
# Housing Prices Page
if (selected == 'Housing Prices Prediction'):
    
    # page title
    st.title('Housing Prices Prediction using ML')
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        area = st.text_input('House Area')
        
    with col2:
        bedrooms = st.text_input('Number of Bedrooms')
    
    with col3:
        bathrooms = st.text_input('Number of bathrooms')
    
    with col1:
        mainroad = st.radio('House have mainroad',('Yes','No'))
    
    with col2:
        guestroom = st.radio('House have guestroom',('Yes','No'))
    
    with col3:
        basement = st.radio('House have basement',('Yes','No'))
    
    with col1:
        airconditioning = st.radio('House have airconditioning',('Yes','No'))
    
    with col2:
        parking = st.text_input('Number of parking in Huuse')


    # import pandas as pd

    # Assuming mainroad is a string variable
    mainroad = pd.Series(mainroad).map({'Yes': 1, 'No': 0})
    guestroom = pd.Series(guestroom).map({'Yes': 1, 'No': 0})
    basement = pd.Series(basement).map({'Yes': 1, 'No': 0})
    airconditioning = pd.Series(airconditioning).map({'Yes': 1, 'No': 0})

    
    # code for Prediction
    prices = ''
    
    # creating a button for Prediction
    input_data=(area, bedrooms, bathrooms, mainroad, guestroom, basement, airconditioning, parking)
   # change the input data to a numpy array
    input_data_as_numpy_array= np.asarray(input_data)

    # reshape the numpy array as we are predicting for only on instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
   
    if st.button('Predict Test Result'):
        prices = House_model.predict(input_data_reshaped)
        
        
    st.success(prices)




# Iris Dataset Classification Page
if (selected == 'Iris Dataset Classification'):

    # page title
    st.title('Iris Dataset Classification using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        SepalLengthCm = st.text_input('Enter Sepal Length in Cm')

    with col2:
        SepalWidthCm = st.text_input('Enter Sepal Width in Cm')

    with col3:
        PetalLengthCm = st.text_input('Enter Petal Length in Cm')

    with col1:
        PetalWidthCm = st.text_input('Enter Petal Width in Cm')

    # Code for Prediction
    class_iris = ''

    # creating a button for Prediction
    if st.button('Test Result'):
        class_iris = Iris_model.predict([[float(SepalLengthCm), float(SepalWidthCm), float(PetalLengthCm), float(PetalWidthCm)]])
        if (class_iris[0] == 0):
            class_iris = "This is in Iris-setosa"
        elif(class_iris[0] == 1):
            class_iris = "This is in Iris-versicolor"
        else:
            class_iris = "This is in Iris-virginica"

            
    st.success(class_iris)
    
    

# # Parkinson's Prediction Page
# if (selected == "Parkinsons Prediction"):
    
#     # page title
#     st.title("Parkinson's Disease Prediction using ML")
    
#     col1, col2, col3, col4, col5 = st.columns(5)  
    
#     with col1:
#         fo = st.text_input('MDVP:Fo(Hz)')
        
#     with col2:
#         fhi = st.text_input('MDVP:Fhi(Hz)')
        
#     with col3:
#         flo = st.text_input('MDVP:Flo(Hz)')
        
#     with col4:
#         Jitter_percent = st.text_input('MDVP:Jitter(%)')
        
#     with col5:
#         Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        
#     with col1:
#         RAP = st.text_input('MDVP:RAP')
        
#     with col2:
#         PPQ = st.text_input('MDVP:PPQ')
        
#     with col3:
#         DDP = st.text_input('Jitter:DDP')
        
#     with col4:
#         Shimmer = st.text_input('MDVP:Shimmer')
        
#     with col5:
#         Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        
#     with col1:
#         APQ3 = st.text_input('Shimmer:APQ3')
        
#     with col2:
#         APQ5 = st.text_input('Shimmer:APQ5')
        
#     with col3:
#         APQ = st.text_input('MDVP:APQ')
        
#     with col4:
#         DDA = st.text_input('Shimmer:DDA')
        
#     with col5:
#         NHR = st.text_input('NHR')
        
#     with col1:
#         HNR = st.text_input('HNR')
        
#     with col2:
#         RPDE = st.text_input('RPDE')
        
#     with col3:
#         DFA = st.text_input('DFA')
        
#     with col4:
#         spread1 = st.text_input('spread1')
        
#     with col5:
#         spread2 = st.text_input('spread2')
        
#     with col1:
#         D2 = st.text_input('D2')
        
#     with col2:
#         PPE = st.text_input('PPE')
        
    
    
#     # code for Prediction
#     parkinsons_diagnosis = ''
    
#     # creating a button for Prediction    
#     if st.button("Parkinson's Test Result"):
#         parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
        
#         if (parkinsons_prediction[0] == 1):
#           parkinsons_diagnosis = "The person has Parkinson's disease"
#         else:
#           parkinsons_diagnosis = "The person does not have Parkinson's disease"
        
#     st.success(parkinsons_diagnosis)


