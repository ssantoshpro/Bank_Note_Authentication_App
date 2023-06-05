# from flask import Flask,request
import pandas as pd
import numpy as np
import pickle
# from flasgger import Swagger
import streamlit as st



#Read ML model
pickle_in = open(r"RFClassifier.pkl",'rb')
RFClassifier = pickle.load(pickle_in)

#Test home page
def welcomePage():
    return "Welcome to testing module"

#stores the parameters
def predict_note_authentication(variance,skewness,curtosis,entropy):
    
    """Let's Authenticate the bank notes
    This is a docstring for specifications.
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
    prediction = RFClassifier.predict([[variance,skewness,curtosis,entropy]])
    print("The Predicted value is : "+str(prediction)) 
    return  prediction



def main():
    with st.sidebar.expander('About'):
        st.write('This Machine learning app uses Random Forest Classifier to predict whether a bank note is legit or not based on the inputs. The front end is built using Streamlit.')
        
    with st.sidebar.expander('Contact'):
        st.write('[GitHub](https://github.com/ssantoshpro/Bank_Note_Authentication_App)')
        st.write('[LinkedIn](https://www.linkedin.com/in/santoshsankaralingam/)')
        st.write('Mail : ssantoshpro@gmail.com')
    # st.title("Bank Note Authentication App")
    html_templ = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">StreamLit Bank Note Authenticator ML App</h2>
    </div>
    """
    st.markdown(html_templ,unsafe_allow_html=True)
    variance = st.number_input("variance")
    skewness = st.number_input("skewness")
    curtosis = st.number_input("curtosis")
    entropy = st.number_input("entropy")
    if st.button("Predict"):
        result = predict_note_authentication(variance,skewness,curtosis,entropy)
        if result:
            st.error('YOU ARE HAVING A {} NOTE 😓'.format("FAKE" if result==1 else 'LEGIT'))
            st.error("Predicted Following value : {}".format(result), icon="🚫")
        else:
            st.success('YOU ARE HAVING A {} NOTE 😇'.format("LEGIT" if result==0 else 'FAKE'))
            st.success("Predicted Following value : {}".format(result), icon="✅")
    if st.button('AUTHOR'):
        st.text("Made by SAMTRON")

if __name__=='__main__':
    main()    