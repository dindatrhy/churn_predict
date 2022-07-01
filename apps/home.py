import streamlit as st
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def app():
    st.header('''by Dinda Tirta Rahayu''')
    image = Image.open("churn_detection.png")
    st.image(image, width=  None)
    
    st.write('''
    
    
    
    ''')
    st.write("To use this application, kindly choose the navigator `Choose Here` in the sidebar. Here is the explanation :")
    st.write("`Churn Prediction Multiple Output` is for app that leave us to input our csv file then the app would process the churn prediction output in multiple case, we can download the output as .csv file too.")
