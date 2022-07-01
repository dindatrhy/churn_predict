import streamlit as st
from multiapp import MultiApp
from apps import home, main_page, direct_testing

app = MultiApp()
st.markdown("""
# Churn Prediction Result App
""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Churn Prediction - csv output", main_page.app)



# The main app
app.run()