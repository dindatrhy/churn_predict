import streamlit as st
import requests as request



def app():


    st.write('Test your sentece sentiment here using the selected topic from the dropdown you picked')

    #Select option
    x_option = ['kemkes2022_omicron', 'kemkes2022_vaksin','kemkes2022_prokes', 'kemkes2022_ppkm', 'kemkes2022_testcovid', 'kemkes2022_booster', 'kemkes2022_local']
    x_collection = st.selectbox("Choose collection", x_option)

    #Input Box
    x_input = st.text_input(label='Input your text here')


    def sentimen_values(text):
        params = {
                    "text" : text,
                    "type" : "sentiment",
                    "col_id" : x_collection
                }
        result_url = "http://65.108.161.63:8000/get_watson_result"
        final_result = request.get(result_url, json=params)
#        st.write(final_result)
        return final_result.json()


    if x_input:
        sentiment_result = sentimen_values(x_input)
        st.write(f'Sentiment : {sentiment_result}')



