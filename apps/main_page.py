import streamlit as st
import numpy as np
import pandas as pd
import numpy as np

from statistics import mode
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder


from statistics import mean, stdev
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold

from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import joblib


import warnings
warnings.filterwarnings("ignore")

def app():

    @st.cache(allow_output_mutation=True)
    def get_df(file):
        # get extension and read file
        extension = file.name.split('.')[1]
        if extension.upper() == 'CSV':
            df = pd.read_csv(file, delimiter=",", header=0, encoding='utf-8')
        return df
    
    file = st.file_uploader("Upload file here", type=['csv'])
    if not file:
        st.write("Upload only .csv file")
        return
    # define our main dataframe
    data = get_df(file)

    #rename dataframe
    data.columns = data.columns.str.lower()

    data = data[['total_trans_ct', 'total_revolving_bal',
       'total_relationship_count', 'total_trans_amt',
       'months_inactive_12_mon', 'total_ct_chng_q4_q1',
       'total_amt_chng_q4_q1', 'avg_open_to_buy', 'customer_age',
       'contacts_count_12_mon']]
    
        


    '''
    Input Model
    '''

    model = joblib.load('./apps/model_churn_1.pkl')

    prediction_test = model.predict(data)

    #get result func
    churn_predict = pd.DataFrame(prediction_test).rename(columns={0:"churn_flag_prediction"}).replace([0,1],["Non Churn", "Churn"]).reset_index()
    
    
    # replace values
    ## mapping sentiment watson values
    data['churn_flag_predict'] = np.nan
    data['churn_flag_predict'] = churn_predict['churn_flag_prediction']

    
    #Vizualization
    #class distribution plotting
    '''
    plt.figure(figsize=(5,4))
    plt.suptitle('Class Distribution', fontsize=15)
    p1 = sns.countplot(data_train['churn_flag'], palette=['#acace6', '#add8e6'])
    p1.set_xlabel('Class', fontsize=12)
    '''


    ##css function
    hide_table_row_index = """
            <style>
            tbody th {display:none}
            .blank {display:none}
            </style>
            """
    #dataframe updated
    st.markdown(hide_table_row_index, unsafe_allow_html=True)
    st.write(data)
    
    @st.cache
    def convert_df(df):
        return df.to_csv().encode('utf-8')


    csv = convert_df(data)

    st.download_button(
        "Press to Download",
        csv,
        "churn_test_predicted.csv",
        "text/csv",
        key='download-csv'
        )
