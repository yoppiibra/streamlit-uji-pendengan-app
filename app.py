import streamlit as st
from streamlit_option_menu import option_menu as om
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import plotly.express as px


df = pd.read_csv('data//hearing_test.csv')
loadeed_model = pickle.load(open('lr_hearing_loss.sav', 'rb'))


with st.sidebar:
    selected= om(menu_title='Main Menu',
                 options = ['Home','Graphs','Project'])
    
if selected =='Home':
    st.title('Hearing Loss Web App')
    st.image('asset//hearing_loss.png', width=700)
    
    if st.checkbox('Check Data'):
        st.dataframe(df, width = 800)
    
if selected =='Graphs':
    st.header('Graphs of Data')
    graphs = st.selectbox('What Kind of Graphs', ['None','Non Interactive', 'Interactive'])
    
    if graphs =='None':
        pass
    
    if graphs =='Non Interactive':
        plt.figure(figsize=(6,4))
        ax = sns.countplot(x='test_result', data=df, palette='spring')
        
        for p in ax.patches:
            height = p.get_height() 
            ax.text(x = p.get_x() + p.get_width()/2, 
                    y = height + 100, 
                    s = '{:.0f}'.format(height), 
                    ha = 'center')
            plt.ylim(0, 3500)
            st.pyplot(ax.get_figure())
        st.write('-------------------------------------------------------')
        
        plt.figure(figsize=(6,4))
        ax = sns.scatterplot(x='age', y='physical_score',hue='test_result', data=df, palette='spring', s=100)
        st.pyplot(ax.get_figure())
        
        st.write('-------------------------------------------------------')
        
        plt.figure(figsize=(6,4))
        matrix = df.corr()
        mask = np.zeros_like(matrix)
        mask[np.triu_indices_from(matrix)]=True
        
        ax = sns.heatmap(matrix, mask=mask, lw=1, annot =True, cmap='Blues')
        st.pyplot(ax.get_figure())
    
    if graphs =='Interactive':
        plt.figure(figsize=(6,4))
        fig = px.scatter(df, x='age', y='physical_score', color='test_result', opacity= 1)        
        st.plotly_chart(fig)
    
        st.write('-------------------------------------------------------')
        
        plt.figure(figsize=(6,3))    
        fig = px.scatter_3d(df, x='age',y='physical_score', z='test_result', 
                            opacity=1, color=df['test_result'].astype(str), 
                            color_discrete_sequence=['black']+ px.colors.qualitative.Plotly,
                            width =500, height=500 
                            )
        
        st.plotly_chart(fig)
        

if selected =='Project':
    st.header('Test Hearing Loss')
    age = st.number_input('Age', min_value=0, max_value=100, step = 1)
    physical_score = st.number_input('Physical Score', min_value=00.00, max_value=100.00, step = 01.00)
    
    def test_result(input_data):
        
        input_data_np_asarray = np.asarray(input_data)
        
        input_data_reshape = input_data_np_asarray.reshape(1, -1)
        
        prediction = loadeed_model.predict(input_data_reshape)[0]
        
        print(prediction)
        
        if prediction ==0:
            return 'The Person is Not Pass'
        else:
            return 'The Person is Pass'
        
    diagnosis = ''
        
    if st.button('Submit'):
        diagnosis = test_result([age, physical_score])
        st.success(diagnosis) 