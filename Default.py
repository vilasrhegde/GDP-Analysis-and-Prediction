import streamlit as st 
import pickle
from streamlit_option_menu import option_menu
import numpy as np
import time
import pandas as pd
import plotly.express as px
from datetime import date





# Implementing the design

st.set_page_config(
    page_title="GDP Prediction & Analysis",
    page_icon="random",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://vilasrhegde.github.io/',
        'Report a bug': "https://www.linkedin.com/in/vilasrhegde/",
        'About': "# Made by Vilas Hegde!"
    }
)
# Loading the saved Random Forest model

@st.cache_resource(show_spinner=True)
def get_model(model_name):
    model = pickle.load(open(model_name,'rb'))
    return model
@st.cache_resource(show_spinner=True)
def get_data(name):
    data=pd.read_csv(name)
    return data

RFmodel=get_model('RFmodel.pkl')
data=get_data('filtered_data.csv')

# ------------------LOGIN---------------------------
st.title("GDP per capita prediction using Machine Learning")

# @st.cache_data
def login(username,password):
    database = {'vilas':'vilas','smruthi':'smruthi','rishith':'rishith','rohan':'rohan'}
    if username in database.keys():
        if database[username]!=password:
            st.error('Username/password did not match ')
            st.stop()
        else:
            st.session_state['username']=username
            return True
if 'username' not in st.session_state:
    with st.expander("Please fill the below credentials to begin",expanded=True):
        with st.form("login"):
            # st.write("Please with the credentials to login")
            username=st.text_input('Enter the username','',placeholder='username')
            psw=st.text_input('Enter the password','',placeholder='password',type='password')
            # Every form must have a submit button.
            submitted = st.form_submit_button("Login",type='secondary',use_container_width=True)
        if submitted:
            login(username,psw)
            

        if not username or not psw:
            st.warning('Please do login before perform any operation')
            st.stop()



# st.write(st.session_state.username)
if 'username' in st.session_state:
    st.sidebar.title('Hi :blue['+ st.session_state.username.capitalize()+'!]')

with st.sidebar:
    selected = option_menu(
        menu_title="Main menu",
        options=["Prediction","Analytics","Help"]
    )
st.header(f"{selected}")
if selected == 'Prediction':
    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        latest_iteration.text(f'App is getting ready... {i+1}%')
        bar.progress(i + 1)
        time.sleep(0.01)
        
    st.warning("Feel free to change the values to predict GDP per capita for any given country!")

    # INPUT VALUES FROM THE USER
    
    cc1,cc2,cc3 = st.columns(3)
    
    att_popl = st.number_input('Population (Example: 7000000)', min_value=1e4, max_value=2e9, value=2e7)

    ac1,ac2= st.columns(2)
    with ac1:
        att_area = st.slider('Area (sq. Km)', min_value= 2.0, max_value= 17e6, value=6e5, step=1e4)
    with ac2:
        att_cost = st.slider('Coastline/Area Ratio', min_value= 0, max_value= 800, value=30, step=10)
    att_dens = st.slider('Population Density (per sq. mile)', min_value= 0, max_value= 12000, value=400, step=10)

    att_migr = st.slider('Annual Net Migration (migrant(s)/1,000 population)', min_value= -20, max_value= 25, value=0, step=2) 
    att_mort = st.slider('Infant mortality (per 1000 births)', min_value= 0, max_value=195, value=40, step=10)
    att_litr = st.slider('Population literacy Percentage', min_value= 0, max_value= 100, value=80, step=5)
    att_phon = st.slider('Phones per 1000', min_value= 0, max_value= 1000, value=250, step=25)
    
    
    with cc1:
        att_arab = st.slider('Arable Land (%)', min_value= 0, max_value= 100, value=25, step=2)
    with cc2:
        att_crop = st.slider('Crops Land (%)', min_value= 0, max_value= 100, value=5, step=2)
    with cc3:
        att_othr = st.slider('Other Land (%)', min_value= 0, max_value= 100, value=70, step=2)
    st.markdown('`Arable, Crops, and Other land are summed up to 100%`')

    #Climate
    att_clim = st.selectbox('Climate', options=('Mostly hot (like: Egypt and Australia)', 'Mostly hot and Tropical (like: China and Cameroon)', 'Mostly cold and Tropical (like: India)', 'Mostly cold and Tropical (like: India)', ' Mostly cold (like: Argentina and Belgium)'))
    if att_clim == 'Mostly hot (like: Egypt and Australia)':
        att_clim=1
    elif att_clim == 'Mostly hot and Tropical (like: China and Cameroon)':
        att_clim=1.5
    elif att_clim == 'Mostly tropical (like: The Bahamas and Thailand)':
        att_clim=2
    elif att_clim == 'Mostly cold and Tropical (like: India)':
        att_clim=2.5
    elif att_clim == 'Mostly cold (like: Argentina and Belgium)':
        att_clim=3

    cc1,cc2 = st.columns(2)
    with cc1:
        att_brth = st.slider('Annual Birth Rate (births/1,000)', min_value= 7, max_value= 50, value=20, step=2)
    with cc2:
        att_deth = st.slider('Annual Death Rate (deaths/1,000)', min_value= 2, max_value= 30, value=10, step=2)
    
    cc1,cc2,cc3 = st.columns(3)
    with cc1:
        att_agrc = st.slider('Agricultural Economy', min_value= 0.0, max_value= 1.0, value=0.15, step=0.05)
    with cc2:
        att_inds = st.slider('Industrial Economy', min_value= 0.0, max_value= 1.0, value=0.25, step=0.05)
    with cc3:
        att_serv = st.slider('Services Economy', min_value= 0.0, max_value= 1.0, value=0.60, step=0.05)
    st.markdown('`Agricultural, Industrial, and Services Economy are summarized to 1`')
    att_regn = st.selectbox('Region', options=('ASIA (EX. NEAR EAST)','BALTICS','C.W. OF IND. STATES','EASTERN EUROPE','LATIN AMER. & CARIB','NEAR EAST','NORTHERN AFRICA','NORTHERN AMERICA','OCEANIA','SUB-SAHARAN AFRICA','WESTERN EUROPE'))

    if att_regn == "ASIA (EX. NEAR EAST)":
        att_regn = 1
    elif att_regn == "BALTICS":
        att_regn = 2
    elif att_regn == "C.W. OF IND. STATES":
        att_regn = 3
    elif att_regn == "EASTERN EUROPE":
        att_regn = 4
    elif att_regn == "LATIN AMER. & CARIB":
        att_regn = 5
    elif att_regn == "NEAR EAST":
        att_regn = 6
    elif att_regn == "NORTHERN AFRICA":
        att_regn = 7
    elif att_regn == "NORTHERN AMERICA":
        att_regn = 8
    elif att_regn == "OCEANIA":
        att_regn = 9
    elif att_regn == "SUB-SAHARAN AFRICA":
        att_regn = 10
    elif att_regn == "WESTERN EUROPE":
        att_regn = 11


    if att_regn == 1:
        att_regn_1 = 1
        att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 2: 
        att_regn_2 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 3: 
        att_regn_3 = 1
        att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 4: 
        att_regn_4 = 1
        att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 5: 
        att_regn_5 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 6: 
        att_regn_6 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 7: 
        att_regn_7 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 8: 
        att_regn_8 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_2 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 9: 
        att_regn_9 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_2 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 10: 
        att_regn_10 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_2 = att_regn_11 = 0
    else: 
        att_regn_11 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_2 = 0


    user_input = np.array([att_dens, att_migr, 
                        att_mort, att_litr, att_phon, 
                            att_brth, att_agrc, att_serv, 
                            att_regn_1, att_regn_2, att_regn_3,
                        att_regn_4, att_regn_5, att_regn_6, att_regn_7, 
                        att_regn_8, att_regn_9, att_regn_10, att_regn_11]).reshape(1,-1)

    # st.write(user_input.shape)

    if (st.button('__**Estimate GDP**__',use_container_width=True,type='primary')):
        prediction=RFmodel.predict(user_input)

        with st.spinner('Prediction is on the way...'):
            time.sleep(2)


        with st.container( ):
            st.balloons()
            st.snow()
            st.header(f'The estimated GDP per capita is: `{float(prediction)}` ')
            st.success(f'R2 Score of the _Random Forest Regressor_ is: __{0.84}__')
            st.info('Generally R2 score __>0.7__ is considered as good', icon="??????")


elif selected=='Analytics':


    
    tab1, tab2, tab3 = st.tabs(["Regional", "EDA", "Performance"])

    d=data.groupby('region')['gdp_per_capita'].mean().sort_values()



    with tab1:
            
        st.plotly_chart(px.bar(d, x='gdp_per_capita',title="Rankings of Regions based on GDP per capita",orientation='h'),theme='streamlit',use_container_width=True)

        st.plotly_chart(px.scatter(data, x="literacy", y="gdp_per_capita",title='GDP per capita v/s Literacy', size='literacy', color="region",
            hover_name="country", log_x=True),use_container_width=False)
        st.info('__GDP__ of a country is highly dependant upon the __literacy__ and vice versa.')

        st.plotly_chart(px.scatter(data, x="agriculture", y="gdp_per_capita", color="region",
                    title='GDP v/s Agriculture (Crops)'),use_container_width=True)
        st.info('Poor countries are more dependant upon harvesting crops than developed countris.')

        st.plotly_chart(px.box(data,x="area",y="gdp_per_capita",points="all"),theme=None,use_container_width=True)
        st.info('As the area increased, the GDP did not kept up.')

        st.plotly_chart(px.bar(data, x='region', y='country'),theme=None,use_container_width=True)
        st.plotly_chart(px.bar(data, x='region', y='gdp_per_capita',color='country',title="GDP of multiple Regions",width=500),use_container_width=True)

    with tab2:
        st.sidebar.markdown('### For Exploratory Data Analysis (EDA)')
        options = st.sidebar.multiselect(
        'Plot your graph by choosing the paremeters',
        ['region', 'gdp_per_capita', 'net_migration', 'phones'],
        ['region', 'gdp_per_capita','net_migration'])
        if len(options)!=3:
            st.warning('Please select only 3.')
            st.stop()
        else:
            st.markdown(f'# The 3D Graph')
            st.write('You selected:', options)

        st.plotly_chart(px.scatter_3d(data, x=options[0], y=options[1], z=options[2],color=options[0],height=720),use_container_width=True  )
        
        st.plotly_chart(px.scatter_matrix(data, dimensions=data[['population', 'area', 'net_migration', 'gdp_per_capita', 'climate']],width=700, height=720,title="Features relationships",color="gdp_per_capita"),use_container_width=True)

        st.info('You can crop and zoom the graph',icon='??????')
        st.plotly_chart(px.imshow(data.corr(),text_auto=True, aspect="auto"),theme=None,use_container_width=True)

        with st.expander("See Observations"):
            st.write('''
            ### From the above graphs:
        - net_migration & gdp_per_capita has good correlations, which means migrants always prefers to move to the countries having better economy and growth which is gdp in our case.
        - climate and populations are less correlated, means people avoid extreme weather and climate places
        - as area increased the amount of migratants also increased, obvious.
            ''')
            c1,c2 = st.columns(2)
            with c1:
                st.write('''
                - `Strong correlations are`,
                    1. infant_mortality & birthrate
                    2. infant_mortality & literacy
                    3. gdp_per_capita & phones
                    4. arable & other than crops
                    5. birthrate & literacy (less literacy = higher the birthrate)
                ''')
            with c2:
                st.write('''
                - `Weak correlations are`,
                    1. infant_mortality & agriculture
                    2. birthrate & phones
                    3. gdp_per_capita & birthrate
                ''')




    with tab3:
        st.header('Performance Awards ????')
        col1, col2, col3 = st.columns(3)
        with col1:

            st.write('''
                1. _`Random Forest` with Feature selection and NO scaling_
            - Mean Absolute Error __(MAE)__: 2451.88
            - Root Mean Squared Error __(RMSE)__: 3580.53
            - R-Squared Score __(R2_Score)__: 0.84''')

        with col2:
            st.write('''

        3. _`Linear Regression` with selected features and scaling_
        - Mean Absolute Error __(MAE)__: 2879.521
        - Root Mean Squared Error __(RMSE)__:3756.43
        - R-Squared Score __(R2_Score)__: 0.83''')
        
        with col3:
            st.write('''

        5. _`SVM` with feature scaling and selection_
        - Mean Absolute Error __(MAE)__: 7040.04
        - Root Mean Squared Error __(RMSE)__: 9794.59
        - R-Squared Score __(R2_Score)__: -0.16
        ''')
            
        with col1:
            st.write('''
        2. _`Gradient Boosting` with selected features and scaling_
        - Mean Absolute Error __(MAE)__: 2467.21
        - Root Mean Squared Error __(RMSE)__: 3789.30
        - R-Squared Score __(R2_Score)__: 0.83''')
            
        with col2:

            
            st.write('''

        4. _`Optimised Random Forest`_
        - Mean Absolute Error __(MAE)__: 3564.04
        - Root Mean Squared Error __(RMSE)__: 5915.82
        - R-Squared Score __(R2_Score)__: 0.73''')



            


elif selected == 'Help':
    with st.container():

        st.write('''
        The :blue[**Gross Domestic Product**] (GDP) is one of the metrics to ensure self-sustained growth for any country. 
        The total monetary value of goods and services flowing through an economy over time is measured by GDP. 
        ''')

        st.write('''
        ### The goal of our project _Generic GDP Prediction and Analysis_ to find the **:red[Patterns]** inside the taken dataset of multiple countries, and to make the **:blue[Prediction]** using Supervized Machine Learning algorithm.        
        ''')
        st.markdown('''
        ## About the Dataset 
        ''')
        st.table(data.describe())
        st.write(data.shape)

        st.write('''
        ### Methodology
        
        :blue[Step 1]. Identifying the :red[research goal] that our project aims and context to deliver and measure of success.
        
        :blue[Step 2]: Data can be stored in a :red[variety of formats], from plain text files to database tables. The objective now is to acquire the related data. Realize the importance and sensitivity of data, and often have norms in place to ensure that everyone has access to just what they want.
        
        :blue[Step 3]: Data Cleansing integrating, and transforming data are the main steps in :red[data preparation].  However, the clean and validated information is gathered from open source. The dataset is clean and hence can be used as it is
        
        :blue[Step 4]: This stage focuses on :red[data exploration].  Extra Trees Regressor class is used to implement a Meta estimator that fits several decision trees on various sub samples. We are also using Plotly visual analysis tool for detailed data exploration.
        
        :blue[Step 5]: Model is designed with clean data and a :red[clear understanding of the content] in order to make better predictions, identify objects, or gain an understanding of the system to model.

        ''')

        st.text('The dataset has 227 entries with 20 features.')


        st.write('''
        ### Validation 
            - The datapoints are near to actual values with the limited data that we have
            - Altough there are slight dissimilarities, did not ruin our analysis
            - Values are bit OLD, not updated to this date. (population, areas, phones etc.)
                 ''')
        






        st.markdown(f":red[????? Vilas Hegde - {date.today().year}]")



