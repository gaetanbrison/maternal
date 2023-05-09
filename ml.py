import streamlit as st
import pandas as pd
from PIL import Image
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import time
from PIL import Image
from sklearn import metrics

image_pregnant = Image.open('pregnant.jpeg')
st.image(image_pregnant, width=1000)


st.title("How do symptoms of maternal physical health impact pregnancy risks?")

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])

if app_mode == 'Summary 🚀':
    st.markdown("##### Objectives")
    st.markdown("We aim to see what indicators of physical health in pregnant women are most associated with high risk pregnancies.")

    st.markdown("Gestastional diabetes (high blood glucose levels in pregnant people) affects 2%-10% of pregnancies for women in the U.S")
    st.markdown("Hypertention occurs in 1 out of every 12-17 pregnacies in women in the U.S. and is only increasing with time.")

    df = pd.read_csv("Maternal Health Risk Data Set.csv")

    num = st.number_input('No. of Rows', 5, 10)

    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head': 
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    st.text('(Rows,Columns)')
    st.write(df.shape)

    st.markdown("##### Key Variables")
    st.markdown("- Age of patient")
    st.markdown("- Systolic Blood Pressure")
    st.markdown("- Diastolic Blood Pressure")
    st.markdown("- Blood Glucose Levels")
    st.markdown("- Body Temperature")
    st.markdown("- Heart Rate")
    st.markdown("- Risk Level (High, Medium, Low")

    st.markdown("*Systolic blood pressure indicates how much pressure your blood is exerting against your artery walls when the heart beats")
    st.markdown("*Diastolic blood pressure indicates how much pressure your blood is exerting against your artery walls while the heart is resting between beats")
    st.markdown("Both are equally important indicators in diagnosing hypertension. The chance of experiencing hypertension increases during pregnancy and can lead to strokes, immediate labor inductions and eclampsia.")

    st.markdown("High blood glucose levels increases the chance of the baby being born too early, weigh too much, or with breathing problems.")

    st.markdown("Over the course of pregnancy your blood volume increases by nearly 50%, so your heart rate speeds up significantly to account for the extra blood flow. So generally having a high heart rate is normal, but in rare cases can lead to  ")

    st.markdown("### Description of Data")
    st.dataframe(df.describe())
    st.markdown("Descriptions for all quantitative data **(rank and streams)** by:")

    st.markdown("Count")
    st.markdown("Mean")
    st.markdown("Standard Deviation")
    st.markdown("Minimum")
    st.markdown("Quartiles")
    st.markdown("Maximum")

    st.markdown("### Missing Values")
    st.markdown("Null or NaN values.")

    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### Completeness")
    st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

    st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)

    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")    
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")












if app_mode == 'Visualization 📊':
    st.subheader("02 Visualization Page - Data Analysis 📊")
    #response = requests.get("https://lookerstudio.google.com/u/0/reporting/97c91d4e-e116-488f-b695-50179f0c7a11/page/pYAKD")
    #html_code = response.text
    #st.write("My Looker Dashboard")
    #html_code = f'<iframe srcdoc="{html_code}" width="100%" height="600" frameborder="0"></iframe>'
    #html(html_code)

    st.markdown("[![Foo](https://i.postimg.cc/1tYLhhnp/Screenshot-2023-05-09-at-15-32-46.png)](https://lookerstudio.google.com/u/0/reporting/6fa4cfe6-2c6f-4fd4-b71a-5e016bc6c231/page/pqZPD)")

    #image_dashboard = Image.open('images/dashboard.png')
    #st.image(image_dashboard)


if app_mode == 'Prediction 📈':

    from codecarbon import OfflineEmissionsTracker



    st.title("Maternal Health Data Analysis - 03 Prediction Page 🧪")

    df = pd.read_csv('Maternal Health Risk Data Set.csv')

    prediction_choices = ['Logistic','KNN']

    prediction_type = st.sidebar.selectbox('Select Type of Prediction', prediction_choices)

    list_variables = df.columns
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)
    new_df= df.drop(labels=select_variable, axis=1) 
    list_var = new_df.columns
    output_multi = st.multiselect("Select Explanatory Variables", list_var)

    new_df2 = new_df[output_multi]
    x =  new_df2
    y = df[select_variable]


    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(df.drop('RiskLevel',axis=1))

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()

    scaler.fit(df.drop('RiskLevel',axis=1))
    scaled_features = scaler.transform(df.drop('RiskLevel',axis=1))

    df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
    x = df[['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']]
    y = df['RiskLevel']

    ### The train_test_split() function splits the data into training and testing sets.
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=train_size)

    if prediction_type == 'Logistic':
    
        from sklearn.linear_model import LogisticRegression
        logtracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
        logtracker.start()
        log_start_time = time.time()
        logmodel = LogisticRegression(multi_class='multinomial', solver='lbfgs')

        logmodel.fit(x_train, y_train)
        logpred = logmodel.predict(x_test)
        logresults = logtracker.stop()
        log_end_time = time.time()
        log_execution_time = log_end_time - log_start_time


        st.write("Execution time:", log_execution_time, "seconds")
        st.write("Carbon Emissions: ",' %.12f kWh' % logresults)

        col1,col2 = st.columns(2)
        col1.subheader("Feature Columns top 25")
        col1.write(x.head(25))
        col2.subheader("Target Column top 25")
        col2.write(y.head(25))

        st.subheader('🎯 Results')

        from sklearn.metrics import classification_report
        st.text(classification_report(y_test,logpred))
        from sklearn.metrics import confusion_matrix

        confusion_matrix(y_test, logpred)
        from sklearn.metrics import accuracy_score

        y_pred = logmodel.predict(x_test)

        accuracy = accuracy_score(y_test, y_pred)

    elif prediction_type == 'KNN':

        from sklearn.neighbors import KNeighborsClassifier

        knntracker = OfflineEmissionsTracker(country_iso_code="FRA") # FRA = France
        knntracker.start()
        knn_start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(x_train,y_train)
        knnpred = knn.predict(x_test)
        knnresults = knntracker.stop()
        knn_end_time = time.time()
        knn_execution_time = knn_end_time - knn_start_time

        st.write("Execution time:", knn_execution_time, "seconds")
        st.write("Carbon Emissions: ",' %.12f kWh' % knnresults)
        from sklearn.metrics import classification_report,confusion_matrix
        st.write(confusion_matrix(y_test,knnpred))
        st.text(metrics.classification_report(y_test,knnpred))

    else:

        st.write('Please select a prediction type')

