########## Libraries ########

# Data Manipulation Libraries

import pandas as pd
import numpy as np

# UI & Driver code Lib

import streamlit as st

###### Trained Model Files ######

import joblib

ohe = joblib.load("ohe.pkl") # trained one hot encoded model

xgb = joblib.load("xgb.pkl") # trained model file

###### Helper Functions ########

def duration_in_hrs(df):
    for indx in range(len(df)):
        if 'h' in df.Duration[indx] and 'm' in df.Duration[indx]:
            h, m =  int(df.Duration[indx].split()[0].strip("h")), int(df.Duration[indx].split()[1].strip("m"))
            if m>=30:
                h+=1
            df.Duration[indx] = h
        elif 'h' in df.Duration[indx]:
            h = int(df.Duration[indx].strip("h"))
            df.Duration[indx] = h
        elif 'm' in df.Duration[indx]:
            h = round(int(df.Duration[indx].strip("m"))/60,2)
            df.Duration[indx] = h

stopnames = ['nonstop', 'ixr', 'bbi', 'blr', 'lko', 'bom', 'cok', 'nag', 'del', 'amd', 'pnq', 'ccu', 'idr', 'gau', 'maa', 'hyd', 'bho', 'jai', 'atq', 'jdh', 'goi', 'bdq', 'trv', 'ixu', 'ixb', 'udr', 'rpr', 'ded', 'vga', 'vns', 'isk', 'ixc', 'pat', 'jlr', 'knu', 'gwl', 'vtz', 'ndc', 'hbx', 'ixz', 'imf', 'ixa', 'stv']

def createclasscol(df, name):
    df['Stop_'+name] = df['Intermediate_Stops'].apply(lambda x: 1 if name in x else 0)

def data_cleaning_preprocessing(df):
    
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
            
    # Data Validation
            
    df.Date_of_Journey = pd.to_datetime(df.Date_of_Journey)
    df.Route = df.Route.str.replace(" ", "")
    df.Dep_Time = pd.to_datetime(df.Dep_Time).dt.hour
    df.Arrival_Time = pd.to_datetime(df.Arrival_Time).dt.hour
    
    duration_in_hrs(df) # applying duration in hrs function in for duration data
    
    df.Duration = df.Duration.astype(float)
    
    df.Total_Stops = df.Total_Stops.str.replace(" stops", "").str.replace(" stop", "")
    df.Total_Stops.replace({'non-stop':0}, inplace=True)
    df.Total_Stops = df.Total_Stops.astype(float)
    
    df.Additional_Info.replace({'in-flight meal not included':'nomeal',
                            'no check-in baggage included':'nocibag',
                            '1 short layover':'oneshortlay',
                            '1 long layover':'onelonglay',
                            '2 long layover':'2longlay',
                            }, inplace=True)

    df.Additional_Info = df.Additional_Info.str.replace(" ","").str.replace("-","")
    
    ############### Feature Engineering & Data Pre-Processing ##################
    
    # Extracting Month name from date
    
    df['Travel_Month'] = df.Date_of_Journey.dt.month_name().str.lower()
    
    del df['Date_of_Journey'] # removing journey date column
    
    
    # Taking a empty column

    df['Intermediate_Stops'] =  np.nan

    # Filling Data in Intermediate stops from Route by removing source & destination

    for indx in range(len(df)):
        stopvals = df.Route[indx].split("→")
        if len(stopvals)>2:
            df['Intermediate_Stops'][indx] = "→".join(stopvals[1:len(stopvals)])
        else:
            df['Intermediate_Stops'][indx] = "nonstop"

    del df['Route'] # Removing route column data
    
    df = df[['Airline', 'Travel_Month', 'Source', 'Destination',
       'Dep_Time', 'Duration', 'Intermediate_Stops', 'Total_Stops', 'Additional_Info']]
    
    for indx in range(len(df)):
        if df.Airline[indx] not in ['jetairways', 'airindia', 'multiplecarriers', 'indigo']:
            df.Airline[indx] = 'other'
            
    df.Dep_Time = np.where((df.Dep_Time>=18) | (df.Dep_Time<=6), 'prime', 'nonprime')
    
    ################## Encodings #####################
    
    ########### Month #######
    
    order = {'march':4, 'may':3, 'june':2, 'april':1}

    for indx in range(len(df)):
        if df.Travel_Month[indx] in order.keys():
            df.Travel_Month[indx] = order[df.Travel_Month[indx]]
        else:
            df.Travel_Month[indx] = 0 # replacing other months 0 for future predictions

    df.Travel_Month = df.Travel_Month.astype(int)
    
    ######## Dep_Time ########
    
    df.Dep_Time.replace({'prime':1, 'nonprime':0}, inplace=True)
    
    ###### Stops ###########
    
    for n in stopnames:
        createclasscol(df, n) # Create Class Function

    del df['Intermediate_Stops'] # Removing Intermeiate Stops Column
    
    ####### Airline , Source, Destination, Additional Info
    
    ohedata = ohe.transform(df.select_dtypes("O")).toarray() # taking onehot encoded data in array

    ohedata = pd.DataFrame(ohedata, columns=ohe.get_feature_names_out()) # converting to data frame

    df = pd.concat([df.select_dtypes(exclude="O"), ohedata], axis=1)
    
    return df

####### UI & Driver Code ##########

# Refer streamlit documentation given below for the ui code

# https://docs.streamlit.io/library/api-reference

# Introduction

st.title("Flight Fare Estimation for Top Airlines....")

st.image("display.png")

st.write("Analyzed and Builted a Predictive Model for Estimation of Flight Fares\nFor the given user data\nSource, Destination, Date of Journey with other particulars like Route, Duration etc.....")

# Data
st.header("Data Taken for Analysis:")
inputdata = pd.read_csv("UserInputData.csv")
st.dataframe(inputdata.head())

### Taking Input From User ###

st.subheader("Enter Below Particulars....")

col1, col2 = st.columns(2)

with col1:
    source = st.selectbox("Select Source City:", inputdata.Source.unique())

with col2:
    dest = st.selectbox("Select Destination City:", inputdata[inputdata.Source==source].Destination.unique())

col3, col4 = st.columns(2)

with col3:
    date = str(st.date_input("Select Date:")).replace("-","/")

if st.button("Estimate Fares"):
    
    # Data Extrcation
    
    raw_input = inputdata[(inputdata.Source == source) & (inputdata.Destination == dest)].drop('Price', axis=1).reset_index(drop=True)
    
    months = set(raw_input.Date_of_Journey.str.split("/", expand=True)[1]) 
    
    # Taking Month Wise Data
      
    if date.split("/")[1] in months:
        raw_input = raw_input[(raw_input.Date_of_Journey.str.split("/", expand=True)[1] == date.split("/")[1])]
        raw_input.Date_of_Journey.loc[:] = date
    
    else:
        raw_input.Date_of_Journey.loc[:] = date

    raw_input = raw_input.reset_index(drop=True)

    st.write("Input Data:")

    st.write("Getting Route Particulars From above data...")

    st.dataframe(raw_input.head())
    
    raw = raw_input.copy()

    inputdata = data_cleaning_preprocessing(raw) # Using Pre-Processing Function
    
    raw_input['Price'] = xgb.predict(inputdata)

    st.subheader("Flights With Minimum Fares:")
    
    for indx in range(len(raw_input)):
        if raw_input.Airline[indx] not in ['Jet Airways', 'Air India', 'Multiple Carriers', 'IndiGo']:
            raw_input.Airline[indx] = 'other'
    
    airlines = []
    prices = []
    for airline in raw_input.Airline.unique():
        st.write("Airline:", airline)
        airlines.append(airline)
        st.write()
        out = raw_input[raw_input.Airline==airline].sort_values(by='Price').reset_index(drop=True)[0:1]
        for indx in range(len(out)):
            prices.append(out.Price[indx])
            out.Price[indx] = "₹ "+ str(round(out.Price[indx]))
        out = out[['Date_of_Journey', 'Source', 'Destination', 'Route', 'Dep_Time', 'Arrival_Time', 'Total_Stops', 'Duration', 'Additional_Info', 'Price']]
        st.dataframe(out)
    
    st.header("Airline Which is Offering Low Fare:")
    st.subheader(airlines[prices.index(min(prices))])
    


