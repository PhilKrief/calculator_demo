import pandas as pd
import streamlit as st

#Page Titles
st.set_page_config(page_title="Calculateur Demo", page_icon=":robot_face:", layout='wide')
st.markdown("<h1 style='text-align: center;'>Demo de l'outil Streamlit et autres 😬</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Bonjour Martin et Sebastien :) </h3>", unsafe_allow_html=True)

if "datafile" not in st.session_state:
   st.session_state["datafile"] = True
if "network" not in st.session_state:
   st.session_state["network"] = True
if "profile" not in st.session_state:
   st.session_state["profile"] = True
if "funddata" not in st.session_state:
   st.session_state["funddata"] = True


#if "profile2" not in st.session_state:
#   st.session_state["profile2"] = True
#Sidebar contents
st.sidebar.title("Sidebar")
st.session_state["datafile"] = st.sidebar.file_uploader("Please upload an excel file: ", type=['xlsx'])
#st.session_state["network"] = st.sidebar.selectbox("Quelle reseau veux-tu analyser? ", options=['GPD', 'SSD', 'Fonds'])
