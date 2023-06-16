import pandas as pd
import streamlit as st

#Page Titles
st.set_page_config(page_title="Calculateur Demo", page_icon=":robot_face:", layout='wide')


# Add the logo image file in the same directory as your script
logo_path = "media/desj.png"

# Create a container to hold the logo and header
header_container = st.container()

# Add the logo to the container
with header_container:
    logo_col, header_col = st.columns([1, 3])
    logo_col.image(logo_path, use_column_width=True)

    # Add the header text
    header_col.markdown("<h1 style='text-align: center;'>Demo de calculateur gestion de patrimoine Desjardins</h1>", unsafe_allow_html=True)



if "datafile" not in st.session_state:
   st.session_state["datafile"] = True
if "network" not in st.session_state:
   st.session_state["network"] = True
if "profile" not in st.session_state:
   st.session_state["profile"] = True
if "funddata" not in st.session_state:
   st.session_state["funddata"] = True
if "periodes" not in st.session_state:
   st.session_state["periodes"] = True


#if "profile2" not in st.session_state:
#   st.session_state["profile2"] = True
#Sidebar contents
st.sidebar.title("Sidebar")
st.session_state["datafile"] = st.sidebar.file_uploader("Please upload an excel file: ", type=['xlsx'])
#st.session_state["network"] = st.sidebar.selectbox("Quelle reseau veux-tu analyser? ", options=['GPD', 'SSD', 'Fonds'])
