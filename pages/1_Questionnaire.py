import pandas as pd
import streamlit as st

# Add the logo image file in the same directory as your script
logo_path = "media/desj.png"

# Create a container to hold the logo and header
header_container = st.container()

# Add the logo to the container
with header_container:
    logo_col, header_col = st.columns([1, 3])
    logo_col.image(logo_path, use_column_width=True)

    # Add the header text
    header_col.markdown("<h2 style='text-align: center;'>Demo de calculateur gestion de patrimoine Desjardins</h2>", unsafe_allow_html=True)
    header_col.markdown("<h3 style='text-align: center;'> Example du questionnaire </h3>", unsafe_allow_html=True)


# Define questions and choices
questions = {
    "Question 1": {
        "text": "What is your investment experience?",
        "choices": ["No experience", "Limited experience", "Moderate experience", "Extensive experience"]
    },
    "Question 2": {
        "text": "What is your investment goal?",
        "choices": ["Preservation of capital", "Income", "Growth", "Aggressive growth"]
    },
    "Question 3": {
        "text": "What is your investment time horizon?",
        "choices": ["Less than 1 year", "1-5 years", "5-10 years", "More than 10 years"]
    },
    "Question 4": {
        "text": "What is your risk tolerance?",
        "choices": ["Very low", "Low", "Moderate", "High", "Very high"]
    }
}

# Define question function
def ask_question(question):
    st.subheader(question["text"])
    choice = st.radio("", question["choices"])
    return choice

# Define questionnaire function
def questionnaire():
    st.write("Please answer the following questions to assess your risk level.")
    risk_level = 0
    for i, question in enumerate(questions.values()):
        choice = ask_question(question)
        if i == 0:
            if choice == "No experience":
                risk_level += 1
            elif choice == "Limited experience":
                risk_level += 2
            elif choice == "Moderate experience":
                risk_level += 3
            elif choice == "Extensive experience":
                risk_level += 4
        elif i == 1:
            if choice == "Preservation of capital":
                risk_level += 1
            elif choice == "Income":
                risk_level += 2
            elif choice == "Growth":
                risk_level += 3
            elif choice == "Aggressive growth":
                risk_level += 4
        elif i == 2:
            if choice == "Less than 1 year":
                risk_level += 1
            elif choice == "1-5 years":
                risk_level += 2
            elif choice == "5-10 years":
                risk_level += 3
            elif choice == "More than 10 years":
                risk_level += 4
        elif i == 3:
            if choice == "Very low":
                risk_level += 1
            elif choice == "Low":
                risk_level += 2
            elif choice == "Moderate":
                risk_level += 3
            elif choice == "High":
                risk_level += 4
            elif choice == "Very high":
                risk_level += 5
        elif i == 4:
            if choice == "Very loww":
                risk_level += 1
            elif choice == "Loww":
                risk_level += 2
            elif choice == "Moderatew":
                risk_level += 3
            elif choice == "Highw":
                risk_level += 4
            elif choice == "Very highw":
                risk_level += 5
    return risk_level


with st.container():
    risk_level = questionnaire()
    st.write("Your risk level is:", risk_level)