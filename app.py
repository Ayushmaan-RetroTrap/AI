import pandas as pd
import streamlit as st

file_path = "Color_Combinations_1M.csv" 
data = pd.read_csv(file_path)
data.columns = data.columns.str.strip().str.lower()  

color1_col = 'color 1 name'
color2_col = 'color 2 name'

st.title("Color Combination Checker")

with st.form(key="color_form"):
    color1 = st.text_input("Enter the first color:")
    color2 = st.text_input("Enter the second color:")
    submit_button = st.form_submit_button(label="Check Match")

if submit_button:
    if not color1 or not color2:
        st.error("Please enter both colors.")
    else:
        match_found = (
            ((data[color1_col] == color1) & (data[color2_col] == color2)) |
            ((data[color1_col] == color2) & (data[color2_col] == color1))
        ).any()

        if match_found:
            st.success(f"The combination of '{color1}' and '{color2}' is a match!")
        else:
            st.warning(f"'{color1}' and '{color2}' do not match.")
