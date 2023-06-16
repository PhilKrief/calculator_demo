import pandas as pd
import streamlit as st
import numpy as np


def calculate_portfolio_returns(allocations, returns):
    mask = returns.isna()
    allocations[mask] = np.nan

    row_sum = allocations.apply(lambda row: row.sum(skipna=True), axis=1)
    norm_aloc = allocations.div(row_sum, axis=0)
    print(norm_aloc)
    portfolio_returns = (returns * norm_aloc).sum(axis=1)

    return portfolio_returns

def allocation_df_prep(allocation, df, returns):
    selected_row = df.loc[allocation]
    allocation_df = pd.DataFrame( columns= selected_row.index, index= returns.Période)
    allocation_df.loc[:,:] = selected_row.values
    return allocation_df

#Page Titles

st.markdown("<h1 style='text-align: center;'>Demo de l'outil Streamlit et autres outil de cette décennie 😬</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Bonjour Martin et Sebastien :) </h3>", unsafe_allow_html=True)



profiles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']


st.session_state["profile"] = st.sidebar.multiselect("Quelle profile veux tu voir? ",options=profiles, default='A')
indice = st.sidebar.checkbox("Veux tu voir l'indices? ")
million = st.sidebar.checkbox("Veux tu voir l'évolution de 1,000,000$ ")

fees = st.sidebar.number_input("Frais annuel (en decimale)")






if st.session_state["datafile"] is not None:
    #read csv
    rendements_df = pd.read_excel(st.session_state["datafile"] ,sheet_name="Rendements bruts")
    indices_df = pd.read_excel(st.session_state["datafile"], sheet_name="Rendements indices")

    allocation_profil = pd.read_excel(st.session_state["datafile"], sheet_name="Allocation").set_index(
    'Code_Rendements').T

else:
    st.warning("you need to upload a csv or excel file.")



rendements_df["Year"] = pd.DatetimeIndex(rendements_df.Période).year
rendements_df["Month"] = pd.DatetimeIndex(rendements_df.Période).month


funds = ['Placement à court terme', 'Obligations gouvernementales', 'Obligations corporatives', 'Actions cdns grande cap', 'Actions cdns pet cap',
         'Actions US Tax', 'Actions EAEO', 'Actions mondiales de PC', 'Actions des marchés émergents', 'Stratégies complémentaires',
         'Stratégie à rendement absolu']

returns_df_calc = rendements_df[funds]
returns_df_calc['Encaisse'] = 0
returns_df_calc.index = rendements_df.Période

st.session_state['funddata'] =  returns_df_calc



indices_df_calc =  indices_df[funds]
indices_df_calc['Encaisse'] = 0
indices_df_calc.index = indices_df.Période

st.session_state['benchdata'] = indices_df_calc

rendement_mandat = pd.DataFrame(index=rendements_df.Période)
rendement_bench = pd.DataFrame(index=indices_df.Période)

for profile in profiles:
    allo = allocation_df_prep(profile, allocation_profil, rendements_df)
    rendement_mandat[profile] = (calculate_portfolio_returns(allo, returns_df_calc))
    allo_bench = allocation_df_prep(profile, allocation_profil, indices_df)
    rendement_bench[profile] = (calculate_portfolio_returns(allo, indices_df_calc))
    

print(rendement_mandat.columns)
graph_df = pd.DataFrame()
graph_df.index = rendement_mandat.index

if fees: 
    monthly = fees / 12
    rendement_mandat = rendement_mandat - monthly

rendement_mandat = ((1 + rendement_mandat).cumprod())
rendement_bench = ((1 + rendement_bench).cumprod())

if million:
    rendement_mandat = rendement_mandat * 1000000
    rendement_bench = rendement_bench * 1000000


if st.session_state["profile"] is not None:
    for profile in st.session_state['profile']:
        #profile = st.session_state['profile']
        graph_df['profile %s'%profile] = rendement_mandat[profile]
        if indice:
            graph_df['indice %s'%profile] = rendement_bench[profile]


    st.line_chart(graph_df)





#st.dataframe(rendement_mandat)

#st.dataframe(rendement_bench)

