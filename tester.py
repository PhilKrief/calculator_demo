import pandas as pd
import streamlit as st
import numpy as np


def calculate_portfolio_returns(allocations, returns):
    mask = returns.isna()
    allocations[mask] = np.nan
    tester = 1
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

def performance_table_2(years,rendement_mandat, rendement_bench, rendements_indices, mandat):
    columns = ["Fonds", "Date de début", "Date de fin", "Rendement brut (période)", "Rendement indice (période)",
               "Rendement brut (annualisée)", "Rendement indice (annualisée)", "Valeur ajoutée (période)", "Valeur ajoutée annualisée",
               "Risque actif annualisé", "Ratio information", "Beta", "Alpha annualisé", "Ratio sharpe", "Coefficient de corrélation",
               "Volatilité annualisée du fonds", "Volatilité annualisée de l'indice"]

    perf_df = pd.DataFrame(columns).rename(columns={0:"Index"})

    time = 12 * years
    inputs = rendement_mandat.reset_index()

    inputs["Période"] = pd.to_datetime(inputs["Période"])
    date_end = inputs.loc[len(inputs)-1, "Période"]
    date_start = inputs.loc[len(inputs)-time, "Période"]

    df_final = pd.DataFrame()

    rendement_mandat = rendement_mandat.add_suffix('_bruts')
    rendement_bench = rendement_bench.add_suffix('_indices')

    df_final["Période"] = inputs["Période"]
    df_final.set_index('Période', inplace=True)
    df_final["Marché monétaire"] = rendements_indices["Marché monétaire"]
    df_final[rendement_mandat.columns] = rendement_mandat[rendement_mandat.columns]
    df_final[rendement_bench.columns] = rendement_bench[rendement_bench.columns]
    df_final.reset_index(inplace=True)


    df_filtered = df_final[(df_final["Période"] <= date_end) & (df_final["Période"] >= date_start)]
    df_filtered[mandat+"_bruts_retours"] = df_filtered[mandat+"_bruts"] +1
    df_filtered[mandat+"_indices_retours"] = df_filtered[mandat+"_indices"]+1
    df_filtered["va"] = df_filtered[mandat+"_bruts"] - df_filtered[mandat+"_indices"]

    rendements_indices_periode = df_filtered[mandat+"_indices_retours"].prod()-1
    rendements_bruts_periode = df_filtered[mandat+"_bruts_retours"].prod()-1
    valeur_ajoute = rendements_bruts_periode - rendements_indices_periode

    rendements_annualisé_brut = (1+rendements_bruts_periode)**(1/(len(df_filtered)/12))-1
    rendements_annualisé_indice = (1+rendements_indices_periode)**(1/(len(df_filtered)/12))-1
    valeur_ajoute_annual = rendements_annualisé_brut - rendements_annualisé_indice
    risque_actif_annual = df_filtered["va"].std()*(12**(1/2))
    information_ratio = valeur_ajoute_annual/risque_actif_annual
    beta = (df_filtered[mandat+"_bruts"].astype(float).cov(df_filtered[mandat+"_indices"].astype(float)))/df_filtered[mandat+"_indices"].astype(float).var()

    df_filtered["Marché monétaire_retours"] = df_filtered["Marché monétaire"]+1
    risk_free = (df_filtered["Marché monétaire_retours"].prod()-1)
    risk_free_annual = (1+risk_free)**(1/(len(df_filtered)/12))-1
    alpha = rendements_annualisé_brut - beta*(rendements_annualisé_indice-risk_free_annual)-risk_free_annual
    stand_dev_fund = df_filtered[mandat+"_bruts"].std()*(12**(.5))
    stand_dev_index = df_filtered[mandat+"_indices"].std()*(12**(.5))
    sharpe = (rendements_annualisé_brut-risk_free_annual)/stand_dev_fund
    coeff_corr = df_filtered[mandat+"_bruts"].astype(float).corr(df_filtered[mandat+"_indices"].astype(float))
    dict_dum = {"Fonds":mandat, "Date de début":date_start, "Date de fin": date_end, "Rendement brut (période)":rendements_bruts_periode,
                "Rendement indice (période)":rendements_indices_periode, "Valeur ajoutée (période)":valeur_ajoute, "Rendement brut ("
                                                                                                           "annualisée)":rendements_annualisé_brut,
                "Rendement indice (annualisée)":rendements_annualisé_indice, "Valeur ajoutée annualisée":valeur_ajoute_annual, "Risque actif "
                                                                                                                               "annualisé":
                    risque_actif_annual,"Ratio information":information_ratio, "Beta":beta, "Alpha annualisé":alpha, "Ratio sharpe":sharpe,
                "Coefficient de corrélation":coeff_corr, "Volatilité annualisée du fonds": stand_dev_fund, "Volatilité annualisée de l'indice":stand_dev_index}
    perf_df[mandat] = perf_df["Index"].map(dict_dum)
    return perf_df

def financial_metric_table(rendement_mandat, rendement_bench, indices_df, mandat):
    data = pd.DataFrame()

    for time in [1,3, 5]:
        financial_metrics = performance_table_2(time,rendement_mandat, rendement_bench, indices_df, mandat).add_suffix('_%d'%time)
        data["Index"] = financial_metrics["Index_%d"%time]
        data = pd.merge(data, financial_metrics,  left_on="Index", right_on="Index_%d"%time, how="left")


    spike_cols = ["Index"] + [col for col in data.columns if mandat in col]
    data = data[spike_cols]

    return data


profiles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']




rendements_df = pd.read_excel("Sources - PowerBI Dashboard - GPD_SSD_VMD RandomData.xlsx",sheet_name="Rendements bruts")
indices_df = pd.read_excel("Sources - PowerBI Dashboard - GPD_SSD_VMD RandomData.xlsx",, sheet_name="Rendements indices")

allocation_profil = pd.read_excel("Sources - PowerBI Dashboard - GPD_SSD_VMD RandomData.xlsx",, sheet_name="Allocation").set_index(
'Code_Rendements').T

rendements_df["Year"] = pd.DatetimeIndex(rendements_df.Période).year
rendements_df["Month"] = pd.DatetimeIndex(rendements_df.Période).month


funds = ['Placement à court terme', 'Obligations gouvernementales', 'Obligations corporatives', 'Actions cdns grande cap', 'Actions cdns pet cap',
         'Actions US Tax', 'Actions EAEO', 'Actions mondiales de PC', 'Actions des marchés émergents', 'Stratégies complémentaires',
         'Stratégie à rendement absolu']

returns_df_calc = rendements_df[funds]
returns_df_calc['Encaisse'] = 0
returns_df_calc.index = rendements_df.Période

indices_df_calc =  indices_df[funds]
indices_df_calc['Encaisse'] = 0
indices_df_calc.index = indices_df.Période


rendement_mandat = pd.DataFrame(index=rendements_df.Période)
rendement_bench = pd.DataFrame(index=indices_df.Période)

for profile in profiles:
    allo = allocation_df_prep(profile, allocation_profil, rendements_df)
    rendement_mandat[profile] = (calculate_portfolio_returns(allo, returns_df_calc))
    allo_bench = allocation_df_prep(profile, allocation_profil, indices_df)
    rendement_bench[profile] = (calculate_portfolio_returns(allo, indices_df_calc))
    

graph_df = pd.DataFrame()
graph_df.index = rendement_mandat.index

if st.session_state["profile"] is not None:
    for profile in st.session_state['profile']:
        #profile = st.session_state['profile']
        graph_df['profile %s'%profile] = rendement_mandat_graph[profile]
        metrique = financial_metric_table(rendement_mandat, rendement_bench, indices_df, profile)
        cols = [i for i in list(metrique.columns) if i != 'Index']
        financial_metrics[cols] = metrique[cols]

rendement_mandat = ((1 + rendement_mandat).cumprod())
rendement_bench = ((1 + rendement_bench).cumprod())




financial_metrics = pd.DataFrame()
financial_metrics['Index'] = ['Fonds', 'Date de début', 'Date de fin', 'Rendement brut (période)', 'Rendement indice (période)', 'Rendement brut (annualisée)', 'Rendement indice (annualisée)', 'Valeur ajoutée (période)', 'Valeur ajoutée annualisée', 'Risque actif annualisé', 'Ratio information', 'Beta', 'Alpha annualisé', 'Ratio sharpe', 'Coefficient de corrélation', 'Volatilité annualisée du fonds', "Volatilité annualisée de l'indice"]


if st.session_state["profile"] is not None:
    for profile in st.session_state['profile']:
        #profile = st.session_state['profile']
        graph_df['profile %s'%profile] = rendement_mandat_graph[profile]
        metrique = financial_metric_table(rendement_mandat, rendement_bench, indices_df, profile)
        cols = [i for i in list(metrique.columns) if i != 'Index']
        financial_metrics[cols] = metrique[cols]
        if indice:
            graph_df['indice %s'%profile] = rendement_bench_graph[profile]

    
    st.line_chart(graph_df)
    st.dataframe(financial_metrics)





#st.dataframe(rendement_mandat)

#st.dataframe(rendement_bench)

