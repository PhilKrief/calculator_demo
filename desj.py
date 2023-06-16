import pandas as pd
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


rendements_df = pd.read_excel("C:/Users/DTQ8851/Desktop/Performance Code/Sources - PowerBI Dashboard - GPD_SSD_VMD v2.xlsx",
                              sheet_name="Rendements bruts")
indices_df = pd.read_excel("C:/Users/DTQ8851/Desktop/Performance Code/Sources - PowerBI Dashboard - GPD_SSD_VMD v2.xlsx",
                           sheet_name="Rendements indices")

rendements_df["Year"] = pd.DatetimeIndex(rendements_df.Période).year
rendements_df["Month"] = pd.DatetimeIndex(rendements_df.Période).month

allocation_profil = pd.read_excel("C:/Users/DTQ8851/Desktop/Apercu des fonds/Source/Sources Apercu test.xlsx", sheet_name="Allocation").set_index(
    'Code_Rendements').T

funds = ['Placement à court terme', 'Obligations gouvernementales', 'Obligations corporatives', 'Actions cdns grande cap', 'Actions cdns pet cap',
         'Actions US Tax', 'Actions EAEO', 'Actions mondiales de PC', 'Actions des marchés émergents', 'Stratégies complémentaires',
         'Stratégie à rendement absolu']

returns_df_calc = rendements_df[funds]
returns_df_calc['Encaisse'] = 0
returns_df_calc.index = rendements_df.Période

bench_df_calc = indices_df[funds]
bench_df_calc['Encaisse'] = 0
bench_df_calc.index = indices_df.Période

rendement_mandat = pd.DataFrame(index=rendements_df.Période)
rendement_bench = pd.DataFrame(index=indices_df.Période)
profiles = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
for profile in profiles:
    allo = allocation_df_prep(profile, allocation_profil, rendements_df)
    rendement_mandat[profile] = (calculate_portfolio_returns(allo, returns_df_calc))

print(rendement_mandat)

rendement_mandat.to_excel("C:/Users/DTQ8851/Desktop/Performance Code/rendement_mandats.xlsx")

standard_devs = pd.DataFrame(columns = ['Années', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'])

#1,3,5,10
years = [1,3,5,10]
standard_devs['Années'] = years

for year in years:
    dummy_df = rendement_mandat.tail(year*12)
    for profile in profiles:
        standard_devs.loc[standard_devs['Années']== year, profile] = dummy_df[profile].std()*((12)**0.5)

standard_devs.to_excel("C:/Users/DTQ8851/Desktop/Performance Code/standard_dev.xlsx")
print(standard_devs)