import pandas as pd
import streamlit as st
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting
import matplotlib.pyplot as plt
import copy 
from datetime import datetime, timedelta
import requests

def get_stock_portfolio_prices(portfolio, key):
    # Define the start and end dates (10 years ago from today)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3653)).strftime('%Y-%m-%d')

    dfs = []
    for stock in portfolio:
        dum = get_stock_prices_daily(stock, key)
        dum = dum[['date', 'close']]
        dum.columns = ['date', stock]
        dum.set_index('date', inplace=True)
        dfs.append(dum)
    merged_df = pd.concat(dfs, axis=1)

    merged_df.to_csv("tester_data.csv")
    return merged_df

def get_stock_prices_daily(stock, key):
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=3653)).strftime('%Y-%m-%d')
    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{stock}?from={start_date}&to={end_date}&apikey={key}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data['historical'])
    return df

def plot_efficient_frontier_and_max_sharpe(mu, S): 
	# Optimize portfolio for max Sharpe ratio and plot it out with efficient frontier curve
	ef = EfficientFrontier(mu, S)
	fig, ax = plt.subplots(figsize=(6,4))
	ef_max_sharpe = copy.deepcopy(ef)
	plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)
	# Find the max sharpe portfolio
	ef_max_sharpe.max_sharpe(risk_free_rate=0)
	ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
	ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")
	# Generate random portfolios
	n_samples = 1000
	w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
	rets = w.dot(ef.expected_returns)
	stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
	sharpes = rets / stds
	ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")
	# Output
	ax.legend()
	return fig

def plot_cov_matrix(S):
	fig, ax = plt.subplots()
	plotting.plot_covariance(S,plot_correlation=True, ax=ax)
	#plt.title("Covariance Matrix")
	plt.savefig("media/COV.png")


key = "d8eabf9ca1dec61aceefd4b4a9b93992"

#Page Titles

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

    header_col.markdown("<h3 style='text-align: center;'> Example des données de risque </h3>", unsafe_allow_html=True)


start_date = st.sidebar.date_input("Date de Debut: ")
end_date = st.sidebar.date_input("Date de fin: ")
owndata = st.sidebar.checkbox("Veux tu utilisé ton propre data? ")
#key = st.sidebar.text_input("API KEY: ")


if owndata:
    try:
        returns = st.session_state['funddata']
        mu = expected_returns.mean_historical_return(returns,returns_data=True, frequency=12)
        S = risk_models.sample_cov(returns,returns_data=True, frequency=12)	
    except:
         st.warning("please insert your data on the app tab")
         
else:
      tickers_string = st.sidebar.text_input('Enter all stock tickers to be included in portfolio separated by commas \
 WITHOUT spaces, e.g. "MA,FB,V,AMZN,JPM,BA"', '').upper()
      tickers = tickers_string.split(',')
      print("here", tickers)
      if len(tickers)>=0:
        df = get_stock_portfolio_prices(tickers, key)
        df = df.loc[::-1]
        mu = expected_returns.mean_historical_return(df)
        print(mu)
        S = risk_models.sample_cov(df)
        print(S)


fig = plot_efficient_frontier_and_max_sharpe(mu, S)
fig.savefig("media/EF.png")

plot_cov_matrix(S)


# Get optimized weights
# Optimize for maximal Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
ef.portfolio_performance(verbose=True)
print(ef)
expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance()
weights_df = pd.DataFrame.from_dict(weights, orient = 'index')
weights_df.columns = ['weights']



col1, col2 = st.columns(2)
with col1:
    st.subheader("Optimized Max Sharpe Portfolio Weights")
    st.dataframe(weights_df)
    
with col2:
    st.subheader('Expected annual return: {}%'.format((expected_annual_return*100).round(2)))
    st.subheader('Annual volatility: {}%'.format((annual_volatility*100).round(2)))
    st.subheader('Sharpe Ratio: {}'.format(sharpe_ratio.round(2)))
        
col3, col4 = st.columns(2)
with col3:
    st.subheader("Optimized Max Sharpe Portfolio Performance")
    st.image("media/EF.png")

with col4:
    st.subheader("Correlation Matrix")
    st.image("media/COV.png")

st.warning("please insert your tickers or select the checkbox")

