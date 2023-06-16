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

st.write("Bonjour Hugo")