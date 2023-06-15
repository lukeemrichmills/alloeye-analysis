import pandas as pd
import matplotlib.pyplot as plt
from src.d01_data.fetch.fetch_timepoints import fetch_timepoints
from src.d00_utils.Conditions import Conditions


df = fetch_timepoints(pid='25', conditions=Conditions.all, blocks=1, trials=0, viewing_type='enc')
print(df.head(10))
