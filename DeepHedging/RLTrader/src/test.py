import datetime

import pandas as pd
import yfinance as yf

now = datetime.datetime.now()

start_date = "2010-01-01"
end_date = "2011-01-01"

data = yf.download('AAPL', start=start_date, end=end_date)
print(type(data))
print(data.columns)
print(data.shape)
print(data[:5])

data.to_csv("./AAPL_data.csv")