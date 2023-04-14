
import datetime as dt
import json
import requests
import pandas as pd

def datacollection():

    url = 'https://api.binance.com/api/v3/klines'
    symbol = 'ADAUSDT'
    interval = '1d'
    start = str(int(dt.datetime(2022,3,1).timestamp()*1000))
    end = str(int(dt.datetime(2022,11,1).timestamp()*1000))
    par = {'symbol': symbol, 'interval': interval, 'startTime': start, 'endTime': end}
    data = pd.DataFrame(json.loads(requests.get(url, params= par).text))
    #format columns name
    data.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume','close_time', 'qav', 'num_trades','taker_base_vol', 'taker_quote_vol', 'ignore']
    data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.datetime]
    data=data.astype(float)

    return data