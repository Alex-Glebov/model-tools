'''
Created on 11 Oct 2025

@author: alex
'''
import numpy as np
#import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
#from select_units import select_units

import logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

#from plot_tools import plot_Vert, plot_volume, plot_prices
#import zigzag.core as zig

def plot_volume(ax:Axes,df:pd.DataFrame,ticker='volume',colour='brown'):
  ax.set_ylim(top=3*max(df[ticker]))
  widtharray= (df.index - df['timestamp0']) if 'timestamp0' in df.columns else df.index.iloc[1]-df.index.iloc[0] 
  ax.bar(x=df.index, height= df[ticker], color=colour, edgecolor='steelblue', width=widtharray, label=ticker)
  ax.set_ylabel(ticker) 
  
def plot_Vert(ax,separator:bool, color='green'):
  for i, idx in enumerate(separator):
    if isinstance(idx, bool):
      if idx == False:
        continue
      else:
        k=i
    else:
      k=idx
    ax.axvline(separator.index[k], color=color, linestyle="--", linewidth=1) 
  
def plot_prices(ax:Axes,df:pd.DataFrame, col:str|None|list=None)->None:
#  fig, ax = plt.subplots()
  colmns = col
  if col is None : # plot OHLCV 
    width = 0.6 * (df.index[1]-df.index[0])# candle body width
    for _, (idx, row) in enumerate(df.iterrows()):
      color = "green" if row["close"] >= row["open"] else "red"
      # Wick (high-low line)
      ax.plot([idx, idx], [row["low"], row["high"]], color=color, linewidth=1) # Candle body
      ax.add_patch(plt.Rectangle((idx - width / 2, min(row["open"], row["close"])), 
          width, 
          abs(row["close"] - row["open"]), 
          color=color, 
          alpha=0.8))
  else:
    if isinstance (col,str):
      colmns=[col]
    for side in [ 'market','side']:
      if side in df.columns:
        break
    for col in colmns:
      if col in df.columns:
#        prp = df[col].where(df[side].str.contains('b'), np.nan)
#        prs = df[col].where(df[side].str.contains('s'),np.nan)
        ax.plot(df.index,df[col], label= f'Price {col}' )
#        ax.scatter(df.index,df[col], label= f'Price {col}' )
#        ax.plot(df.index,prp, color='green',label= 'Price buy ' )
#        ax.plot(df.index,prs, color='red', label= 'Price sell' )
    ax.set_ylabel("Price")
  

#def plot_OHLCV(df:pd.DataFrame,price_column:str,gain=0.02,decline=-0.02, prices:str|None=None):
#  ticker_close=df[price_column]
#  df_size=len(ticker_close)
#  if 'timestamp' in df.columns:
#    df.set_index('timestamp', inplace=True) # Create a copy of your DataFrame with converted timestamps
#  title_timeframe, major_locator, minor_locator, major_fmt = select_units(df)
#  logger.info(f"database has {df_size} records")
#  ind = zig.peak_valley_pivots(X=ticker_close, up_thresh=gain, down_thresh=-math.fabs(decline))
#  ind[0] = 0
#  ind[-1] = 0
#  PeakValley = pd.Series(ind, index=df.index, name='PeakValley')
#  df['PeakValley'] = PeakValley
#  entr_ideal = PeakValley == -1
##  entr_ideal1 = PeakValley.shift(-1) == -1
#  exts_ideal = PeakValley == 1
##  exts_ideal1 = PeakValley.shift(-1) == 1
#  logger.info(f"Teoretical entr {entr_ideal.sum()}")
#  logger.info(f"Teoretical exts {exts_ideal.sum()}") # setup function call
## calculate volume components
##  calcVolumes(df=ticker_1m)
##  calcPriceVolumes(df=df)
##  df=calcVolumesML(df=df,fields_names= fields_names_kraken)
## Convert masks to integer positions
##  start_idx = df.index[entr_ideal].to_numpy()
##  stop_idx = df.index[exts_ideal].to_numpy() # Combine them (True when start OR stop)
##  reset_mask = entr_ideal | exts_ideal # Create running sum that resets on True
## Pick some random indices for vertical bars
##use date range
##  date_from = df.index[0] # df['timestamp'].iloc[0]   
##  date_to = df.index[-1] #df['timestamp'].iloc[-1] #filter df 
##  df.where((date_from <= df.index) & (df.index <= date_to)) #  df.dropna(how='all',inplace=True)
##  title_timeframe, major_locator, minor_locator, major_fmt = select_units(df)
#  selected_peaks = df['PeakValley'] == 1
#  selected_valleys = df['PeakValley'] == -1 # Calculate time difference and determine best timeframe
#  fig, ax1 = plt.subplots()
#  ax1.xaxis.set_major_locator(major_locator)
#  ax1.xaxis.set_minor_locator(minor_locator)
#  ax1.xaxis.set_major_formatter(major_fmt) # Rotate dates for better readability
#  plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
#  ax2 = ax1.twinx()
#  plt.title(f"Exits - {title_timeframe}")
#  plt.xlabel("Date")
#  # Set major and minor ticks
## Optional: Improve grid
#  ax1.grid(True, linestyle="--", alpha=0.4, which='major')
#  ax1.grid(True, linestyle=":", alpha=0.2, which='minor')
##  df[['side','type']].head(20)
#  plot_Vert(ax1, selected_valleys, color='green')
#  plot_Vert(ax1, selected_peaks, color='red')
#  plot_prices_OMWC(ax1, df, col=None) 
##  plot_prices(ax1, df, col=prices) 
##t_volume(ax2, df,ticker='Vol+',colour='#907F00')
##  plot_volume(ax2, df,ticker='Vol-',colour='#90FF90')
##  plot_volume(ax2, df,ticker='vSm',colour='#FF9000')
##  plot_volume(ax2, df,ticker='vSl',colour='#FF9090')
#  trendvolfield = 'volume'
##  plot_volume(ax2, df, ticker=trendvolfield, colour='red') #  plot_volume(ax2, df,ticker='vBm',colour='blue')
##  plot_volume(ax2, df, ticker='vSm',colour='blue')
##  ax2.plot(df.index, pd.Series(0, index=df.index), color='black') #  plot_volume(ax2, df,ticker='Vol+',colour='blue')
##  plot_volume(ax2, df,ticker='Vol-',colour='brown')
#  plt.tight_layout() #  fig.autofmt_xdate()
#  plt.show()
#  pass
#  return
def plot_peaks(ax:Axes,df:pd.DataFrame,colours=["green","red"], col_name='peaks')->Axes:
#  _valls = (df[col_name] == -1).index # Calculate time difference and determine best timeframe
#  _peaks = (df[col_name] ==  1).index.to_pydatetime()
  _p = df[col_name] >  0
  _v = df[col_name] <  0
  _peaks = pd.to_datetime(_p[_p].index)
  _valls = pd.to_datetime(_v[_v].index)
#  plot_Vert(ax, _valls, color=colours[0])
#  plot_Vert(ax, _peaks, color=colours[1])
  ymin, ymax = ax.get_ylim()
  ax.vlines(
      x=_peaks,              # positions (datetime objects or floats)
      ymin=ymin , #df['price'].min(),     # bottom of lines
      ymax=ymax,  #df['price'].max(),     # top of lines (or any fixed values)
      colors=colours[1],               # single color, or list of colors
      linestyles='--',            # '--', ':', '-' etc.
      linewidth=1.2,
      alpha=0.7,
      label='Sell'
  )
  ax.vlines(
      x=_valls,              # positions (datetime objects or floats)
      ymin=ymin , #df['price'].min(),     # bottom of lines
      ymax=ymax,  #df['price'].max(),     # top of lines (or any fixed values)
      colors=colours[0],               # single color, or list of colors
      linestyles='--',            # '--', ':', '-' etc.
      linewidth=1.2,
      alpha=0.7,
      label='Buy'
  )
#  peak_line =f"{col_name}_line"
#  if peak_line in df.columns:
#    ax.plot(df.index,df[peak_line],color=colours[0])
  return ax
 
def plot_prices_OMWC (ax:Axes,df:pd.DataFrame,col:str|None|list=None, shift=0, colours=["green","red"])->None:
  if col is None : # plot OHLCV 
    v_start='open'
    v_stop = 'close'
    v_high = 'weight'
    v_low  = 'mean'
    use_timestamp= 'timestamp0' in df.columns
    idx0 = df.index[0] -(df.index[-1]-df.index[0])/len(df.index)
    for i, (idx, row) in enumerate(df.iterrows()):
      idx0=df['timestamp0'].iloc[i] if use_timestamp else idx0 
      #idx-width_average 
      width = 1 * (idx-idx0)# candle body width
      color = colours[0] if row[v_stop] >= row[v_start] else colours[1]
#      color = colours[0] if row[v_stop] >= row[v_high] else colours[1]
      # Wick (high-low line)
      x_plot=idx0 + shift*width/2.0
      ax.plot([x_plot,x_plot], [row[v_start], row[v_high]], color=color, linewidth=1) # open spike
      x_plot=idx  + shift*width/2.0
      ax.plot([x_plot,x_plot], [row[v_stop] , row[v_high]], color=color, linewidth=3) # close spike
      
      ax.add_patch(plt.Rectangle(((idx0 + shift * width / 2.0 ), min(row[v_high], row[v_low])), 
        width,                            # time step 
          abs(row[v_high] - row[v_low]),  # delta height
          color=color, 
          alpha=0.8))
      idx0=idx  # pr3ep for next step 
    plt.legend()
  else:
    pass
#    if isinstance (col,str):
#      colmns=[col]
#    for col in colmns:
#      if col in df.columns:
#        ax.plot(df.index,df[col], label= f'Price {col}' )
#    ax.set_ylabel("Price")
