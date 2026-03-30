import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
import os
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import urllib3

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 1. 路径与配置 ---
st.set_page_config(page_title="S&P 500 Trading timing system", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if __file__ else "."
SNP500_FILE = os.path.join(BASE_DIR, "snp500.csv")
COMPONENTS_PARQUET = os.path.join(BASE_DIR, "sp500_master.parquet")
GLOBAL_PARQUET = os.path.join(BASE_DIR, "global_indices_master.parquet")
METRICS_CACHE_FILE = os.path.join(BASE_DIR, "market_calculated_metrics.csv")
SMH_XLP_CACHE_FILE = os.path.join(BASE_DIR, "smh_xlp_cache.csv")
START_DISPLAY_DATE = "2020-01-01"

# 全球股指 Tickers
GLOBAL_TICKERS = [
    "^GSPC", "^GDAXI", "^STOXX50E", "^FCHI", "^FTSE", "^N225", "^KS11", "^STI", 
    "^NSEI", "^BSESN", "^VNINDEX", "^AXJO", "^SPTSX60", "^GSPTSE", "^BVSP", 
    "^SSMI", "^HSI", "000300.SS", "^TWII", "^AEX", "^OMX", "^OMXC20", 
    "FTSEMIB.MI", "^IBEX", "^TASI", "^JN0U.JO", "^OMXH25", "^BFX", "^OSEAX"
]

# ================= 2. 数据处理引擎 =================

def fix_ticker(symbol):
    return symbol.replace('.', '-')

def get_cboe_data(ticker):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{ticker}_History.csv"
    cache_file = os.path.join(BASE_DIR, f"{ticker}_History.csv")
    df = None
    
    try:
        response = requests.get(url, headers=headers, timeout=20, verify=False)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text))
            df.to_csv(cache_file, index=False)
        else:
            if os.path.exists(cache_file):
                df = pd.read_csv(cache_file)
    except:
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file)
            
    if df is not None and not df.empty:
        cols = df.columns.tolist()
        date_col = next((c for c in cols if c.upper() == 'DATE'), 'DATE')
        val_col = next((c for c in cols if c.upper() == 'CLOSE' or c.upper() == ticker.upper()), cols[-1])
        res = df[[date_col, val_col]].rename(columns={date_col: 'date', val_col: ticker.lower()})
        res['date'] = pd.to_datetime(res['date']).dt.tz_localize(None)
        return res
    return None

def calculate_breadth_and_lows(df_prices, prefix=""):
    if df_prices.empty: return pd.DataFrame(), 0
    df = df_prices.dropna(axis=1, how='all')
    count = df.shape[1]
    if count == 0: return pd.DataFrame(), 0
    period = 50 if prefix == "sp_" else 60
    ma_core = df.rolling(period, min_periods=10).mean()
    breadth_core = (df > ma_core).sum(axis=1) / count * 100
    low3m = df.rolling(63, min_periods=10).min()
    new_low = (df <= low3m).sum(axis=1) / count * 100
    
    res_dict = {f'{prefix}breadth': breadth_core, f'{prefix}new_low': new_low}
    
    if prefix == "sp_":
        ma200 = df.rolling(200, min_periods=50).mean()
        breadth_200 = (df > ma200).sum(axis=1) / count * 100
        res_dict['sp_breadth_200'] = breadth_200
        
    res = pd.DataFrame(res_dict, index=df.index)
    return res, count

def robust_sync_and_calc(msg_container):
    raw_symbols = pd.read_csv(SNP500_FILE)['Symbol'].tolist()
    sp_symbols = [fix_ticker(s) for s in raw_symbols]
    
    def fetch_incremental(tickers, label, parquet_file):
        base_df = pd.DataFrame()
        start_date = "2008-01-01" 
        
        if os.path.exists(parquet_file):
            base_df = pd.read_parquet(parquet_file)
            if not base_df.empty:
                if base_df.index.min() > pd.to_datetime('2008-06-01'):
                    base_df = pd.DataFrame()
                else:
                    start_date = (base_df.index.max() - timedelta(days=5)).strftime('%Y-%m-%d')
                    
        msg_container.info(f"正在抓取 {label} 价格数据 (起点: {start_date})...")
        df_res = pd.DataFrame()
        batch_size = 40
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            try:
                data = yf.download(batch, start=start_date, progress=False, auto_adjust=True)
                if data.empty: continue
                prices = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
                if not isinstance(data.columns, pd.MultiIndex): prices.columns = batch
                df_res = pd.concat([df_res, prices], axis=1)
            except: continue
            
        if not df_res.empty:
            df_res.index = pd.to_datetime(df_res.index).tz_localize(None)
            if not base_df.empty:
                merged = pd.concat([base_df, df_res])
                merged = merged[~merged.index.duplicated(keep='last')].sort_index()
                return merged
            return df_res
        return base_df

    sp_prices = fetch_incremental(sp_symbols, "S&P 500 成份股", COMPONENTS_PARQUET)
    global_prices = fetch_incremental(GLOBAL_TICKERS, "全球主要股指", GLOBAL_PARQUET)
    
    sp_metrics, sp_count = calculate_breadth_and_lows(sp_prices, prefix="sp_")
    global_metrics, gl_count = calculate_breadth_and_lows(global_prices, prefix="gl_")
    
    if sp_metrics.empty and global_metrics.empty: return None, 0, 0
    all_metrics = pd.merge(sp_metrics, global_metrics, left_index=True, right_index=True, how='outer').sort_index()
    all_metrics.index.name = 'date'
    all_metrics.to_csv(METRICS_CACHE_FILE)
    
    if not sp_prices.empty: sp_prices.to_parquet(COMPONENTS_PARQUET)
    if not global_prices.empty: global_prices.to_parquet(GLOBAL_PARQUET)
    return all_metrics.reset_index(), sp_count, gl_count

def fetch_fred(series_id):
    cache_file = os.path.join(BASE_DIR, f"{series_id}.csv")
    
    def process_df(df_in):
        df = df_in.copy()
        col_map = {c: c.upper() for c in df.columns}
        df = df.rename(columns=col_map)
        
        date_col = 'DATE' if 'DATE' in df.columns else None
        val_col = series_id.upper() if series_id.upper() in df.columns else None
        
        if date_col and val_col:
            df['date'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
            df[series_id.lower()] = pd.to_numeric(df[val_col], errors='coerce')
            return df[['date', series_id.lower()]].dropna(subset=[series_id.lower()])
        return pd.DataFrame()

    if os.path.exists(cache_file):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df = pd.read_csv(cache_file)
                res = process_df(df)
                if not res.empty: return res
        except: pass

    try:
        import pandas_datareader.data as web
        df = web.DataReader(series_id, 'fred', "1994-01-01", datetime.now())
        if not df.empty:
            df = df.reset_index()
            df = df.rename(columns={'DATE': 'date', series_id: series_id.lower()})
            df.to_csv(cache_file, index=False)
            return df[['date', series_id.lower()]]
    except: pass
    
    try:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=15)
        if res.status_code == 200 and series_id in res.text:
            df = pd.read_csv(StringIO(res.text))
            processed = process_df(df)
            if not processed.empty:
                df.to_csv(cache_file, index=False)
                return processed
    except: pass
    
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            return process_df(df)
        except: pass
        
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_spx_weekly_data():
    cache_file = os.path.join(BASE_DIR, "SPX_Weekly_Cache.csv")
    df_weekly = pd.DataFrame()
    
    if os.path.exists(cache_file):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df_weekly = pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
                if not df_weekly.empty: return df_weekly
        except: pass
        
    try:
        yf_data = yf.download('^GSPC', start='1927-12-01', progress=False)
        if not yf_data.empty:
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
            df_weekly = yf_data.resample('W-MON').agg({
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last'
            }).dropna()
            df_weekly.to_csv(cache_file)
            return df_weekly
    except: pass
    
    if os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_ndx_weekly_data():
    cache_file = os.path.join(BASE_DIR, "NDX_Weekly_Cache.csv")
    df_weekly = pd.DataFrame()
    
    if os.path.exists(cache_file):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df_weekly = pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
                if not df_weekly.empty: return df_weekly
        except: pass
        
    try:
        yf_data = yf.download('^NDX', start='1985-10-01', progress=False)
        if not yf_data.empty:
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
            df_weekly = yf_data.resample('W-MON').agg({
                'High': 'max', 
                'Low': 'min', 
                'Close': 'last'
            }).dropna()
            df_weekly.to_csv(cache_file)
            return df_weekly
    except: pass
    
    if os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_iwm_daily_data():
    cache_file = os.path.join(BASE_DIR, "IWM_Daily_Cache.csv")
    df_daily = pd.DataFrame()
    
    if os.path.exists(cache_file):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df_daily = pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
                if not df_daily.empty: return df_daily
        except: pass
        
    try:
        yf_data = yf.download('IWM', start='1990-01-01', progress=False)
        if not yf_data.empty:
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
            df_daily = yf_data[['Open', 'High', 'Low', 'Close']].dropna()
            df_daily.to_csv(cache_file)
            return df_daily
    except: pass
    
    if os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_dxy_daily_data():
    cache_file = os.path.join(BASE_DIR, "DXY_Daily_Cache.csv")
    df_daily = pd.DataFrame()
    
    if os.path.exists(cache_file):
        try:
            mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - mtime < timedelta(days=1):
                df_daily = pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
                if not df_daily.empty: return df_daily
        except: pass
        
    try:
        yf_data = yf.download('DX-Y.NYB', start='2006-01-01', progress=False)
        if not yf_data.empty:
            if isinstance(yf_data.columns, pd.MultiIndex):
                yf_data.columns = yf_data.columns.get_level_values(0)
            df_daily = yf_data[['Open', 'High', 'Low', 'Close']].dropna()
            df_daily.to_csv(cache_file)
            return df_daily
    except: pass
    
    if os.path.exists(cache_file):
        try:
            return pd.read_csv(cache_file, parse_dates=['Date']).set_index('Date')
        except: pass
    return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_g4_m2_data():
    """直接读取本地的 全球M2同比.csv 文件"""
    file_path = os.path.join(BASE_DIR, "全球M2同比.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            if 'Date' in df.columns and 'Value' in df.columns:
                df = df.rename(columns={'Date': 'date', 'Value': 'g4_m2_yoy'})
            else:
                df.columns = ['date', 'g4_m2_yoy']
                
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            df['g4_m2_yoy'] = pd.to_numeric(df['g4_m2_yoy'], errors='coerce')
            df = df.dropna().sort_values('date').reset_index(drop=True)
            return df[['date', 'g4_m2_yoy']]
        except Exception as e:
            pass
    return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def get_fci_data():
    """抓取四大核心市场波动率并计算综合金融条件指数 (FCI) 250日百分位总和"""
    df_vix = fetch_fred("VIXCLS")        # 美股波动率 VIX
    df_ovx = fetch_fred("OVXCLS")        # 原油波动率 OVX
    df_hy = fetch_fred("BAMLH0A0HYM2")   # 高收益债信用利差
    
    df_move = pd.DataFrame()
    move_file = os.path.join(BASE_DIR, "move.csv")
    if os.path.exists(move_file):
        try:
            move_raw = pd.read_csv(move_file)
            if 'Date' in move_raw.columns and 'Value' in move_raw.columns:
                move_raw = move_raw.rename(columns={'Date': 'date', 'Value': 'move'})
                move_raw['date'] = pd.to_datetime(move_raw['date']).dt.tz_localize(None)
                move_raw['move'] = pd.to_numeric(move_raw['move'], errors='coerce')
                df_move = move_raw[['date', 'move']].dropna().sort_values('date')
        except:
            pass

    # 合并到一个连续的交易日时间轴上
    dates = pd.date_range(start="2000-01-01", end=datetime.now(), freq='B')
    df_fci = pd.DataFrame({'date': dates})
    
    if not df_vix.empty: df_fci = pd.merge(df_fci, df_vix, on='date', how='left')
    if not df_ovx.empty: df_fci = pd.merge(df_fci, df_ovx, on='date', how='left')
    if not df_hy.empty: df_fci = pd.merge(df_fci, df_hy, on='date', how='left')
    if not df_move.empty: df_fci = pd.merge(df_fci, df_move, on='date', how='left')
    
    # 填充非交易日的缺失值
    df_fci = df_fci.ffill()
    
    cols_to_calc = [c for c in ['vixcls', 'ovxcls', 'bamlh0a0hym2', 'move'] if c in df_fci.columns]
    
    # 计算每个指标的 250 日滚动百分位数 (0-100)
    for col in cols_to_calc:
        df_fci[f'{col}_pct'] = df_fci[col].rolling(window=250, min_periods=100).apply(lambda x: (x <= x[-1]).mean() * 100, raw=True)
        
    pct_cols = [f'{col}_pct' for col in cols_to_calc]
    if pct_cols:
        df_fci['fci_score'] = df_fci[pct_cols].sum(axis=1)
        return df_fci[['date', 'fci_score']].dropna()
        
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_full_dataset():
    msg_box = st.empty()
    spy_df = yf.download("SPY", start="1994-01-01", progress=False, auto_adjust=True)
    spy = spy_df[['Close']].reset_index()
    spy.columns = ['date', 'spy']
    spy['date'] = pd.to_datetime(spy['date']).dt.tz_localize(None)
    df_merged = spy.copy()
    
    for t in ['COR1M', 'VIXEQ', 'VIX', 'VVIX', 'DSPX', 'SKEW']:
        df_t = get_cboe_data(t)
        if df_t is not None: df_merged = pd.merge(df_merged, df_t, on='date', how='left')
        
    try:
        dix_df = pd.read_csv("https://squeezemetrics.com/monitor/static/DIX.csv")
        dix_df['date'] = pd.to_datetime(dix_df['date']).dt.tz_localize(None)
        df_merged = pd.merge(df_merged, dix_df[['date', 'dix', 'gex']], on='date', how='left')
    except: pass

    try:
        vix_yf = yf.download("^VIX", start="1994-01-01", progress=False, auto_adjust=True)
        if not vix_yf.empty:
            v_high = vix_yf['High']
            if isinstance(v_high, pd.DataFrame): 
                v_high = v_high.iloc[:, 0]
            vh_df = pd.DataFrame({'date': vix_yf.index, 'vix_high': v_high.values})
            vh_df['date'] = pd.to_datetime(vh_df['date']).dt.tz_localize(None)
            df_merged = pd.merge(df_merged, vh_df, on='date', how='left')
    except:
        pass

    # === 全自动数据补齐机制：加载本地 PMI 和 NMI ===
    def _load_ism(filename, col_name):
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                d_col = next(c for c in df.columns if 'date' in c.lower())
                v_col = next(c for c in df.columns if c != d_col)
                df['date'] = pd.to_datetime(df[d_col]).dt.tz_localize(None)
                df = df.set_index('date').resample('ME').last()
                df[col_name] = df[v_col]
                return df[[col_name]].reset_index()
            except:
                pass
                
        if filename == "ISM_PMI.csv":
            try:
                url = "https://www.econdb.com/api/series/PMIUS/?format=csv"
                headers = {'User-Agent': 'Mozilla/5.0'}
                res = requests.get(url, headers=headers, timeout=10)
                if res.status_code == 200:
                    df = pd.read_csv(StringIO(res.text))
                    df.columns = ['date', col_name]
                    df['date'] = pd.to_datetime(df['date'])
                    df.to_csv(path, index=False)
                    df = df.set_index('date').resample('ME').last()
                    return df[[col_name]].reset_index()
            except:
                pass
                
        return pd.DataFrame()
        
    pmi_df = _load_ism("ISM_PMI.csv", "pmi")
    nmi_df = _load_ism("ISM_NMI.csv", "nmi")
    
    if not pmi_df.empty and not nmi_df.empty:
        ism_merged = pd.merge(pmi_df, nmi_df, on='date', how='outer').sort_values('date').ffill()
        ism_merged['pmi_nmi_sum'] = ism_merged['pmi'] + ism_merged['nmi']
        ism_merged['pmi_nmi_yoy'] = ism_merged['pmi_nmi_sum'].pct_change(12) * 100
        df_merged = pd.merge(df_merged, ism_merged[['date', 'pmi_nmi_yoy']], on='date', how='left')
    
    df_merged = df_merged.sort_values('date').ffill()
    
    try:
        if os.path.exists(SMH_XLP_CACHE_FILE):
            smh_xlp_df = pd.read_csv(SMH_XLP_CACHE_FILE)
            smh_xlp_df['date'] = pd.to_datetime(smh_xlp_df['date']).dt.tz_localize(None)
            if smh_xlp_df['date'].max().date() < datetime.now().date() - timedelta(days=2):
                raise ValueError("Cache outdated")
        else:
            raise ValueError("No cache")
    except:
        msg_box.info("正在抓取 SMH 和 XLP 数据...")
        sx_data = yf.download(['SMH', 'XLP'], start="2008-01-01", progress=False, auto_adjust=True)
        if not sx_data.empty:
            prices = sx_data['Close'] if isinstance(sx_data.columns, pd.MultiIndex) else sx_data
            smh_xlp_df = pd.DataFrame({'date': prices.index})
            smh_xlp_df['date'] = pd.to_datetime(smh_xlp_df['date']).dt.tz_localize(None)
            smh_xlp_df['smh_xlp_ratio'] = prices['SMH'].values / prices['XLP'].values
            smh_xlp_df['smh_xlp_yoy'] = smh_xlp_df['smh_xlp_ratio'].pct_change(252) * 100
            smh_xlp_df.to_csv(SMH_XLP_CACHE_FILE, index=False)
            
    if 'smh_xlp_df' in locals() and not smh_xlp_df.empty:
        df_merged = pd.merge(df_merged, smh_xlp_df[['date', 'smh_xlp_yoy']], on='date', how='left')

    try:
        msg_box.info("正在抓取 HYG 数据...")
        hyg_df = yf.download("HYG", start="1994-01-01", progress=False, auto_adjust=True)
        if not hyg_df.empty:
            prices = hyg_df['Close'] if isinstance(hyg_df.columns, pd.MultiIndex) else hyg_df[['Close']]
            if not isinstance(hyg_df.columns, pd.MultiIndex): prices.columns = ['HYG']
            hyg_res = pd.DataFrame({'date': prices.index, 'hyg': prices['HYG'].values})
            hyg_res['date'] = pd.to_datetime(hyg_res['date']).dt.tz_localize(None)
            df_merged = pd.merge(df_merged, hyg_res, on='date', how='left')
    except:
        pass
    
    metrics, sp_count, gl_count = None, 0, 0
    if os.path.exists(METRICS_CACHE_FILE):
        metrics = pd.read_csv(METRICS_CACHE_FILE)
        if 'date' not in metrics.columns and 'index' in metrics.columns: metrics.rename(columns={'index': 'date'}, inplace=True)
        if 'date' in metrics.columns:
            metrics['date'] = pd.to_datetime(metrics['date']).dt.tz_localize(None)
            if metrics['date'].max().date() < datetime.now().date() - timedelta(days=2):
                metrics, sp_count, gl_count = robust_sync_and_calc(msg_box)
            else:
                sp_count = pd.read_parquet(COMPONENTS_PARQUET).dropna(axis=1, how='all').shape[1] if os.path.exists(COMPONENTS_PARQUET) else 0
                gl_count = pd.read_parquet(GLOBAL_PARQUET).dropna(axis=1, how='all').shape[1] if os.path.exists(GLOBAL_PARQUET) else 0
        else: metrics, sp_count, gl_count = robust_sync_and_calc(msg_box)
    else: metrics, sp_count, gl_count = robust_sync_and_calc(msg_box)
    
    if metrics is not None and 'date' in metrics.columns:
        metrics['date'] = pd.to_datetime(metrics['date']).dt.tz_localize(None)
        df_merged = pd.merge(df_merged, metrics, on='date', how='left')

    df_dff = fetch_fred("DFF")
    df_dgs2 = fetch_fred("DGS2")
    df_unrate = fetch_fred("UNRATE")
    
    df_cape = pd.DataFrame()
    cape_file = os.path.join(BASE_DIR, "ie_data.xls")
    try:
        download_cape = True
        if os.path.exists(cape_file):
            mtime = datetime.fromtimestamp(os.path.getmtime(cape_file))
            if datetime.now() - mtime < timedelta(days=20):
                download_cape = False
        if download_cape:
            url = "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/3228b83a-7bad-4e69-b405-71e3a1ca6351/ie_data.xls?ver=1772737019751"
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=20)
            if res.status_code == 200:
                with open(cape_file, 'wb') as f:
                    f.write(res.content)
                    
        if os.path.exists(cape_file):
            c_df = pd.read_excel(cape_file, sheet_name='Data', skiprows=7, usecols=[0, 12], names=['raw_date', 'cape'], engine='xlrd')
            c_df = c_df.dropna(subset=['cape'])
            def parse_shiller(d):
                try:
                    s = f"{float(d):.2f}"
                    y, m = s.split('.')
                    if m == '1': m = '10'
                    return pd.to_datetime(f"{y}-{m}-01")
                except:
                    return pd.NaT
            c_df['date'] = c_df['raw_date'].apply(parse_shiller)
            df_cape = c_df.dropna(subset=['date'])[['date', 'cape']]
            df_cape['cape'] = pd.to_numeric(df_cape['cape'], errors='coerce')
    except:
        pass
        
    df_cuts = pd.DataFrame()
    rate_cut_file = os.path.join(BASE_DIR, "全球央行降息比例.csv")
    if os.path.exists(rate_cut_file):
        try:
            cuts = pd.read_csv(rate_cut_file)
            cuts['date'] = pd.to_datetime(cuts['Date']).dt.tz_localize(None)
            cuts['rate_cut_ratio'] = pd.to_numeric(cuts['Value'], errors='coerce')
            df_cuts = cuts[['date', 'rate_cut_ratio']]
        except:
            pass

    df_fwd_pe = pd.DataFrame()
    fwd_pe_file = os.path.join(BASE_DIR, "标普500远期市盈率.csv")
    if os.path.exists(fwd_pe_file):
        try:
            pe_df = pd.read_csv(fwd_pe_file)
            pe_df['date'] = pd.to_datetime(pe_df['Date']).dt.tz_localize(None)
            pe_df['forward_pe'] = pd.to_numeric(pe_df['Value'], errors='coerce')
            df_fwd_pe = pe_df[['date', 'forward_pe']]
        except:
            pass

    date_range = pd.DataFrame({'date': pd.date_range(start="1994-01-01", end=datetime.now())})
    df_macro = date_range.copy()
    if not df_cape.empty: df_macro = pd.merge(df_macro, df_cape, on='date', how='outer')
    if not df_fwd_pe.empty: df_macro = pd.merge(df_macro, df_fwd_pe, on='date', how='outer')
    if not df_dff.empty: df_macro = pd.merge(df_macro, df_dff, on='date', how='outer')
    if not df_dgs2.empty: df_macro = pd.merge(df_macro, df_dgs2, on='date', how='outer')
    if not df_unrate.empty: df_macro = pd.merge(df_macro, df_unrate, on='date', how='outer')
    if not df_cuts.empty: df_macro = pd.merge(df_macro, df_cuts, on='date', how='outer')
    if not pmi_df.empty: df_macro = pd.merge(df_macro, pmi_df[['date', 'pmi']], on='date', how='outer')
    
    # 融入新加入的 FCI 数据
    df_fci = get_fci_data()
    if not df_fci.empty: df_macro = pd.merge(df_macro, df_fci, on='date', how='outer')
    
    df_macro = df_macro.sort_values('date').ffill()
    df_merged = pd.merge(df_merged, df_macro, on='date', how='left')

    msg_box.empty()
    return df_merged.sort_values('date').dropna(subset=['spy']), sp_count, gl_count

# ================= 3. 信号逻辑计算 =================

def calculate_channel(df, col, k_sd, regime_split=None):
    if col not in df.columns: return None, None, None
    res, up, lo = pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float), pd.Series(index=df.index, dtype=float)
    def fit(idx):
        sub = df.loc[idx].dropna(subset=[col])
        if len(sub) < 20: return
        X = (sub['date'] - sub['date'].min()).dt.days.values.reshape(-1, 1)
        model = LinearRegression().fit(X, sub[col].values)
        trend = model.predict(X).flatten()
        std = (sub[col].values - trend).std()
        res.loc[sub.index], up.loc[sub.index], lo.loc[sub.index] = trend, trend + k_sd * std, trend - k_sd * std
    if regime_split:
        fit(df[df['date'] <= regime_split].index); fit(df[df['date'] > regime_split].index)
    else: fit(df.index)
    return res, up, lo

def get_signals(df):
    data = df.copy()
    
    data['spy_yoy'] = data['spy'].pct_change(252) * 100
    data['spy_roc_63'] = data['spy'].pct_change(63) * 100
    
    data['sig_buy_breadth200_20'] = 0
    if 'sp_breadth_200' in data.columns:
        prev = data['sp_breadth_200'].shift(1)
        curr = data['sp_breadth_200']
        cross_mask = (prev < 20) & (curr >= 20)
        
        last_idx = -100
        sig_arr = np.zeros(len(data), dtype=int)
        cross_vals = cross_mask.values
        for i in range(len(data)):
            if cross_vals[i]:
                if i - last_idx > 10:
                    sig_arr[i] = 1
                    last_idx = i
        data['sig_buy_breadth200_20'] = sig_arr

    if 'spy_roc_63' in data.columns:
        data['spy_roc_63_ma3'] = data['spy_roc_63'].rolling(3).mean()
        data['sig_buy_roc63'] = 0
        armed_roc = False
        last_roc_buy_idx = -100
        for i in range(len(data)):
            if pd.notna(data['spy_roc_63'].iloc[i]) and data['spy_roc_63'].iloc[i] < -9:
                armed_roc = True
            if armed_roc and pd.notna(data['spy_roc_63_ma3'].iloc[i]) and data['spy_roc_63'].iloc[i] > data['spy_roc_63_ma3'].iloc[i]:
                if i - last_roc_buy_idx > 5:
                    data.at[data.index[i], 'sig_buy_roc63'] = 1
                    last_roc_buy_idx = i
                armed_roc = False
    
    data['vix_spread'] = data['vixeq'] - data['vix']
    _, up_s, _ = calculate_channel(data, 'vix_spread', 1.9)
    data['spread_ma3'] = data['vix_spread'].rolling(3).mean()
    
    data['sig_reduce_lev'] = 0
    armed_v = False
    for i in range(len(data)):
        if data['vix_spread'].iloc[i] > up_s.iloc[i]: armed_v = True
        if armed_v and data['vix_spread'].iloc[i] < data['spread_ma3'].iloc[i]:
            data.at[data.index[i], 'sig_reduce_lev'] = 1; armed_v = False
            
    r_date = pd.Timestamp('2023-05-15')
    _, up_c, _ = calculate_channel(data, 'cor1m', 1.2, regime_split=r_date)
    _, up_c2, _ = calculate_channel(data[data['date']>r_date], 'cor1m', 1.4)
    if up_c is not None: up_c.update(up_c2)
    data['cor_ma3'] = data['cor1m'].rolling(3).mean()
    data['sig_buy_cor'] = 0
    armed_c = False
    for i in range(len(data)):
        if data['cor1m'].iloc[i] > up_c.iloc[i]: 
            armed_c = True
        if armed_c and i > 0:
            if data['cor1m'].iloc[i-1] >= data['cor_ma3'].iloc[i-1] and data['cor1m'].iloc[i] < data['cor_ma3'].iloc[i]:
                data.at[data.index[i], 'sig_buy_cor'] = 1
                armed_c = False
            
    data['sp_breadth_ma3'] = data['sp_breadth'].rolling(3).mean()
    data['sig_buy_sp_breadth'] = 0
    armed_sb = False
    for i in range(len(data)):
        if data['sp_breadth'].iloc[i] < 25: armed_sb = True
        if armed_sb and data['sp_breadth'].iloc[i] > data['sp_breadth_ma3'].iloc[i]:
            data.at[data.index[i], 'sig_buy_sp_breadth'] = 1; armed_sb = False
            
    data['sp_nl_ma3'] = data['sp_new_low'].rolling(3).mean()
    data['sp_nl_ma3_ma3'] = data['sp_nl_ma3'].rolling(3).mean()
    data['sig_buy_sp_nl'] = 0
    armed_sn = False
    for i in range(len(data)):
        if data['sp_nl_ma3'].iloc[i] > 24: armed_sn = True
        if armed_sn and data['sp_nl_ma3'].iloc[i] < data['sp_nl_ma3_ma3'].iloc[i]:
            data.at[data.index[i], 'sig_buy_sp_nl'] = 1; armed_sn = False
            
    data['gex_ma5'] = data['gex'].rolling(5).mean()
    data['gex_ma5_ma5'] = data['gex_ma5'].rolling(5).mean()
    data['sig_buy_gex'] = 0
    armed_g = False
    for i in range(len(data)):
        if data['gex_ma5'].iloc[i] < 0: armed_g = True
        if armed_g and data['gex_ma5'].iloc[i] > data['gex_ma5_ma5'].iloc[i] and data['gex_ma5'].shift(1).iloc[i] <= data['gex_ma5_ma5'].shift(1).iloc[i]:
            data.at[data.index[i], 'sig_buy_gex'] = 1; armed_g = False
            
    data['vvix_ma3'] = data['vvix'].rolling(3).mean()
    data['sig_buy_vvix'] = 0
    armed_vv = False
    for i in range(len(data)):
        if data['vvix'].iloc[i] > 140: armed_vv = True
        if armed_vv and data['vvix'].iloc[i] < data['vvix_ma3'].iloc[i]:
            data.at[data.index[i], 'sig_buy_vvix'] = 1; armed_vv = False
            
    data['vix_ma3'] = data['vix'].rolling(3).mean()
    data['sig_buy_vix'] = 0
    armed_vbuy = False
    for i in range(len(data)):
        if data['vix'].iloc[i] > 30: armed_vbuy = True
        if armed_vbuy and data['vix'].iloc[i] < data['vix_ma3'].iloc[i]:
            data.at[data.index[i], 'sig_buy_vix'] = 1; armed_vbuy = False
            
    if 'gl_breadth' in data.columns:
        ma3 = data['gl_breadth'].rolling(3).mean()
        data['sig_buy_gl_breadth60'] = 0
        armed_gb = False
        for i in range(len(data)):
            if data['gl_breadth'].iloc[i] < 15: armed_gb = True
            if armed_gb and data['gl_breadth'].iloc[i] > ma3.iloc[i]:
                data.at[data.index[i], 'sig_buy_gl_breadth60'] = 1; armed_gb = False
                
    if 'gl_new_low' in data.columns:
        data['gl_nl_ma3'] = data['gl_new_low'].rolling(3).mean()
        data['gl_nl_ma3_ma3'] = data['gl_nl_ma3'].rolling(3).mean()
        data['sig_buy_gl_nl'] = 0
        armed_gn = False
        for i in range(len(data)):
            if data['gl_nl_ma3'].iloc[i] > 25: armed_gn = True
            if armed_gn and data['gl_nl_ma3'].iloc[i] < data['gl_nl_ma3_ma3'].iloc[i]:
                data.at[data.index[i], 'sig_buy_gl_nl'] = 1; armed_gn = False

    up_d = pd.Series(index=data.index, dtype=float)
    data['sig_buy_dspx'] = 0
    data['sig_reduce_lev_dspx'] = 0
    if 'dspx' in data.columns:
        r1_start, r1_end = pd.Timestamp('2022-01-13'), pd.Timestamp('2023-12-07')
        r2_start, r2_end = pd.Timestamp('2023-12-08'), pd.Timestamp('2025-02-18')
        r3_start = pd.Timestamp('2025-02-19')
        
        _, up_d1, _ = calculate_channel(data[(data['date']>=r1_start) & (data['date']<=r1_end)], 'dspx', 1.5)
        _, up_d2, _ = calculate_channel(data[(data['date']>=r2_start) & (data['date']<=r2_end)], 'dspx', 1.5)
        _, up_d3, _ = calculate_channel(data[data['date']>=r3_start], 'dspx', 1.5)
        
        if up_d1 is not None: up_d.update(up_d1)
        if up_d2 is not None: up_d.update(up_d2)
        if up_d3 is not None: up_d.update(up_d3)
        
        data['dspx_up'] = up_d
        data['dspx_ma3'] = data['dspx'].rolling(3).mean()
        data['spy_ret_10'] = data['spy'].pct_change(10) * 100
        
        armed_d = False
        for i in range(len(data)):
            if pd.notna(data['dspx_up'].iloc[i]) and data['dspx'].iloc[i] > data['dspx_up'].iloc[i]:
                armed_d = True
            if armed_d and pd.notna(data['dspx_ma3'].iloc[i]) and data['dspx'].iloc[i] < data['dspx_ma3'].iloc[i]:
                ret = data['spy_ret_10'].iloc[i]
                if pd.notna(ret):
                    if ret < -1.5:
                        data.at[data.index[i], 'sig_buy_dspx'] = 1
                    elif ret > 1.5:
                        data.at[data.index[i], 'sig_reduce_lev_dspx'] = 1
                armed_d = False

    if 'smh_xlp_yoy' in data.columns:
        data['sig_buy_smh_xlp'] = 0
        data['sig_sell_smh_xlp'] = 0
        rolling_min = data['smh_xlp_yoy'].rolling(window=31, center=True, min_periods=1).min()
        rolling_max = data['smh_xlp_yoy'].rolling(window=31, center=True, min_periods=1).max()
        buy_cond = (data['smh_xlp_yoy'] < -12) & (data['smh_xlp_yoy'] == rolling_min)
        sell_cond = (data['smh_xlp_yoy'] > 60) & (data['smh_xlp_yoy'] == rolling_max)
        data.loc[buy_cond, 'sig_buy_smh_xlp'] = 1
        data.loc[sell_cond, 'sig_sell_smh_xlp'] = 1

    if 'skew' in data.columns:
        data['skew_3m_chg'] = data['skew'].pct_change(63) * 100
        data['skew_3m_chg_ma3'] = data['skew_3m_chg'].rolling(3).mean()
        data['spy_ma20'] = data['spy'].rolling(20).mean()
        data['sig_buy_skew'] = 0
        armed_skew = False
        for i in range(len(data)):
            if pd.notna(data['skew_3m_chg'].iloc[i]) and data['skew_3m_chg'].iloc[i] < -12:
                armed_skew = True
            if armed_skew and pd.notna(data['skew_3m_chg_ma3'].iloc[i]) and data['skew_3m_chg'].iloc[i] > data['skew_3m_chg_ma3'].iloc[i]:
                if pd.notna(data['spy'].iloc[i]) and pd.notna(data['spy_ma20'].iloc[i]) and data['spy'].iloc[i] < data['spy_ma20'].iloc[i]:
                    data.at[data.index[i], 'sig_buy_skew'] = 1
                armed_skew = False

    data['sig_buy_spy_yoy_0'] = 0
    if 'spy_yoy' in data.columns:
        prev_yoy = data['spy_yoy'].shift(1)
        curr_yoy = data['spy_yoy']
        cross_mask = (prev_yoy < 0) & (curr_yoy >= 0)
        
        last_idx = -100
        yoy_sigs = np.zeros(len(data), dtype=int)
        cross_arr = cross_mask.values
        for i in range(len(data)):
            if cross_arr[i]:
                if i - last_idx > 10:
                    yoy_sigs[i] = 1
                    last_idx = i
        data['sig_buy_spy_yoy_0'] = yoy_sigs

    data['sig_buy_vix_40'] = 0
    if 'vix_high' not in data.columns:
        data['vix_high'] = data['vix']
        
    armed_v40 = False
    last_vix_trigger = -100
    for i in range(len(data)):
        if (pd.notna(data['vix_high'].iloc[i]) and data['vix_high'].iloc[i] > 40) or (pd.notna(data['vix'].iloc[i]) and data['vix'].iloc[i] > 40):
            armed_v40 = True
        if armed_v40 and pd.notna(data['vix'].iloc[i]) and pd.notna(data['vix_ma3'].iloc[i]):
            if data['vix'].iloc[i] < data['vix_ma3'].iloc[i]:
                if i - last_vix_trigger > 10:
                    data.at[data.index[i], 'sig_buy_vix_40'] = 1
                    last_vix_trigger = i
                armed_v40 = False
                
    # === 新增：FCI 跌破 350 买入信号 (10天冷却) ===
    data['sig_buy_fci_350'] = 0
    if 'fci_score' in data.columns:
        prev_fci = data['fci_score'].shift(1)
        curr_fci = data['fci_score']
        cross_under_350 = (prev_fci >= 350) & (curr_fci < 350)
        
        last_idx = -100
        fci_sigs = np.zeros(len(data), dtype=int)
        cross_arr = cross_under_350.values
        for i in range(len(data)):
            if cross_arr[i]:
                if i - last_idx > 10:
                    fci_sigs[i] = 1
                    last_idx = i
        data['sig_buy_fci_350'] = fci_sigs

    if 'forward_pe' in data.columns:
        data['fwd_pe_ma3'] = data['forward_pe'].rolling(3).mean()
        
    if 'hyg' in data.columns:
        data['hyg_ma120'] = data['hyg'].rolling(120).mean()

    cooldown_cols_5d = [
        'sig_reduce_lev',         
        'sig_buy_cor',            
        'sig_buy_sp_breadth',     
        'sig_buy_vix',            
        'sig_buy_gl_breadth60',   
        'sig_buy_dspx',           
        'sig_reduce_lev_dspx',    
        'sig_buy_skew',           
        'sig_buy_gex'             
    ]
    for c in cooldown_cols_5d:
        if c in data.columns:
            arr = data[c].values
            res = np.zeros(len(arr), dtype=int)
            last_idx = -100
            for i in range(len(arr)):
                if arr[i] == 1:
                    if i - last_idx > 5:
                        res[i] = 1
                        last_idx = i
            data[c] = res

    return data, up_s, up_c, up_d

# ================= 4. UI 渲染引擎 =================

if "force_refresh" not in st.session_state:
    st.session_state.force_refresh = False

if st.session_state.force_refresh:
    load_full_dataset.clear()
    get_spx_weekly_data.clear() 
    get_ndx_weekly_data.clear() 
    get_iwm_daily_data.clear()
    get_dxy_daily_data.clear()
    get_g4_m2_data.clear()
    get_fci_data.clear()
    for cache_f in [METRICS_CACHE_FILE, COMPONENTS_PARQUET, GLOBAL_PARQUET, SMH_XLP_CACHE_FILE,
                    os.path.join(BASE_DIR, "SPX_Weekly_Cache.csv"), os.path.join(BASE_DIR, "NDX_Weekly_Cache.csv"),
                    os.path.join(BASE_DIR, "IWM_Daily_Cache.csv"), os.path.join(BASE_DIR, "DXY_Daily_Cache.csv"), 
                    os.path.join(BASE_DIR, "G4_M2_Cache.csv"), os.path.join(BASE_DIR, "G4_M2_Daily_Cache.csv")]:
        if os.path.exists(cache_f): os.remove(cache_f)
    st.session_state.force_refresh = False
    st.rerun()

df_raw, sp_count, gl_count = load_full_dataset()
latest_date_str = df_raw['date'].max().strftime('%Y-%m-%d')

c_title, c_btn = st.columns([5, 1])
with c_title:
    st.title(f"S&P 500 Trading timing system (Latest: {latest_date_str})")
with c_btn:
    st.write("")
    if st.button("🔄 强制全量刷新数据", use_container_width=True):
        st.session_state.force_refresh = True
        st.rerun()

df_final, up_s, up_c, up_d = get_signals(df_raw)

# ================= TAB 设计 =================
tab1, tab2 = st.tabs(["📊 中短期择时 (2020至今)", "📈 长周期择时 (2009至今)"])

# ----------------- Tab 1: 中短期择时 -----------------
with tab1:
    base_df = df_final[df_final['date'] >= START_DISPLAY_DATE].copy()
    base_df.loc[base_df['date'] < '2020-07-01', 'sig_buy_cor'] = 0

    if 'sig_buy_dspx' in base_df.columns and 'sig_reduce_lev_dspx' in base_df.columns:
        d_buy_arr = base_df['sig_buy_dspx'].values
        d_sell_arr = base_df['sig_reduce_lev_dspx'].values
        last_d_buy, last_d_sell = -100, -100
        for i in range(len(base_df)):
            is_b, is_s = d_buy_arr[i] == 1, d_sell_arr[i] == 1
            if is_b and is_s: d_sell_arr[i], is_s = 0, False
            if is_s and (i - last_d_buy <= 5): d_sell_arr[i], is_s = 0, False
            if is_b and (i - last_d_sell <= 5): d_buy_arr[i], is_b = 0, False
            if is_b: last_d_buy = i
            if is_s: last_d_sell = i
        base_df['sig_buy_dspx'], base_df['sig_reduce_lev_dspx'] = d_buy_arr, d_sell_arr

    buy_cols = ['sig_buy_cor', 'sig_buy_sp_breadth', 'sig_buy_sp_nl', 'sig_buy_gex', 'sig_buy_vvix', 'sig_buy_vix', 'sig_buy_gl_breadth60', 'sig_buy_gl_nl', 'sig_buy_dspx', 'sig_buy_skew']
    active_cols = [c for c in buy_cols if c in base_df.columns]
    base_df['daily_signal_count'] = base_df[active_cols].sum(axis=1)
    current_trigger = int(base_df['daily_signal_count'].iloc[-1])

    st.markdown(f'<div style="font-family: \'Open Sans\', verdana, arial, sans-serif; font-size: 22px; font-weight: bold; color: black; margin-top: 15px; margin-bottom: 10px;">1. SPY 汇总图 (当前触发: {current_trigger}个)</div>', unsafe_allow_html=True)
    
    st.markdown("##### 信号共振参数设置")
    c_p1, c_p2 = st.columns(2)
    with c_p1: lookback = st.slider("共振窗口 (Days)", 1, 5, 2, key="slider1")
    with c_p2: min_sigs = st.slider("最小共振指标数", 1, 10, 4, key="slider2") 
    time_range = st.radio("主图展示周期 (下方子图维持全量显示):", ["3m", "6m", "1y", "2y", "All"], index=4, horizontal=True)

    base_df['score'] = base_df[active_cols].rolling(lookback).sum().sum(axis=1)
    base_df['final_buy_raw'] = (base_df['score'] >= min_sigs).astype(int)
    reduce_vix = base_df['sig_reduce_lev'].fillna(0).astype(int)
    reduce_dspx = base_df.get('sig_reduce_lev_dspx', pd.Series(0, index=base_df.index)).fillna(0).astype(int)
    base_df['final_reduce_lev_raw'] = (reduce_vix | reduce_dspx).astype(int)

    final_buy_arr = base_df['final_buy_raw'].values
    final_sell_arr = base_df['final_reduce_lev_raw'].values
    last_buy_idx, last_sell_idx = -100, -100
    for i in range(len(base_df)):
        is_buy, is_sell = final_buy_arr[i] == 1, final_sell_arr[i] == 1
        if is_buy and is_sell: final_sell_arr[i], is_sell = 0, False
        if is_buy and (i - last_sell_idx <= 10): final_buy_arr[i], is_buy = 0, False
        if is_sell and (i - last_buy_idx <= 10): final_sell_arr[i], is_sell = 0, False
        if is_buy: last_buy_idx = i
        if is_sell: last_sell_idx = i
    base_df['final_buy'], base_df['final_reduce_lev'] = final_buy_arr, final_sell_arr

    latest_date_dt = base_df['date'].max()
    x_max_padded = latest_date_dt + pd.Timedelta(days=50) 
    if time_range == "3m": start_limit = latest_date_dt - pd.DateOffset(months=3)
    elif time_range == "6m": start_limit = latest_date_dt - pd.DateOffset(months=6)
    elif time_range == "1y": start_limit = latest_date_dt - pd.DateOffset(years=1)
    elif time_range == "2y": start_limit = latest_date_dt - pd.DateOffset(years=2)
    else: start_limit = pd.to_datetime(START_DISPLAY_DATE)

    plot_df = base_df[base_df['date'] >= start_limit].copy()
    spy_range = [np.log10(plot_df['spy'].min() * 0.97), np.log10(plot_df['spy'].max() * 1.03)]

    fig_main = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01, row_heights=[0.7, 0.3], specs=[[{"secondary_y": True}], [{"secondary_y": False}]])
    shading_configs = [(1, 2, "rgba(255, 255, 255, 0)", "1-2个信号"), (3, 4, "rgba(255, 165, 0, 0.95)", "3-4个信号"), (5, 6, "rgba(255, 0, 0, 0.95)", "5-6个信号"), (7, 9, "rgba(160, 32, 240, 0.95)", "7-9个信号")]
    for low, high, color, label in shading_configs:
        mask = (plot_df['daily_signal_count'] >= low) & (plot_df['daily_signal_count'] <= high)
        y_vals = np.where(mask, 1, 0)
        fig_main.add_trace(go.Bar(x=plot_df['date'], y=y_vals, name=label, marker_color=color, showlegend=True, hoverinfo='skip', offset=0, base=0, width=1000*3600*24, marker_line_width=0), row=1, col=1, secondary_y=False)

    fig_main.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['spy'], name="SPY Price", line=dict(color='black', width=4)), row=1, col=1, secondary_y=True)
    buys = plot_df[plot_df['final_buy']==1]
    if not buys.empty: fig_main.add_trace(go.Scatter(x=buys['date'], y=buys['spy'] * 0.97, mode='markers', marker=dict(symbol='triangle-up', size=18, color='red'), name="买入共振"), row=1, col=1, secondary_y=True)
    sells = plot_df[plot_df['final_reduce_lev']==1]
    if not sells.empty: fig_main.add_trace(go.Scatter(x=sells['date'], y=sells['spy'] * 1.03, mode='markers', marker=dict(symbol='triangle-down', size=18, color='green'), name="降低杠杆"), row=1, col=1, secondary_y=True)

    fig_main.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['daily_signal_count'], fill='tozeroy', mode='none', fillcolor='rgba(220, 20, 60, 0.8)', name="信号强度"), row=2, col=1)
    fig_main.add_hline(y=3, line_dash="dash", line_color="black", line_width=1, opacity=0.5, row=2, col=1)

    fig_main.update_layout(
        template="plotly_white", height=800, showlegend=True, hovermode="x unified", bargap=0, 
        margin=dict(l=65, r=150, t=20, b=50, autoexpand=False), 
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.025)
    )
    fig_main.update_xaxes(range=[start_limit, x_max_padded], showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")
    fig_main.update_yaxes(showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")
    fig_main.update_yaxes(row=1, col=1, range=[0, 1], visible=False, secondary_y=False)
    fig_main.update_yaxes(row=1, col=1, type="log", title="Price (Log Scale)", range=spy_range, side="left", secondary_y=True)
    fig_main.update_yaxes(row=2, col=1, title="Count (0-9)", range=[0, 9.5])
    st.plotly_chart(fig_main, use_container_width=True)

    def create_standalone_chart(title, y_col, y_name, line_color, sig_col=None, sig_name=None, hline=None, hline_color=None, extra_line_col=None, extra_line_name=None, extra_line_dash='dash', show_zero=False, lgroup=None, indicator_opacity=1.0, ma3_col=None, ma3_color='gray', ma3_dash='dot', y_range=None, main_visible=True, ma3_of_ma3_col=None, ma3_width=1):
        if y_col not in base_df.columns: return None
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=base_df['date'], y=base_df[y_col], name=y_name, line=dict(color=line_color, width=2), opacity=indicator_opacity, legendgroup=lgroup, visible=main_visible), secondary_y=False)
        if extra_line_col is not None: fig.add_trace(go.Scatter(x=base_df['date'], y=extra_line_col.loc[base_df.index], name=extra_line_name, line=dict(dash=extra_line_dash, color=hline_color if hline_color else 'red', width=1), opacity=indicator_opacity, legendgroup=lgroup), secondary_y=False)
        if ma3_col is not None: fig.add_trace(go.Scatter(x=base_df['date'], y=ma3_col.loc[base_df.index], name="MA3 Line", line=dict(dash=ma3_dash, color=ma3_color, width=ma3_width), opacity=indicator_opacity, legendgroup=lgroup), secondary_y=False)
        if ma3_of_ma3_col is not None: fig.add_trace(go.Scatter(x=base_df['date'], y=ma3_of_ma3_col.loc[base_df.index], name="MA3 of MA3", line=dict(dash='dot', color=line_color, width=1), opacity=indicator_opacity, legendgroup=lgroup), secondary_y=False)
        if hline is not None: fig.add_trace(go.Scatter(x=[base_df['date'].min(), base_df['date'].max()], y=[hline, hline], mode='lines', line=dict(dash='dash', width=1, color=hline_color if hline_color else "red"), name=f"{hline} 阈值", opacity=indicator_opacity, legendgroup=lgroup), secondary_y=False)
        if show_zero: fig.add_trace(go.Scatter(x=[base_df['date'].min(), base_df['date'].max()], y=[0, 0], mode='lines', line=dict(width=1, color="black"), name="0轴", opacity=indicator_opacity, legendgroup=lgroup), secondary_y=False)
        fig.add_trace(go.Scatter(x=base_df['date'], y=base_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        if sig_col and sig_col in base_df.columns:
            pts = base_df[base_df[sig_col]==1]
            if not pts.empty:
                is_buy = 'buy' in sig_col
                offset = 0.97 if is_buy else 1.03
                sym = 'triangle-up' if is_buy else 'triangle-down'
                clr = 'red' if is_buy else 'green'
                fig.add_trace(go.Scatter(x=pts['date'], y=pts['spy'] * offset, mode='markers', marker=dict(symbol=sym, size=14, color=clr, line=dict(width=1, color="white")), name=sig_name), secondary_y=True)
        
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), template="plotly_white", height=750, margin=dict(l=65, r=150, t=50, b=50, autoexpand=False), showlegend=True, hovermode="x unified", legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.025))
        fig.update_xaxes(range=[base_df['date'].min(), x_max_padded], showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")
        fig.update_yaxes(type="log", secondary_y=True, showgrid=False, zeroline=False)
        fig.update_yaxes(showgrid=False, secondary_y=False, zeroline=False)
        if y_range is not None: fig.update_yaxes(range=y_range, secondary_y=False)
        return fig

    charts_config = [
        ("2. VIX Spread 监控: Spread > 1.9SD -> 跌破 MA3 (降低杠杆触发) | 5天内不再触发", "vix_spread", "VIX Spread", "blue", 'sig_reduce_lev', "降低杠杆", None, "orange", up_s, "1.9SD Bound", 'dash', False, "vix_grp", 0.8, base_df['spread_ma3'] if 'spread_ma3' in base_df.columns else None, "gray", 'dot', [0, 45]),
        ("3. COR1M 买入: COR1M > Regime Upper -> 刚跌破 MA3 (触发一次) | 5天内不再触发", "cor1m", "COR1M", "purple", 'sig_buy_cor', "买入信号", None, "red", up_c, "Regime Upper", 'dash', False, "cor_grp", 0.8, base_df['cor_ma3'] if 'cor_ma3' in base_df.columns else None, "gray", 'dot', [0, 140]),
        (f"4. S&P 500 50日宽度: 跌破 25% -> 涨破 MA3 (采样: {sp_count}只) | 5天内不再触发", "sp_breadth", "50d Breadth", "navy", 'sig_buy_sp_breadth', "买入信号", 25, "red", None, None, 'dash', False, "spb_grp", 0.8, base_df['sp_breadth'].rolling(3).mean() if 'sp_breadth' in base_df.columns else None, "gray", 'dot', [0, 250]),
        (f"5. S&P 500 3月新低: MA3 > 24% -> 跌破 MA3的MA3 (采样: {sp_count}只)", "sp_new_low", "New Low Ratio", "brown", 'sig_buy_sp_nl', "买入信号", 24, "red", None, None, 'solid', False, None, 1.0, base_df['sp_nl_ma3'] if 'sp_nl_ma3' in base_df.columns else None, "orange", 'solid', [0, 100]),
        ("6. GEX 筹码买入: MA5 < 0 -> 涨破 MA5的MA5 (触发) | 5天内不再触发", "gex_ma5", "GEX MA5", "darkcyan", 'sig_buy_gex', "买入信号", None, "darkcyan", base_df['gex_ma5_ma5'] if 'gex_ma5_ma5' in base_df.columns else None, "Signal Line", 'dot', True, "gex_grp", 1.0, None, "gray", 'dot', [-5000000000, 40000000000]),
        ("7. VVIX 买入: VVIX > 140 -> 跌破 MA3 (触发)", "vvix", "VVIX Index", "darkorange", 'sig_buy_vvix', "买入信号", 140, "red", None, None, 'dash', False, "vvix_grp", 1.0, base_df['vvix_ma3'] if 'vvix_ma3' in base_df.columns else None, "magenta", 'dot', [70, 250]),
        ("8. VIX 买入: VIX > 30 -> 跌破 MA3 (触发) | 5天内不再触发", "vix", "VIX Index", "crimson", 'sig_buy_vix', "买入信号", 30, "red", None, None, 'dash', False, "vix_buy_grp", 1.0, base_df['vix_ma3'] if 'vix_ma3' in base_df.columns else None, "gray", 'dot', None),
        (f"9. 全球股指 60日宽度: 跌破 15% -> 涨破 MA3 (采样: {gl_count}只) | 5天内不再触发", "gl_breadth", "Global 60d Breadth", "DeepSkyBlue", 'sig_buy_gl_breadth60', "全球买入信号", 15, "Peru", None, None, 'dash', False, None, 0.8, base_df['gl_breadth'].rolling(3).mean() if 'gl_breadth' in base_df.columns else None, "dimgray", 'dot', [0, 250]),
        (f"10. 全球股指 3月新低: MA3 > 25% -> MA3 跌破 MA3的MA3 (采样: {gl_count}只)", "gl_new_low", "Global New Low Ratio", "darkred", 'sig_buy_gl_nl', "全球买入信号", 25, "red", None, None, 'solid', False, None, 1.0, base_df['gl_nl_ma3'] if 'gl_nl_ma3' in base_df.columns else None, "orange", 'solid', [0, 120])
    ]

    for cfg in charts_config:
        if cfg[1] in ["sp_new_low", "gl_new_low"]:
            c = create_standalone_chart(*cfg, main_visible="legendonly", ma3_of_ma3_col=base_df[f"{cfg[1][:3]}nl_ma3_ma3"] if f"{cfg[1][:3]}nl_ma3_ma3" in base_df.columns else None, ma3_width=3)
        else:
            c = create_standalone_chart(*cfg)
        if c: st.plotly_chart(c, use_container_width=True)

    if 'dspx' in base_df.columns:
        fig_dspx = create_standalone_chart(title="11. DSPX 隐含离散度: 突破上轨 -> 跌破 MA3 (结合SPY动量过滤) | 5天内不再触发", y_col="dspx", y_name="DSPX Index", line_color="blue", sig_col="sig_buy_dspx", sig_name="DSPX 底部买入信号", hline=None, hline_color="red", extra_line_col=up_d, extra_line_name="Regime 1.5SD Upper Bound", extra_line_dash='dash', show_zero=False, lgroup="dspx_grp", indicator_opacity=0.8, ma3_col=base_df['dspx_ma3'] if 'dspx_ma3' in base_df.columns else None, ma3_color="gray", ma3_dash='dot', y_range=[20, 70], main_visible=True, ma3_of_ma3_col=None)
        if fig_dspx and 'sig_reduce_lev_dspx' in base_df.columns:
            pts = base_df[base_df['sig_reduce_lev_dspx']==1]
            if not pts.empty: fig_dspx.add_trace(go.Scatter(x=pts['date'], y=pts['spy'] * 1.03, mode='markers', marker=dict(symbol='triangle-down', size=14, color='green', line=dict(width=1, color="white")), name="DSPX 顶部降杠杆"), secondary_y=True)
            st.plotly_chart(fig_dspx, use_container_width=True)

    if 'smh_xlp_yoy' in base_df.columns:
        fig_smh = create_standalone_chart(title="12. SMH/XLP YoY: 跌破 -12% 寻底买入 / 突破 +60% 寻顶降杠杆 (未来函数 ±15天)", y_col="smh_xlp_yoy", y_name="SMH/XLP YoY (%)", line_color="teal", sig_col="sig_buy_smh_xlp", sig_name="SMH/XLP 底部买入", hline=-12, hline_color="red", extra_line_col=pd.Series(60, index=base_df.index), extra_line_name="+60% 阈值", extra_line_dash='dash', show_zero=False, lgroup="smh_grp", indicator_opacity=0.8, ma3_col=None, ma3_color="gray", ma3_dash='dot', y_range=[-50, 250], main_visible=True, ma3_of_ma3_col=None)
        if fig_smh: st.plotly_chart(fig_smh, use_container_width=True)

    if 'skew_3m_chg' in base_df.columns:
        fig_skew = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_skew.add_trace(go.Scatter(x=base_df['date'], y=base_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        fig_skew.add_trace(go.Scatter(x=base_df['date'], y=base_df['skew_3m_chg'], name="SKEW 3M Chg (%)", line=dict(color="SkyBlue", width=2), opacity=0.8, legendgroup="skew_grp"), secondary_y=False)
        
        if 'skew_3m_chg_ma3' in base_df.columns:
            fig_skew.add_trace(go.Scatter(x=base_df['date'], y=base_df['skew_3m_chg_ma3'], name="SKEW MA3 Line", line=dict(color="SkyBlue", dash='dash', width=1), opacity=0.8, legendgroup="skew_grp"), secondary_y=False)
            
        fig_skew.add_trace(go.Scatter(x=[base_df['date'].min(), base_df['date'].max()], y=[-12, -12], mode='lines', line=dict(dash='dash', width=1, color="gray"), name="-12 阈值", opacity=0.8, legendgroup="skew_grp"), secondary_y=False)
        fig_skew.add_trace(go.Scatter(x=base_df['date'], y=base_df['spy_roc_63'], name="SPY 3-Month ROC (%)", line=dict(color="red", width=2)), secondary_y=False)
        
        if 'sig_buy_skew' in base_df.columns:
            pts = base_df[base_df['sig_buy_skew']==1]
            if not pts.empty:
                fig_skew.add_trace(go.Scatter(x=pts['date'], y=pts['spy'] * 0.97, mode='markers', marker=dict(symbol='triangle-up', size=14, color='red', line=dict(width=1, color="white")), name="SKEW 底部买入"), secondary_y=True)
        
        fig_skew.update_layout(
            title=dict(text="<b>13. SKEW 3M 跌幅极值: < -12% 且涨破自身 MA3 且 SPY 位于 MA20 之下 (买入) | 5天内不再触发</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
            template="plotly_white", 
            height=750, 
            margin=dict(l=65, r=150, t=50, b=50, autoexpand=False), 
            showlegend=True, 
            hovermode="x unified", 
            legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.025)
        )
        
        fig_skew.update_xaxes(range=[base_df['date'].min(), x_max_padded], showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")
        fig_skew.update_yaxes(showgrid=False, secondary_y=False, zeroline=False, range=[-30, 80])
        fig_skew.update_yaxes(type="log", secondary_y=True, showgrid=False, zeroline=False)
        
        st.plotly_chart(fig_skew, use_container_width=True)
        
    if 'hyg' in base_df.columns and 'hyg_ma120' in base_df.columns:
        valid_hyg = base_df[['hyg', 'hyg_ma120']].dropna()
        if not valid_hyg.empty:
            current_hyg = valid_hyg['hyg'].iloc[-1]
            current_ma120 = valid_hyg['hyg_ma120'].iloc[-1]
            state_str = "正常" if current_hyg >= current_ma120 else "信用预警 (HYG 跌破半年线)"
            
            fig_hyg = make_subplots(specs=[[{"secondary_y": True}]])
            
            x_warn, y_warn = [], []
            dates = base_df['date'].values
            hygs = base_df['hyg'].values
            ma120s = base_df['hyg_ma120'].values
            
            current_state = None
            start_date = None
            
            for d, h, m in zip(dates, hygs, ma120s):
                if pd.isna(h) or pd.isna(m): continue
                state = (h < m)
                if current_state is None:
                    current_state = state
                    start_date = d
                elif current_state != state:
                    if current_state: 
                        x_warn.extend([start_date, start_date, d, d, None])
                        y_warn.extend([0, 500, 500, 0, None])
                    current_state = state
                    start_date = d
            
            if start_date is not None and current_state:
                x_warn.extend([start_date, start_date, x_max_padded, x_max_padded, None])
                y_warn.extend([0, 500, 500, 0, None])
                
            fig_hyg.add_trace(go.Scatter(x=x_warn, y=y_warn, fill='toself', fillcolor="rgba(144, 238, 144, 0.2)", line=dict(width=0), mode='none', name="预警状态背景 (<MA120)", hoverinfo='skip'), secondary_y=False)
            
            fig_hyg.add_trace(go.Scatter(x=base_df['date'], y=base_df['hyg'], name="HYG Close", line=dict(color="blue", width=2)), secondary_y=False)
            fig_hyg.add_trace(go.Scatter(x=base_df['date'], y=base_df['hyg_ma120'], name="HYG MA120", line=dict(color="orange", width=2, dash='dash')), secondary_y=False)
            fig_hyg.add_trace(go.Scatter(x=base_df['date'], y=base_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
            
            fig_hyg.update_layout(title=dict(text=f"<b>14. 信用市场交叉验证: HYG vs SPY (当前状态: {state_str})</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), template="plotly_white", height=750, margin=dict(l=65, r=150, t=50, b=50, autoexpand=False), showlegend=True, hovermode="x unified", legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.025))
            fig_hyg.update_xaxes(range=[base_df['date'].min(), x_max_padded], showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")
            fig_hyg.update_yaxes(type="log", secondary_y=True, showgrid=False, zeroline=False)
            fig_hyg.update_yaxes(showgrid=False, secondary_y=False, zeroline=False)
            
            fig_hyg.update_yaxes(range=[50, 100], secondary_y=False)

            st.plotly_chart(fig_hyg, use_container_width=True)


# ----------------- Tab 2: 长周期择时 -----------------
with tab2:
    lt_df = df_final[df_final['date'] >= '2009-01-01'].copy()
    lt_x_max_padded = lt_df['date'].max() + pd.Timedelta(days=50)
    
    lt_df_1994 = df_final[df_final['date'] >= '1994-01-01'].copy()
    lt_x_max_padded_1994 = lt_df_1994['date'].max() + pd.Timedelta(days=50)
    
    cape_trend = None
    cape_up = None
    cape_lo = None
    if 'cape' in lt_df.columns:
        cape_trend, cape_up, cape_lo = calculate_channel(lt_df, 'cape', 1.4)
        lt_df['cape_lo'] = cape_lo
        sig_cape = np.zeros(len(lt_df), dtype=int)
        state = 0 
        for i in range(1, len(lt_df)):
            c = lt_df['cape'].iloc[i]
            c_prev = lt_df['cape'].iloc[i-1]
            lo = lt_df['cape_lo'].iloc[i]
            
            if pd.isna(c) or pd.isna(lo): continue
            
            if c >= lo:
                state = 0
            else:
                if state == 0:
                    state = 1
                if state == 1 and c > c_prev:
                    sig_cape[i] = 1
                    state = 2
        lt_df['sig_cape_buy'] = sig_cape

    layout_template = dict(
        template="plotly_white", height=750, showlegend=True, hovermode="x unified",
        margin=dict(l=65, r=120, t=50, b=50, autoexpand=False), 
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )
    axis_opts = dict(showgrid=False, zeroline=False, showspikes=True, spikemode="across", spikesnap="cursor", spikedash="dash", spikecolor="gray")

    # === 图1 长周期择时信号汇总 ===
    lt_sigs = ['sig_buy_breadth200_20', 'sig_buy_spy_yoy_0', 'sig_buy_roc63', 'sig_cape_buy', 'sig_buy_vix_40']
    active_lt_sigs = [col for col in lt_sigs if col in lt_df.columns]
    
    lt_df['lt_combined_buy'] = 0
    if active_lt_sigs:
        lt_df['lt_combined_buy'] = (lt_df[active_lt_sigs].sum(axis=1) > 0).astype(int)
        
    fig_lt0 = go.Figure()
    fig_lt0.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)))
    
    if 'lt_combined_buy' in lt_df.columns:
        pts_lt_combined = lt_df[lt_df['lt_combined_buy'] == 1]
        if not pts_lt_combined.empty:
            fig_lt0.add_trace(go.Scatter(
                x=pts_lt_combined['date'], 
                y=pts_lt_combined['spy'] * 0.95, 
                mode='markers', 
                marker=dict(symbol='triangle-up', size=16, color='red'), 
                name="长周期综合买入信号"
            ))
            
    fig_lt0.update_layout(
        title=dict(text="<b>1. 长周期择时信号汇总 (SPY对数坐标) | 包含 200日宽度 / PMI宏观 / ROC动量 / 估值 / VIX恐慌</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
        **layout_template,
        yaxis=dict(type="log", **axis_opts)
    )
    fig_lt0.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts)
    st.plotly_chart(fig_lt0, use_container_width=True)

    # === 图2: SPY + 200日宽度 ===
    if 'sp_breadth_200' in lt_df.columns:
        current_breadth = lt_df['sp_breadth_200'].iloc[-1] if not lt_df['sp_breadth_200'].empty else 50
        state_str = "进攻状态" if current_breadth >= 50 else "防守状态"

        fig_lt1 = make_subplots(specs=[[{"secondary_y": True}]])
        
        x_attack, y_attack, x_defense, y_defense = [], [], [], []
        if not lt_df.empty:
            dates = lt_df['date'].values
            breadths = lt_df['sp_breadth_200'].values
            current_state = None 
            start_date = None
            
            for d, b in zip(dates, breadths):
                if pd.isna(b): continue
                state = (b >= 50)
                if current_state is None:
                    current_state = state
                    start_date = d
                elif current_state != state:
                    if current_state: 
                        x_attack.extend([start_date, start_date, d, d, None])
                        y_attack.extend([0, 250, 250, 0, None])
                    else: 
                        x_defense.extend([start_date, start_date, d, d, None])
                        y_defense.extend([0, 250, 250, 0, None])
                    current_state = state
                    start_date = d
            
            if start_date is not None:
                if current_state:
                    x_attack.extend([start_date, start_date, lt_x_max_padded, lt_x_max_padded, None])
                    y_attack.extend([0, 250, 250, 0, None])
                else:
                    x_defense.extend([start_date, start_date, lt_x_max_padded, lt_x_max_padded, None])
                    y_defense.extend([0, 250, 250, 0, None])

        fig_lt1.add_trace(go.Scatter(x=x_attack, y=y_attack, fill='toself', fillcolor="rgba(255, 230, 230, 0.4)", line=dict(width=0), mode='none', name="进攻状态背景 (>=50)", hoverinfo='skip'), secondary_y=False)
        fig_lt1.add_trace(go.Scatter(x=x_defense, y=y_defense, fill='toself', fillcolor="rgba(230, 255, 230, 0.4)", line=dict(width=0), mode='none', name="防守状态背景 (<50)", hoverinfo='skip'), secondary_y=False)
        fig_lt1.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['sp_breadth_200'], name="200-day Breadth (%)", line=dict(color="navy", width=2)), secondary_y=False)
        fig_lt1.add_hline(y=50, line_dash="dash", line_color="black", name="牛熊分界 (50%)", secondary_y=False)
        fig_lt1.add_hline(y=20, line_dash="dash", line_color="red", name="极度悲观 (20%)", secondary_y=False)
        fig_lt1.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        if 'sig_buy_breadth200_20' in lt_df.columns:
            pts_b200 = lt_df[lt_df['sig_buy_breadth200_20']==1]
            if not pts_b200.empty:
                fig_lt1.add_trace(go.Scatter(x=pts_b200['date'], y=pts_b200['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red'), name="宽度突破20买入"), secondary_y=True)
        
        fig_lt1.update_layout(title=dict(text=f"<b>2. SPY 200日长线宽度 (目前: {state_str}) | 买入规则: 宽度上穿 20%</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt1.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        fig_lt1.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt1.update_yaxes(secondary_y=False, range=[0, 250], **axis_opts) 
        st.plotly_chart(fig_lt1, use_container_width=True)

    # === 图3: SPY YoY vs PMI+NMI YoY ===
    if 'spy_yoy' in lt_df.columns:
        try:
            pmi_df_local = pd.read_csv(os.path.join(BASE_DIR, "ISM_PMI.csv"))
            nmi_df_local = pd.read_csv(os.path.join(BASE_DIR, "ISM_NMI.csv"))
            
            d_col_p = next(c for c in pmi_df_local.columns if 'date' in c.lower())
            v_col_p = next(c for c in pmi_df_local.columns if c != d_col_p)
            pmi_df_local['date'] = pd.to_datetime(pmi_df_local[d_col_p]).dt.tz_localize(None)
            pmi_df_local = pmi_df_local.set_index('date').resample('ME').last()
            pmi_df_local['pmi'] = pmi_df_local[v_col_p]
            
            d_col_n = next(c for c in nmi_df_local.columns if 'date' in c.lower())
            v_col_n = next(c for c in nmi_df_local.columns if c != d_col_n)
            nmi_df_local['date'] = pd.to_datetime(nmi_df_local[d_col_n]).dt.tz_localize(None)
            nmi_df_local = nmi_df_local.set_index('date').resample('ME').last()
            nmi_df_local['nmi'] = nmi_df_local[v_col_n]
            
            ism_plot = pd.merge(pmi_df_local[['pmi']], nmi_df_local[['nmi']], left_index=True, right_index=True, how='outer').sort_index().ffill()
            ism_plot['pmi_nmi_sum'] = ism_plot['pmi'] + ism_plot['nmi']
            ism_plot['pmi_nmi_yoy'] = ism_plot['pmi_nmi_sum'].pct_change(12) * 100
            ism_plot = ism_plot.reset_index()
            ism_plot = ism_plot[(ism_plot['date'] >= '2009-01-01')]
        except Exception:
            ism_plot = pd.DataFrame()

        fig_lt2 = go.Figure()
        
        fig_lt2.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5), yaxis="y1"))
        
        fig_lt2.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_yoy'], name="SPY YoY (%)", fill='tozeroy', mode='lines', line=dict(color="blue", width=1.5), fillcolor="rgba(0, 0, 255, 0.2)", yaxis="y2"))
        
        if not ism_plot.empty and 'pmi_nmi_yoy' in ism_plot.columns:
            fig_lt2.add_trace(go.Scatter(x=ism_plot['date'], y=ism_plot['pmi_nmi_yoy'], name="PMI+NMI YoY (%)", line=dict(color="red", width=2, dash='solid'), yaxis="y3"))
            
        if 'sig_buy_spy_yoy_0' in lt_df.columns:
            pts_yoy = lt_df[lt_df['sig_buy_spy_yoy_0'] == 1]
            if not pts_yoy.empty:
                fig_lt2.add_trace(go.Scatter(x=pts_yoy['date'], y=pts_yoy['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red'), name="SPY YoY上穿0买入", yaxis="y1"))

        fig_lt2.update_layout(
            title=dict(text="<b>3. 宏观基本面定价: SPY YoY vs (PMI+NMI) YoY | 买入规则: SPY YoY 上穿 0 轴</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
            **layout_template,
            yaxis=dict(type="log", side="left", **axis_opts), 
            yaxis2=dict(overlaying="y", side="right", range=[-40, 80], **axis_opts), 
            yaxis3=dict(overlaying="y", side="right", anchor="free", position=0.975, range=[-30, 50], **axis_opts) 
        )
        fig_lt2.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        st.plotly_chart(fig_lt2, use_container_width=True)

    # === 图4: SPY YoY + SPY + ISM PMI ===
    if 'spy_yoy' in lt_df.columns:
        try:
            pmi_df_local2 = pd.read_csv(os.path.join(BASE_DIR, "ISM_PMI.csv"))
            d_col_p2 = next(c for c in pmi_df_local2.columns if 'date' in c.lower())
            v_col_p2 = next(c for c in pmi_df_local2.columns if c != d_col_p2)
            pmi_df_local2['date'] = pd.to_datetime(pmi_df_local2[d_col_p2]).dt.tz_localize(None)
            pmi_df_local2 = pmi_df_local2.set_index('date').resample('ME').last().reset_index()
            pmi_df_local2['pmi'] = pmi_df_local2[v_col_p2]
            pmi_plot = pmi_df_local2[(pmi_df_local2['date'] >= '2009-01-01')]
        except Exception:
            pmi_plot = pd.DataFrame()
            
        fig_lt_new4 = go.Figure()

        fig_lt_new4.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5), yaxis="y1"))
        fig_lt_new4.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_yoy'], name="SPY YoY (%)", fill='tozeroy', mode='lines', line=dict(color="blue", width=1.5), fillcolor="rgba(0, 0, 255, 0.2)", yaxis="y2"))
        
        if not pmi_plot.empty and 'pmi' in pmi_plot.columns:
            fig_lt_new4.add_trace(go.Scatter(x=pmi_plot['date'], y=pmi_plot['pmi'], name="ISM PMI", line=dict(color="red", width=2), yaxis="y3"))

        max_yoy = lt_df['spy_yoy'].abs().max() if pd.notna(lt_df['spy_yoy'].abs().max()) else 60
        range_yoy = [-max_yoy * 1.1, max_yoy * 1.1]
        
        if not pmi_plot.empty and 'pmi' in pmi_plot.columns:
            max_pmi_dev = (pmi_plot['pmi'] - 50).abs().max()
            if pd.isna(max_pmi_dev): max_pmi_dev = 20
        else:
            max_pmi_dev = 20
        max_pmi_dev = max(10, max_pmi_dev * 1.1)
        range_pmi = [50 - max_pmi_dev, 50 + max_pmi_dev]

        fig_lt_new4.update_layout(
            title=dict(text="<b>4. 宏观经济共振: SPY YoY vs ISM PMI | SPY YoY 0轴与 ISM PMI 50横向对齐</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
            **layout_template,
            yaxis=dict(type="log", side="left", **axis_opts), 
            yaxis2=dict(overlaying="y", side="right", range=range_yoy, **axis_opts), 
            yaxis3=dict(overlaying="y", side="right", anchor="free", position=0.975, range=range_pmi, **axis_opts) 
        )
        fig_lt_new4.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        st.plotly_chart(fig_lt_new4, use_container_width=True)

    # === 数据加载辅助函数：扩散指数 ===
    def load_diffusion_data(filename, val_col_name):
        path = os.path.join(BASE_DIR, filename)
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                d_col = next(c for c in df.columns if 'date' in c.lower())
                v_col = next(c for c in df.columns if c != d_col)
                df['date'] = pd.to_datetime(df[d_col]).dt.tz_localize(None)
                df = df.set_index('date').resample('ME').last().reset_index()
                df[val_col_name] = pd.to_numeric(df[v_col], errors='coerce')
                return df[(df['date'] >= '2009-01-01')]
            except Exception:
                pass
        return pd.DataFrame()

    # === 新增 图5: SPY YoY + SPY + 全球PMI扩散指数 ===
    if 'spy_yoy' in lt_df.columns:
        diff_idx_plot = load_diffusion_data("全球PMI扩散指数.csv", "diff_idx")
        
        fig_lt_new5 = go.Figure()

        fig_lt_new5.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5), yaxis="y1"))
        fig_lt_new5.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_yoy'], name="SPY YoY (%)", fill='tozeroy', mode='lines', line=dict(color="blue", width=1.5), fillcolor="rgba(0, 0, 255, 0.2)", yaxis="y2"))
        
        if not diff_idx_plot.empty and 'diff_idx' in diff_idx_plot.columns:
            fig_lt_new5.add_trace(go.Scatter(x=diff_idx_plot['date'], y=diff_idx_plot['diff_idx'], name="全球PMI扩散指数", line=dict(color="red", width=2), yaxis="y3"))

        max_yoy = lt_df['spy_yoy'].abs().max() if pd.notna(lt_df['spy_yoy'].abs().max()) else 60
        range_yoy = [-max_yoy * 1.1, max_yoy * 1.1]
        
        if not diff_idx_plot.empty and 'diff_idx' in diff_idx_plot.columns:
            max_diff_dev = (diff_idx_plot['diff_idx'] - 50).abs().max()
            if pd.isna(max_diff_dev): max_diff_dev = 50
        else:
            max_diff_dev = 50
        max_diff_dev = max(10, max_diff_dev * 1.1)
        range_diff = [50 - max_diff_dev, 50 + max_diff_dev]

        fig_lt_new5.update_layout(
            title=dict(text="<b>5. 宏观经济共振: SPY YoY vs 全球PMI扩散指数 | 0轴与50横向对齐</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
            **layout_template,
            yaxis=dict(type="log", side="left", **axis_opts), 
            yaxis2=dict(overlaying="y", side="right", range=range_yoy, **axis_opts), 
            yaxis3=dict(overlaying="y", side="right", anchor="free", position=0.975, range=range_diff, **axis_opts) 
        )
        fig_lt_new5.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        st.plotly_chart(fig_lt_new5, use_container_width=True)

    # === 新增 图6: SPY YoY + SPY + 全球PMI年变动扩散指数 ===
    if 'spy_yoy' in lt_df.columns:
        diff_yoy_idx_plot = load_diffusion_data("全球PMI年变动扩散指数.csv", "diff_yoy_idx")
        
        fig_lt_new6 = go.Figure()

        fig_lt_new6.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5), yaxis="y1"))
        fig_lt_new6.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_yoy'], name="SPY YoY (%)", fill='tozeroy', mode='lines', line=dict(color="blue", width=1.5), fillcolor="rgba(0, 0, 255, 0.2)", yaxis="y2"))
        
        if not diff_yoy_idx_plot.empty and 'diff_yoy_idx' in diff_yoy_idx_plot.columns:
            fig_lt_new6.add_trace(go.Scatter(x=diff_yoy_idx_plot['date'], y=diff_yoy_idx_plot['diff_yoy_idx'], name="全球PMI年变动扩散指数", line=dict(color="red", width=2), yaxis="y3"))

        max_yoy = lt_df['spy_yoy'].abs().max() if pd.notna(lt_df['spy_yoy'].abs().max()) else 60
        range_yoy = [-max_yoy * 1.1, max_yoy * 1.1]
        
        if not diff_yoy_idx_plot.empty and 'diff_yoy_idx' in diff_yoy_idx_plot.columns:
            max_diff_yoy_dev = (diff_yoy_idx_plot['diff_yoy_idx'] - 50).abs().max()
            if pd.isna(max_diff_yoy_dev): max_diff_yoy_dev = 50
        else:
            max_diff_yoy_dev = 50
        max_diff_yoy_dev = max(10, max_diff_yoy_dev * 1.1)
        range_diff_yoy = [50 - max_diff_yoy_dev, 50 + max_diff_yoy_dev]

        fig_lt_new6.update_layout(
            title=dict(text="<b>6. 宏观经济共振: SPY YoY vs 全球PMI年变动扩散指数 | 0轴与50横向对齐</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")),
            **layout_template,
            yaxis=dict(type="log", side="left", **axis_opts), 
            yaxis2=dict(overlaying="y", side="right", range=range_yoy, **axis_opts), 
            yaxis3=dict(overlaying="y", side="right", anchor="free", position=0.975, range=range_diff_yoy, **axis_opts) 
        )
        fig_lt_new6.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        st.plotly_chart(fig_lt_new6, use_container_width=True)

    # === 图7: SPY 3-month Chg ===
    if 'spy_roc_63' in lt_df.columns:
        fig_lt3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt3.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_roc_63'], name="SPY 3-Month Chg (%)", fill='tozeroy', line=dict(color="MediumOrchid")), secondary_y=False)
        
        if 'spy_roc_63_ma3' in lt_df.columns:
            fig_lt3.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy_roc_63_ma3'], name="ROC MA3", line=dict(color="gray", dash="dot", width=1)), secondary_y=False)

        fig_lt3.add_hline(y=-9, line_dash="dash", line_color="red", name="-9% 极度超卖线", secondary_y=False)
        fig_lt3.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        if 'sig_buy_roc63' in lt_df.columns:
            pts_roc = lt_df[lt_df['sig_buy_roc63']==1]
            if not pts_roc.empty: 
                fig_lt3.add_trace(go.Scatter(x=pts_roc['date'], y=pts_roc['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red'), name="ROC底部买入"), secondary_y=True)
        
        fig_lt3.update_layout(title=dict(text="<b>7. 极限动量偏离: SPY 3-Month Change (ROC 63) vs SPY | 买入规则: ROC < -9% 且上穿自身 MA3 | 5天内不再触发</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt3.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts) 
        fig_lt3.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt3.update_yaxes(secondary_y=False, range=[-30, 60], **axis_opts)
        st.plotly_chart(fig_lt3, use_container_width=True)

    # === 图8: 估值周期 SPY vs CAPE (主副图联动) ===
    if 'cape' in lt_df.columns:
        if cape_trend is not None and cape_up is not None and cape_lo is not None:
            max_abs_res = (lt_df['cape'] - cape_trend).abs().max()
            cape_up_max = cape_trend + max_abs_res
            cape_lo_max = cape_trend - max_abs_res
            lt_df['cape_pct'] = (lt_df['cape'] - cape_lo_max) / (cape_up_max - cape_lo_max) * 100
        
        fig_lt4 = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3], 
            specs=[[{"secondary_y": True}], [{"secondary_y": False}]]
        )
        
        if cape_up is not None:
            fig_lt4.add_trace(go.Scatter(x=lt_df['date'], y=cape_up, name="CAPE +1.4SD 上轨", line=dict(color="gray", dash="dash", width=1), legendgroup="cape"), row=1, col=1, secondary_y=False)
        if cape_lo is not None:
            fig_lt4.add_trace(go.Scatter(x=lt_df['date'], y=cape_lo, name="CAPE -1.4SD 下轨", line=dict(color="gray", dash="dash", width=1), legendgroup="cape"), row=1, col=1, secondary_y=False)
            
        fig_lt4.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['cape'], name="Shiller CAPE", line=dict(color="brown", width=2), legendgroup="cape"), row=1, col=1, secondary_y=False)
        fig_lt4.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5), legendgroup="spy"), row=1, col=1, secondary_y=True)
        
        pts_cape = lt_df[lt_df.get('sig_cape_buy', 0) == 1]
        if not pts_cape.empty:
            fig_lt4.add_trace(go.Scatter(x=pts_cape['date'], y=pts_cape['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red'), name="CAPE极度低估拐头买入", legendgroup="spy"), row=1, col=1, secondary_y=True)

        if 'cape_pct' in lt_df.columns:
            fig_lt4.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['cape_pct'], name="CAPE 通道百分位 (%)", line=dict(color="purple", width=2), legendgroup="pct"), row=2, col=1)
            
            fig_lt4.add_hrect(y0=0, y1=30, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, row=2, col=1, name="低估区 (0~30%)", showlegend=True)
            fig_lt4.add_hrect(y0=70, y1=100, fillcolor="rgba(144, 238, 144, 0.3)", line_width=0, row=2, col=1, name="高估区 (70~100%)", showlegend=True)
            
            fig_lt4.add_hline(y=100, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
            fig_lt4.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)

        custom_layout = layout_template.copy()
        custom_layout["height"] = 900 
        
        fig_lt4.update_layout(title=dict(text="<b>8. 估值周期: SPY vs Shiller CAPE Ratio (上图: 绝对估值通道 | 下图: 自适应百分位满量程)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **custom_layout)
        
        fig_lt4.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], row=1, col=1, **axis_opts)
        fig_lt4.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], row=2, col=1, **axis_opts)
        
        fig_lt4.update_yaxes(type="log", secondary_y=True, row=1, col=1, **axis_opts)
        fig_lt4.update_yaxes(secondary_y=False, range=[10, 60], row=1, col=1, **axis_opts)
        fig_lt4.update_yaxes(title_text="Percentile (%)", range=[-20, 120], row=2, col=1, **axis_opts)
        
        st.plotly_chart(fig_lt4, use_container_width=True)

    # === 图9: 估值动态 SPY vs 标普500远期市盈率 ===
    if 'forward_pe' in lt_df.columns:
        fig_lt5_new = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt5_new.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['forward_pe'], name="Forward P/E", line=dict(color="darkgoldenrod", width=2)), secondary_y=False)
        
        if 'fwd_pe_ma3' in lt_df.columns:
            fig_lt5_new.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['fwd_pe_ma3'], name="Forward P/E MA3", line=dict(color="gray", dash='dot', width=1)), secondary_y=False)

        line_start = pd.Timestamp('2020-02-01')
        df_lines = lt_df[lt_df['date'] >= line_start]
        if not df_lines.empty:
            fig_lt5_new.add_trace(go.Scatter(x=df_lines['date'], y=[19]*len(df_lines), name="y=19 (低估线)", line=dict(color="green", dash='dash', width=1)), secondary_y=False)
            fig_lt5_new.add_trace(go.Scatter(x=df_lines['date'], y=[23]*len(df_lines), name="y=23 (高估线)", line=dict(color="red", dash='dash', width=1)), secondary_y=False)
            
        fig_lt5_new.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)

        fig_lt5_new.update_layout(title=dict(text="<b>9. 估值动态: SPY vs 标普500远期市盈率</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt5_new.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts)
        fig_lt5_new.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt5_new.update_yaxes(secondary_y=False, range=[10, 45], **axis_opts) 
        st.plotly_chart(fig_lt5_new, use_container_width=True)

    # === 图10: 风险偏好 SPY vs SMH/XLP YoY ===
    if 'smh_xlp_yoy' in lt_df.columns:
        fig_lt5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt5.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['smh_xlp_yoy'], name="SMH/XLP YoY (%)", fill='tozeroy', line=dict(color="teal", width=2)), secondary_y=False)
        fig_lt5.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        fig_lt5.add_hline(y=60, line_dash="dash", line_color="red", name="y=60 高位过热", secondary_y=False)
        fig_lt5.add_hline(y=-20, line_dash="dash", line_color="green", name="y=-20 极度悲观", secondary_y=False)

        fig_lt5.update_layout(title=dict(text="<b>10. 风险偏好定价: SPY vs SMH/XLP YoY (市场定价周期情况)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        
        lt_x_max_padded_100 = lt_df['date'].max() + pd.Timedelta(days=100)
        fig_lt5.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded_100], domain=[0, 0.94], **axis_opts)
        
        fig_lt5.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt5.update_yaxes(secondary_y=False, range=[-50, 150], **axis_opts)
        st.plotly_chart(fig_lt5, use_container_width=True)

    # === 图11: 综合金融条件指数 (FCI) ===
    if 'fci_score' in lt_df.columns:
        fig_lt_fci = make_subplots(specs=[[{"secondary_y": True}]])
        
        x_red, y_red, x_green, y_green = [], [], [], []
        if not lt_df.empty:
            dates = lt_df['date'].values
            fcis = lt_df['fci_score'].values
            current_state = None 
            start_date = None
            lt_x_max_padded_fci = lt_df['date'].max() + pd.Timedelta(days=150)
            
            for d, f in zip(dates, fcis):
                if pd.isna(f): continue
                state = (f >= 200)
                if current_state is None:
                    current_state = state
                    start_date = d
                elif current_state != state:
                    if current_state: 
                        x_green.extend([start_date, start_date, d, d, None])
                        y_green.extend([0, 1000, 1000, 0, None])
                    else: 
                        x_red.extend([start_date, start_date, d, d, None])
                        y_red.extend([0, 1000, 1000, 0, None])
                    current_state = state
                    start_date = d
            
            if start_date is not None:
                if current_state:
                    x_green.extend([start_date, start_date, lt_x_max_padded_fci, lt_x_max_padded_fci, None])
                    y_green.extend([0, 1000, 1000, 0, None])
                else:
                    x_red.extend([start_date, start_date, lt_x_max_padded_fci, lt_x_max_padded_fci, None])
                    y_red.extend([0, 1000, 1000, 0, None])

        fig_lt_fci.add_trace(go.Scatter(x=x_red, y=y_red, fill='toself', fillcolor="rgba(255, 230, 230, 0.4)", line=dict(width=0), mode='none', name="FCI < 200 (淡红背景)"), secondary_y=False)
        fig_lt_fci.add_trace(go.Scatter(x=x_green, y=y_green, fill='toself', fillcolor="rgba(230, 255, 230, 0.4)", line=dict(width=0), mode='none', name="FCI >= 200 (淡绿背景)"), secondary_y=False)
        
        fig_lt_fci.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['fci_score'], name="FCI (综合金融条件指数)", line=dict(color="darkorange", width=2)), secondary_y=False)
        fig_lt_fci.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        if 'sig_buy_fci_350' in lt_df.columns:
            pts_fci = lt_df[lt_df['sig_buy_fci_350'] == 1]
            if not pts_fci.empty:
                fig_lt_fci.add_trace(go.Scatter(x=pts_fci['date'], y=pts_fci['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red', line=dict(width=1, color="white")), name="FCI跌破350买入信号"), secondary_y=True)

        fig_lt_fci.add_hline(y=350, line_dash="dash", line_color="red", name="极度恐慌/收紧 (350)", secondary_y=False)
        fig_lt_fci.add_hline(y=50, line_dash="dash", line_color="green", name="极度贪婪/宽松 (50)", secondary_y=False)

        new_title = "<b>11. 综合金融条件指数 (FCI) | 规则: 跌破350触发买入(冷却10天); <200淡红背景, >=200淡绿背景</b>"
        fig_lt_fci.update_layout(title=dict(text=new_title, x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        
        fig_lt_fci.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded_fci], domain=[0, 0.94], **axis_opts)
        
        fig_lt_fci.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt_fci.update_yaxes(secondary_y=False, range=[0, 1000], **axis_opts)
        
        st.plotly_chart(fig_lt_fci, use_container_width=True)

    # === 图12: 流动性周期 ISM PMI vs 降息比例 ===
    if 'pmi' in lt_df.columns and 'rate_cut_ratio' in lt_df.columns:
        fig_lt9 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_lt9.add_trace(go.Scatter(x=lt_df['date'], y=[50]*len(lt_df), line=dict(width=0), showlegend=False, hoverinfo='skip'), secondary_y=False)
        fig_lt9.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['pmi'], name="ISM PMI", line=dict(color="darkgreen", width=2.5), fill='tonexty', fillcolor="rgba(0, 100, 0, 0.3)"), secondary_y=False)
        
        shifted_date = lt_df['date'] + pd.DateOffset(months=10)
        fig_lt9.add_trace(go.Scatter(x=shifted_date, y=lt_df['rate_cut_ratio'], name="全球央行降息比例 (%)", line=dict(color="red", width=4)), secondary_y=True)
        
        fig_lt9.add_hline(y=50, line_dash="dash", line_color="green", name="PMI 荣枯线", secondary_y=False)
        
        fig_lt9.update_layout(title=dict(text="<b>12. 商业与流动性周期: ISM PMI vs 全球央行降息比例 (降息比例领先PMI约10个月)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        
        max_date_plus_10m = lt_df['date'].max() + pd.DateOffset(months=10)
        fig_lt9.update_xaxes(range=['2009-01-01', max_date_plus_10m], domain=[0, 0.94], **axis_opts)
        fig_lt9.update_yaxes(secondary_y=False, range=[30, 70], **axis_opts)
        fig_lt9.update_yaxes(secondary_y=True, range=[0, 100], **axis_opts)
        st.plotly_chart(fig_lt9, use_container_width=True)

    # === 图13: 就业周期 SPY vs UNRATE (1994年至今) ===
    if 'unrate' in lt_df_1994.columns:
        fig_lt10 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt10.add_trace(go.Scatter(x=lt_df_1994['date'], y=lt_df_1994['unrate'], name="US Unemployment Rate (%)", fill='tozeroy', line=dict(color="purple", width=2)), secondary_y=False)
        fig_lt10.add_trace(go.Scatter(x=lt_df_1994['date'], y=lt_df_1994['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        fig_lt10.update_layout(title=dict(text="<b>13. 经济与就业周期: SPY vs 失业率 (UNRATE)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt10.update_xaxes(range=[lt_df_1994['date'].min(), lt_x_max_padded_1994], domain=[0, 0.94], **axis_opts)
        fig_lt10.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt10.update_yaxes(secondary_y=False, range=[3, 14], **axis_opts) 
        st.plotly_chart(fig_lt10, use_container_width=True)

    # === 图14: 货币周期 SPY vs DFF & DGS2 (2000年至今) ===
    if 'dff' in lt_df_1994.columns and 'dgs2' in lt_df_1994.columns:
        lt_df_2000 = lt_df_1994[lt_df_1994['date'] >= '2000-01-01'].copy()
        lt_x_max_padded_2000 = lt_df_2000['date'].max() + pd.Timedelta(days=50) if not lt_df_2000.empty else lt_x_max_padded_1994
        
        fig_lt11 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt11.add_trace(go.Scatter(x=lt_df_2000['date'], y=lt_df_2000['dff'], name="Fed Funds Rate (DFF)", line=dict(color="blue", width=2, dash='dot')), secondary_y=False)
        fig_lt11.add_trace(go.Scatter(x=lt_df_2000['date'], y=lt_df_2000['dgs2'], name="US 2Y Treasury (DGS2)", line=dict(color="red", width=2)), secondary_y=False)
        fig_lt11.add_trace(go.Scatter(x=lt_df_2000['date'], y=lt_df_2000['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        fig_lt11.update_layout(title=dict(text="<b>14. 货币政策周期: SPY vs 联邦基金利率 (DFF) & 2年期美债 (DGS2)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt11.update_xaxes(range=[lt_df_2000['date'].min(), lt_x_max_padded_2000], domain=[0, 0.94], **axis_opts)
        fig_lt11.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt11.update_yaxes(secondary_y=False, **axis_opts)
        st.plotly_chart(fig_lt11, use_container_width=True)

    # === 图15: 极度恐慌买入 VIX > 40 vs SPY ===
    if 'vix' in lt_df.columns:
        fig_lt12 = make_subplots(specs=[[{"secondary_y": True}]])
        fig_lt12.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['vix'], name="VIX Close", line=dict(color="crimson", width=2)), secondary_y=False)
        
        if 'vix_ma3' in lt_df.columns:
            fig_lt12.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['vix_ma3'], name="VIX MA3", line=dict(color="gray", dash='dot', width=1)), secondary_y=False)
            
        fig_lt12.add_hline(y=40, line_dash="dash", line_color="red", name="恐慌阈值 (40)", secondary_y=False)
        
        fig_lt12.add_trace(go.Scatter(x=lt_df['date'], y=lt_df['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        if 'sig_buy_vix_40' in lt_df.columns:
            pts_vix40 = lt_df[lt_df['sig_buy_vix_40'] == 1]
            if not pts_vix40.empty:
                fig_lt12.add_trace(go.Scatter(x=pts_vix40['date'], y=pts_vix40['spy']*0.95, mode='markers', marker=dict(symbol='triangle-up', size=16, color='red'), name="VIX极值回落买入"), secondary_y=True)
                
        fig_lt12.update_layout(title=dict(text="<b>15. 极度恐慌抛售买入: SPY vs VIX | 买入规则: VIX 盘中突破 40 后跌破自身 MA3 | 10天内不再触发</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        fig_lt12.update_xaxes(range=[lt_df['date'].min(), lt_x_max_padded], domain=[0, 0.94], **axis_opts)
        fig_lt12.update_yaxes(type="log", secondary_y=True, **axis_opts)
        fig_lt12.update_yaxes(secondary_y=False, **axis_opts) 
        st.plotly_chart(fig_lt12, use_container_width=True)

    # === 图16: 信用市场交叉验证图 (HYG vs SPY) 2012至今 ===
    df_hyg_2012 = df_final[df_final['date'] >= '2012-02-01'].copy()
    if 'hyg' in df_hyg_2012.columns and 'hyg_ma120' in df_hyg_2012.columns:
        valid_hyg_lt = df_hyg_2012[['hyg', 'hyg_ma120']].dropna()
        if not valid_hyg_lt.empty:
            current_hyg_lt = valid_hyg_lt['hyg'].iloc[-1]
            current_ma120_lt = valid_hyg_lt['hyg_ma120'].iloc[-1]
            state_str_lt = "正常" if current_hyg_lt >= current_ma120_lt else "信用预警 (HYG 跌破半年线)"
            
            fig_hyg_lt = make_subplots(specs=[[{"secondary_y": True}]])
            
            x_warn_lt, y_warn_lt = [], []
            dates_lt = df_hyg_2012['date'].values
            hygs_lt = df_hyg_2012['hyg'].values
            ma120s_lt = df_hyg_2012['hyg_ma120'].values
            
            current_state_lt = None
            start_date_lt = None
            lt_x_max_padded_hyg = df_hyg_2012['date'].max() + pd.Timedelta(days=50)
            
            for d, h, m in zip(dates_lt, hygs_lt, ma120s_lt):
                if pd.isna(h) or pd.isna(m): continue
                state = (h < m)
                if current_state_lt is None:
                    current_state_lt = state
                    start_date_lt = d
                elif current_state_lt != state:
                    if current_state_lt: 
                        x_warn_lt.extend([start_date_lt, start_date_lt, d, d, None])
                        y_warn_lt.extend([0, 500, 500, 0, None])
                    current_state_lt = state
                    start_date_lt = d
            
            if start_date_lt is not None and current_state_lt:
                x_warn_lt.extend([start_date_lt, start_date_lt, lt_x_max_padded_hyg, lt_x_max_padded_hyg, None])
                y_warn_lt.extend([0, 500, 500, 0, None])
                
            fig_hyg_lt.add_trace(go.Scatter(x=x_warn_lt, y=y_warn_lt, fill='toself', fillcolor="rgba(144, 238, 144, 0.2)", line=dict(width=0), mode='none', name="预警状态背景 (<MA120)", hoverinfo='skip'), secondary_y=False)
            
            fig_hyg_lt.add_trace(go.Scatter(x=df_hyg_2012['date'], y=df_hyg_2012['hyg'], name="HYG Close", line=dict(color="blue", width=2)), secondary_y=False)
            fig_hyg_lt.add_trace(go.Scatter(x=df_hyg_2012['date'], y=df_hyg_2012['hyg_ma120'], name="HYG MA120", line=dict(color="orange", width=2, dash='dash')), secondary_y=False)
            fig_hyg_lt.add_trace(go.Scatter(x=df_hyg_2012['date'], y=df_hyg_2012['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
            
            fig_hyg_lt.update_layout(title=dict(text=f"<b>16. 信用市场交叉验证: HYG vs SPY (当前状态: {state_str_lt})</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
            fig_hyg_lt.update_xaxes(range=[df_hyg_2012['date'].min(), lt_x_max_padded_hyg], domain=[0, 0.94], **axis_opts)
            fig_hyg_lt.update_yaxes(type="log", secondary_y=True, **axis_opts)
            fig_hyg_lt.update_yaxes(secondary_y=False, **axis_opts)
            
            fig_hyg_lt.update_yaxes(range=[40, 110], secondary_y=False)

            st.plotly_chart(fig_hyg_lt, use_container_width=True)

    # === 新增 图17: 美元指数 (DXY) 日线对数通道 ===
    def plot_dxy_daily_trend_chart(df_plot, df_spy, title, anchor1, anchor2, parallel_anchor, extend_days=100):
        last_date = df_plot['Date'].max()
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=extend_days)
        extended_dates = pd.concat([df_plot['Date'], pd.Series(future_dates)]).reset_index(drop=True)
        
        x_ext = pd.to_datetime(extended_dates).apply(lambda d: d.toordinal()).values
        x_orig = pd.to_datetime(df_plot['Date']).apply(lambda d: d.toordinal()).values
        
        idx1 = np.abs(df_plot['Date'] - pd.to_datetime(anchor1)).argmin()
        idx2 = np.abs(df_plot['Date'] - pd.to_datetime(anchor2)).argmin()
        idx_p = np.abs(df_plot['Date'] - pd.to_datetime(parallel_anchor)).argmin()
        
        x1, x2, xp = x_orig[idx1], x_orig[idx2], x_orig[idx_p]
        
        y1, y2 = np.log(df_plot['Low'].iloc[idx1]), np.log(df_plot['Low'].iloc[idx2])
        yp = np.log(df_plot['High'].iloc[idx_p])
            
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        c_lower = y1 - m * x1
        c_upper = yp - m * xp
        
        lower_bound = np.exp(m * x_ext + c_lower)
        upper_bound = np.exp(m * x_ext + c_upper)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(go.Scatter(x=extended_dates, y=upper_bound, name="上轨 (High)", line=dict(color="red", dash="dash", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=extended_dates, y=lower_bound, name="下轨 (Low)", line=dict(color="green", dash="dash", width=2)), secondary_y=False)
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name="DXY Close", line=dict(color="blue", width=2)), secondary_y=False)
        
        fig.add_trace(go.Scatter(x=df_spy['date'], y=df_spy['spy'], name="SPY (Log)", line=dict(color='black', width=3.5)), secondary_y=True)
        
        custom_layout = layout_template.copy()
        custom_layout["height"] = 750
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **custom_layout)
        
        x_max_padded_chart = extended_dates.max() + pd.Timedelta(days=10)
        fig.update_xaxes(range=[df_plot['Date'].min(), x_max_padded_chart], domain=[0, 0.94], **axis_opts)
        
        # Plotly中 log坐标轴的范围设定需要使用 np.log10
        fig.update_yaxes(type="log", secondary_y=False, range=[np.log10(70), np.log10(240)], **axis_opts)
        fig.update_yaxes(type="log", secondary_y=True, **axis_opts)
        
        return fig

    df_dxy_daily = get_dxy_daily_data()
    if not df_dxy_daily.empty:
        df_dxy_daily = df_dxy_daily.reset_index()
        fig_lt17_dxy = plot_dxy_daily_trend_chart(
            df_dxy_daily,
            df_final,  # 传入包含 SPY 数据的 DataFrame
            title="17. 美元指数 (DXY) 日线对数通道 (2006年起)",
            anchor1="2008-07-14", 
            anchor2="2021-05-24", 
            parallel_anchor="2008-11-20", 
            extend_days=100
        )
        st.plotly_chart(fig_lt17_dxy, use_container_width=True)

    # ================= 图 18, 19, 20, 21 (SPX & NDX 周线趋势对数通道扩展版) =================
    
    def plot_extended_trend_chart(df_plot, title, anchor1, anchor2, parallel_anchor, base_line='lower', index_name="SPX", extend_weeks=30):
        last_date = df_plot['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=extend_weeks, freq='W-MON')
        extended_dates = pd.concat([df_plot['Date'], pd.Series(future_dates)]).reset_index(drop=True)
        
        x_ext = pd.to_datetime(extended_dates).apply(lambda d: d.toordinal()).values
        x_orig = pd.to_datetime(df_plot['Date']).apply(lambda d: d.toordinal()).values
        
        idx1 = np.abs(df_plot['Date'] - pd.to_datetime(anchor1)).argmin()
        idx2 = np.abs(df_plot['Date'] - pd.to_datetime(anchor2)).argmin()
        idx_p = np.abs(df_plot['Date'] - pd.to_datetime(parallel_anchor)).argmin()
        
        x1, x2, xp = x_orig[idx1], x_orig[idx2], x_orig[idx_p]
        
        if base_line == 'lower':
            y1, y2 = np.log(df_plot['Low'].iloc[idx1]), np.log(df_plot['Low'].iloc[idx2])
            yp = np.log(df_plot['High'].iloc[idx_p])
        else: 
            y1, y2 = np.log(df_plot['High'].iloc[idx1]), np.log(df_plot['High'].iloc[idx2])
            yp = np.log(df_plot['Low'].iloc[idx_p])
            
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        c_base = y1 - m * x1
        c_parallel = yp - m * xp
        c_mid = (c_base + c_parallel) / 2
        
        if base_line == 'lower':
            c_lower, c_upper = c_base, c_parallel
        else:
            c_upper, c_lower = c_base, c_parallel
            
        lower_bound = np.exp(m * x_ext + c_lower)
        upper_bound = np.exp(m * x_ext + c_upper)
        mid_bound = np.exp(m * x_ext + c_mid)
        
        lower_orig = np.exp(m * x_orig + c_lower)
        upper_orig = np.exp(m * x_orig + c_upper)
        percentile = (df_plot['Close'] - lower_orig) / (upper_orig - lower_orig) * 100
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3], 
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=extended_dates, y=upper_bound, name="上轨 (High)", line=dict(color="gray", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=extended_dates, y=mid_bound, name="中轨 (Mid)", line=dict(color="orange", dash="dot", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=extended_dates, y=lower_bound, name="下轨 (Low)", line=dict(color="gray", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name=f"{index_name} Close", line=dict(color="black", width=2.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=percentile, name="通道百分位 (%)", line=dict(color="purple", width=2)), row=2, col=1)
        
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, row=2, col=1, name="低估区 (0~30%)", showlegend=True)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(144, 238, 144, 0.3)", line_width=0, row=2, col=1, name="高估区 (70~100%)", showlegend=True)
        
        fig.add_hline(y=100, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
        
        custom_layout = layout_template.copy()
        custom_layout["height"] = 900
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **custom_layout)
        
        x_max_padded_chart = extended_dates.max() + pd.Timedelta(days=10)
        fig.update_xaxes(range=[df_plot['Date'].min(), x_max_padded_chart], domain=[0, 0.94], row=1, col=1, **axis_opts)
        fig.update_xaxes(range=[df_plot['Date'].min(), x_max_padded_chart], domain=[0, 0.94], row=2, col=1, **axis_opts)
        
        fig.update_yaxes(type="log", row=1, col=1, **axis_opts)
        fig.update_yaxes(title_text="Percentile (%)", range=[-20, 120], row=2, col=1, **axis_opts)
        
        return fig

    df_spx_weekly = get_spx_weekly_data()
    if not df_spx_weekly.empty:
        df_spx_weekly = df_spx_weekly.reset_index()
        
        df_long = df_spx_weekly[df_spx_weekly['Date'] >= '1930-01-01'].copy()
        if not df_long.empty:
            fig_lt14 = plot_extended_trend_chart(
                df_long, 
                title="18. SPX 长期趋势周线对数通道 (1930年起 | 包含未来100周)", 
                anchor1="1942-04-27", 
                anchor2="2009-03-02", 
                parallel_anchor="1937-03-01",
                base_line="lower",
                index_name="SPX",
                extend_weeks=100
            )
            st.plotly_chart(fig_lt14, use_container_width=True)
            
        df_recent = df_spx_weekly[df_spx_weekly['Date'] >= '2008-01-01'].copy()
        if not df_recent.empty:
            fig_lt15 = plot_extended_trend_chart(
                df_recent, 
                title="19. SPX 中短期趋势周线对数通道 (2008年起 | 包含未来30周)", 
                anchor1="2009-07-13", 
                anchor2="2023-10-27", 
                parallel_anchor="2014-11-28",
                base_line="lower",
                index_name="SPX",
                extend_weeks=30
            )
            st.plotly_chart(fig_lt15, use_container_width=True)

    df_ndx_weekly = get_ndx_weekly_data()
    if not df_ndx_weekly.empty:
        df_ndx_weekly = df_ndx_weekly.reset_index()
        
        df_ndx_long = df_ndx_weekly[df_ndx_weekly['Date'] >= '1985-10-01'].copy()
        if not df_ndx_long.empty:
            fig_lt16 = plot_extended_trend_chart(
                df_ndx_long,
                title="20. NDX 长期趋势周线对数通道 (1985年起 | 包含未来100周)",
                anchor1="1987-09-28",
                anchor2="2021-11-21",
                parallel_anchor="2009-03-02",
                base_line="upper",   
                index_name="NDX",
                extend_weeks=100
            )
            st.plotly_chart(fig_lt16, use_container_width=True)
            
        df_ndx_mid = df_ndx_weekly[df_ndx_weekly['Date'] >= '2010-01-01'].copy()
        if not df_ndx_mid.empty:
            fig_lt17 = plot_extended_trend_chart(
                df_ndx_mid,
                title="21. NDX 中短期趋势周线对数通道 (2010年起 | 包含未来30周)",
                anchor1="2010-07-01",
                anchor2="2023-01-06",
                parallel_anchor="2024-12-16",
                base_line="lower",
                index_name="NDX",
                extend_weeks=30
            )
            st.plotly_chart(fig_lt17, use_container_width=True)

    # ================= 图 22 (IWM 日线趋势通道) =================
    
    def plot_iwm_daily_trend_chart(df_plot, title, anchor1, anchor2, parallel_anchor, extend_days=100):
        last_date = df_plot['Date'].max()
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=extend_days)
        extended_dates = pd.concat([df_plot['Date'], pd.Series(future_dates)]).reset_index(drop=True)
        
        x_ext = pd.to_datetime(extended_dates).apply(lambda d: d.toordinal()).values
        x_orig = pd.to_datetime(df_plot['Date']).apply(lambda d: d.toordinal()).values
        
        idx1 = np.abs(df_plot['Date'] - pd.to_datetime(anchor1)).argmin()
        idx2 = np.abs(df_plot['Date'] - pd.to_datetime(anchor2)).argmin()
        idx_p = np.abs(df_plot['Date'] - pd.to_datetime(parallel_anchor)).argmin()
        
        x1, x2, xp = x_orig[idx1], x_orig[idx2], x_orig[idx_p]
        
        y1, y2 = np.log(df_plot['High'].iloc[idx1]), np.log(df_plot['High'].iloc[idx2])
        yp = np.log(df_plot['Low'].iloc[idx_p])
            
        m = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
        c_upper = y1 - m * x1
        c_lower = yp - m * xp
        c_25 = c_lower + 0.25 * (c_upper - c_lower)
        c_75 = c_lower + 0.75 * (c_upper - c_lower)
        
        lower_bound = np.exp(m * x_ext + c_lower)
        upper_bound = np.exp(m * x_ext + c_upper)
        line25_bound = np.exp(m * x_ext + c_25)
        line75_bound = np.exp(m * x_ext + c_75)
        
        percentile = (np.log(df_plot['Close']) - (m * x_orig + c_lower)) / (c_upper - c_lower) * 100
        
        fig = make_subplots(
            rows=2, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.03, 
            row_heights=[0.7, 0.3], 
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        fig.add_trace(go.Scatter(x=extended_dates, y=upper_bound, name="上轨 (High)", line=dict(color="gray", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=extended_dates, y=line75_bound, name="75% 分轨", line=dict(color="orange", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=extended_dates, y=line25_bound, name="25% 分轨", line=dict(color="orange", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=extended_dates, y=lower_bound, name="下轨 (Low)", line=dict(color="gray", dash="dash", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Close'], name="IWM Close", line=dict(color="black", width=1.5)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_plot['Date'], y=percentile, name="通道百分位 (%)", line=dict(color="purple", width=1.5)), row=2, col=1)
        
        fig.add_hrect(y0=0, y1=25, fillcolor="rgba(255, 0, 0, 0.15)", line_width=0, row=2, col=1, name="低估区 (0~25%)", showlegend=True)
        fig.add_hrect(y0=75, y1=100, fillcolor="rgba(144, 238, 144, 0.3)", line_width=0, row=2, col=1, name="高估区 (75~100%)", showlegend=True)
        
        fig.add_hline(y=100, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
        fig.add_hline(y=75, line_dash="dash", line_color="orange", line_width=1, row=2, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="orange", line_width=1, row=2, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=1, row=2, col=1)
        
        custom_layout = layout_template.copy()
        custom_layout["height"] = 900
        fig.update_layout(title=dict(text=f"<b>{title}</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **custom_layout)
        
        x_max_padded_chart = extended_dates.max() + pd.Timedelta(days=10)
        fig.update_xaxes(range=[df_plot['Date'].min(), x_max_padded_chart], domain=[0, 0.94], row=1, col=1, **axis_opts)
        fig.update_xaxes(range=[df_plot['Date'].min(), x_max_padded_chart], domain=[0, 0.94], row=2, col=1, **axis_opts)
        
        fig.update_yaxes(type="log", row=1, col=1, **axis_opts)
        
        fig.update_yaxes(title_text="Percentile (%)", range=[20, 80], row=2, col=1, **axis_opts)
        
        return fig

    df_iwm_daily = get_iwm_daily_data()
    if not df_iwm_daily.empty:
        df_iwm_daily = df_iwm_daily.reset_index()
        fig_lt18 = plot_iwm_daily_trend_chart(
            df_iwm_daily,
            title="22. IWM 长期趋势日线对数通道 (2000年起 | 包含未来100天)",
            anchor1="2006-05-05", 
            anchor2="2021-11-08", 
            parallel_anchor="2020-03-19", 
            extend_days=100
        )
        st.plotly_chart(fig_lt18, use_container_width=True)

    # ================= 图 23 (全球宏观流动性 G4 M2 YoY vs ISM PMI) =================
    df_g4 = get_g4_m2_data()
    
    pmi_raw_clean = pd.DataFrame()
    pmi_cache = os.path.join(BASE_DIR, "ISM_PMI.csv")
    
    if os.path.exists(pmi_cache):
        try:
            pmi_raw = pd.read_csv(pmi_cache)
            d_col_p = next(c for c in pmi_raw.columns if 'date' in c.lower())
            v_col_p = next(c for c in pmi_raw.columns if c != d_col_p)
            pmi_raw['date'] = pd.to_datetime(pmi_raw[d_col_p]).dt.tz_localize(None)
            pmi_raw_clean = pmi_raw.set_index('date').resample('ME').last().reset_index()
            pmi_raw_clean['pmi'] = pmi_raw_clean[v_col_p]
            pmi_raw_clean = pmi_raw_clean[pmi_raw_clean['date'] >= '2009-01-01']
        except Exception:
            pass
    else:
        try:
            url = "https://www.econdb.com/api/series/PMIUS/?format=csv"
            headers = {'User-Agent': 'Mozilla/5.0'}
            res = requests.get(url, headers=headers, timeout=10)
            if res.status_code == 200:
                df = pd.read_csv(StringIO(res.text))
                df.columns = ['date', 'pmi']
                df['date'] = pd.to_datetime(df['date'])
                df.to_csv(pmi_cache, index=False)
                pmi_raw_clean = df.set_index('date').resample('ME').last().reset_index()
                pmi_raw_clean = pmi_raw_clean[pmi_raw_clean['date'] >= '2009-01-01']
        except Exception:
            pass

    if not df_g4.empty or not pmi_raw_clean.empty:
        fig_lt19 = make_subplots(specs=[[{"secondary_y": True}]])
        
        has_pmi = not pmi_raw_clean.empty and 'pmi' in pmi_raw_clean.columns
        has_m2 = not df_g4.empty and 'g4_m2_yoy' in df_g4.columns
        
        if has_pmi:
            fig_lt19.add_trace(go.Scatter(x=pmi_raw_clean['date'], y=pmi_raw_clean['pmi'], name="ISM PMI", line=dict(color="darkgreen", width=2.5)), secondary_y=False)
            fig_lt19.add_hline(y=50, line_dash="dash", line_color="green", name="PMI 荣枯线", secondary_y=False)
            
        df_g4_plot = pd.DataFrame()
        if has_m2:
            df_g4_plot = df_g4[df_g4['date'] >= '2009-01-01']
            if not df_g4_plot.empty:
                fig_lt19.add_trace(go.Scatter(x=df_g4_plot['date'], y=df_g4_plot['g4_m2_yoy'], name="G4 M2 YoY (%)", fill='tozeroy', line=dict(color="rgba(255, 69, 0, 0.8)", width=2)), secondary_y=True)
                fig_lt19.add_hline(y=0, line_dash="solid", line_color="gray", name="M2 零轴", secondary_y=True)
            
        fig_lt19.update_layout(title=dict(text="<b>23. 全球宏观流动性与经济周期: G4 M2 YoY (基于本地CSV) vs ISM PMI (最新可用)</b>", x=0, xref='container', xanchor='left', font=dict(size=22, color="black")), **layout_template)
        
        max_dt = pd.Timestamp('2009-01-01')
        if has_pmi: max_dt = max(max_dt, pmi_raw_clean['date'].max())
        if has_m2 and not df_g4_plot.empty: max_dt = max(max_dt, df_g4_plot['date'].max())
        lt_x_max_padded_m2 = max_dt + pd.Timedelta(days=50)
        fig_lt19.update_xaxes(range=['2009-01-01', lt_x_max_padded_m2], domain=[0, 0.94], **axis_opts)
        
        delta_pmi = 20
        if has_pmi:
            pmi_max_diff = (pmi_raw_clean['pmi'] - 50).abs().max()
            if pd.notna(pmi_max_diff):
                delta_pmi = max(20, pmi_max_diff * 1.1)

        m2_limit = 30
        if has_m2 and not df_g4_plot.empty:
            max_m2 = df_g4_plot['g4_m2_yoy'].abs().max()
            if pd.notna(max_m2):
                m2_limit = max(30, max_m2 * 1.1)

        fig_lt19.update_yaxes(title_text="ISM PMI", range=[50 - delta_pmi, 50 + delta_pmi], secondary_y=False, **axis_opts)
        fig_lt19.update_yaxes(title_text="G4 M2 YoY (%)", range=[-m2_limit, m2_limit], secondary_y=True, **axis_opts)
        
        st.plotly_chart(fig_lt19, use_container_width=True)