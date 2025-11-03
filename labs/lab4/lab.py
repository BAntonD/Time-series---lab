import os, sys, math, json, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss, acf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Перевірка hurst
try:
    from hurst import compute_Hc
    _HURST_AVAILABLE = True
except Exception:
    _HURST_AVAILABLE = False

# --- Утиліти для роботи з файлами та візуалізаціями ---

# Створює структуру папок для збереження результатів
def ensure_dirs(base='outputs'):
    os.makedirs(base, exist_ok=True)
    plots = os.path.join(base, 'plots'); os.makedirs(plots, exist_ok=True)
    csv = os.path.join(base, 'csv'); os.makedirs(csv, exist_ok=True)
    analysis = os.path.join(base, 'analysis'); os.makedirs(analysis, exist_ok=True)
    return base, plots, csv, analysis

# Зберігає серію у CSV
def save_series_csv(series, path):
    pd.DataFrame({'value': np.asarray(series)}).to_csv(path, index=False)

# Зберігає графіки у файл
def save_plot(fig_or_ax, fname):
    try:
        if hasattr(fig_or_ax, 'savefig'):
            fig = fig_or_ax
        else:
            fig = plt.gcf()
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
    except Exception:
        plt.savefig(fname, dpi=150)
    plt.close('all')

# Зберігає графіки серії з опційним трендом
def safe_plot_series(series, title, fname=None, trend=None):
    plt.figure(figsize=(10,4))
    plt.plot(series, label='series')
    if trend is not None:
        plt.plot(trend, label='trend', linewidth=2)
    plt.title(title); plt.legend()
    if fname:
        save_plot(plt.gcf(), fname)
    else:
        plt.show()
    plt.close('all')

# Зберігає гістограми серій
def save_hist(series, bins=30, title='hist', fname=None):
    plt.figure(figsize=(8,4))
    plt.hist(series, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    if fname:
        save_plot(plt.gcf(), fname)
    else:
        plt.show()
    plt.close('all')

# --- Робота з пропущеними даними ---

# Замінює нулі на NaN та інтерполює серію
def fix_zeros_as_nan_and_interpolate(arr):
    s = pd.Series(arr).astype(float).copy()
    zeros_idx = (s == 0)
    if zeros_idx.sum() == 0:
        return s.values
    s[zeros_idx] = np.nan
    s = s.interpolate(method='linear', limit_direction='both')
    s = s.ffill().bfill()
    return s.values

# --- Читання реальних даних ---

# Просте читання Excel файлу з валютними даними
def try_read_real_excel_simple():
    default_paths = [
        'Oschadbank (USD).xls',
        'Oschadbank (USD).xlsx',
        'Oschadbank (USD).xlsm'
    ]
    expected_cols = ['Купівля', 'Продаж', 'КурсНбу']
    for p in default_paths:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
            except Exception as e:
                print(f"Не можу прочитати {p}: {e}")
                return None, None
            found = [c for c in expected_cols if c in df.columns]
            if found:
                return df[found].copy(), p
            nums = df.select_dtypes(include=[np.number]).columns.tolist()
            if nums:
                return df[nums].copy(), p
    return None, None

# --- Генерація синтетичних серій ---

# Генерує тренди
def generate_trend(kind: str, params: dict, n: int):
    i = np.arange(n, dtype=float)
    if kind == 'linear':
        return params.get('m',0.0)*i + params.get('c',0.0)
    if kind == 'quadratic':
        a=params.get('a',0); b=params.get('b',0); c=params.get('c',0)
        return a*i**2 + b*i + c
    if kind == 'cubic':
        a=params.get('a',0); b=params.get('b',0); c=params.get('c',0); d=params.get('d',0)
        return a*i**3 + b*i**2 + c*i + d
    raise ValueError("Unknown trend")

# енерує шум
def generate_noise(kind: str, params: dict, n: int):
    if kind == 'normal':
        return np.random.normal(loc=params.get('loc',0.0), scale=params.get('scale',1.0), size=n)
    if kind == 'uniform':
        return np.random.uniform(low=params.get('low',-1.0), high=params.get('high',1.0), size=n)
    raise ValueError("Unknown noise")

# Синтез серії, тренд + шум
def synthesize_series(trend_kind, trend_params, noise_kind, noise_params, n):
    trend = generate_trend(trend_kind, trend_params, n)
    noise = generate_noise(noise_kind, noise_params, n)
    return trend + noise, trend, noise

# Синтетична серія на основі параметрів STL (тренд + сезонність + шум)
def gen_synth_from_stl(n, trend_slope=0.0, seasonal_amp=1.0, period=12, noise_sigma=1.0, baseline=0.0, phase=0):
    t = np.arange(n)
    trend = baseline + trend_slope * t
    seasonal = seasonal_amp * np.sin(2*np.pi*(t+phase)/period)
    noise = np.random.normal(0, noise_sigma, size=n)
    return trend + seasonal + noise, trend, seasonal, noise

# --- Статистика та Hurst ---

# Базові статистики серії (mean, std, skew, kurtosis, min/max)
def compute_stats(arr):
    s = pd.Series(arr).dropna().astype(float)
    return {
        'n': int(len(s)),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'var': float(s.var()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'skew': float(s.skew()),
        'kurtosis': float(s.kurtosis())
    }


# Обчислення експоненти Херста (H) для серії
def compute_hurst(series):
    if not _HURST_AVAILABLE:
        return None
    try:
        s = pd.Series(series).dropna().astype(float)
    except Exception as e:
        return {'error': f'bad input: {e}'}
    if len(s) < 20:
        return {'error': 'too short'}
    if np.isclose(s.max(), s.min()):
        return {'error': 'constant series'}
    if (s <= 0).any():
        s = s + (abs(s.min()) + 1e-8)
    try:
        H, c, _ = compute_Hc(s, kind='price', simplified=True)
        if not (np.isfinite(H) and 0 < H < 2):
            return {'error': f'bad H: {H}'}
        return {'H': float(H), 'c': float(c)}
    except FloatingPointError as e:
        return {'error': f'FloatingPointError: {e}'}
    except Exception as e:
        return {'error': f'hurst failed: {e}'}

# --- Тестування стаціонарності ---

# ADF тест на стаціонарність
def adf_test(series):
    s = pd.Series(series).dropna().astype(float)
    try:
        res = adfuller(s, autolag='AIC')
        return {'adf_stat': float(res[0]), 'pvalue': float(res[1]), 'used_lags': int(res[2]), 'nobs': int(res[3]), 'crit_vals': res[4]}
    except Exception as e:
        return {'error': str(e)}

# KPSS тест на стаціонарність
def kpss_test(series):
    s = pd.Series(series).dropna().astype(float)
    try:
        res = kpss(s, regression='c', nlags='auto')
        return {'kpss_stat': float(res[0]), 'pvalue': float(res[1]), 'lags': int(res[2]), 'crit_vals': res[3]}
    except Exception as e:
        return {'error': str(e)}

# --- Декомпозиція ---
def decompose_stl(ts_series, period=12, robust=True):
    s = pd.Series(ts_series).dropna().astype(float)
    stl = STL(s, period=period, robust=robust)
    res = stl.fit()
    return res

# --- Аномалії ---

# Аномалії по IQR
def detect_anomalies_iqr(series, k=1.5):
    s = pd.Series(series).dropna().astype(float)
    q1 = s.quantile(0.25); q3 = s.quantile(0.75); iqr = q3 - q1
    lower = q1 - k*iqr; upper = q3 + k*iqr
    s_full = pd.Series(series)
    mask = (s_full < lower) | (s_full > upper)
    mask = mask.fillna(False).values.astype(bool)
    return mask, (lower, upper, q1, q3, iqr)

# Аномалії через Isolation Forest
def detect_anomalies_isolationforest(series, contamination=0.01, random_state=42):
    arr = np.asarray(pd.Series(series).fillna(method='ffill').fillna(method='bfill'), dtype=float).reshape(-1,1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(arr)
    labels = clf.predict(arr)
    mask = labels == -1
    scores = clf.decision_function(arr)
    return mask, scores

# Аномалії через ковзне вікно + MNK
def detect_anomalies_sliding_mnk(series, n_wind=5, Q=1.6, min_votes=2):
    arr = np.asarray(series, dtype=float)
    n = len(arr)
    votes = np.zeros(n, dtype=int)
    if n < n_wind:
        return votes.astype(bool)
    ref = arr[:n_wind]; scv_ref = np.std(ref) if np.std(ref)>0 else 1e-9
    for j in range(0, n - n_wind + 1):
        window = arr[j:j+n_wind]; scv_win = np.std(window)
        if scv_win > Q * scv_ref:
            votes[j:j+n_wind] += 1
    for i in range(n):
        left = max(0, i - n_wind//2); right = min(n, i + n_wind//2 + 1)
        local = arr[left:right]
        if len(local) > 1:
            med = np.median(local); std = np.std(local) if np.std(local)>0 else 1e-9
            if abs(arr[i] - med) > 2 * std:
                votes[i] += 1
    return votes >= min_votes

# Очищення серії за маскою аномалій
def clean_using_mask(series, mask, method='median'):
    s = pd.Series(series).astype(float).copy()
    n = len(s)
    if method == 'median':
        k = 2
        for idx in np.where(mask)[0]:
            left = max(0, idx - k); right = min(n, idx + k + 1)
            window = s.iloc[left:right].values
            window = window[~np.isnan(window)]
            if len(window) > 0:
                s.iloc[idx] = np.median(window)
            else:
                s.iloc[idx] = np.nan
        s = s.interpolate().ffill().bfill()
    elif method == 'interpolate':
        s[mask] = np.nan; s = s.interpolate().ffill().bfill()
    elif method == 'drop':
        s[mask] = np.nan; s = s.dropna()
    else:
        raise ValueError("Unknown method")
    return s.values

# --- Фільтри ---

# ABF(Adaptive Basic Filter)
def ABF(S0, alpha=None, beta=None):
    iter_n = len(S0); Yin = np.asarray(S0, dtype=float).ravel(); YoutAB = np.zeros(iter_n)
    if iter_n < 2: return Yin.copy()
    T0 = 1.0; Yspeed_retro = (Yin[1]-Yin[0])/T0; Yextra = Yin[0] + Yspeed_retro
    alfa = alpha if alpha is not None else (2*(2*1-1)/(1*(1+1))); beta_val = beta if beta is not None else (6/1*(1+1))
    YoutAB[0] = Yin[0] + alfa * (Yin[0])
    for i in range(1, iter_n):
        YoutAB[i] = Yextra + alfa*(Yin[i] - Yextra)
        Yspeed = Yspeed_retro + (beta_val / T0) * (Yin[i] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i] + Yspeed_retro
        if alpha is None:
            alfa = (2*(2*i - 1))/(i*(i+1))
        else:
            alfa = alpha
        if beta is None:
            beta_val = 6/(i*(i+1))
        else:
            beta_val = beta
    return YoutAB

# ABGF(Adaptive Basic Filter)
def ABGF(S0, alpha=None, beta=None, gamma=None):
    iter_n = len(S0); Yin = np.asarray(S0, dtype=float).ravel(); Yout = np.zeros(iter_n)
    if iter_n < 3: return Yin.copy()
    T0 = 1.0; Yspeed_retro = (Yin[1]-Yin[0])/T0; Yaccel_retro = (Yin[2]-2*Yin[1]+Yin[0])/(T0*T0)
    Yextra = Yin[0] + Yspeed_retro*T0 + Yaccel_retro*0.5*T0*T0
    alpha_local = alpha if alpha is not None else (3*(3*1*1-3*1+2)/(1*(1+1)*(1+2)))
    beta_local  = beta  if beta  is not None else (18*(2*1-1)/(T0*(1+1)*(1+2)*1))
    gamma_local = gamma if gamma is not None else (60/(T0*T0*(1+1)*(1+2)*1))
    Yout[0] = Yin[0]
    for i in range(1, iter_n):
        Yout[i] = Yextra + alpha_local*(Yin[i] - Yextra)
        Yspeed = Yspeed_retro + (beta_local / T0) * (Yin[i] - Yextra)
        Yaccel = Yaccel_retro + (gamma_local / (T0*T0)) * (Yin[i] - Yextra)
        Yspeed_retro, Yaccel_retro = Yspeed, Yaccel
        Yextra = Yout[i] + Yspeed_retro*T0 + Yaccel*0.5*T0*T0
        if alpha is None:
            alpha_local = 3*(3*i*i - 3*i + 2)/(i*(i+1)*(i+2))
        if beta is None:
            beta_local = 18*(2*i - 1)/(T0*(i+1)*(i+2)*i)
        if gamma is None:
            gamma_local = 60/(T0*T0*(i+1)*(i+2)*i)
    return Yout

# Класичний одновимірний Калман
def kalman_1d(z, Q=1e-3, R=1.0, x0=None, P0=None):
    z = np.asarray(z, dtype=float); n=len(z); x=np.zeros(n); P=np.zeros(n)
    x[0] = x0 if x0 is not None else z[0]; P[0] = P0 if P0 is not None else 1.0
    for k in range(1,n):
        x_pred = x[k-1]; P_pred = P[k-1] + Q
        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (z[k] - x_pred)
        P[k] = (1 - K) * P_pred
    return x

# Оцінка точності фільтра (bias, RMSE, MAE)
def evaluate_filter(true, estimate):
    true = np.asarray(true).ravel()[:len(estimate)]; estimate = np.asarray(estimate).ravel()
    bias = np.mean(estimate - true); rmse = math.sqrt(np.mean((estimate-true)**2)); mae = np.mean(np.abs(estimate-true))
    return {'bias': float(bias), 'rmse': float(rmse), 'mae': float(mae)}

# - Автотюнінг фільтрів -

# Підібрати параметри ABF
def autotune_abf(series, cleaned, alpha_range, beta_range, w_bias=1.0, w_rmse=0.5):
    best_score=None; best_params=None; best_est=None
    for a in alpha_range:
        for b in beta_range:
            est = ABF(cleaned, alpha=a, beta=b)
            m = evaluate_filter(series, est)
            score = w_bias*abs(m['bias']) + w_rmse*m['rmse']
            if best_score is None or score < best_score:
                best_score = score; best_params=(a,b); best_est=est.copy()
    return best_params, best_est, best_score

# Підібрати параметри ABGF
def autotune_abgf(series, cleaned, alpha_range, beta_range, gamma_range, w_bias=1.0, w_rmse=0.5):
    best_score=None; best_params=None; best_est=None
    for a in alpha_range:
        for b in beta_range:
            for g in gamma_range:
                est = ABGF(cleaned, alpha=a, beta=b, gamma=g)
                m = evaluate_filter(series, est)
                score = w_bias*abs(m['bias']) + w_rmse*m['rmse']
                if best_score is None or score < best_score:
                    best_score = score; best_params=(a,b,g); best_est=est.copy()
    return best_params, best_est, best_score

#  Підібрати параметри Калман
def autotune_kalman(series, cleaned, Q_range, R_range, w_bias=1.0, w_rmse=0.5):
    best_score=None; best_params=None; best_est=None
    for Q in Q_range:
        for R in R_range:
            est = kalman_1d(cleaned, Q=Q, R=R)
            m = evaluate_filter(series, est)
            score = w_bias*abs(m['bias']) + w_rmse*m['rmse']
            if best_score is None or score < best_score:
                best_score = score; best_params=(Q,R); best_est=est.copy()
    return best_params, best_est, best_score

# --- Поліноміальна апроксимація ---

# Поліноміальна апроксимація (МНК)
def fit_polynomial_mnk(series, degree=2):
    x=np.arange(len(series)); y=np.asarray(series,dtype=float)
    coeffs = np.polyfit(x, y, deg=degree)
    p = np.poly1d(coeffs); y_fit = p(x)
    return y_fit, coeffs

# Прогноз серії за поліномом
def extrapolate_polynomial(coeffs, n, extrapol_ratio=0.5):
    extra = int(math.ceil(n*extrapol_ratio)); x_full=np.arange(n+extra); p=np.poly1d(coeffs); return p(x_full)

# MAE, RMSE, MSE, R2
def evaluate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred); mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse); r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# -- Порівняння серій ---
# відстань між серіями по автокореляції
def acf_distance(a, b, nlags=20):
    a_acf = acf(pd.Series(a).dropna(), nlags=nlags, fft=False)
    b_acf = acf(pd.Series(b).dropna(), nlags=nlags, fft=False)
    L = min(len(a_acf), len(b_acf))
    return float(np.linalg.norm(a_acf[:L] - b_acf[:L]))

# --- Кластеризація ---

# Обчислення ознак серії для кластеризації
def compute_features_for_series(name, series, seasonal_strength=None, hurst_res=None, adf_res=None):
    stats = compute_stats(series)
    feat = {
        'name': name,
        'mean': stats['mean'], 'std': stats['std'], 'skew': stats['skew'], 'kurtosis': stats['kurtosis'],
        'seasonal_strength': (seasonal_strength if seasonal_strength is not None else np.nan),
        'hurst': (hurst_res.get('H') if isinstance(hurst_res, dict) and 'H' in hurst_res else np.nan),
        'adf_pvalue': (adf_res.get('pvalue') if isinstance(adf_res, dict) and 'pvalue' in adf_res else np.nan)
    }
    return feat

# KMeans кластеризація серій за ознаками + PCA візуалізація
def run_simple_clustering(features_df, k=3, out_plots=None):
    X = features_df.select_dtypes(include=[np.number]).fillna(0).values
    if X.shape[0] < 2:
        return None
    k = min(k, X.shape[0])
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    features_df['cluster'] = labels
    # PCA plot
    pca = PCA(n_components=2)
    Z = pca.fit_transform(X)
    plt.figure(figsize=(6,5))
    for lab in np.unique(labels):
        mask = labels == lab
        plt.scatter(Z[mask,0], Z[mask,1], label=f'c{lab}')
    plt.title('Clusters (PCA 2D)')
    plt.legend()
    if out_plots:
        save_plot(plt.gcf(), os.path.join(out_plots, 'clusters_pca.png'))
    plt.close('all')
    return features_df

# --- Головний пайплайн ---
def main():
    np.random.seed(42)
    out_base, out_plots, out_csv, out_analysis = ensure_dirs('outputs')
    summary = {'synth': [], 'real': {}, 'filters': {}, 'decomposition': {}, 'clusters': {}}

    # 1) Генерація базових синтетичних серій
    combos = [
        ('cubic', {'a':1e-9,'b':1e-7,'c':1e-4,'d':0.0}, 'normal', {'loc':0.0,'scale':1.0}, 'cubic_normal'),
        ('cubic', {'a':1e-9,'b':1e-7,'c':1e-4,'d':0.0}, 'uniform', {'low':-2.0,'high':2.0}, 'cubic_uniform'),
        ('linear', {'m':0.001,'c':0.0}, 'normal', {'loc':0.0,'scale':0.5}, 'linear_normal'),
        ('linear', {'m':0.001,'c':0.0}, 'uniform', {'low':-1.0,'high':1.0}, 'linear_uniform'),
    ]
    n = 500
    synth_series_list = []
    for trend_k, trend_p, noise_k, noise_p, name in combos:
        # Генеруємо синтетичну серію
        series, trend, noise = synthesize_series(trend_k, trend_p, noise_k, noise_p, n)
        synth_series_list.append((name, series, trend, noise))

        # Зберігаємо CSV та графіки
        save_series_csv(series, os.path.join(out_csv, f"{name}.csv"))
        safe_plot_series(series, title=f"SYNTH {name}", fname=os.path.join(out_plots, f"{name}.png"), trend=trend)
        save_hist(noise, bins=40, title=f"{name} noise", fname=os.path.join(out_plots, f"{name}_noise_hist.png"))

        # Додаємо статистику в summary
        summary['synth'].append({'name': name, 'stats': compute_stats(series)})

    # 2) Обробка реальних даних
    real_df, real_path = try_read_real_excel_simple()
    feature_rows = []
    if real_df is not None:
        print("Found real:", real_path)
        for col in real_df.columns:
            raw_vals = pd.Series(real_df[col].values.astype(float)).ravel()

            # Заміна нулів на NaN та інтерполяція серії
            vals = fix_zeros_as_nan_and_interpolate(raw_vals)

            # Зберігаємо серію та графіки
            save_series_csv(vals, os.path.join(out_csv, f"real_{col}.csv"))
            safe_plot_series(vals, title=f"REAL {col}", fname=os.path.join(out_plots, f"real_{col}.png"))
            save_hist(vals, bins=40, title=f"REAL {col} hist", fname=os.path.join(out_plots, f"real_{col}_hist.png"))
            summary['real'][col] = {'stats': compute_stats(vals)}

            # STL-декомпозиція та параметри
            period = 12 if len(vals) > 24 else 4
            try:
                stl_res = decompose_stl(vals, period=period, robust=True)
                plt.figure(figsize=(9,4)); plt.plot(stl_res.trend, label='trend'); plt.plot(stl_res.seasonal, label='seasonal'); plt.legend()
                save_plot(plt.gcf(), os.path.join(out_plots, f"stl_{col}.png"))

                # Сила сезонності
                seasonal_strength = 1 - np.var(stl_res.resid) / np.var(stl_res.trend + stl_res.resid) if np.var(stl_res.trend + stl_res.resid)!=0 else np.nan
                summary['decomposition'][col] = {'seasonal_strength': float(seasonal_strength)}

                # Нахил тренду та амплітуда сезонності
                trend = stl_res.trend
                trend_idx = np.arange(len(trend)); mask = ~np.isnan(trend)
                slope = float(np.polyfit(trend_idx[mask], trend[mask], 1)[0]) if mask.sum() >= 2 else 0.0
                seasonal_amp = (np.nanmax(stl_res.seasonal) - np.nanmin(stl_res.seasonal)) / 2.0
                noise_sigma = float(np.nanstd(stl_res.resid))
            except Exception as e:
                summary['decomposition'][col] = {'error': str(e)}
                slope = 0.0; seasonal_amp = 0.0; noise_sigma = float(np.nanstd(vals))

            # Обчислення властивостей серії
            hurst_res = compute_hurst(vals) if _HURST_AVAILABLE else None
            adf_res = adf_test(vals)
            kpss_res = kpss_test(vals)
            summary['real'][col].update({'hurst': hurst_res, 'adf': adf_res, 'kpss': kpss_res})

            # Виявлення аномалій різними методами
            mask_iqr, iqr_info = detect_anomalies_iqr(vals, k=1.5)
            mask_if, scores_if = detect_anomalies_isolationforest(vals, contamination=0.015)
            mask_slide = detect_anomalies_sliding_mnk(vals, n_wind=6, Q=3, min_votes=2)

            # збереження графіків аномалій
            plt.figure(figsize=(10,3)); plt.plot(vals, label='raw'); plt.scatter(np.where(mask_iqr)[0], vals[mask_iqr], color='red', label='IQR'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"outliers_{col}_iqr.png"))
            plt.figure(figsize=(10,3)); plt.plot(vals, label='raw'); plt.scatter(np.where(mask_if)[0], vals[mask_if], color='red', label='IF'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"outliers_{col}_if.png"))
            plt.figure(figsize=(10,3)); plt.plot(vals, label='raw'); plt.scatter(np.where(mask_slide)[0], vals[mask_slide], color='red', label='slide'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"outliers_{col}_slide.png"))

            # Очищення даних на основі sliding mask
            cleaned = clean_using_mask(vals, mask_slide, method='median')
            save_series_csv(cleaned, os.path.join(out_csv, f"real_{col}_cleaned.csv"))
            safe_plot_series(cleaned, title=f"REAL {col} cleaned", fname=os.path.join(out_plots, f"real_{col}_cleaned.png"))

            # Автоналаштування фільтрів
            w_bias = 3.0; w_rmse = 0.5
            best_abf_params, best_abf_est, best_abf_score = autotune_abf(vals, cleaned, [None,0.02,0.05,0.1,0.2], [None,0.01,0.05,0.1], w_bias, w_rmse)
            best_abgf_params, best_abgf_est, best_abgf_score = autotune_abgf(vals, cleaned, [None,0.1,0.2], [None,0.01,0.05], [None,0.001,0.01], w_bias, w_rmse)
            best_kalman_params, best_kalman_est, best_kalman_score = autotune_kalman(vals, cleaned, [1e-6,1e-4,1e-3], [1e-2,1e-1,1.0], w_bias, w_rmse)
            summary['filters'][col] = {
                'abf_params': best_abf_params, 'abf_score': best_abf_score, 'abf_metrics': evaluate_filter(vals, best_abf_est),
                'abgf_params': best_abgf_params, 'abgf_score': best_abgf_score, 'abgf_metrics': evaluate_filter(vals, best_abgf_est),
                'kalman_params': best_kalman_params, 'kalman_score': best_kalman_score, 'kalman_metrics': evaluate_filter(vals, best_kalman_est)
            }

            # Збереження графіків фільтрів
            plt.figure(figsize=(10,4)); plt.plot(vals, label='orig', alpha=0.6); plt.plot(cleaned, label='cleaned'); plt.plot(best_abf_est, label='abf'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"{col}_filters_abf.png"))
            plt.figure(figsize=(10,4)); plt.plot(vals, label='orig', alpha=0.6); plt.plot(best_abgf_est, label='abgf'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"{col}_filters_abgf.png"))
            plt.figure(figsize=(10,4)); plt.plot(vals, label='orig', alpha=0.6); plt.plot(best_kalman_est, label='kalman'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"{col}_filters_kalman.png"))

            # Поліноміальна апроксимація та екстраполяція
            y_fit, coeffs = fit_polynomial_mnk(cleaned, degree=2)
            metrics_fit = evaluate_metrics(cleaned[:len(y_fit)], y_fit)
            y_full = extrapolate_polynomial(coeffs, len(cleaned), extrapol_ratio=0.5)
            plt.figure(figsize=(10,4)); plt.plot(cleaned, label='cleaned'); plt.plot(y_fit, label='poly_fit'); plt.plot(range(len(cleaned), len(y_full)), y_full[len(cleaned):], '--', label='forecast'); plt.legend()
            save_plot(plt.gcf(), os.path.join(out_plots, f"{col}_polyfit.png"))
            summary['real'][col].update({'poly_fit_metrics': metrics_fit, 'poly_coeffs':[float(x) for x in coeffs.tolist()]})

            # Генерація синтетичної серії на основі STL-параметрів реальної серії
            synth_stl, st_trend, st_seasonal, st_noise = gen_synth_from_stl(len(cleaned), trend_slope=slope, seasonal_amp=seasonal_amp, period=period, noise_sigma=noise_sigma, baseline=float(np.nanmean(cleaned)))
            name_stl = f"synth_stl_{col}"
            save_series_csv(synth_stl, os.path.join(out_csv, f"{name_stl}.csv"))
            safe_plot_series(synth_stl, title=f"SYNTH_STL {col}", fname=os.path.join(out_plots, f"{name_stl}.png"))
            summary['synth'].append({'name': name_stl, 'stats': compute_stats(synth_stl), 'from_real': col})

            # Вибір найкращої синтетичної серії серед всіх (оригінальні + STL)
            alpha = 0.8; beta = 0.2
            sreal_stats = compute_stats(cleaned)
            best_name = None; best_score = None
            candidates = []
            for sname, sseries, strend, snoise in synth_series_list:
                candidates.append((sname, sseries))
            candidates.append((name_stl, synth_stl))
            for sname, sseries in candidates:
                minlen = min(len(cleaned), len(sseries))
                if minlen < 10:
                    continue
                mse = float(np.mean((cleaned[:minlen] - sseries[:minlen])**2))
                norm_mse = mse / (sreal_stats['var'] + 1e-9)
                acfd = acf_distance(cleaned[:minlen], sseries[:minlen], nlags=20)
                acf_norm = acfd / (1.0 + acfd)
                score = alpha * norm_mse + beta * acf_norm
                if best_score is None or score < best_score:
                    best_score = score; best_name = sname; best_metrics = {'mse': mse, 'norm_mse': norm_mse, 'acf_distance': acfd}
            summary['real'][col].update({'best_synth_match': {'name': best_name, 'score': float(best_score)}, 'compare_to_synth': best_metrics})

            # Збір ознак для кластеризації
            feature_rows.append(compute_features_for_series(col, cleaned, seasonal_strength=(seasonal_strength if 'seasonal_strength' in locals() else np.nan), hurst_res=hurst_res, adf_res=adf_res))

    else:
        print("No real data found: running only synth analyses.")
        for name, series, trend, noise in synth_series_list:
            try:
                stl = decompose_stl(series, period=12 if len(series)>48 else 4, robust=True)
                plt.figure(figsize=(7,3)); plt.plot(stl.trend); plt.plot(stl.seasonal); save_plot(plt.gcf(), os.path.join(out_plots, f"stl_{name}.png"))
                summary['synth'].append({'name': name, 'seasonal_strength': float(1 - np.var(stl.resid) / np.var(stl.trend + stl.resid))})
            except Exception as e:
                summary['synth'].append({'name': name, 'error': str(e)})

    # Виконання кластеризації на зібраних ознаках (якщо вони є)
    if len(feature_rows) > 0:
        feats_df = pd.DataFrame(feature_rows)
        feats_df.to_csv(os.path.join(out_analysis, 'features.csv'), index=False)
        clustered = run_simple_clustering(feats_df, k=min(3, max(2, len(feats_df))), out_plots=out_plots)
        if clustered is not None:
            clustered.to_csv(os.path.join(out_analysis, 'features_with_clusters.csv'), index=False)
            summary['clusters']['features_file'] = os.path.join(out_analysis, 'features_with_clusters.csv')

    # Збереження підсумку всіх результатів
    with open(os.path.join(out_base, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("Done. Results in 'outputs/' (summary.json, csv, plots).")

if __name__ == '__main__':
    main()
