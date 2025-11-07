import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import warnings
warnings.filterwarnings('ignore')

# --- Утиліти ---

def try_read_file():
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
            else:
                print(f"Файл {p} прочитано, але очікувані колонки не знайдені: {list(df.columns)[:10]}")
                return None, None

    print("Файл з реальними даними не знайдено у поточній папці.")
    return None, None


# --- EDA ---

def eda_report(series: pd.Series, name: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    n = len(series)
    print(f"\n--- EDA для {name} ---")
    print(f"Кількість точок: {n}")
    print(series.describe().to_string())
    print(f"Кількість нулів: {(series==0).sum()}")

    plt.figure(figsize=(10,4))
    plt.plot(series, marker='.', linewidth=0.8)
    plt.title(f"{name} — часовий ряд (raw)")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_timeseries.png"))
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(series.dropna(), bins=40, alpha=0.7)
    plt.title(f"{name} — гістограма")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_hist.png"))
    plt.close()


# --- Детекція аномалій ---

def detect_anomalies_iqr(series: pd.Series, k=1.5):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask, (lower, upper, q1, q3, iqr)


def detect_anomalies_isolationforest(series: pd.Series, contamination=0.01, random_state=42):
    X = series.values.reshape(-1,1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X)
    labels = clf.predict(X)
    mask = labels == -1
    scores = clf.decision_function(X)
    return mask, scores


def detect_anomalies_sliding_mnk(series: pd.Series, n_wind=6, Q=3, min_votes=2):
    arr = series.values.copy()
    n = len(arr)
    votes = np.zeros(n, dtype=int)
    if n < n_wind:
        return votes.astype(bool)

    ref = arr[:n_wind]
    scv_ref = np.std(ref, ddof=0)
    if scv_ref == 0:
        scv_ref = 1e-9

    for j in range(0, n - n_wind + 1):
        window = arr[j:j+n_wind]
        scv_win = np.std(window, ddof=0)
        if scv_win > Q * scv_ref:
            votes[j:j+n_wind] += 1

    for i in range(n):
        left = max(0, i - n_wind//2)
        right = min(n, i + n_wind//2 + 1)
        local = arr[left:right]
        if len(local) > 1:
            med = np.median(local)
            std = np.std(local)
            if std == 0:
                std = 1e-9
            if abs(arr[i] - med) > 2 * std:
                votes[i] += 1

    mask = votes >= min_votes
    return mask


# --- Очищення ----

def clean_using_mask(series: pd.Series, mask: np.ndarray, method='median'):
    s = series.copy().astype(float)
    n = len(s)
    if method == 'median':
        k = 2
        for idx in np.where(mask)[0]:
            left = max(0, idx - k)
            right = min(n, idx + k + 1)
            window = s[left:right].copy()
            window = window[~np.isnan(window)]
            if len(window) > 0:
                s.iloc[idx] = np.median(window)
            else:
                s.iloc[idx] = np.nan
        s = s.interpolate().ffill().bfill()
    elif method == 'interpolate':
        s[mask] = np.nan
        s = s.interpolate().ffill().bfill()
    elif method == 'drop':
        s[mask] = np.nan
        s = s.dropna()
    else:
        raise ValueError("Unknown method")
    return s


# --- Нормалізація ----

def normalize_minmax(series: pd.Series, feature_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    arr = series.values.reshape(-1,1)
    arr_scaled = scaler.fit_transform(arr).reshape(-1)
    return pd.Series(arr_scaled, index=series.index), scaler


# --- Ковзне середнє ----

def compute_moving_averages(series: pd.Series, windows=(3,5,7,14)):
    ma = {}
    for w in windows:
        ma[w] = series.rolling(window=w, min_periods=1, center=False).mean()
    return ma


# --- Стаціонарність ---

def test_stationarity_adf(series: pd.Series, title='series'):
    print(f"\nADF Test for {title}:")
    res = adfuller(series.dropna(), autolag='AIC')
    print(f"ADF stat = {res[0]:.6f}, p-value = {res[1]:.6f}")
    for key,val in res[4].items():
        print(f"  Critical {key}: {val:.3f}")
    if res[1] > 0.05:
        print("-> Non-stationary (p>0.05)")
    else:
        print("-> Stationary (p<=0.05)")
    return res


# --- Arima grid search (просте) -------------------------

def select_arima_order(series: pd.Series, p_range=(0,1,2), d_range=(0,1), q_range=(0,1,2)):
    best_aic = np.inf
    best_order = None
    for p in p_range:
        for d in d_range:
            for q in q_range:
                try:
                    m = ARIMA(series, order=(p,d,q)).fit()
                    aic = m.aic
                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p,d,q)
                except Exception:
                    continue
    print(f"Best ARIMA order by AIC: {best_order} (AIC={best_aic:.2f})")
    return best_order


# --- Оцінка метрик ---

def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, label="model"):
    n = min(len(y_true), len(y_pred))
    y_true = np.array(y_true[:n])
    y_pred = np.array(y_pred[:n])
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (np.where(y_true==0, 1e-9, y_true)))) * 100
    print(f"[{label}] MAE={mae:.6f}, RMSE={rmse:.6f}, MAPE={mape:.3f}%")
    return {'MAE':mae, 'RMSE':rmse, 'MAPE':mape}


# --- Прогноз/екстраполяція ---

def forecast_arima(series: pd.Series, order, extra_steps):
    model = ARIMA(series, order=order)
    fit = model.fit()
    forecast = fit.forecast(steps=extra_steps)
    return fit, forecast


# --- Головний метод ---

def process_series(name: str, series: pd.Series, out_dir: str, run_residuals=False, sliding_params=(6,3,2)):
    os.makedirs(out_dir, exist_ok=True)
    n = len(series)
    print(f"\n== Обробка: {name} (n={n}) ==")

    # 0) Чистимонулі -> NaN -> інтерполяція
    zeros_count = (series == 0).sum()
    print(f"Знайдено нулів: {zeros_count}. Замінюю 0->NaN -> interpolate...")
    s = series.replace(0, np.nan).astype(float)
    s = s.interpolate(method='linear').ffill().bfill()

    # EDA
    eda_report(s, name + '_cleanzeros', out_dir)

    # 1) Детекція аномалій
    mask_iqr, _ = detect_anomalies_iqr(s, k=1.5)
    mask_if, _ = detect_anomalies_isolationforest(s, contamination=0.015)
    n_wind, Q, min_votes = sliding_params
    mask_slide = detect_anomalies_sliding_mnk(s, n_wind=n_wind, Q=Q, min_votes=min_votes)

    print(f"IQR outliers: {mask_iqr.sum()}, IF outliers: {mask_if.sum()}, sliding outliers: {mask_slide.sum()}")

    plt.figure(figsize=(10,4))
    plt.plot(s, label='cleaned (zeros->NaN->interp)')
    plt.scatter(np.where(mask_slide)[0], s[mask_slide], color='red', label='slide outliers')
    plt.legend()
    plt.title(f'{name} - sliding outliers ()')
    plt.savefig(os.path.join(out_dir, f"{name}_outliers_slide.png"))
    plt.close()

    # 2) Очищення (використаємо sliding result як базовий)
    series_clean = clean_using_mask(s, mask_slide, method='median')
    plt.figure(figsize=(10,4))
    plt.plot(s, alpha=0.6, label='after_zero_clean')
    plt.plot(series_clean, label='cleaned', linewidth=1)
    plt.legend()
    plt.title(f'{name} - after zero-clean vs cleaned')
    plt.savefig(os.path.join(out_dir, f"{name}_afterzero_vs_clean.png"))
    plt.close()

    # 3) Нормалізація (для деяких методів, але зберігаємо оригінал для ARIMA)
    series_norm, scaler = normalize_minmax(series_clean, feature_range=(-1,1))
    series_for_models = series_clean

    # 4) Ковзні середні
    windows = (3,5,7,14)
    ma = compute_moving_averages(series_for_models, windows=windows)
    plt.figure(figsize=(12,4))
    plt.plot(series_for_models, label='clean')
    for w in windows:
        plt.plot(ma[w], label=f'MA_{w}')
    plt.legend()
    plt.title(f'{name} - Moving averages ')
    plt.savefig(os.path.join(out_dir, f"{name}_moving_averages.png"))
    plt.close()

    # 5) Декомпозиція
    try:
        decomp = seasonal_decompose(series_for_models.dropna(), period=max(2, min(12, int(n/6))))
        fig = decomp.plot()
        fig.set_size_inches(10,8)
        fig.suptitle(f'{name} - decomposition ')
        fig.savefig(os.path.join(out_dir, f"{name}_decomposition.png"))
        plt.close(fig)
    except Exception as e:
        print('Decomposition failed:', e)

    # 6) ADF тест
    adf_res = test_stationarity_adf(series_for_models, title=name + ' ')

    # 7) Підбір порядку ARIMA (невелика сітка)
    p_range = (0,1,2)
    d_range = (0,1)
    q_range = (0,1,2)
    best_order = select_arima_order(series_for_models, p_range, d_range, q_range)
    if best_order is None:
        best_order = (1,1,1)

    # 8) Backtest: візьмемо останні T_hold як тест
    T_hold = max(7, int(0.15 * n))
    train = series_for_models[:-T_hold]
    test = series_for_models[-T_hold:]

    try:
        model = ARIMA(train, order=best_order).fit()
        pred = model.predict(start=test.index[0], end=test.index[-1], typ='levels')
    except Exception as e:
        print('ARIMA fit/predict failed:', e)
        pred = pd.Series([np.nan]*len(test), index=test.index)

    # Оцінка backtest
    metrics_back = evaluate_metrics(test.values, pred.values, label=f'{name}_ARIMA_backtest')

    plt.figure(figsize=(10,4))
    plt.plot(train.index, train, label='train')
    plt.plot(test.index, test, label='actual_test')
    plt.plot(pred.index, pred, label='pred_test')
    plt.legend()
    plt.title(f'{name} - ARIMA backtest ')
    plt.savefig(os.path.join(out_dir, f"{name}_arima_backtest.png"))
    plt.close()

    # Діагностика залишків моделі
    resid_diag = None
    if run_residuals:
        try:
            resid = model.resid.dropna()
            fig = plt.figure(figsize=(10,4))
            ax = fig.add_subplot(121)
            sm.graphics.tsa.plot_acf(resid, lags=40, ax=ax)
            ax2 = fig.add_subplot(122)
            sm.graphics.tsa.plot_pacf(resid, lags=40, ax=ax2)
            plt.suptitle(f'{name} - residuals ACF/PACF')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{name}_resid_acf_pacf.png"))
            plt.close(fig)

            lb = acorr_ljungbox(resid, lags=[10], return_df=True)
            print(f"Ljung-Box (lag=10) p-value: {lb['lb_pvalue'].values[0]:.6f}")
            resid_diag = {'ljungbox_pvalue': float(lb['lb_pvalue'].values[0])}
        except Exception as e:
            print('Residual diagnostics failed:', e)

    # 9) Прогноз/екстраполяція на задані множники
    ratios = [0.5, 1.0, 1.5, 2.0]
    results = {}
    for r in ratios:
        extra = int(math.ceil(n * r))
        try:
            fit, fc = forecast_arima(series_for_models, best_order, extra)
            results[r] = fc
            # plot
            plt.figure(figsize=(10,4))
            plt.plot(series_for_models.index, series_for_models, label='observed')

            try:
                last_index = series_for_models.index[-1]
                if pd.api.types.is_datetime64_any_dtype(series_for_models.index):
                    fc_index = pd.date_range(start=last_index, periods=extra+1, closed='right', freq=pd.infer_freq(series_for_models.index) or None)
                    if len(fc_index) != len(fc):
                        fc_index = None
                else:
                    fc_index = None
            except Exception:
                fc_index = None

            if fc_index is None:
                fc_index = np.arange(len(series_for_models), len(series_for_models)+len(fc))
            plt.plot(fc_index, fc, linestyle='--', label=f'forecast r={r}')
            plt.legend()
            plt.title(f'{name} - Forecast r={r} (extra={extra})')
            plt.savefig(os.path.join(out_dir, f"{name}_forecast_r{str(r).replace('.','')}.png"))
            plt.close()
        except Exception as e:
            print(f'Forecast failed for r={r}:', e)
            results[r] = None

    # 10) Збереження метрик
    metrics = {
        'name': name,
        'n': n,
        'zeros_replaced': int(zeros_count),
        'iqr_outliers': int(mask_iqr.sum()),
        'if_outliers': int(mask_if.sum()),
        'slide_outliers': int(mask_slide.sum()),
        'arima_order': str(best_order),
        'backtest_mae': metrics_back.get('MAE'),
        'backtest_rmse': metrics_back.get('RMSE')
    }

    if resid_diag is not None:
        metrics.update(resid_diag)

    return metrics, results


# --- Метод запуску ---

def main():
    out_plot_dir = 'lab5_plots/'
    os.makedirs(out_plot_dir, exist_ok=True)

    df, path = try_read_file()
    if df is None:
        print('Немає даних. Виконайте скрипт у папці з Oschadbank (USD).xls')
        return

    print(f'Зчитано файл: {path}. Колонки: {list(df.columns)}')

    all_metrics = []
    all_forecasts = {}
    for col in df.columns:
        series = pd.Series(df[col].values.astype(float))
        metrics, forecasts = process_series(col, series, out_plot_dir, run_residuals=False, sliding_params=(6,3,2))
        all_metrics.append(metrics)
        all_forecasts[col] = forecasts

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv('lab5_metrics_summary.csv', index=False)

    print('\nГотово. Збережено графіки у папці lab5_plots/ та метрики у lab5_metrics_summary.csv')


if __name__ == '__main__':
    main()
