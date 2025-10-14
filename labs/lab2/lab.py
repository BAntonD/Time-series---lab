import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings

# 1) Зчитування даних (функція)
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

            # Залишимо лише наявні з очікуваних колонок в стандартному порядку
            found = [c for c in expected_cols if c in df.columns]
            if found:
                return df[found].copy(), p
            else:
                print(f"Файл {p} прочитано, але очікувані колонки не знайдені: {list(df.columns)[:10]}")
                return None, None

    print("Файл з реальними даними не знайдено у поточній папці.")
    return None, None


# 2) Візуалізація
def eda_report(series: pd.Series, name: str, path_plot: str):
    os.makedirs(os.path.dirname(path_plot), exist_ok=True)
    n = len(series)
    print(f"\n--- EDA для {name} ---")
    print(f"Кількість точок: {n}")
    print(series.describe().to_string())
    print(f"Кількість нулів: {(series==0).sum()}")

    plt.figure(figsize=(10,4))
    plt.plot(series, marker='.', linewidth=0.6)
    plt.title(f"{name} — часовий ряд (raw)")
    plt.ylabel(name)
    plt.tight_layout()
    plt.savefig(path_plot + f"{name}_timeseries.png")
    plt.close()

    plt.figure(figsize=(6,4))
    plt.hist(series.dropna(), bins=40, alpha=0.7)
    plt.title(f"{name} — гістограма")
    plt.tight_layout()
    plt.savefig(path_plot + f"{name}_hist.png")
    plt.close()

# 3) Детекція аномалій: IQR
def detect_anomalies_iqr(series: pd.Series, k=1.5):
    """
    Простий детектор на основі IQR:
    - повертає маску (True — точка вважається аномальною)
    - k=1.5 стандартний;
    """
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (series < lower) | (series > upper)
    return mask, (lower, upper, q1, q3, iqr)

# 4) Детекція: IsolationForest
def detect_anomalies_isolationforest(series: pd.Series, contamination=0.01, random_state=42):
    """
    contamination — очікувана частка аномалій (напр., 0.01 = 1%).
    """
    X = series.values.reshape(-1,1)
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(X)
    labels = clf.predict(X)  # 1 = inlier, -1 = outlier
    mask = labels == -1
    scores = clf.decision_function(X)  # корисно для візуалізації
    return mask, scores

# 5) Детекція: sliding-window MNK (твій підхід)
def detect_anomalies_sliding_mnk(series: pd.Series, n_wind=5, Q=1.6, min_votes=2):
    """
    Пом'якшена sliding-window MNK:
    - n_wind: розмір вікна
    - Q: поріг (як раніше)
    - min_votes: мінімальна кількість "спрацьовувань" для вважати точку аномальною
    Повертає маску аномалій.
    """
    arr = series.values.copy()
    n = len(arr)
    votes = np.zeros(n, dtype=int)
    if n < n_wind:
        return votes.astype(bool)

    # еталонне вікно — перші n_wind точок
    ref = arr[:n_wind]
    scv_ref = np.std(ref, ddof=0)
    if scv_ref == 0:
        scv_ref = 1e-9

    for j in range(0, n - n_wind + 1):
        window = arr[j:j+n_wind]
        scv_win = np.std(window, ddof=0)
        if scv_win > Q * scv_ref:
            for k in range(j, j+n_wind):
                votes[k] += 1

    # додаткова перевірка: якщо значення відхиляється від локальної медіани на > 2*std => додатковий голос
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

# 6) Очищення за маскою
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

# 7) Нормалізація
def normalize_minmax(series: pd.Series, feature_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    arr = series.values.reshape(-1,1)
    arr_scaled = scaler.fit_transform(arr).reshape(-1)
    return pd.Series(arr_scaled, index=series.index), scaler

def normalize_max(series: pd.Series):
    mx = series.max()
    if mx == 0:
        mx = 1e-9
    return series / mx, mx

# 8) Поліноміальна регресія МНК
def fit_polynomial_mnk(series: pd.Series, degree=2):
    x = np.arange(len(series))
    y = series.values
    coeffs = np.polyfit(x, y, deg=degree)  # highest->lowest
    p = np.poly1d(coeffs)
    y_fit = p(x)
    return pd.Series(y_fit, index=series.index), coeffs

# 9) Екстраполяція на 0.5 інтервалу вибірки
def extrapolate_polynomial(series_fit: pd.Series, coeffs, extrapol_ratio=0.5):
    n = len(series_fit)
    extra = int(np.ceil(n * extrapol_ratio))
    x_full = np.arange(n + extra)
    p = np.poly1d(coeffs)
    y_full = p(x_full)
    idx = np.arange(n + extra)
    return pd.Series(y_full, index=idx)

# 10) Оцінка якості (MAE, MSE, RMSE, R2)
def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, label="model"):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE={mae:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")
    return {'MAE':mae, 'MSE':mse, 'RMSE':rmse, 'R2':r2}


# 11) Головна програма
def main():
    out_plot_dir = "data_plot/"
    os.makedirs(out_plot_dir, exist_ok=True)

    df, path = try_read_real_excel_simple()
    if df is None:
        print("Закінчую: немає реальних даних.")
        return

    print(f"Зчитано файл: {path}. Колонки у DataFrame: {list(df.columns)}")
    # Візьмемо колонку 'Купівля' як приклад
    col = 'Купівля'
    if col not in df.columns:
        col = df.columns[0]

    series_raw = pd.Series(df[col].values.astype(float))
    eda_report(series_raw, col, out_plot_dir)

    # 1) Детекція аномалій різними методами
    print("\n-- Детекція IQR (k=1.5) --")
    mask_iqr, iqr_info = detect_anomalies_iqr(series_raw, k=1.5)
    print(f"Знайдено IQR аномалій: {mask_iqr.sum()}")
    # збережемо візуалізацію
    plt.figure(figsize=(10,4))
    plt.plot(series_raw, label='raw')
    plt.scatter(np.where(mask_iqr)[0], series_raw[mask_iqr], color='red', label='IQR outliers')
    plt.legend()
    plt.title('IQR outliers')
    plt.savefig(out_plot_dir + 'outliers_iqr.png')
    plt.close()

    print("\n-- Детекція IsolationForest (contamination=0.015) --")
    mask_if, scores_if = detect_anomalies_isolationforest(series_raw, contamination=0.015)
    print(f"Знайдено IsolationForest аномалій: {mask_if.sum()}")
    plt.figure(figsize=(10,4))
    plt.plot(series_raw, label='raw')
    plt.scatter(np.where(mask_if)[0], series_raw[mask_if], color='red', label='IF outliers')
    plt.legend()
    plt.title('IsolationForest outliers')
    plt.savefig(out_plot_dir + 'outliers_if.png')
    plt.close()

    print("\n-- Детекція sliding-window MNK (n_wind=6, Q=3) --")
    mask_slide = detect_anomalies_sliding_mnk(series_raw, n_wind=6, Q=3, min_votes=2)
    print(f"Знайдено sliding-window аномалій: {mask_slide.sum()}")
    plt.figure(figsize=(10,4))
    plt.plot(series_raw, label='raw')
    plt.scatter(np.where(mask_slide)[0], series_raw[mask_slide], color='red', label='slide outliers')
    plt.legend()
    plt.title('Sliding-window outliers')
    plt.savefig(out_plot_dir + 'outliers_slide.png')
    plt.close()

    # 2) Очищення: виберемо метод median-interp (локальна медіана)
    series_clean_iqr = clean_using_mask(series_raw, mask_iqr, method='median')
    series_clean_if = clean_using_mask(series_raw, mask_if, method='median')
    series_clean_slide = clean_using_mask(series_raw, mask_slide, method='median')

    for name, s in [('iqr', series_clean_iqr), ('if', series_clean_if), ('slide', series_clean_slide)]:
        plt.figure(figsize=(10,4))
        plt.plot(series_raw, label='raw', alpha=0.6)
        plt.plot(s, label=f'clean_{name}', linewidth=1)
        plt.legend()
        plt.title(f'Raw vs cleaned ({name})')
        plt.savefig(out_plot_dir + f'raw_vs_clean_{name}.png')
        plt.close()

    # 3) Нормалізація
    series_for_model = series_clean_slide.copy()
    series_norm, scaler = normalize_minmax(series_for_model, feature_range=(-1,1))

    # 4) Поліном МНК
    degree = 2
    y_fit, coeffs = fit_polynomial_mnk(series_for_model, degree=degree)
    # зберегти графік апроксимації
    plt.figure(figsize=(10,4))
    plt.plot(series_for_model, label='cleaned (slide)')
    plt.plot(y_fit, label=f'poly_fit_deg{degree}', linewidth=2)
    plt.legend()
    plt.title('Поліноміальна апроксимація (МНК)')
    plt.savefig(out_plot_dir + f'poly_fit_deg{degree}.png')
    plt.close()

    # 5) Екстраполяція на 0.5 інтервалу вибірки
    y_full = extrapolate_polynomial(y_fit, coeffs, extrapol_ratio=0.5)
    plt.figure(figsize=(10,4))
    n = len(series_for_model)
    plt.plot(range(n), series_for_model, label='observed')
    plt.plot(range(n, len(y_full)), y_full[n:], label='forecast', linestyle='--')
    plt.legend()
    plt.title('Прогноз (екстраполяція поліномом)')
    plt.savefig(out_plot_dir + 'poly_forecast.png')
    plt.close()

    # 6) Оцінка якості: порівняння до/після очищення
    print("\n-- Оцінка моделей (до/після очищення) --")
    metrics_raw = evaluate_metrics(series_raw.values, np.poly1d(np.polyfit(np.arange(len(series_raw)), series_raw.values, degree))(np.arange(len(series_raw))), label='raw_polyfit')
    metrics_clean = evaluate_metrics(series_for_model.values, y_fit.values, label='clean_slide_polyfit')

    # Зберігання метрики у CSV для звіту
    metrics_df = pd.DataFrame([metrics_raw, metrics_clean])
    metrics_df.index = ['raw_polyfit', 'clean_slide_polyfit']
    metrics_df.to_csv('metrics_comparison.csv')

    print("\nГотово. Збережено графіки у папці 'data_plot/' та 'metrics_comparison.csv'.")

if __name__ == "__main__":
    main()
