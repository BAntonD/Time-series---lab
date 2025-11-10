import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Налаштування виводу/шляхів
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)



# Читання файлу
def try_read_file(preferred_names=None):
    cwd = os.getcwd()
    if preferred_names is None:
        preferred_names = [
            'Oschadbank (USD).xls',
            'Oschadbank (USD).xlsx',
            'Oschadbank (USD).xlsm',
            'AirPassengers.csv',
            'daily-website-visitors.csv',
            'data.csv'
        ]

    found_any = False
    for name in preferred_names:
        p = os.path.join(cwd, name)
        if os.path.exists(p):
            found_any = True
            try:
                if p.lower().endswith(('.xls', '.xlsx', '.xlsm')):
                    df = pd.read_excel(p)
                else:
                    df = pd.read_csv(p)
            except Exception as e:
                raise RuntimeError(f"Файл знайдено за шляхом {p}, але не вдалось прочитати: {e}")

            # знайдемо числову колонку для таймсері
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) == 0:
                # спробуємо назви колонок, які часто зустрічаються
                for possible in ['#Passengers', 'Passengers', 'Купівля', 'Продаж', 'Unique.Visits', 'value']:
                    if possible in df.columns:
                        numeric_cols.append(possible)
                        break

            if len(numeric_cols) == 0:
                raise RuntimeError(f"Файл {p} прочитано, але у ньому не знайдено числових колонок, які можна використати як часовий ряд. Стовпці: {list(df.columns)}")

            col = numeric_cols[0]

            # шукаємо колонку з датами для індексу
            date_cols = [c for c in df.columns if 'date' in str(c).lower() or 'дата' in str(c).lower() or 'month' in str(c).lower()]
            if len(date_cols) > 0:
                df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], dayfirst=True, errors='coerce')
                df.set_index(date_cols[0], inplace=True)
            else:
                # якщо дат немає — використовуємо простий RangeIndex
                df.index = pd.RangeIndex(start=0, stop=len(df), step=1)

            ts = df[[col]].copy()
            ts.columns = ['value']
            print(f"Прочитано файл: {p}, використана колонка: '{col}'")
            return ts, p

    if not found_any:
        raise FileNotFoundError(
            "Файл з даними не знайдено у поточній теці. Помістіть ваш файл (наприклад, "
            "'Oschadbank (USD).xls' або 'AirPassengers.csv') у ту саму папку, де знаходиться цей скрипт, і повторіть запуск."
        )



# Очищення нулів з подальною інтерполяцією
def clean_zeros_and_interpolate(ts, method='linear'):
    df = ts.copy()
    zeros = (df['value'] == 0).sum()
    if zeros > 0:
        print(f"Знайдено {zeros} нулів у даних. Замінюю 0 -> NaN і інтерполюю методом '{method}'.")
    else:
        print("Нулів у даних не знайдено.")
    df.loc[df['value'] == 0, 'value'] = np.nan
    df['value'] = df['value'].interpolate(method=method).ffill().bfill()
    return df

# Моделі експоненціального згладжування
def fit_simple_exp_smoothing(ts):
    model = SimpleExpSmoothing(ts['value'])
    fit = model.fit()
    fitted = fit.fittedvalues
    forecast = fit.forecast(len(ts))
    return fitted, forecast, fit


def fit_holt(ts):
    model = Holt(ts['value'])
    fit = model.fit()
    fitted = fit.fittedvalues
    forecast = fit.forecast(len(ts))
    return fitted, forecast, fit


def fit_holt_winters(ts, seasonal_periods=None, trend='add', seasonal='add'):
    if seasonal_periods is None:
        seasonal_periods = 12
    model = ExponentialSmoothing(ts['value'], seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal)
    fit = model.fit()
    fitted = fit.fittedvalues
    forecast = fit.forecast(len(ts))
    return fitted, forecast, fit

# Метрики
def compute_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    try:
        r2 = r2_score(true, pred)
    except Exception:
        r2 = float('nan')
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# Візуалізація і збереження
def plot_save(ts, fitted, forecast, title, filename_base):
    plt.figure(figsize=(10, 4))
    plt.plot(ts.index, ts['value'], label='Original')
    plt.plot(ts.index, fitted, label='Fitted')
    plt.plot(ts.index, forecast, label='Forecast')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    outpng = os.path.join(OUTPUT_DIR, f"{filename_base}.png")
    plt.savefig(outpng)
    plt.close()
    print(f"Графік збережено: {outpng}")

# Головна функція
def main():
    try:
        ts, used_path = try_read_file()
    except Exception as e:
        print("Помилка при читанні файлу:", e)
        sys.exit(1)

    # Попередній аналіз
    print("\n=== Попередній аналіз вхідних даних ===")
    print(f"Перші 5 рядків:\n{ts.head()}")
    print(f"Кількість точок: {len(ts)}")
    print(f"Опис статистики:\n{ts['value'].describe()}")

    # Очистка
    ts_clean = clean_zeros_and_interpolate(ts, method='linear')
    ts_clean.to_csv(os.path.join(OUTPUT_DIR, 'ts_clean.csv'))
    print(f"Очищені дані збережено: {os.path.join(OUTPUT_DIR, 'ts_clean.csv')}")

    # Перевірка індексу
    if not np.issubdtype(ts_clean.index.dtype, np.datetime64):
        ts_clean = ts_clean.reset_index(drop=True)
        ts_clean.index = pd.RangeIndex(start=0, stop=len(ts_clean), step=1)

    # Навчання моделей
    print("\n=== Навчання моделей ===")
    fitted_s, forecast_s, fit_s = fit_simple_exp_smoothing(ts_clean)
    fitted_h, forecast_h, fit_h = fit_holt(ts_clean)
    seasonal_periods = 12
    if len(ts_clean) < 24:
        seasonal_periods = max(2, len(ts_clean) // 3)
    fitted_hw, forecast_hw, fit_hw = fit_holt_winters(ts_clean, seasonal_periods=seasonal_periods)

    # Метрики
    metrics = {}
    metrics['Simple_fitted'] = compute_metrics(ts_clean['value'], fitted_s)
    metrics['Holt_fitted'] = compute_metrics(ts_clean['value'], fitted_h)
    metrics['HW_fitted'] = compute_metrics(ts_clean['value'], fitted_hw)
    metrics['Simple_forecast'] = compute_metrics(ts_clean['value'], forecast_s)
    metrics['Holt_forecast'] = compute_metrics(ts_clean['value'], forecast_h)
    metrics['HW_forecast'] = compute_metrics(ts_clean['value'], forecast_hw)

    # Збереження результатів
    results_df = pd.DataFrame({
        'original': ts_clean['value'],
        'fitted_simple': fitted_s,
        'forecast_simple': forecast_s,
        'fitted_holt': fitted_h,
        'forecast_holt': forecast_h,
        'fitted_hw': fitted_hw,
        'forecast_hw': forecast_hw,
    }, index=ts_clean.index)
    results_csv_path = os.path.join(OUTPUT_DIR, 'results_models.csv')
    results_df.to_csv(results_csv_path)
    print(f"Результати моделей збережено: {results_csv_path}")

    # Графіки
    plot_save(ts_clean, fitted_s, forecast_s, 'Simple Exponential Smoothing', 'simple_exp_smoothing')
    plot_save(ts_clean, fitted_h, forecast_h, 'Holt (Double) Exponential Smoothing', 'holt_smoothing')
    plot_save(ts_clean, fitted_hw, forecast_hw, 'Holt-Winters Seasonal Smoothing', 'holt_winters_smoothing')

    # Звіт
    report_lines = []
    report_lines.append("Звіт по лабораторній роботі №6 — Експоненційне згладжування\n")
    report_lines.append(f"Вхідні дані: {used_path}\n")
    report_lines.append(f"Кількість точок: {len(ts_clean)}\n")
    report_lines.append("\n--- Метрики (Fitted) ---\n")
    for k in ['Simple_fitted', 'Holt_fitted', 'HW_fitted']:
        m = metrics[k]
        report_lines.append(f"{k}: MAE={m['MAE']:.4f}, MSE={m['MSE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}\n")
    report_lines.append("\n--- Метрики (Forecast vs Real) ---\n")
    for k in ['Simple_forecast', 'Holt_forecast', 'HW_forecast']:
        m = metrics[k]
        report_lines.append(f"{k}: MAE={m['MAE']:.4f}, MSE={m['MSE']:.4f}, RMSE={m['RMSE']:.4f}, R2={m['R2']:.4f}\n")

    mse_fitted = {
        'Simple': metrics['Simple_fitted']['MSE'],
        'Holt': metrics['Holt_fitted']['MSE'],
        'HW': metrics['HW_fitted']['MSE'],
    }
    best_model = min(mse_fitted, key=mse_fitted.get)
    report_lines.append(f"\nНайкраща модель за MSE (на fitted): {best_model} (MSE={mse_fitted[best_model]:.4f})\n")

    report_path = os.path.join(OUTPUT_DIR, 'report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(report_lines))
    print(f"Текстовий звіт збережено: {report_path}")

    # Короткий вивід у консоль
    print("\n=== Короткий звіт (консоль) ===")
    for line in report_lines:
        print(line.strip())
    print("\nГотово. Дивіться папку 'output/' для графіків, CSV та звіту.")


if __name__ == "__main__":
    main()
