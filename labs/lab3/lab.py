import os
import numpy as np
import math as mt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import ttest_1samp
import json

# Читаємо файл
def File_read(filename_hint=None):
    default_names = [
        'Oschadbank (USD).xls',
        'Oschadbank (USD).xlsx',
        'Oschadbank (USD).xlsm'
    ]
    if filename_hint:
        default_names.insert(0, filename_hint)

    expected_cols = ['Купівля', 'Продаж', 'КурсНбу']

    for p in default_names:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
            except Exception as e:
                print("Не можливо прочитати", p, ":", e)
                return None, None, None
            print("Файл прочитано:", p)
            found = [c for c in expected_cols if c in df.columns]
            if not found:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    col = numeric_cols[0]
                    print("Використовую перший числовий стовпець:", col)
                    return df[[col]].copy(), p, col
                return None, None, None
            return df[found].copy(), p, found
    print("Файл з реальними даними не знайдено у поточній папці.")
    return None, None, None

# МНК для визначення тренду даних та оцінки відхилень від нього
def MNK_Stat_characteristics(S0):
    iter_n = len(S0)
    Yin = np.zeros((iter_n, 1))
    F = np.ones((iter_n, 3))
    for i in range(iter_n):
        Yin[i,0] = float(S0[i])
        F[i,1] = float(i)
        F[i,2] = float(i*i)
    C = np.linalg.pinv(F.T.dot(F)).dot(F.T).dot(Yin)
    Yout = F.dot(C)
    return Yout

# Обчислює статистичні характеристики похибок даних від тренду та малює гістограму
def Stat_characteristics_in(SL, Text, filename=None):
    Yout = MNK_Stat_characteristics(SL)
    iter_n = len(Yout)
    SL0 = np.zeros((iter_n))
    for i in range(iter_n):
        SL0[i] = SL[i] - Yout[i,0]
    mS = np.median(SL0)
    dS = np.var(SL0)
    scvS = mt.sqrt(dS)
    print("------------", Text, "-------------")
    print("кількість елементів вибірки=", iter_n)
    print("медіана похибок (після вирахування тренду)=", mS)
    print("дисперсія похибок =", dS)
    print("СКВ похибок=", scvS)
    print("-----------------------------------------------------")
    if filename:
        plt.figure()
        plt.hist(SL0, bins=20, facecolor="blue", alpha=0.6)
        plt.title("Гістограма помилок: " + Text)
        plt.savefig(filename)
        plt.close()

# Відображає графік вхідних даних і тренду для візуальної оцінки похибок та аномалій
def Plot_AV(S0_L, SV_L, Text, filename=None):
    plt.figure(figsize=(10,4))
    plt.plot(SV_L, label='Вимір (вхідні)', linewidth=1)
    if S0_L is not None:
        plt.plot(S0_L, label='Тренд / Модель', linewidth=1)
    plt.ylabel(Text)
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()
    plt.close()

# Обчислює основні метрики точності моделі (MAE, MSE, RMSE, R²) для оцінки відповідності тренду даним
def kpi_model(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2}

# LSM
def LSM(S0):
    iter_n = len(S0)
    Yin = np.zeros((iter_n,1))
    F = np.ones((iter_n,3))
    for i in range(iter_n):
        Yin[i,0] = float(S0[i])
        F[i,1] = float(i)
        F[i,2] = float(i*i)
    C = np.linalg.pinv(F.T.dot(F)).dot(F.T).dot(Yin)
    Yout = F.dot(C)
    return Yout.ravel()

# ABF
def ABF(S0, alpha=None, beta=None):
    iter_n = len(S0)
    Yin = np.asarray(S0, dtype=float).ravel()
    YoutAB = np.zeros((iter_n))
    T0 = 1
    if iter_n < 2:
        return Yin.copy()
    Yspeed_retro = (Yin[1] - Yin[0]) / T0 if iter_n>1 else 0.0
    Yextra = Yin[0] + Yspeed_retro
    alfa = alpha if alpha is not None else (2 * (2 * 1 - 1) / (1 * (1 + 1)))
    beta_val = beta if beta is not None else ((6 / 1) * (1 + 1))
    YoutAB[0] = Yin[0] + alfa * (Yin[0])
    for i in range(1, iter_n):
        YoutAB[i] = Yextra + alfa * (Yin[i] - Yextra)
        Yspeed = Yspeed_retro + (beta_val / T0) * (Yin[i] - Yextra)
        Yspeed_retro = Yspeed
        Yextra = YoutAB[i] + Yspeed_retro
        if alpha is None:
            alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        else:
            alfa = alpha
        if beta is None:
            beta_val = 6 / (i * (i + 1))
        else:
            beta_val = beta
    return YoutAB

# ABGF
def ABGF(S0, alpha=None, beta=None, gamma=None):
    iter_n = len(S0)
    Yin = np.asarray(S0, dtype=float).ravel()
    Yout_ABG = np.zeros((iter_n))
    T0 = 1
    if iter_n < 3:
        return Yin.copy()
    Yspeed_retro = (Yin[1] - Yin[0]) / T0
    Yaccel_retro = (Yin[2] - 2 * Yin[1] + Yin[0]) / (T0 * T0)
    Yextra = Yin[0] + (Yspeed_retro * T0) + (Yaccel_retro * 0.5 * T0 * T0)
    alpha_local = alpha if alpha is not None else (3 * (3 * 1 * 1 - 3 * 1 + 2) / (1 * (1 + 1) * (1 + 2)))
    beta_local = beta if beta is not None else (18 * (2 * 1 - 1) / (T0 * (1 + 1) * (1 + 2) * 1))
    gamma_local = gamma if gamma is not None else (60 / (T0 * T0 * (1 + 1) * (1 + 2) * 1))
    Yout_ABG[0] = Yin[0]
    for i in range(1, iter_n):
        Yout_ABG[i] = Yextra + alpha_local * (Yin[i] - Yextra)
        Yspeed = Yspeed_retro + (beta_local / T0) * (Yin[i] - Yextra)
        Yaccel = Yaccel_retro + (gamma_local / (T0 * T0)) * (Yin[i] - Yextra)
        Yspeed_retro = Yspeed
        Yaccel_retro = Yaccel
        Yextra = Yout_ABG[i] + (Yspeed_retro * T0) + (Yaccel * 0.5 * T0 * T0)
        if alpha is None:
            alpha_local = 3 * (3 * i * i - 3 * i + 2) / (i * (i + 1) * (i + 2))
        else:
            alpha_local = alpha
        if beta is None:
            beta_local = 18 * (2 * i - 1) / (T0 * (i + 1) * (i + 2) * i)
        else:
            beta_local = beta
        if gamma is None:
            gamma_local = 60 / (T0 * T0 * (i + 1) * (i + 2) * i)
        else:
            gamma_local = gamma
    return Yout_ABG

# 1D Kalman
def kalman_1d(z, Q, R, x0=None, P0=None):
    n = len(z)
    x = np.zeros(n)
    P = np.zeros(n)
    x[0] = x0 if x0 is not None else z[0]
    P[0] = P0 if P0 is not None else 1.0
    for k in range(1, n):
        x_pred = x[k-1]
        P_pred = P[k-1] + Q
        K = P_pred / (P_pred + R)
        x[k] = x_pred + K * (z[k] - x_pred)
        P[k] = (1 - K) * P_pred
    return x

# Детектор аномалій: медіанний фільтр зі скользячим вікном
def Sliding_Window_AV_Detect_sliding_wind(S0, n_Wind):
    S = S0.copy().astype(float)
    iter_n = len(S)
    if iter_n < n_Wind:
        return S
    j_Wind = iter_n - n_Wind + 1
    Midi = np.zeros((iter_n))
    for j in range(j_Wind):
        S0_Wind = S[j:j+n_Wind]
        idx = j + n_Wind - 1
        Midi[idx] = np.median(S0_Wind)
    S0_Midi = Midi.copy()
    for j in range(n_Wind):
        S0_Midi[j] = S[j]
    return S0_Midi

# Оцінка швидкості тренду методом найменших квадратів для детекції аномалій
def LSM_AV_Detect(S0):
    iter_n = len(S0)
    Yin = np.zeros((iter_n,1))
    F = np.ones((iter_n,3))
    for i in range(iter_n):
        Yin[i,0] = float(S0[i])
        F[i,1] = float(i)
        F[i,2] = float(i*i)
    C = np.linalg.pinv(F.T.dot(F)).dot(F.T).dot(Yin)
    return C[1,0]

# Детектор аномалій: поєднання скользячого вікна та оцінки тренду LSM
def Sliding_Window_AV_Detect_LSM(S0, Q, n_Wind):
    S = S0.copy().astype(float)
    iter_n = len(S)
    if iter_n < n_Wind:
        return S
    Speed_standart = abs(LSM_AV_Detect(S0))
    Yout_S0 = LSM(S0)
    j_Wind = iter_n - n_Wind + 1
    for j in range(j_Wind):
        S0_Wind = S[j:j+n_Wind]
        dS = np.var(S0_Wind)
        scvS = mt.sqrt(dS) if dS >= 0 else 0.0
        Speed_standart_1 = abs(Speed_standart * mt.sqrt(iter_n))
        Speed_1 = abs(Q * Speed_standart * mt.sqrt(n_Wind) * scvS)
        idx = j + n_Wind - 1
        if Speed_1 > Speed_standart_1:
            S[idx] = Yout_S0[idx]
    return S

# Оцінка роботи фільтра: bias, RMSE та MAE між істинними та оціненими даними
def evaluate_filter(true, estimate):
    true = np.asarray(true).ravel()[:len(estimate)]
    estimate = np.asarray(estimate).ravel()
    bias = np.mean(estimate - true)
    rmse = np.sqrt(np.mean((estimate - true)**2))
    return {'bias': bias, 'rmse': rmse, 'mae': np.mean(np.abs(estimate - true))}

# Автопідбір параметрів фільтра ABF за мінімізацією комбінованого скору bias+RMSE
def autotune_abf(series, cleaned, alpha_range, beta_range, w_bias=1.0, w_rmse=0.5):
    best_score = None
    best_params = None
    best_est = None
    for a in alpha_range:
        for b in beta_range:
            est = ABF(cleaned, alpha=a, beta=b)
            m = evaluate_filter(series, est)
            score = w_bias * abs(m['bias']) + w_rmse * m['rmse']
            if (best_score is None) or (score < best_score):
                best_score = score
                best_params = (a,b)
                best_est = est.copy()
    return best_params, best_est, best_score

# Автопідбір параметрів фільтра ABGF за мінімізацією комбінованого скору bias+RMSE
def autotune_abgf(series, cleaned, alpha_range, beta_range, gamma_range, w_bias=1.0, w_rmse=0.5):
    best_score = None
    best_params = None
    best_est = None
    for a in alpha_range:
        for b in beta_range:
            for g in gamma_range:
                est = ABGF(cleaned, alpha=a, beta=b, gamma=g)
                m = evaluate_filter(series, est)
                score = w_bias * abs(m['bias']) + w_rmse * m['rmse']
                if (best_score is None) or (score < best_score):
                    best_score = score
                    best_params = (a,b,g)
                    best_est = est.copy()
    return best_params, best_est, best_score

# Автопідбір параметрів фільтра Калман за мінімізацією комбінованого скору bias+RMSE
def autotune_kalman(series, cleaned, Q_range, R_range, w_bias=1.0, w_rmse=0.5):
    best_score = None
    best_params = None
    best_est = None
    for Q in Q_range:
        for R in R_range:
            est = kalman_1d(cleaned, Q=Q, R=R)
            m = evaluate_filter(series, est)
            score = w_bias * abs(m['bias']) + w_rmse * m['rmse']
            if (best_score is None) or (score < best_score):
                best_score = score
                best_params = (Q,R)
                best_est = est.copy()
    return best_params, best_est, best_score

# Побудова графіка порівняння оригінальних, очищених і згладжених даних та збереження результату
def plot_compare(original, cleaned, smoothed, title_prefix, out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    n = len(original)
    x = np.arange(n)
    plt.figure(figsize=(12,5))
    plt.plot(x, original, label='Оригінал', linewidth=1)
    plt.plot(x, cleaned, label='Очищено (детектор)', linewidth=1)
    plt.plot(x, smoothed, label='Згладжено (фільтр)', linewidth=1)
    plt.title(title_prefix)
    plt.legend()
    plt.grid(True)
    fname = os.path.join(out_dir, "{}.png".format(title_prefix.replace(' ','_')))
    plt.savefig(fname, bbox_inches='tight')
    plt.show()
    plt.close()
    print("Збережено:", fname)

def main_pipeline():
    # 1. Читання даних

    print("Пошук файлу у поточній папці...")
    df, fname, found_cols = File_read()
    if df is None:
        print("Не знайдено файл чи стовпець для читання. Помістіть 'Oschadbank (USD).xls' поруч із цим скриптом.")
        return

    # Вибір колонки для аналізу (Купівля/Продаж/курс НБУ)
    if isinstance(found_cols, list) and len(found_cols) > 1:
        print("Знайдено колонки:", found_cols)
        if 'Купівля' in found_cols:
            col = 'Купівля'
        else:
            col = found_cols[0]
    elif isinstance(found_cols, list) and len(found_cols) == 1:
        col = found_cols[0]
    else:
        col = found_cols
    if col not in df.columns:
        col = df.columns[0]

    # Перетворення в numpy-ряд і видалення NaN
    series = df[col].values.astype(float).ravel()
    idx = ~np.isnan(series)
    if np.count_nonzero(~idx) > 0:
        print("Знайдено і видалено", np.count_nonzero(~idx), "NaN значень.")
    series = series[idx]
    n = len(series)
    print("Довжина ряду після очищення NaN:", n)

    # Видалення нульових значень (якщо мало)
    zeros_count = np.sum(series == 0)
    print("Кількість нульових значень:", zeros_count, "({:.2f}%)".format(100 * zeros_count / max(1, n)))
    if zeros_count > 0 and zeros_count / n < 0.2:
        nonzero_idx = series != 0
        series = series[nonzero_idx]
        n = len(series)
        print("Нульові значення видалено. Нова довжина:", n)

    # 2. Візуалізація та статистичний аналіз
    os.makedirs('results', exist_ok=True)
    Plot_AV(series, series, "Оригінальні дані: " + col, filename='results/original_series.png')
    Stat_characteristics_in(series, "Оригінальні дані: " + col, filename='results/hist_original.png')

    # 3. Детекція аномалій
    n_Wind = 5
    s_sliding = Sliding_Window_AV_Detect_sliding_wind(series, n_Wind)
    cleaned = s_sliding.copy()

    # 4. Автопідбір параметрів фільтрів (grid search)
    print("\nПочинаємо автопідбір параметрів (grid search) — пріоритет: мінімум |bias|")
    w_bias = 3.0   # пріоритет на незміщеність
    w_rmse = 0.5   # пріоритет на RMSE

    # Пошук оптимальних параметрів ABF
    alpha_range_abf = [None, 0.02, 0.05, 0.1, 0.2]
    beta_range_abf = [None, 0.01, 0.05, 0.1]
    best_abf_params, best_abf_est, best_abf_score = autotune_abf(series, cleaned, alpha_range_abf, beta_range_abf, w_bias=w_bias, w_rmse=w_rmse)
    abf_metrics = evaluate_filter(series, best_abf_est)
    print("Best ABF params:", best_abf_params, "score=", best_abf_score)

    # Пошук оптимальних параметрів ABGF
    alpha_range_abgf = [None, 0.1, 0.2, 0.3]
    beta_range_abgf = [None, 0.01, 0.05]
    gamma_range_abgf = [None, 0.001, 0.01, 0.05]
    best_abgf_params, best_abgf_est, best_abgf_score = autotune_abgf(series, cleaned, alpha_range_abgf, beta_range_abgf, gamma_range_abgf, w_bias=w_bias, w_rmse=w_rmse)
    abgf_metrics = evaluate_filter(series, best_abgf_est)
    print("Best ABGF params:", best_abgf_params, "score=", best_abgf_score)

    # Пошук оптимальних параметрів Калманівського фільтра
    Q_range = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
    R_range = [1e-2, 1e-1, 1.0, 5.0, 10.0]
    best_kalman_params, best_kalman_est, best_kalman_score = autotune_kalman(series, cleaned, Q_range, R_range, w_bias=w_bias, w_rmse=w_rmse)
    kalman_metrics = evaluate_filter(series, best_kalman_est)
    print("Best Kalman params:", best_kalman_params, "score=", best_kalman_score)

    # Лінійна регресія (LSM) як тренд
    lsm_trend = LSM(cleaned)
    lsm_metrics = evaluate_filter(series, lsm_trend)

    # 5. Збереження результатів
    summary = {
        'w_bias': w_bias, 'w_rmse': w_rmse,
        'abf_params': best_abf_params, 'abf_metrics': abf_metrics, 'abf_score': best_abf_score,
        'abgf_params': best_abgf_params, 'abgf_metrics': abgf_metrics, 'abgf_score': best_abgf_score,
        'kalman_params': best_kalman_params, 'kalman_metrics': kalman_metrics, 'kalman_score': best_kalman_score,
        'lsm_metrics': lsm_metrics
    }
    with open('results/autotune_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 6. Графічне порівняння
    plot_compare(series, cleaned, best_abf_est, 'best_abf', out_dir='results')
    plot_compare(series, cleaned, best_abgf_est, 'best_abgf', out_dir='results')
    plot_compare(series, cleaned, best_kalman_est, 'best_kalman', out_dir='results')
    plot_compare(series, cleaned, lsm_trend, 'lsm_trend', out_dir='results')

    # 7. t-тест на значущість bias
    def bias_ttest(true, estimate, label):
        diff = np.asarray(estimate).ravel() - np.asarray(true).ravel()[:len(estimate)]
        t_stat, p = ttest_1samp(diff, 0.0)
        print("t-test for bias ({}): t={:.4f}, p={:.4g} (p<0.05 -> bias значущий)".format(label, t_stat, p))

    print("\nПеревірка значущості bias:")
    bias_ttest(series, best_abf_est, 'ABF')
    bias_ttest(series, best_abgf_est, 'ABGF')
    bias_ttest(series, best_kalman_est, 'Kalman')
    bias_ttest(series, lsm_trend, 'LSM')

    # 8. Збереження остаточного CSV
    out_df = pd.DataFrame({
        'original': series,
        'cleaned_sliding': cleaned,
        'best_abf': best_abf_est,
        'best_abgf': best_abgf_est,
        'best_kalman': best_kalman_est,
        'lsm_trend': lsm_trend
    })
    out_csv = 'results/pipeline_autotune_bias_priority_results.csv'
    out_df.to_csv(out_csv, index=False)
    print("\nЗбережено CSV з результатами:", out_csv)

if __name__ == '__main__':
    main_pipeline()
