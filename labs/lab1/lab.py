import os
import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# ----------------------------- Утиліти -----------------------------------

def ensure_dirs(base='outputs'):
    """Створює папки outputs/csv і outputs/plots, якщо їх немає."""
    csv_dir = os.path.join(base, 'csv')
    plots_dir = os.path.join(base, 'plots')
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    return csv_dir, plots_dir

def save_series_csv(series, path):
    """Зберігає ряд у CSV з колонкою 'value'."""
    pd.DataFrame({'value': series}).to_csv(path, index=False)

def safe_plot(series, trend=None, title='series', fname=None):
    """Малює лінійний графік ряду (та тренду, якщо задано) і зберігає у файл."""
    plt.figure(figsize=(10,4))
    plt.plot(series, label='series')
    if trend is not None:
        plt.plot(trend, label='trend', linewidth=2)
    plt.title(title)
    plt.xlabel('t')
    plt.ylabel('value')
    plt.grid(True)
    plt.legend()
    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
    plt.close()

def save_hist(series, bins=30, title='hist', fname=None):
    """Малює гістограму і зберігає у файл."""
    plt.figure(figsize=(8,4))
    plt.hist(series, bins=bins, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(True, linestyle='--', alpha=0.3)
    if fname:
        plt.tight_layout()
        plt.savefig(fname, dpi=150)
    plt.close()

# ------------------------- Генератори -------------------------------------

def generate_trend(kind: str, params: dict, n: int):
    """
    Генерує детерміновану частину (тренд).
    kind: 'linear', 'quadratic', 'cubic'
    params: словник параметрів
    """
    i = np.arange(n, dtype=float)
    if kind == 'linear':
        m = params.get('m', 0.0)
        c = params.get('c', 0.0)
        return m * i + c
    if kind == 'quadratic':
        a = params.get('a', 0.0)
        b = params.get('b', 0.0)
        c = params.get('c', 0.0)
        return a * i**2 + b * i + c
    if kind == 'cubic':
        a = params.get('a', 0.0)
        b = params.get('b', 0.0)
        c = params.get('c', 0.0)
        d = params.get('d', 0.0)
        return a * i**3 + b * i**2 + c * i + d
    raise ValueError(f"Unknown trend kind: {kind}")

def generate_noise(kind: str, params: dict, n: int):
    """
    Генерує шум.
    kind: 'normal' або 'uniform'
    params: словник параметрів (normal: loc, scale; uniform: low, high)
    """
    if kind == 'normal':
        loc = params.get('loc', 0.0)
        scale = params.get('scale', 1.0)
        return np.random.normal(loc=loc, scale=scale, size=n)
    if kind == 'uniform':
        low = params.get('low', 0.0)
        high = params.get('high', 1.0)
        return np.random.uniform(low=low, high=high, size=n)
    raise ValueError(f"Unknown noise kind: {kind}")

def synthesize_series(trend_kind, trend_params, noise_kind, noise_params, n):
    """
    Створює синтетичний ряд: trend + noise
    Повертає tuple (series, trend, noise)
    """
    trend = generate_trend(trend_kind, trend_params, n)
    noise = generate_noise(noise_kind, noise_params, n)
    series = trend + noise
    return series, trend, noise

# ------------------------- Статистика -------------------------------------

def compute_stats(arr):
    """Обчислює базові числові характеристики ряду і повертає словник."""
    s = pd.Series(arr).dropna().astype(float)
    stats = {
        'n': int(len(s)),
        'mean': float(s.mean()),
        'median': float(s.median()),
        'var': float(s.var()),
        'std': float(s.std()),
        'min': float(s.min()),
        'max': float(s.max()),
        'skew': float(s.skew()),
        'kurtosis': float(s.kurtosis()),
    }
    return stats

def print_stats(label, stats):
    """Друк статистик у зручному форматі."""
    print(f"--- {label} ---")
    for k, v in stats.items():
        print(f"{k:8s} : {v}")
    print("-" * 40)

# ----------------------- Зчитування реальних даних --------------------------

def try_read_real_excel_simple():
    """
    Шукає файл Oschadbank (USD).xlsx/.xls/.xlsm у поточній теці і повертає
    DataFrame з колонками ['Купівля', 'Продаж', 'КурсНбу'] якщо вони є.
    Повертає (None, None) при відсутності.
    """
    default_paths = [
        'Oschadbank (USD).xlsx',
        'Oschadbank (USD).xls',
        'Oschadbank (USD).xlsm'
    ]
    expected_cols = ['Купівля', 'Продаж', 'КурсНбу']

    for p in default_paths:
        if os.path.exists(p):
            try:
                df = pd.read_excel(p)
            except Exception as e:
                # Якщо не вдається прочитати (наприклад, для .xls потрібен xlrd)
                print(f"Не можу прочитати {p}: {e}")
                return None, None

            found = [c for c in expected_cols if c in df.columns]
            if found:
                # Повертаємо DataFrame тільки з наявними колонками у потрібному порядку
                return df[found], p
            else:
                print(f"Файл {p} прочитано, але очікувані колонки не знайдені: {list(df.columns)[:10]}")
                return None, None

    print("Файл з реальними даними не знайдено у поточній папці.")
    return None, None

# -------------------------- Основний pipeline ----------------------------------

def main():
    # Фіксуємо зерно генератора для відтворюваності результатів
    np.random.seed(42)

    # Створюємо папки для результатів
    csv_dir, plots_dir = ensure_dirs('outputs')

    # Параметри: довжина синтетичного ряду
    n = 1000

    # Комбінації: тренд × шум (рівень 2 — normal & uniform; linear & cubic)
    combos = [
        ('cubic', {'a':1e-9, 'b':1e-7, 'c':1e-4, 'd':0.0}, 'normal', {'loc':0.0, 'scale':1.0}, 'cubic_normal'),
        ('cubic', {'a':1e-9, 'b':1e-7, 'c':1e-4, 'd':0.0}, 'uniform', {'low':-2.0, 'high':2.0}, 'cubic_uniform'),
        ('linear', {'m':0.001, 'c':0.0}, 'normal', {'loc':0.0, 'scale':0.5}, 'linear_normal'),
        ('linear', {'m':0.001, 'c':0.0}, 'uniform', {'low':-1.0, 'high':1.0}, 'linear_uniform'),
    ]

    results_summary = []
    print("\nГенеруємо синтетичні серії та рахуємо статистики...\n")

    for trend_k, trend_p, noise_k, noise_p, name in combos:
        # 1) Синтезуємо серію
        series, trend, noise = synthesize_series(trend_k, trend_p, noise_k, noise_p, n)

        # 2) Обчислюємо статистики для повної серії, для шуму і для тренду
        stats_series = compute_stats(series)
        stats_noise = compute_stats(noise)
        stats_trend = compute_stats(trend)

        # 3) Друкуємо статистики
        print_stats(f"SYNTH:{name} (series)", stats_series)
        print_stats(f"SYNTH:{name} (noise)", stats_noise)
        print_stats(f"SYNTH:{name} (trend)", stats_trend)

        # 4) Зберігаємо CSV та графіки:
        csv_path = os.path.join(csv_dir, f"{name}.csv")
        save_series_csv(series, csv_path)

        plot_path = os.path.join(plots_dir, f"{name}.png")
        safe_plot(series, trend=trend, title=f"{name} (series with trend)", fname=plot_path)

        # 5) Окремі графіки: сам тренд і гістограма шуму
        trend_plot = os.path.join(plots_dir, f"{name}_trend.png")
        safe_plot(trend, trend=None, title=f"{name} (trend only)", fname=trend_plot)

        hist_noise = os.path.join(plots_dir, f"{name}_noise_hist.png")
        save_hist(noise, bins=40, title=f"{name} noise histogram", fname=hist_noise)

        # 6) Додаємо до зведення
        results_summary.append({
            'name': name,
            'trend': trend_k,
            'noise': noise_k,
            'stats_series': stats_series,
            'stats_noise': stats_noise,
            'stats_trend': stats_trend,
            'csv': csv_path,
            'plot': plot_path,
            'trend_plot': trend_plot,
            'noise_hist': hist_noise
        })

    # Обробка реальних даних (Oschadbank)
    real_df, real_path = try_read_real_excel_simple()

    if real_df is not None:
        print("\nОбробка реальних даних...\n")
        for col in real_df.columns:
            # Серія реальних даних (без NaN)
            series_real = pd.Series(real_df[col].dropna().values).astype(float)

            # Обчислюємо статистики і друкуємо
            stats_real = compute_stats(series_real)
            print_stats(f"REAL:{col}", stats_real)

            # Зберігаємо CSV і графік часового ряду
            csv_path = os.path.join(csv_dir, f"real_{col}.csv")
            pd.DataFrame({'value': series_real}).to_csv(csv_path, index=False)

            plot_path = os.path.join(plots_dir, f"real_{col}.png")
            safe_plot(series_real.values, trend=None, title=f"Real_{col}", fname=plot_path)

            # Гістограма реальних значень (корисно для аналізу розподілу)
            hist_path = os.path.join(plots_dir, f"real_{col}_hist.png")
            save_hist(series_real.values, bins=40, title=f"Real {col} histogram", fname=hist_path)

            # Додаємо у зведення
            results_summary.append({
                'name': f"real_{col}",
                'trend': 'real',
                'noise': 'real',
                'stats_series': stats_real,
                'csv': csv_path,
                'plot': plot_path,
                'hist': hist_path
            })

        # Просте порівняння: шукаємо синтетичну серію з найменшою різницею по mean і var
        print("\nПорівняння синтетичних і реальних (по mean і var):")
        for real_col in real_df.columns:
            sreal = compute_stats(real_df[real_col].dropna().astype(float).values)
            best = None
            best_score = None
            for res in results_summary:
                # розглядаємо лише синтетичні записи
                if res['name'].startswith('cubic') or res['name'].startswith('linear'):
                    synth_stats = res['stats_series']
                    # простий score: сума квадратів різниць mean і var
                    score = ((synth_stats['mean'] - sreal['mean'])**2) + ((synth_stats['var'] - sreal['var'])**2)
                    if best_score is None or score < best_score:
                        best_score = score
                        best = res
            print(f"Real '{real_col}' best matched by synthetic '{best['name']}' (score={best_score:.4g})")
    else:
        print("\nРеальні дані не оброблені (файл не знайдено або колонки відсутні).")

    # Формуємо текстовий звіт (summary.txt)
    summary_lines = []
    summary_lines.append("Lab Work Level 2 — Summary\n\n")
    for r in results_summary:
        summary_lines.append(f"NAME: {r['name']}\n")
        summary_lines.append(f"  trend: {r['trend']}, noise: {r['noise']}\n")
        st = r.get('stats_series')
        if st:
            for k, v in st.items():
                summary_lines.append(f"    {k:8s} : {v}\n")
        if 'csv' in r:
            summary_lines.append(f"  csv: {r['csv']}\n")
        if 'plot' in r:
            summary_lines.append(f"  plot: {r['plot']}\n")
        if 'noise_hist' in r:
            summary_lines.append(f"  noise_hist: {r['noise_hist']}\n")
        if 'trend_plot' in r:
            summary_lines.append(f"  trend_plot: {r['trend_plot']}\n")
        if 'hist' in r:
            summary_lines.append(f"  hist: {r['hist']}\n")
        summary_lines.append("\n")

    summary_path = os.path.join('outputs', 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.writelines(summary_lines)

    print(f"\nГотово. Результати збережено у папці 'outputs/'. Звіт: {summary_path}\n")

if __name__ == '__main__':
    main()
