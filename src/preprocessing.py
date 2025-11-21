import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

"""
preprocessing.py - Комплексный финансовый анализ v2.6
УЛУЧШЕНО: волатильность считается как IQR/медиана (стандарт для финансов)
Это дает более адекватные значения без зависимости от экстремальных выбросов
"""

def load_data(filepath='T_cashback_dataset.xlsx'):
    """Загружает датасет"""
    print("Загружаем датасет...")
    df = pd.read_excel(filepath)
    print(f"✓ Загружено {len(df)} строк, {df['ключ_клиента'].nunique()} уникальных клиентов")
    print(f"  Колонки: {list(df.columns[:10])}...")
    return df


def calculate_client_baseline(df):
    """
    Рассчитывает базовую статистику и доверительные интервалы
    ИСПРАВЛЕНО: волатильность считается через IQR (Interquartile Range)
    Это стандарт в финансовой аналитике и намного более адекватен
    """
    print("\nРасчет базовой статистики (улучшенный метод волатильности)...")
    
    oboroty_cols = [col for col in df.columns if col.startswith('оборот_')]
    activation_cols = [col for col in df.columns if col.startswith('активация_')]
    
    print(f"  Найдено колонок оборотов: {len(oboroty_cols)}")
    
    baseline_stats = []
    
    for client_id in df['ключ_клиента'].unique():
        client_data = df[df['ключ_клиента'] == client_id]
        
        # Получаем оборот: каждая строка = один месяц для одного клиента
        oborots = client_data[oboroty_cols].values.flatten()
        oborots = oborots[oborots > 0]  # Берем только ненулевые значения
        
        if len(oborots) == 0:
            continue
        
        # Статистика по месячным оборотам
        mean_oborot = oborots.mean()
        median_oborot = np.median(oborots)
        std_oborot = oborots.std()
        
        if np.isnan(mean_oborot) or np.isnan(std_oborot):
            continue
        
        # Доверительные интервалы: процентили (15% и 85%)
        ci_lower = np.percentile(oborots, 15)
        ci_upper = np.percentile(oborots, 85)
        
        # УЛУЧШЕНО: волатильность через IQR (Interquartile Range)
        # IQR = Q3 - Q1 (разница между 75-м и 25-м процентилями)
        # Это более адекватная мера волатильности, не чувствительна к выбросам
        q75, q25 = np.percentile(oborots, [65, 35])
        iqr = q75 - q25
        
        # Волатильность = IQR / медиана
        # Нормализуем на медиану, чтобы можно было сравнивать клиентов
        cv = iqr / median_oborot if median_oborot > 0 else 0
        
        # Концентрация расходов (топ-3 категории)
        all_spending = client_data[oboroty_cols].values.flatten()
        all_spending = all_spending[all_spending > 0]
        
        if len(all_spending) > 0:
            top3_sum = np.sort(all_spending)[-3:].sum() if len(all_spending) >= 3 else all_spending.sum()
            concentration = top3_sum / all_spending.sum()
        else:
            concentration = 0
        
        # Регулярность транзакций
        transactions = (client_data[oboroty_cols].values.flatten() > 0).sum()
        
        baseline_stats.append({
            'ключ_клиента': client_id,
            'оборот_mean': mean_oborot,
            'оборот_std': std_oborot,
            'cv': cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'транзакции_кол': transactions,
            'концентрация': concentration,
            'возраст': client_data['возраст'].iloc[0],
            'регион': 'неизвестен'
        })
    
    baseline_df = pd.DataFrame(baseline_stats)
    baseline_df = baseline_df.dropna()  # Удаляем NaN
    
    print(f"✓ Рассчитана статистика для {len(baseline_df)} клиентов")
    print(f"  Средний оборот/месяц: {baseline_df['оборот_mean'].mean():.0f} р.")
    print(f"  Средняя волатильность (IQR): {baseline_df['cv'].mean():.2f}")
    print(f"  Средний ДИ: [{baseline_df['ci_lower'].mean():.0f}, {baseline_df['ci_upper'].mean():.0f}]")
    
    return baseline_df


def identify_anomalies(df, baseline_df):
    """
    Выявляет клиентов с аномалиями
    """
    print("\nВыявление аномалий...")
    
    oboroty_cols = [col for col in df.columns if col.startswith('оборот_')]
    
    anomalies = []
    for idx, row in df.iterrows():
        client_id = row['ключ_клиента']
        oborots = row[oboroty_cols].values
        oborots = oborots[oborots > 0]
        
        if len(oborots) == 0:
            continue
        
        current_oborot = oborots.mean()
        
        baseline = baseline_df[baseline_df['ключ_клиента'] == client_id]
        if len(baseline) == 0:
            continue
        
        ci_lower = baseline['ci_lower'].values[0]
        ci_upper = baseline['ci_upper'].values[0]
        mean_oborot = baseline['оборот_mean'].values[0]
        
        # Проверка аномалии
        is_anomaly = (current_oborot < ci_lower) or (current_oborot > ci_upper)
        deviation_pct = ((current_oborot - mean_oborot) / (mean_oborot + 1)) * 100
        
        if is_anomaly:
            anomaly_type = "высокие расходы" if current_oborot > ci_upper else "низкие расходы"
            anomalies.append({
                'ключ_клиента': client_id,
                'тип': anomaly_type,
                'текущий_оборот': current_oborot,
                'ожидаемый_диапазон': f"[{ci_lower:.0f}, {ci_upper:.0f}]",
                'отклонение_%': abs(deviation_pct),
                'приоритет': 'высокий' if abs(deviation_pct) > 30 else 'средний'
            })
    
    anomalies_df = pd.DataFrame(anomalies)
    print(f"✓ Обнаружено {len(anomalies_df)} аномалий")
    if len(anomalies_df) > 0:
        print(f"  Высокие расходы: {len(anomalies_df[anomalies_df['тип']=='высокие расходы'])}")
        print(f"  Низкие расходы: {len(anomalies_df[anomalies_df['тип']=='низкие расходы'])}")
    
    return anomalies_df


def segment_clients_hierarchical(baseline_df, max_cohorts=8):
    """
    Иерархическая кластеризация с ОГРАНИЧЕНИЕМ на количество когорт
    max_cohorts контролирует максимальное число кластеров
    """
    print(f"\nКогортная сегментация (максимум {max_cohorts} когорт)...")
    
    # Подготовка признаков
    features = baseline_df[['оборот_mean', 'cv', 'концентрация', 'транзакции_кол']].copy()
    
    # Заполняем NaN (если остались)
    features = features.fillna(features.mean())
    
    # Проверяем на NaN после заполнения
    if features.isna().any().any():
        print("  Внимание: остались NaN значения, используем медиану")
        features = features.fillna(features.median())
    
    # Нормализация
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Проверяем на NaN после масштабирования
    if np.isnan(features_scaled).any():
        print("  Ошибка: NaN после масштабирования, пропускаем выбросы")
        valid_idx = ~np.isnan(features_scaled).any(axis=1)
        features_scaled = features_scaled[valid_idx]
        baseline_df = baseline_df.iloc[valid_idx].reset_index(drop=True)
    
    # Иерархическая кластеризация с ЖЕСТКИМ ОГРАНИЧЕНИЕМ на количество
    clustering = AgglomerativeClustering(
        n_clusters=max_cohorts,
        linkage='ward'
    )
    labels = clustering.fit_predict(features_scaled)
    
    baseline_df['когорта'] = labels
    
    # Проверяем размеры когорт
    cohort_sizes = baseline_df['когорта'].value_counts().sort_index()
    print(f"✓ Создано {len(cohort_sizes)} когорт:")
    for cohort_id, size in cohort_sizes.items():
        pct = size/len(baseline_df)*100
        print(f"  Когорта {cohort_id}: {size} клиентов ({pct:.1f}%)")
    
    return baseline_df


def calculate_cohort_profiles(baseline_df):
    """Рассчитывает профили когорт"""
    print("\nРасчет профилей когорт...")
    
    cohort_profiles = baseline_df.groupby('когорта').agg({
        'ключ_клиента': 'count',
        'оборот_mean': ['mean', 'median', 'std'],
        'cv': 'mean',
        'концентрация': 'mean',
        'транзакции_кол': 'mean',
        'возраст': 'mean'
    }).round(2)
    
    cohort_profiles.columns = ['размер_когорты', 'средний_оборот', 'медиана_оборота', 
                               'волатильность_оборота', 'средний_cv', 'средняя_концентрация',
                               'средние_транзакции', 'средний_возраст']
    
    print("\nПрофили когорт:")
    for cohort_id, row in cohort_profiles.iterrows():
        print(f"\nКогорта {cohort_id} ({int(row['размер_когорты'])} клиентов):")
        print(f"  Средний оборот: {row['средний_оборот']:.0f} р/месяц")
        print(f"  Волатильность (IQR): {row['средний_cv']:.2f}")
        print(f"  Концентрация расходов: {row['средняя_концентрация']:.1%}")
    
    return cohort_profiles


def save_results(baseline_df, anomalies_df, cohort_profiles, output_dir='./data'):
    """Сохраняет результаты"""
    print(f"\nСохранение результатов в {output_dir}...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_df.to_csv(f'{output_dir}/client_baseline.csv', index=False)
    print(f"✓ {output_dir}/client_baseline.csv")
    
    anomalies_df.to_csv(f'{output_dir}/anomalies.csv', index=False)
    print(f"✓ {output_dir}/anomalies.csv ({len(anomalies_df)} записей)")
    
    cohort_profiles.to_csv(f'{output_dir}/cohort_profiles.csv')
    print(f"✓ {output_dir}/cohort_profiles.csv")


def main():
    print("="*60)
    print("CASHBACK ANALYTICS - ФИНАНСОВАЯ МОДЕЛЬ v2.6")
    print("="*60)
    
    try:
        # 1. Загрузка
        df = load_data('T_cashback_dataset.xlsx')
        
        # 2. Базовая статистика
        baseline_df = calculate_client_baseline(df)
        
        if len(baseline_df) == 0:
            print("\nОШИБКА: нет валидных данных для анализа")
            return False
        
        # 3. Выявление аномалий
        anomalies_df = identify_anomalies(df, baseline_df)
        
        # 4. Когортная сегментация
        baseline_df = segment_clients_hierarchical(baseline_df, max_cohorts=8)
        
        # 5. Профили когорт
        cohort_profiles = calculate_cohort_profiles(baseline_df)
        
        # 6. Сохранение
        save_results(baseline_df, anomalies_df, cohort_profiles)
        
        print("\n" + "="*60)
        print("✓ PIPELINE УСПЕШНО ЗАВЕРШЕН")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)