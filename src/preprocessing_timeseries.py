import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

"""
preprocessing.py - Pipeline обработки данных для Cashback Analytics (с анализом временных рядов)
Функции:
1. Загрузка и валидация
2. Расчет статистики по временным периодам
3. Расчет доверительных интервалов 90%
4. Выявление аномалий в расходах
5. Сохранение результатов
"""

def load_data(filepath='T_cashback_dataset.xlsx'):
    """Загружает Excel датасет с временными рядами"""
    print("📥 Загружаем датасет...")
    df = pd.read_excel(filepath)
    print(f"✓ Загружено {len(df)} строк, {len(df.columns)} столбцов")
    return df

def validate_data(df):
    """Валидация датасета"""
    print("\n✓ Валидация данных:")
    
    # Проверяем пропуски
    missing = df.isnull().sum().sum()
    print(f"  - Пропусков: {missing}")
    
    # Проверяем уникальных клиентов
    unique_clients = df['ключ_клиента'].nunique()
    print(f"  - Уникальных клиентов: {unique_clients}")
    
    # Проверяем количество записей на клиента (периодов времени)
    records_per_client = df.groupby('ключ_клиента').size()
    print(f"  - Записей на клиента: min={records_per_client.min()}, max={records_per_client.max()}, среднее={records_per_client.mean():.1f}")
    
    return True

def calculate_client_statistics(df):
    """
    Рассчитывает статистику по каждому клиенту за все временные периоды:
    - Среднее значение оборота
    - Стандартное отклонение (дисперсия)
    - Доверительный интервал 90%
    - Минимум, максимум, количество периодов
    """
    print("\n📊 Расчет статистики по клиентам...")
    
    # Берем обороты по всем категориям
    oboroty_cols = [col for col in df.columns if col.startswith('оборот_')]
    
    # Для каждой записи считаем общий оборот
    df['общий_оборот'] = df[oboroty_cols].sum(axis=1)
    
    # Группируем по клиенту и считаем статистику
    client_stats = df.groupby('ключ_клиента').agg({
        'общий_оборот': ['mean', 'std', 'min', 'max', 'count'],
        'возраст': 'first',
        'регион_проживания': 'first',
        'город_проживания': 'first',
        'пол': 'first'
    }).reset_index()
    
    # Переименовываем столбцы
    client_stats.columns = ['ключ_клиента', 'оборот_mean', 'оборот_std', 
                            'оборот_min', 'оборот_max', 'периодов',
                            'возраст', 'регион', 'город', 'пол']
    
    # ========== ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ 90% ==========
    # z-score для 90% доверия (95% с одной стороны)
    z_score = stats.norm.ppf(0.95)  # z ≈ 1.645
    
    # Заполняем NaN в std нулями (если всего 1 запись на клиента)
    client_stats['оборот_std'] = client_stats['оборот_std'].fillna(0)
    
    # Доверительный интервал
    client_stats['ci_lower'] = client_stats['оборот_mean'] - z_score * client_stats['оборот_std']
    client_stats['ci_upper'] = client_stats['оборот_mean'] + z_score * client_stats['оборот_std']
    
    # Обороты не могут быть отрицательны
    client_stats['ci_lower'] = client_stats['ci_lower'].clip(lower=0)
    
    # ========== КОЭФФИЦИЕНТ ВАРИАЦИИ (для выявления нестабильности) ==========
    # CV = std / mean (показывает волатильность относительно среднего)
    client_stats['cv'] = (client_stats['оборот_std'] / (client_stats['оборот_mean'] + 1))
    
    print(f"✓ Рассчитана статистика для {len(client_stats)} клиентов")
    print(f"\n  Статистика доверительных интервалов 90%:")
    print(f"    - Средний интервал: [{client_stats['ci_lower'].mean():.2f}, {client_stats['ci_upper'].mean():.2f}]")
    print(f"    - Максимальный интервал: [{client_stats['ci_lower'].min():.2f}, {client_stats['ci_upper'].max():.2f}]")
    
    return client_stats

def calculate_anomaly_metrics(df, client_stats):
    """
    Рассчитывает метрики для выявления аномалий:
    - Выход за границы доверительного интервала
    - Резкое изменение от предыдущего периода
    - Аномальные значения кэшбэка (NaN обрабатываются как норма)
    """
    print("\n🚨 Выявление аномалий...")
    
    cashback_cols = [col for col in df.columns if col.startswith('кэшбэк_')]
    activation_cols = [col for col in df.columns if col.startswith('активация_')]
    
    # Добавляем общий кэшбэк
    df['общий_кэшбэк'] = df[cashback_cols].sum(axis=1)
    
    # Активированные категории
    df['активированные_категории'] = (df[activation_cols] == 1).sum(axis=1)
    
    # Коэффициент активации
    доступные = (df[activation_cols] >= 0).sum(axis=1)
    df['коэф_активации'] = df['активированные_категории'] / (доступные + 1)
    
    # Объединяем с доверительными интервалами
    df_with_ci = df.merge(
        client_stats[['ключ_клиента', 'ci_lower', 'ci_upper', 'оборот_mean', 'cv']], 
        on='ключ_клиента'
    )
    
    # Флаг аномалии: выход за границы CI
    df_with_ci['is_anomaly'] = (
        (df_with_ci['общий_оборот'] < df_with_ci['ci_lower']) | 
        (df_with_ci['общий_оборот'] > df_with_ci['ci_upper'])
    ).astype(int)
    
    # Отклонение от среднего (в процентах)
    df_with_ci['deviation_pct'] = (
        (df_with_ci['общий_оборот'] - df_with_ci['оборот_mean']) / 
        (df_with_ci['оборот_mean'] + 1) * 100
    )
    
    anomalies = df_with_ci[df_with_ci['is_anomaly'] == 1]
    print(f"✓ Обнаружено {len(anomalies)} аномальных записей ({len(anomalies)/len(df_with_ci)*100:.2f}%)")
    
    return df_with_ci

def generate_report(client_stats, df_with_anomalies):
    """Генерирует отчет с инсайтами"""
    print("\n" + "="*60)
    print("📈 ИНСАЙТЫ ПО АНОМАЛИЯМ")
    print("="*60)
    
    # Клиенты с высокой волатильностью (CV > 0.5)
    high_cv = client_stats[client_stats['cv'] > 0.5]
    print(f"\n🔴 Клиенты с высокой волатильностью (CV > 0.5): {len(high_cv)}")
    print(f"   Потенциал для отправки уведомлений: {len(high_cv)}")
    
    # Клиенты с узким доверительным интервалом (стабильные)
    stable = client_stats[client_stats['cv'] < 0.2]
    print(f"\n🟢 Стабильные клиенты (CV < 0.2): {len(stable)}")
    
    # Аномальные периоды
    anomalies_high = df_with_anomalies[
        (df_with_anomalies['is_anomaly'] == 1) & 
        (df_with_anomalies['общий_оборот'] > df_with_anomalies['ci_upper'])
    ]
    anomalies_low = df_with_anomalies[
        (df_with_anomalies['is_anomaly'] == 1) & 
        (df_with_anomalies['общий_оборот'] < df_with_anomalies['ci_lower'])
    ]
    
    print(f"\n⬆️  Аномально высокие расходы: {len(anomalies_high)}")
    print(f"⬇️  Аномально низкие расходы: {len(anomalies_low)}")

def save_results(client_stats, df_with_anomalies, output_dir='./data'):
    """Сохраняет результаты в CSV"""
    print(f"\n💾 Сохранение результатов в {output_dir}...")
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохраняем статистику клиентов
    client_stats.to_csv(f'{output_dir}/client_statistics.csv', index=False)
    print(f"✓ {output_dir}/client_statistics.csv")
    
    # Сохраняем данные с аномалиями
    df_with_anomalies.to_csv(f'{output_dir}/data_with_anomalies.csv', index=False)
    print(f"✓ {output_dir}/data_with_anomalies.csv")
    
    # Сохраняем только аномалии для приложения
    anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == 1]
    anomalies.to_csv(f'{output_dir}/anomalies.csv', index=False)
    print(f"✓ {output_dir}/anomalies.csv ({len(anomalies)} записей)")

def main():
    """Основной pipeline"""
    print("="*60)
    print("CASHBACK ANALYTICS - PREPROCESSING (TIME SERIES)")
    print("="*60)
    
    try:
        # 1. Загрузка
        df = load_data('T_cashback_dataset.xlsx')
        
        # 2. Валидация
        validate_data(df)
        
        # 3. Расчет статистики
        client_stats = calculate_client_statistics(df)
        
        # 4. Выявление аномалий
        df_with_anomalies = calculate_anomaly_metrics(df, client_stats)
        
        # 5. Отчет
        generate_report(client_stats, df_with_anomalies)
        
        # 6. Сохранение
        save_results(client_stats, df_with_anomalies)
        
        print("\n" + "="*60)
        print("✅ PIPELINE ЗАВЕРШЕН УСПЕШНО")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
