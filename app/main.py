import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="VTB Avatar - Финансовое здоровье",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; padding: 20px; border-radius: 10px; text-align: center;}
    .cohort-header {font-size: 24px; font-weight: bold; margin: 20px 0;}
    .anomaly-high {color: #ff6b6b; font-weight: bold;}
    .anomaly-low {color: #4dabf7; font-weight: bold;}
</style>
""", unsafe_allow_html=True)


# ========== ЗАГРУЗКА ДАННЫХ ==========

@st.cache_data
def load_all_data():
    """Загружает базовые данные, аномалии и профили когорт"""
    try:
        baseline_df = pd.read_csv('data/client_baseline.csv')
        anomalies_df = pd.read_csv('data/anomalies.csv')
        cohort_profiles = pd.read_csv('data/cohort_profiles.csv', index_col=0)
        
        print(f"✓ Загружено: {len(baseline_df)} клиентов, {len(anomalies_df)} аномалий")
        return baseline_df, anomalies_df, cohort_profiles
    except FileNotFoundError:
        st.error("❌ Файлы данных не найдены. Сначала запустите preprocessing_v2.py")
        st.stop()


baseline_df, anomalies_df, cohort_profiles = load_all_data()

# Создаём список ID клиентов
@st.cache_data
def get_client_ids(baseline_df):
    return sorted(baseline_df['ключ_клиента'].unique().tolist())

client_ids = get_client_ids(baseline_df)


# ========== ГЛАВНЫЙ ИНТЕРФЕЙС ==========

st.title("💰 VTB Avatar - Финансовое здоровье")
st.markdown("**Анализ поведения клиентов и мониторинг аномалий**")


# Боковое меню
st.sidebar.header("⚙️ Навигация")
selected_tab = st.sidebar.radio("Выберите раздел:", [
    "📊 Личный профиль",
    "👥 Анализ когорт",
    "🚨 Мониторинг аномалий",
    "📈 Прогнозирование"
])


# ========== ТАБ 1: ЛИЧНЫЙ ПРОФИЛЬ ==========

if selected_tab == "📊 Личный профиль":
    st.header("📊 Финансовый профиль клиента")
    
    col1, col2 = st.columns(2)
    with col1:
        client_id = st.selectbox("Выберите клиента:", options=client_ids, index=0)
    
    # Получаем данные клиента
    client_data = baseline_df[baseline_df['ключ_клиента'] == client_id]
    if len(client_data) == 0:
        st.error("Клиент не найден")
    else:
        client = client_data.iloc[0]
        cohort_id = int(client['когорта']) if 'когорта' in client.index else 0
        
        # КПИ
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Средний оборот/месяц", f"{client['оборот_mean']:.0f} р.")
        with col2:
            st.metric("📊 Волатильность", f"{client['cv']:.2f}")
        with col3:
            st.metric("🎯 Когорта", f"#{cohort_id}")
        with col4:
            st.metric("👤 Возраст", f"{int(client['возраст'])} лет")
        
        # Детали
        st.subheader("📋 Подробный профиль")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Диапазон расходов (90% ДИ):**")
            st.write(f"  min: {client['ci_lower']:.0f} р.")
            st.write(f"  max: {client['ci_upper']:.0f} р.")
        
        with col2:
            st.write(f"**Поведение расходов:**")
            st.write(f"  Концентрация: {client['концентрация']:.1%} (топ-3 категории)")
            st.write(f"  Регион: {client['регион']}")
        
        # Статус аномалии
        is_anomaly = len(anomalies_df[anomalies_df['ключ_клиента'] == client_id]) > 0
        if is_anomaly:
            anomaly = anomalies_df[anomalies_df['ключ_клиента'] == client_id].iloc[0]
            if anomaly['тип'] == 'высокие расходы':
                st.warning(f"⬆️ **Аномалия: ВЫСОКИЕ расходы** (на {anomaly['отклонение_%']:.0f}%)")
            else:
                st.info(f"⬇️ **Аномалия: НИЗКИЕ расходы** (на {anomaly['отклонение_%']:.0f}%)")
        else:
            st.success("✅ Расходы в норме (в пределах доверительного интервала)")
        
        # Рекомендации
        st.subheader("💡 Рекомендации")
        
        if client['cv'] > 0.5:
            st.warning("📌 Волатильность расходов выше средней. Рекомендуется планирование бюджета")
        
        if client['концентрация'] > 0.6:
            st.info("📌 Расходы концентрированы в 3-х категориях. Рекомендуется диверсификация")
        
        if is_anomaly and anomaly['тип'] == 'высокие расходы':
            st.warning("📌 Зафиксирована аномалия расходов. Рекомендуется проверить бюджет")


# ========== ТАБ 2: АНАЛИЗ КОГОРТ ==========

elif selected_tab == "👥 Анализ когорт":
    st.header("👥 Сегментация клиентов по финансовому здоровью")
    
    st.subheader("📊 Распределение клиентов по когортам")
    
    # Таблица когорт
    cohort_display = cohort_profiles.copy()
    cohort_display.columns = ['Размер когорты', 'Средний оборот', 'Медиана оборота',
                              'Волатильность', 'Волат-ть (CV)', 'Концентрация',
                              'Ср. транзакции', 'Ср. возраст']
    
    st.dataframe(cohort_display.round(0), use_container_width=True)
    
    # Визуализация когорт
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = px.bar(
            x=cohort_display.index,
            y=cohort_display['Размер когорты'],
            title="Размер когорт",
            labels={'x': 'Когорта', 'y': 'Количество клиентов'}
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(
            x=cohort_display['Средний оборот'],
            y=cohort_display['Волат-ть (CV)'],
            size=cohort_display['Размер когорты'],
            title="Оборот vs Волатильность",
            labels={'x': 'Средний оборот', 'y': 'Коэффициент вариации'},
            text=cohort_display.index
        )
        fig2.update_traces(textposition='top center')
        st.plotly_chart(fig2, use_container_width=True)
    
    # Характеристика каждой когорты
    st.subheader("🎯 Характеристики когорт")
    
    for cohort_id in sorted(cohort_display.index):
        with st.expander(f"Когорта {cohort_id} ({int(cohort_display.loc[cohort_id, 'Размер когорты'])} клиентов)"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Средний оборот", f"{cohort_display.loc[cohort_id, 'Средний оборот']:.0f} р.")
            with col2:
                st.metric("Волатильность", f"{cohort_display.loc[cohort_id, 'Волат-ть (CV)']:.2f}")
            with col3:
                st.metric("Концентрация", f"{cohort_display.loc[cohort_id, 'Концентрация']:.1%}")
            with col4:
                st.metric("Ср. возраст", f"{cohort_display.loc[cohort_id, 'Ср. возраст']:.0f}")


# ========== ТАБ 3: МОНИТОРИНГ АНОМАЛИЙ ==========

elif selected_tab == "🚨 Мониторинг аномалий":
    st.header("🚨 Система мониторинга и уведомлений")
    
    st.write(f"**Всего аномалий выявлено: {len(anomalies_df)}**")
    
    # Статистика аномалий
    col1, col2, col3 = st.columns(3)
    with col1:
        high_count = len(anomalies_df[anomalies_df['тип'] == 'высокие расходы'])
        st.metric("⬆️ Высокие расходы", high_count)
    with col2:
        low_count = len(anomalies_df[anomalies_df['тип'] == 'низкие расходы'])
        st.metric("⬇️ Низкие расходы", low_count)
    with col3:
        high_priority = len(anomalies_df[anomalies_df['приоритет'] == 'высокий'])
        st.metric("🔴 Высокий приоритет", high_priority)
    
    # Фильтры
    st.subheader("🔍 Фильтры")
    col1, col2 = st.columns(2)
    
    with col1:
        anomaly_type = st.multiselect(
            "Тип аномалии",
            options=['высокие расходы', 'низкие расходы'],
            default=['высокие расходы', 'низкие расходы']
        )
    
    with col2:
        priority = st.multiselect(
            "Приоритет",
            options=['высокий', 'средний'],
            default=['высокий', 'средний']
        )
    
    # Отфильтрованные аномалии
    filtered_anomalies = anomalies_df[
        (anomalies_df['тип'].isin(anomaly_type)) &
        (anomalies_df['приоритет'].isin(priority))
    ].sort_values('отклонение_%', ascending=False)
    
    st.subheader(f"📋 Аномалии ({len(filtered_anomalies)} шт.)")
    
    if len(filtered_anomalies) > 0:
        # Таблица аномалий
        display_cols = ['ключ_клиента', 'тип', 'текущий_оборот', 'отклонение_%', 'приоритет']
        st.dataframe(
            filtered_anomalies[display_cols].round(0),
            use_container_width=True,
            hide_index=True
        )
        
        # График аномалий
        fig = px.bar(
            filtered_anomalies.sort_values('отклонение_%'),
            x='отклонение_%',
            y='ключ_клиента',
            color='тип',
            orientation='h',
            title="Величина отклонений от нормы",
            labels={'отклонение_%': 'Отклонение (%)', 'ключ_клиента': 'Клиент'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет аномалий с выбранными фильтрами")


# ========== ТАБ 4: ПРОГНОЗИРОВАНИЕ ==========

elif selected_tab == "📈 Прогнозирование":
    st.header("📈 Сценарное моделирование")
    
    client_id = st.selectbox("Выберите клиента для прогноза:", 
        options=client_ids, index=0, key="forecast_selector")
    
    client_data = baseline_df[baseline_df['ключ_клиента'] == client_id]
    if len(client_data) == 0:
        st.error("Клиент не найден")
    else:
        client = client_data.iloc[0]
        
        st.info(f"👤 Клиент {client_id} | Когорта #{int(client['когорта'])}")
        
        # Параметры прогноза
        col1, col2, col3 = st.columns(3)
        with col1:
            growth_rate = st.slider("Рост расходов (%)", -20, 50, 10)
        with col2:
            volatility_change = st.slider("Изменение волатильности (%)", -30, 30, 0)
        with col3:
            months = st.slider("Период (месяцы)", 1, 12, 6)
        
        # Прогноз
        current_mean = client['оборот_mean']
        current_cv = client['cv']
        
        scenarios = []
        for m in range(0, months + 1):
            mean_forecast = current_mean * (1 + growth_rate/100) ** m
            cv_forecast = current_cv * (1 + volatility_change/100) ** m
            ci_lower_forecast = max(0, mean_forecast * (1 - 1.645 * cv_forecast))
            ci_upper_forecast = mean_forecast * (1 + 1.645 * cv_forecast)
            
            scenarios.append({
                'месяц': m,
                'оборот': mean_forecast,
                'волатильность': cv_forecast,
                'ci_lower': ci_lower_forecast,
                'ci_upper': ci_upper_forecast
            })
        
        scenarios_df = pd.DataFrame(scenarios)
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.area(
                scenarios_df,
                x='месяц',
                y=['ci_lower', 'оборот', 'ci_upper'],
                title="Прогноз оборота с доверительным интервалом",
                labels={'месяц': 'Месяц', 'value': 'Оборот (р.)'}
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(
                scenarios_df,
                x='месяц',
                y='волатильность',
                title="Прогноз волатильности",
                markers=True,
                labels={'месяц': 'Месяц', 'волатильность': 'Коэф. вариации'}
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Итоги
        st.subheader("📊 Итоги прогноза")
        
        final_scenario = scenarios_df[scenarios_df['месяц'] == months].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Прогноз оборота", f"{final_scenario['оборот']:.0f} р.")
        with col2:
            st.metric("Изменение", f"{(final_scenario['оборот']/current_mean - 1)*100:.1f}%")
        with col3:
            st.metric("Волатильность", f"{final_scenario['волатильность']:.2f}")
        with col4:
            st.metric("Диапазон ДИ", f"[{final_scenario['ci_lower']:.0f}, {final_scenario['ci_upper']:.0f}]")


# Footer
st.markdown("""
---
**VTB Avatar** | Финансовое здоровье и когортный анализ  
Данные обновлены: """ + datetime.now().strftime("%Y-%m-%d %H:%M"))
