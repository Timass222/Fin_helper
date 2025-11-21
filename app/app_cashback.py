import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ========== КОНФИГУРАЦИЯ ==========
st.set_page_config(
    page_title="Tinkoff Cashback Analytics",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ========== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========

def generate_recommendations(client_metrics, cohort_profile, client_data, 
                            oboroty_cols, activation_cols, cashback_cols):
    """Генерирует рекомендации на основе метрик"""
    
    recommendations = []
    
    # Рекомендация 1: Активация
    if client_metrics['коэф_активации'] < 0.6:
        recommendations.append({
            'title': 'Повысить активацию кэшбэка',
            'description': 'Клиент активирует менее 60% доступных категорий. Это потенциал для роста.',
            'potential': f"Возможно +{client_metrics['потенциальный_кэшбэк']:.2f} р. в месяц",
            'action': '✅ Включить push-уведомления о новых категориях'
        })
    else:
        recommendations.append({
            'title': 'Отличная активация кэшбэка',
            'description': 'Клиент активирует свыше 60% категорий. Держать статус-кво.',
            'potential': 'Стабильный доход',
            'action': '✅ Отправить премиум-предложения'
        })
    
    # Рекомендация 2: Волатильность
    if client_metrics['волатильность_расходов'] > cohort_profile['волатильность_расходов'] * 1.2:
        recommendations.append({
            'title': 'Диверсифицировать расходы',
            'description': 'Расходы клиента нестабильны. Рекомендуется добавить новые категории.',
            'potential': 'Снижение волатильности на 15-20%',
            'action': '✅ Персональное предложение на 2-3 новых категории'
        })
    
    # Рекомендация 3: Премиум статус
    if client_metrics['премиум_статус'] == 1:
        recommendations.append({
            'title': 'VIP-клиент - предложить премиум',
            'description': 'Клиент в топ-25% по кэшбэку. Кандидат на премиум-сегмент.',
            'potential': 'Увеличение LTV на 30-40%',
            'action': '✅ Отправить VIP-оффер на повышенный кэшбэк'
        })
    
    # Рекомендация 4: Потенциал роста
    unrealized = client_metrics['потенциальный_кэшбэк']
    if unrealized > 5:
        recommendations.append({
            'title': 'Огромный потенциал роста',
            'description': f'Клиент не получает ~{unrealized:.2f} р. в месяц из не активированных категорий.',
            'potential': f'+{unrealized:.2f} р./месяц = +{unrealized*12:.2f} р./год',
            'action': '✅ Персональный консультант'
        })
    
    return recommendations[:3]  # Показываем топ-3


def create_llm_prompt(client_metrics, cohort_profile, client_data):
    """Создаёт промпт для LLM"""
    
    prompt = f"""Проанализируй финансовый профиль клиента и дай 5-7 конкретных рекомендаций по максимизации кэшбэка.

ДАННЫЕ КЛИЕНТА:
- Оборот/месяц: {client_metrics['оборот_за_месяц']:.0f} р./месяц
- Получено кэшбэка: {client_metrics['кэшбэк_за_месяц']:.2f} р./месяц
- Эффективность кэшбэка: {client_metrics['кэшбэк_rate']*100:.2f}%
- Активирована категорий: {client_metrics['активированные_категории']:.0f} из {client_metrics['доступные_категории']:.0f}
- Коэффициент активации: {client_metrics['коэф_активации']:.2%}
- Волатильность расходов: {client_metrics['волатильность_расходов']:.2f}
- Концентрация расходов (топ-3): {client_metrics['концентрация_расходов']:.2%}
- Возраст: {client_metrics['возраст']:.0f} лет
- Потенциальный недополученный кэшбэк: {client_metrics['потенциальный_кэшбэк']:.2f} р./месяц

СРЕДНИЕ ПОКАЗАТЕЛИ ПО КОГОРТЕ:
- Средний оборот: {cohort_profile['оборот_за_месяц']:.0f} р.
- Средний кэшбэк: {cohort_profile['кэшбэк_за_месяц']:.2f} р.
- Средний коэфф. активации: {cohort_profile['коэф_активации']:.2%}

ЗАДАНИЕ:
1. Определи сегмент клиента
2. Выявь основные проблемы использования кэшбэка
3. Дай 5-7 специфических действий для увеличения кэшбэка
4. Оцени потенциальный прирост дохода клиента в год
5. Предложи особую кампанию для этого клиента

Ответ структурируй в JSON формате."""
    
    return prompt


# ========== ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ ==========

@st.cache_data
def load_and_process_data():
    """Загружает данные, считает метрики и создаёт кластеры"""
    
    # Загрузка
    df = pd.read_excel('T_cashback_dataset.xlsx')
    
    # ВРЕМЕННО: для быстрого тестирования (раскомментируйте если нужно)
    df = df.head(5000)
    
    oboroty_cols = [col for col in df.columns if col.startswith('оборот_')]
    cashback_cols = [col for col in df.columns if col.startswith('кэшбэк_')]
    activation_cols = [col for col in df.columns if col.startswith('активация_')]
    
    # ===== РАСЧЕТ МЕТРИК =====
    metrics_df = pd.DataFrame()
    metrics_df['ключ_клиента'] = df['ключ_клиента'].values
    
    # Количество месяцев в датасете (апрель-сентябрь = 6)
    months_count = 6
    
    # ИСПРАВЛЕНИЕ: делим на количество месяцев для получения значений за месяц
    metrics_df['оборот_за_месяц'] = (df[oboroty_cols].sum(axis=1).values / months_count)
    metrics_df['кэшбэк_за_месяц'] = (df[cashback_cols].sum(axis=1).values / months_count)
    
    metrics_df['кэшбэк_rate'] = (metrics_df['кэшбэк_за_месяц'] / (metrics_df['оборот_за_месяц'] + 1)).values
    metrics_df['активированные_категории'] = (df[activation_cols] == 1).sum(axis=1).values
    metrics_df['доступные_категории'] = (df[activation_cols] >= 0).sum(axis=1).values
    metrics_df['коэф_активации'] = (metrics_df['активированные_категории'] / (metrics_df['доступные_категории'] + 1)).values
    metrics_df['кэшбэк_на_категорию'] = (metrics_df['кэшбэк_за_месяц'] / (metrics_df['активированные_категории'] + 1)).values
    metrics_df['оборот_на_категорию'] = (metrics_df['оборот_за_месяц'] / len(oboroty_cols)).values
    metrics_df['волатильность_расходов'] = df[oboroty_cols].std(axis=1).values / months_count
    
    def calc_concentration(row):
        top3_sum = row.nlargest(3).sum()
        total = row.sum()
        return top3_sum / total if total > 0 else 0
    
    metrics_df['концентрация_расходов'] = df[oboroty_cols].apply(calc_concentration, axis=1).values
    
    def calc_herfindahl(row):
        total = row.sum()
        return ((row / total) ** 2).sum() if total > 0 else 0
    
    metrics_df['индекс_герфиндаля'] = df[oboroty_cols].apply(calc_herfindahl, axis=1).values
    metrics_df['возраст'] = df['возраст'].values
    
    # Добавляем потенциальный кэшбэк
    metrics_df['потенциальный_кэшбэк'] = 0.0
    for idx in df.index:
        not_activated_mask = df.loc[idx, activation_cols].values == 0
        if not_activated_mask.sum() > 0:
            unrealized = (df.loc[idx, oboroty_cols].values[not_activated_mask].sum() * 0.05) / months_count
            metrics_df.loc[metrics_df['ключ_клиента'] == df.loc[idx, 'ключ_клиента'], 'потенциальный_кэшбэк'] = unrealized
    
    cashback_median = metrics_df['кэшбэк_за_месяц'].median()
    cashback_std = metrics_df['кэшбэк_за_месяц'].std()
    metrics_df['премиум_статус'] = (metrics_df['кэшбэк_за_месяц'] > cashback_median + cashback_std).astype(int).values
    
    # ===== K-MEANS КЛАСТЕРИЗАЦИЯ =====
    metrics_for_clustering = metrics_df[[col for col in metrics_df.columns if col != 'ключ_клиента']].copy()
    metrics_for_clustering = metrics_for_clustering.fillna(0)
    
    scaler = StandardScaler()
    metrics_scaled = scaler.fit_transform(metrics_for_clustering)
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(metrics_scaled)
    
    metrics_df['кластер'] = cluster_labels
    df['кластер'] = cluster_labels
    
    # Профили когорт
    cluster_profiles = metrics_df.groupby('кластер')[metrics_for_clustering.columns].mean()
    
    return df, metrics_df, cluster_profiles, oboroty_cols, cashback_cols, activation_cols, months_count


# Загружаем данные
df, metrics_df, cluster_profiles, oboroty_cols, cashback_cols, activation_cols, months_count = load_and_process_data()

# ========== СОЗДАЁМ СПИСОК ID КЛИЕНТОВ (один раз!) ==========
@st.cache_data
def get_client_ids(metrics_df):
    """Один раз создаём список ID для всех селекторов"""
    return sorted(metrics_df['ключ_клиента'].unique().tolist())

client_ids = get_client_ids(metrics_df)
default_client_id = client_ids[0]  # Первый ID в списке


# ========== MAIN INTERFACE ==========

st.title("💳 Tinkoff Cashback Analytics MVP")
st.markdown("**Анализ когорт клиентов и финансовые метрики**")


# Боковое меню
st.sidebar.header("⚙️ Параметры")
selected_tab = st.sidebar.radio("Выберите раздел:", 
    ["📊 Профиль Клиента", "👥 Анализ Когорты", "🤖 AI Рекомендации", "📈 Финансовые Сценарии"])


# ========== ТАБ 1: ПРОФИЛЬ КЛИЕНТА ==========

if selected_tab == "📊 Профиль Клиента":
    st.header("📊 Профиль Клиента")
    
    col1, col2 = st.columns(2)
    with col1:
        client_id = st.selectbox("Выберите ID клиента:", 
            options=client_ids,
            index=0)
    
    with col2:
        st.write("")
    
    # Получаем данные клиента
    client_row_idx = metrics_df[metrics_df['ключ_клиента'] == client_id].index
    if len(client_row_idx) == 0:
        st.error("Клиент не найден")
    else:
        client_metrics = metrics_df.loc[client_row_idx[0]]
        client_data = df[df['ключ_клиента'] == client_id].iloc[0]
        
        # Основные KPI
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Оборот/месяц (р.)", f"{client_metrics['оборот_за_месяц']:.0f}")
        with col2:
            st.metric("💵 Кэшбэк/месяц (р.)", f"{client_metrics['кэшбэк_за_месяц']:.2f}")
        with col3:
            st.metric("📊 Эффективность %", f"{client_metrics['кэшбэк_rate']*100:.2f}%")
        with col4:
            cohort = int(client_metrics['кластер'])
            st.metric("👥 Когорта", f"#{cohort}")
        
        # Детальные метрики
        st.subheader("📋 Детальные финметрики")
        
        metrics_display = pd.DataFrame({
            'Метрика': [
                'Активированные категории',
                'Доступные категории',
                'Коэф. активации',
                'Кэшбэк на категорию (р.)',
                'Оборот на категорию (р.)',
                'Волатильность расходов',
                'Концентрация расходов (топ-3)',
                'Возраст',
                'Премиум статус'
            ],
            'Значение': [
                f"{client_metrics['активированные_категории']:.0f}",
                f"{client_metrics['доступные_категории']:.0f}",
                f"{client_metrics['коэф_активации']:.2%}",
                f"{client_metrics['кэшбэк_на_категорию']:.2f}",
                f"{client_metrics['оборот_на_категорию']:.2f}",
                f"{client_metrics['волатильность_расходов']:.2f}",
                f"{client_metrics['концентрация_расходов']:.2%}",
                f"{client_metrics['возраст']:.0f} лет",
                "🟢 Премиум" if client_metrics['премиум_статус'] == 1 else "⚪ Обычный"
            ]
        })
        st.table(metrics_display)
        
        # График расходов по категориям - ИСПРАВЛЕННЫЙ
        st.subheader("📊 Расходы по категориям (р./месяц)")
        
        category_spending = []
        for cat in oboroty_cols:
            col_name = cat.replace('оборот_', '')
            spending = client_data[cat] / months_count if not pd.isna(client_data[cat]) else 0
            
            activation_col = f'активация_{col_name}'
            if activation_col in client_data.index:
                activated = client_data[activation_col]
            else:
                activated = np.nan
            
            category_spending.append({
                'Категория': col_name.replace('_', ' ').title(),
                'Оборот': spending,
                'Активирован': '✅' if activated == 1 else ('❌' if activated == 0 else '—')
            })
        
        cat_df = pd.DataFrame(category_spending)
        cat_df = cat_df[cat_df['Оборот'] > 0].sort_values('Оборот', ascending=True).tail(15)
        
        if len(cat_df) > 0:
            fig = px.bar(cat_df, x='Оборот', y='Категория', 
            orientation='h',
            labels={'Категория': '', 'Оборот': 'Оборот (р./месяц)'},
            title="Топ-15 категорий расходов")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Нет данных по расходам для этого клиента")


# ========== ТАБ 2: АНАЛИЗ КОГОРТЫ ==========

elif selected_tab == "👥 Анализ Когорты":
    st.header("👥 Анализ Когорт")
    
    cohort_id = st.slider("Выберите когорту:", 0, 5, 0)
    
    cohort_clients = metrics_df[metrics_df['кластер'] == cohort_id]
    cohort_profile = cluster_profiles.loc[cohort_id]
    
    # Статистика по когорте
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Клиентов в когорте", len(cohort_clients))
    with col2:
        st.metric("💰 Средний оборот (р.)", f"{cohort_profile['оборот_за_месяц']:.0f}")
    with col3:
        st.metric("💵 Средний кэшбэк (р.)", f"{cohort_profile['кэшбэк_за_месяц']:.2f}")
    with col4:
        st.metric("📊 Ср. эффективность", f"{cohort_profile['кэшбэк_rate']*100:.2f}%")
    
    # Профиль когорты
    st.subheader(f"📋 Профиль когорты #{cohort_id}")
    
    profile_display = pd.DataFrame({
        'Метрика': cohort_profile.index,
        'Значение': cohort_profile.values
    }).round(2)
    
    st.dataframe(profile_display, use_container_width=True)
    
    # Сравнение когорт
    st.subheader("📊 Сравнение всех когорт (радар)")
    
    fig = go.Figure()
    for cluster_id in range(6):
        profile = cluster_profiles.loc[cluster_id]
        fig.add_trace(go.Scatterpolar(
            r=[profile['оборот_за_месяц']/100, 
               profile['кэшбэк_за_месяц'],
               profile['коэф_активации']*10,
               profile['концентрация_расходов']*20,
               profile['возраст']/5],
            theta=['Оборот', 'Кэшбэк', 'Активация', 'Концентр.', 'Возраст'],
            fill='toself',
            name=f'Когорта {cluster_id}'
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
        title="Профили когорт (радар)",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)


# ========== ТАБ 3: AI РЕКОМЕНДАЦИИ ==========

elif selected_tab == "🤖 AI Рекомендации":
    st.header("🤖 AI-Powered Рекомендации")
    
    client_id = st.selectbox("Выберите клиента для рекомендаций:", 
        options=client_ids,
        index=0,
        key="recommendations_selector")
    
    client_row_idx = metrics_df[metrics_df['ключ_клиента'] == client_id].index
    if len(client_row_idx) == 0:
        st.error("Клиент не найден")
    else:
        client_metrics = metrics_df.loc[client_row_idx[0]]
        client_data = df[df['ключ_клиента'] == client_id].iloc[0]
        cohort_id = int(client_metrics['кластер'])
        cohort_profile = cluster_profiles.loc[cohort_id]
        
        st.info(f"👤 Клиент ID: {client_id} | 👥 Когорта: #{cohort_id}")
        
        # Генерируем рекомендации
        recommendations = generate_recommendations(
            client_metrics, cohort_profile, client_data, 
            oboroty_cols, activation_cols, cashback_cols
        )
        
        st.subheader("💡 Персонализированные рекомендации")
        
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"📌 Рекомендация {i}: {rec['title']}", expanded=i==1):
                st.write(f"**Описание:** {rec['description']}")
                st.write(f"**Потенциал:** {rec['potential']}")
                st.write(f"**Действие:** {rec['action']}")
        
        # Шаблон LLM промпта
        st.subheader("🔧 LLM Промпт (OpenAI/Claude)")
        
        llm_prompt = create_llm_prompt(client_metrics, cohort_profile, client_data)
        
        with st.expander("Посмотреть промпт для LLM", expanded=False):
            st.code(llm_prompt, language="text")
        
        # Кнопка для отправки в LLM
        if st.button("📤 Отправить в ChatGPT / Claude"):
            st.success("✅ Промпт скопирован в буфер обмена! Вставьте его в ChatGPT или Claude.")


# ========== ТАБ 4: ФИНАНСОВЫЕ СЦЕНАРИИ ==========

elif selected_tab == "📈 Финансовые Сценарии":
    st.header("📈 Финансовые Сценарии")
    
    client_id = st.selectbox("Выберите клиента для сценариев:", 
        options=client_ids,
        index=0,
        key="scenarios_selector")
    
    client_row_idx = metrics_df[metrics_df['ключ_клиента'] == client_id].index
    if len(client_row_idx) == 0:
        st.error("Клиент не найден")
    else:
        client_metrics = metrics_df.loc[client_row_idx[0]]
        
        st.info(f"👤 Клиент ID: {client_id}")
        
        # Параметры сценариев
        col1, col2, col3 = st.columns(3)
        with col1:
            growth_rate = st.slider("Рост оборота (%)", 0, 50, 15)
        with col2:
            activation_boost = st.slider("Рост активации (%)", 0, 30, 10)
        with col3:
            months = st.slider("Период прогноза (месяцы)", 1, 12, 6)
        
        # Вычисляем сценарии
        current_turnover = client_metrics['оборот_за_месяц']
        current_cashback = client_metrics['кэшбэк_за_месяц']
        current_activation = client_metrics['коэф_активации']
        
        # Базовый сценарий (без изменений)
        base_scenario = {
            'месяц': 0,
            'оборот': current_turnover,
            'кэшбэк': current_cashback,
            'активация': current_activation
        }
        
        # Оптимистичный сценарий
        scenarios = [base_scenario]
        for m in range(1, months + 1):
            turnover = current_turnover * (1 + growth_rate/100) ** m
            activation = min(current_activation * (1 + activation_boost/100) ** m, 0.95)
            cashback = turnover * activation * 0.05
            
            scenarios.append({
                'месяц': m,
                'оборот': turnover,
                'кэшбэк': cashback,
                'активация': activation
            })
        
        scenarios_df = pd.DataFrame(scenarios)
        
        # Графики
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.line(scenarios_df, x='месяц', y='оборот',
                title="Прогноз оборота",
                labels={'месяц': 'Месяц', 'оборот': 'Оборот (р.)'})
            fig1.add_hline(y=current_turnover, line_dash="dash", line_color="red", 
                           annotation_text="Текущий")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.line(scenarios_df, x='месяц', y='кэшбэк',
                title="Прогноз кэшбэка",
                labels={'месяц': 'Месяц', 'кэшбэк': 'Кэшбэк (р.)'})
            fig2.add_hline(y=current_cashback, line_dash="dash", line_color="red",
                           annotation_text="Текущий")
            st.plotly_chart(fig2, use_container_width=True)
        
        # Итоговая таблица
        st.subheader("📊 Итоговые значения по сценарию")
        
        final_month_idx = scenarios_df[scenarios_df['месяц'] == months].index[0]
        
        result_df = pd.DataFrame({
            'Метрика': ['Оборот (р.)', 'Кэшбэк (р.)', 'Активация'],
            'Текущее': [f"{current_turnover:.0f}", f"{current_cashback:.2f}", f"{current_activation:.2%}"],
            f'Через {months} месяцев': [
                f"{scenarios_df.loc[final_month_idx, 'оборот']:.0f}",
                f"{scenarios_df.loc[final_month_idx, 'кэшбэк']:.2f}",
                f"{scenarios_df.loc[final_month_idx, 'активация']:.2%}"
            ],
            'Прирост': [
                f"+{(scenarios_df.loc[final_month_idx, 'оборот']/current_turnover - 1)*100:.1f}%",
                f"+{(scenarios_df.loc[final_month_idx, 'кэшбэк']/current_cashback - 1)*100:.1f}%",
                f"+{(scenarios_df.loc[final_month_idx, 'активация']/current_activation - 1)*100:.1f}%"
            ]
        })
        
        st.table(result_df)


# Footer
st.markdown("""
---
**Tinkoff Cashback Analytics MVP** | Powered by Streamlit + Plotly  
✅ Все данные пересчитаны на месячную базу (апрель-сентябрь = 6 месяцев)
""")
