import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from streamlit_autorefresh import st_autorefresh
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="EcoPower Hub - Revolution Énergétique",
    page_icon="🌍⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh toutes les 30 secondes
st_autorefresh(interval=30000, key="data_refresh")

# CSS personnalisé avec effets avancés
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: #ffffff;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00ff87, #60efff, #0061ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 20px rgba(0, 255, 135, 0.3);
        margin: 0;
        padding: 20px 0;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .section-title {
        font-size: 2rem;
        background: linear-gradient(90deg, #00ff87, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 40px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(0, 255, 135, 0.3);
    }
    
    .highlight-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour générer des données réalistes
def generate_realistic_data():
    """Génère des données énergétiques réalistes pour une maison"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    n = len(dates)
    
    np.random.seed(42)
    
    # Consommation de base
    base_power = 200.0 + 50.0 * np.sin(2 * np.pi * dates.hour / 24.0)
    seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * dates.dayofyear / 365.0 - np.pi/2.0)
    temperature = 15.0 + 10.0 * np.sin(2 * np.pi * dates.dayofyear / 365.0) + np.random.randn(n) * 5.0
    
    heating_effect = np.maximum(0.0, 18.0 - temperature) * 30.0
    global_power = base_power * seasonal_factor + heating_effect + np.random.randn(n) * 20.0
    global_power = np.maximum(50.0, global_power)
    
    # Consommation par pièce
    living_room = 60.0 + 40.0 * np.sin(2 * np.pi * dates.hour / 24.0) + np.random.randn(n) * 10.0
    
    kitchen = np.zeros(n, dtype=float)
    kitchen += 80.0 * ((dates.hour >= 7) & (dates.hour <= 9)).astype(float)
    kitchen += 120.0 * ((dates.hour >= 12) & (dates.hour <= 14)).astype(float)
    kitchen += 150.0 * ((dates.hour >= 19) & (dates.hour <= 21)).astype(float)
    kitchen += np.random.randn(n) * 15.0
    
    bedroom = 40.0 + 20.0 * ((dates.hour >= 22) | (dates.hour <= 7)).astype(float) + np.random.randn(n) * 8.0
    bathroom = 100.0 * ((dates.hour >= 7) & (dates.hour <= 9)).astype(float)
    bathroom += 80.0 * ((dates.hour >= 20) & (dates.hour <= 22)).astype(float)
    bathroom += np.random.randn(n) * 20.0
    
    standby = 50.0 + np.random.randn(n) * 5.0
    other_power = bathroom + standby

    raw_data = pd.DataFrame({
        'datetime': dates,
        'Global_active_power': global_power.astype(float),
        'Global_active_power_Wh': (global_power * 1000.0).astype(float),
        'Living_room_power': living_room.astype(float),
        'Kitchen_power': kitchen.astype(float),
        'Bedroom_power': bedroom.astype(float),
        'Bathroom_power': bathroom.astype(float),
        'Standby_power': standby.astype(float),
        'Other_power': other_power.astype(float),
        'hour': dates.hour.astype(int),
        'dayofweek': dates.dayofweek.astype(int),
        'month': dates.month.astype(int),
        'temperature': temperature.astype(float)
    })
    
    # Statistiques globales
    daily_avg = raw_data.groupby(raw_data['datetime'].dt.date)['Global_active_power'].mean().mean()
    
    global_stats = {
        'avg_daily_consumption_kWh': float(round(daily_avg * 24.0 / 1000.0, 1)),
        'peak_power': float(round(global_power.max() / 1000.0, 1)),
        'surconsommation_rate': 12.5
    }
    
    # Données mensuelles
    monthly_data = raw_data.groupby(raw_data['datetime'].dt.to_period('M')).agg({
        'Global_active_power_Wh': 'sum',
        'Living_room_power': 'mean',
        'Kitchen_power': 'mean',
        'Bedroom_power': 'mean'
    }).reset_index()
    
    monthly_data['year'] = monthly_data['datetime'].dt.year.astype(int)
    monthly_data['month'] = monthly_data['datetime'].dt.month.astype(int)
    monthly_data['Global_active_power'] = (monthly_data['Global_active_power_Wh'] / (1000.0 * 24.0 * 30.0)).astype(float)
    
    # Données journalières
    daily_data = raw_data.groupby(raw_data['datetime'].dt.date).agg({
        'Global_active_power_Wh': 'sum',
        'temperature': 'mean'
    }).reset_index()
    daily_data.columns = ['date', 'Global_active_power_Wh', 'temperature']
    daily_data['Global_active_power_Wh'] = daily_data['Global_active_power_Wh'].astype(float)
    daily_data['temperature'] = daily_data['temperature'].astype(float)
    
    return {
        'raw_data': raw_data,
        'global_stats': global_stats,
        'monthly_data': monthly_data,
        'daily_data': daily_data
    }

# Fonction pour charger ou générer les données
@st.cache_data
def load_or_generate_data():
    try:
        with open('energy_data.pkl', 'rb') as f:
            data = pickle.load(f)
            if 'Other_power' not in data['raw_data'].columns:
                data['raw_data']['Other_power'] = data['raw_data']['Bathroom_power'] + 50.0
            return data
    except:
        data = generate_realistic_data()
        with open('energy_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        return data

# Chargement des données
try:
    data = load_or_generate_data()
except Exception as e:
    st.error(f"Erreur : {e}")
    data = generate_realistic_data()

# En-tête principal
st.markdown('<h1 class="main-title">🏠Consommation énergétique </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; opacity: 0.8;">Pilotez votre consommation en temps réel</p>', unsafe_allow_html=True)

# Barre de contrôle
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🔄 ACTUALISER", use_container_width=True):
        st.rerun()
with col2:
    dark_mode = st.toggle("🌙 MODE SOMBRE", value=True)
with col3:
    notifications = st.toggle("🔔 NOTIFICATIONS", value=True)
with col4:
    if st.button("💾 SAUVEGARDER", use_container_width=True):
        st.success("Données sauvegardées !")

# Section 1: MAISON INTERACTIVE
st.markdown('<div class="section-title">🏠 MAISON INTELLIGENTE</div>', unsafe_allow_html=True)

# État initial des appareils
if 'appliances' not in st.session_state:
    st.session_state.appliances = {
        'heating': {'on': True, 'temp': 20, 'power': 800.0},
        'lights_living': {'on': True, 'power': 150.0},
        'lights_kitchen': {'on': True, 'power': 120.0},
        'lights_bedroom': {'on': False, 'power': 80.0},
        'lights_bathroom': {'on': False, 'power': 60.0},
        'tv': {'on': True, 'power': 100.0},
        'fridge': {'on': True, 'power': 150.0},
        'computer': {'on': False, 'power': 120.0},
        'washing_machine': {'on': False, 'power': 2000.0},
        'oven': {'on': False, 'power': 2200.0},
        'water_heater': {'on': True, 'power': 1200.0},
        'standby_devices': {'on': True, 'power': 80.0}
    }

# Calculer la consommation actuelle
def calculate_current_power():
    total = 0.0
    for appliance, state in st.session_state.appliances.items():
        if state['on']:
            total += state['power']
    return total

current_power = calculate_current_power()

# Données de consommation par pièce
room_powers = {
    'Salon': 0.0,
    'Cuisine': 0.0,
    'Chambre': 0.0,
    'Salle de bain': 0.0,
    'Veilles': 0.0
}

# Assigner la puissance aux pièces
if st.session_state.appliances['heating']['on']:
    room_powers['Salon'] += st.session_state.appliances['heating']['power'] * 0.4
    room_powers['Chambre'] += st.session_state.appliances['heating']['power'] * 0.3
    room_powers['Cuisine'] += st.session_state.appliances['heating']['power'] * 0.3

if st.session_state.appliances['lights_living']['on']:
    room_powers['Salon'] += st.session_state.appliances['lights_living']['power']
if st.session_state.appliances['lights_kitchen']['on']:
    room_powers['Cuisine'] += st.session_state.appliances['lights_kitchen']['power']
if st.session_state.appliances['lights_bedroom']['on']:
    room_powers['Chambre'] += st.session_state.appliances['lights_bedroom']['power']
if st.session_state.appliances['lights_bathroom']['on']:
    room_powers['Salle de bain'] += st.session_state.appliances['lights_bathroom']['power']

if st.session_state.appliances['tv']['on']:
    room_powers['Salon'] += st.session_state.appliances['tv']['power']
if st.session_state.appliances['fridge']['on']:
    room_powers['Cuisine'] += st.session_state.appliances['fridge']['power']
if st.session_state.appliances['computer']['on']:
    room_powers['Chambre'] += st.session_state.appliances['computer']['power']
if st.session_state.appliances['washing_machine']['on']:
    room_powers['Salle de bain'] += st.session_state.appliances['washing_machine']['power'] * 0.7
    room_powers['Cuisine'] += st.session_state.appliances['washing_machine']['power'] * 0.3
if st.session_state.appliances['oven']['on']:
    room_powers['Cuisine'] += st.session_state.appliances['oven']['power']
if st.session_state.appliances['water_heater']['on']:
    room_powers['Salle de bain'] += st.session_state.appliances['water_heater']['power']

room_powers['Veilles'] = st.session_state.appliances['standby_devices']['power'] if st.session_state.appliances['standby_devices']['on'] else 0.0

# Créer la visualisation de la maison
fig_house = go.Figure()

# Structure de la maison
house_parts = {
    'walls': {'x': [0, 12, 12, 0, 0], 'y': [0, 0, 8, 8, 0], 'color': 'white', 'alpha': 0.1},
    'roof': {'x': [0, 6, 12], 'y': [8, 12, 8], 'color': '#ff6b6b', 'alpha': 0.2},
    'living_room': {'x': [1, 5, 5, 1], 'y': [4, 4, 7, 7], 'color': '#00ff87', 'name': 'Salon'},
    'kitchen': {'x': [7, 11, 11, 7], 'y': [4, 4, 7, 7], 'color': '#00d4ff', 'name': 'Cuisine'},
    'bedroom': {'x': [1, 5, 5, 1], 'y': [1, 1, 3, 3], 'color': '#ff6b6b', 'name': 'Chambre'},
    'bathroom': {'x': [7, 11, 11, 7], 'y': [1, 1, 3, 3], 'color': '#ffd93d', 'name': 'Salle de bain'},
    'hallway': {'x': [5, 7, 7, 5], 'y': [1, 1, 7, 7], 'color': '#9d4edd', 'name': 'Couloir'}
}

# Dessiner la maison
for part_name, part_data in house_parts.items():
    if part_name in ['walls', 'roof']:
        fillcolor = f'rgba(255,255,255,{part_data["alpha"]})' if part_name == 'walls' else f'rgba(255,107,107,{part_data["alpha"]})'
        
        fig_house.add_trace(go.Scatter(
            x=part_data['x'], y=part_data['y'],
            fill="toself",
            fillcolor=fillcolor,
            line=dict(color=part_data['color'], width=2),
            hoverinfo="skip",
            showlegend=False
        ))
    else:
        room_name = part_data['name']
        power_value = room_powers.get(room_name, 0.0)
        alpha = 0.2 + min(0.6, power_value / 1000.0 * 0.6)
        
        color_map = {
            '#00ff87': (0, 255, 135),
            '#00d4ff': (0, 212, 255),
            '#ff6b6b': (255, 107, 107),
            '#ffd93d': (255, 217, 61),
            '#9d4edd': (157, 78, 221)
        }
        
        rgb = color_map.get(part_data['color'], (255, 255, 255))
        
        fig_house.add_trace(go.Scatter(
            x=part_data['x'], y=part_data['y'],
            fill="toself",
            fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},{alpha})',
            line=dict(color=part_data['color'], width=1.5),
            name=room_name,
            text=f"{room_name}: {power_value:.0f} W",
            hoverinfo="text",
            showlegend=True
        ))

# Ajouter les labels
for part_name, part_data in house_parts.items():
    if part_name not in ['walls', 'roof']:
        center_x = sum(part_data['x'][:2]) / 2.0
        center_y = sum(part_data['y'][:2]) / 2.0
        room_name = part_data['name']
        power_value = room_powers.get(room_name, 0.0)
        
        fig_house.add_annotation(
            x=center_x,
            y=center_y + 0.5,
            text=room_name.upper(),
            showarrow=False,
            font=dict(size=11, color="white", weight="bold"),
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor=part_data['color'],
            borderwidth=1,
            borderpad=3
        )
        
        fig_house.add_annotation(
            x=center_x,
            y=center_y - 0.2,
            text=f"{power_value:.0f}W",
            showarrow=False,
            font=dict(size=10, color=part_data['color']),
            bgcolor="rgba(0,0,0,0.4)"
        )

fig_house.update_layout(
    title="🏠 CONSOMMATION PAR PIÈCE",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 13]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 14]),
    showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)'),
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=20, r=20, t=80, b=20)
)

# Afficher la maison
house_col1, house_col2 = st.columns([3, 2])

with house_col1:
    st.plotly_chart(fig_house, use_container_width=True)

with house_col2:
    with st.container():
        st.markdown("### 🔧 CONTRÔLE DES APPAREILS")
        
        # Contrôles
        with st.expander("🌡️ CHAUFFAGE ET CLIMATISATION", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                heating_on = st.toggle("Chauffage", 
                                     value=st.session_state.appliances['heating']['on'],
                                     key="heating_toggle")
                st.session_state.appliances['heating']['on'] = heating_on
                
                if heating_on:
                    temp = st.slider("Température", 16, 24, 
                                   st.session_state.appliances['heating']['temp'],
                                   key="temp_slider")
                    st.session_state.appliances['heating']['temp'] = temp
            
            with col2:
                water_on = st.toggle("Chauffe-eau", 
                                   value=st.session_state.appliances['water_heater']['on'],
                                   key="water_toggle")
                st.session_state.appliances['water_heater']['on'] = water_on
        
        with st.expander("💡 ÉCLAIRAGE"):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.appliances['lights_living']['on'] = st.toggle("Salon", 
                    value=st.session_state.appliances['lights_living']['on'])
                st.session_state.appliances['lights_kitchen']['on'] = st.toggle("Cuisine", 
                    value=st.session_state.appliances['lights_kitchen']['on'])
            with cols[1]:
                st.session_state.appliances['lights_bedroom']['on'] = st.toggle("Chambre", 
                    value=st.session_state.appliances['lights_bedroom']['on'])
                st.session_state.appliances['lights_bathroom']['on'] = st.toggle("Salle de bain", 
                    value=st.session_state.appliances['lights_bathroom']['on'])
        
        with st.expander("📺 ÉLECTROMÉNAGER"):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.appliances['tv']['on'] = st.toggle("TV", 
                    value=st.session_state.appliances['tv']['on'])
                st.session_state.appliances['computer']['on'] = st.toggle("Ordinateur", 
                    value=st.session_state.appliances['computer']['on'])
            with cols[1]:
                st.session_state.appliances['fridge']['on'] = st.toggle("Réfrigérateur", 
                    value=st.session_state.appliances['fridge']['on'])
                st.session_state.appliances['standby_devices']['on'] = st.toggle("Mode veille", 
                    value=st.session_state.appliances['standby_devices']['on'])
        
        with st.expander("🧺 GROS ÉLECTROMÉNAGER"):
            cols = st.columns(2)
            with cols[0]:
                st.session_state.appliances['washing_machine']['on'] = st.toggle("Lave-linge", 
                    value=st.session_state.appliances['washing_machine']['on'])
            with cols[1]:
                st.session_state.appliances['oven']['on'] = st.toggle("Four", 
                    value=st.session_state.appliances['oven']['on'])
        
        # Résumé
        st.markdown("---")
        st.markdown("**CONSOMMATION ACTUELLE**")
        st.markdown(f"<div style='font-size: 2rem; color: #00ff87;'>{current_power:,.0f} W</div>", unsafe_allow_html=True)
        st.caption(f"Coût horaire: €{(current_power * 0.18 / 1000.0):.3f}")

# Section 2: MÉTRIQUES
st.markdown('<div class="section-title">📊 IMPACT ET ÉCONOMIES</div>', unsafe_allow_html=True)

# Calculer les métriques
total_energy_kwh = data['raw_data']['Global_active_power_Wh'].sum() / 1000000.0
co2_saved_kg = total_energy_kwh * 400.0
avg_daily_kwh = data['global_stats']['avg_daily_consumption_kWh']
monthly_cost = avg_daily_kwh * 30.0 * 0.18

# Métriques
metrics_cols = st.columns(4)

with metrics_cols[0]:
    with st.container():
        st.metric("⚡ kW ACTUELS", f"{current_power/1000.0:.1f}", 
                 f"{sum(1 for a in st.session_state.appliances.values() if a['on'])}/12 appareils")

with metrics_cols[1]:
    with st.container():
        st.metric("💰 COÛT MENSUEL", f"€{monthly_cost:.0f}", 
                 f"{avg_daily_kwh:.1f} kWh/jour")

with metrics_cols[2]:
    with st.container():
        st.metric("🌍 CO₂ ÉVITÉ", f"{co2_saved_kg/1000.0:.0f} tonnes", 
                 f"Équiv. {int(co2_saved_kg/1000.0 * 50)} arbres")

with metrics_cols[3]:
    with st.container():
        efficiency = 100.0 - (current_power / 5000.0 * 100.0) if current_power < 5000.0 else 0.0
        st.metric("🏆 EFFICACITÉ", f"{efficiency:.0f}%", 
                 f"Classement: {int(efficiency/10.0)}/10")

# Section 3: SIMULATEUR
st.markdown('<div class="section-title">🎯 SIMULATEUR D\'ÉCONOMIES</div>', unsafe_allow_html=True)

sim_col1, sim_col2 = st.columns([2, 1])

with sim_col1:
    with st.container():
        st.markdown("### 💡 OPTIMISEZ VOTRE CONSOMMATION")
        
        scenario_led = st.checkbox("Remplacer par des LED (-80W)")
        scenario_heating = st.checkbox("Baisser le chauffage de 1°C (-150W)")
        scenario_standby = st.checkbox("Couper les veilles (-50W)")
        scenario_schedule = st.checkbox("Programmer les gros appareils (-200W)")
        scenario_solar = st.checkbox("Ajouter des panneaux solaires (-300W)")
        
        # Calcul
        savings_watts = 0.0
        if scenario_led:
            savings_watts += 80.0
        if scenario_heating:
            savings_watts += 150.0
        if scenario_standby:
            savings_watts += 50.0
        if scenario_schedule:
            savings_watts += 200.0
        if scenario_solar:
            savings_watts += 300.0
        
        daily_savings_kwh = savings_watts * 24.0 / 1000.0
        monthly_savings_eur = daily_savings_kwh * 30.0 * 0.18
        annual_savings_eur = monthly_savings_eur * 12.0

with sim_col2:
    with st.container():
        st.markdown("### 📈 RÉSULTATS")
        
        # Afficher les résultats avec les widgets Streamlit
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Économie puissance", f"{savings_watts:.0f} W")
        
        with col2:
            st.metric("Économies mensuelles", f"€{monthly_savings_eur:.0f}")
        
        with col3:
            st.metric("Économies annuelles", f"€{annual_savings_eur:.0f}")
        
        if st.button("💾 ENREGISTRER CE SCÉNARIO", use_container_width=True):
            st.success("Scénario enregistré avec succès!")

# Section 4: VISUALISATIONS
st.markdown('<div class="section-title">📈 ANALYSES DÉTAILLÉES</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📅 HISTORIQUE", "📊 RÉPARTITION", "🔮 PRÉVISIONS"])

with tab1:
    if 'datetime' in data['raw_data'].columns:
        sample_data = data['raw_data'].iloc[-168:]
        fig_history = go.Figure()
        
        fig_history.add_trace(go.Scatter(
            x=sample_data['datetime'],
            y=sample_data['Global_active_power'],
            mode='lines',
            name='Consommation',
            line=dict(color='#00ff87', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 135, 0.1)'
        ))
        
        fig_history.update_layout(
            title="📅 CONSOMMATION HEURE PAR HEURE (7 derniers jours)",
            xaxis_title="Date et Heure",
            yaxis_title="Puissance (W)",
            height=400,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_history, use_container_width=True)

with tab2:
    categories = {
        'Chauffage': room_powers['Salon'] * 0.4 + room_powers['Chambre'] * 0.3 + room_powers['Cuisine'] * 0.3,
        'Éclairage': sum(st.session_state.appliances[f'lights_{room}']['power'] 
                        for room in ['living', 'kitchen', 'bedroom', 'bathroom'] 
                        if st.session_state.appliances[f'lights_{room}']['on']),
        'Électroménager': sum(st.session_state.appliances[app]['power'] 
                            for app in ['tv', 'fridge', 'computer', 'washing_machine', 'oven'] 
                            if st.session_state.appliances[app]['on']),
        'Eau chaude': st.session_state.appliances['water_heater']['power'] if st.session_state.appliances['water_heater']['on'] else 0.0,
        'Veilles': room_powers['Veilles']
    }
    
    filtered_categories = {k: v for k, v in categories.items() if v > 0}
    
    if filtered_categories:
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(filtered_categories.keys()),
            values=list(filtered_categories.values()),
            hole=0.4,
            marker_colors=['#00ff87', '#00d4ff', '#ff6b6b', '#ffd93d', '#9d4edd']
        )])
        
        fig_pie.update_layout(
            title="📊 RÉPARTITION PAR CATÉGORIE",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

with tab3:
    col_pred1, col_pred2 = st.columns(2)
    
    with col_pred1:
        hours = list(range(24))
        typical_pattern = [200.0 + 100.0 * np.sin(2.0 * np.pi * h / 24.0) for h in hours]
        
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=hours,
            y=typical_pattern,
            mode='lines+markers',
            name='Prévision typique',
            line=dict(color='#00d4ff', width=3)
        ))
        
        fig_forecast.add_trace(go.Scatter(
            x=hours,
            y=[p * 0.7 for p in typical_pattern],
            mode='lines',
            name='Avec optimisations',
            line=dict(color='#00ff87', width=3, dash='dash')
        ))
        
        fig_forecast.update_layout(
            title="🔮 PRÉVISION JOURNALIÈRE",
            xaxis_title="Heure de la journée",
            yaxis_title="Puissance (W)",
            height=350,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
    
    with col_pred2:
        with st.container():
            st.markdown("### 🎯 RECOMMANDATIONS")
            
            recommendations = [
                ("💡 Éteignez les lumières inutiles", "-50W"),
                ("🔥 Baissez le chauffage la nuit", "-100W"),
                ("🔌 Débranchez les chargeurs", "-20W"),
                ("⏰ Utilisez les heures creuses", "-30% coût")
            ]
            
            for text, impact in recommendations:
                with st.container():
                    st.markdown(f"**{text}**")
                    st.caption(impact)

# Section 5: DÉFIS
st.markdown('<div class="section-title">🏆 DÉFIS QUOTIDIENS</div>', unsafe_allow_html=True)

challenge_cols = st.columns(3)

with challenge_cols[0]:
    with st.container():
        st.markdown("### 🌱 DÉFI DU JOUR")
        st.markdown("**Consommer moins de 4kWh**")
        st.progress(0.75)
        st.caption("Progression: 75% • Récompense: 50 points")

with challenge_cols[1]:
    with st.container():
        st.markdown("### ⚡ OBJECTIF MENSUEL")
        st.markdown("**-15% de consommation**")
        st.progress(0.60)
        st.caption("Progression: 60% • Économie cible: €45")

with challenge_cols[2]:
    with st.container():
        st.markdown("### 👥 CLASSEMENT")
        st.markdown("**#3 sur 25 maisons**")
        st.caption("Prochain niveau: 150 points")

# Section 6: ACTIONS RAPIDES
st.markdown('<div class="section-title">⚡ ACTIONS IMMÉDIATES</div>', unsafe_allow_html=True)

quick_cols = st.columns(4)

with quick_cols[0]:
    if st.button("🌙 MODE NUIT", use_container_width=True):
        st.session_state.appliances['lights_living']['on'] = False
        st.session_state.appliances['lights_kitchen']['on'] = False
        st.session_state.appliances['heating']['temp'] = 18
        st.success("Mode nuit activé !")
        st.rerun()

with quick_cols[1]:
    if st.button("🏠 MODE ABSENCE", use_container_width=True):
        for appliance in ['lights_living', 'lights_kitchen', 'lights_bedroom', 
                         'lights_bathroom', 'tv', 'computer']:
            st.session_state.appliances[appliance]['on'] = False
        st.session_state.appliances['heating']['temp'] = 16
        st.success("Mode absence activé !")
        st.rerun()

with quick_cols[2]:
    if st.button("💰 MODE ÉCONOMIE", use_container_width=True):
        st.session_state.appliances['heating']['temp'] = 19
        st.session_state.appliances['standby_devices']['on'] = False
        st.success("Mode économie activé !")
        st.rerun()

with quick_cols[3]:
    if st.button("🔄 RÉINITIALISER", use_container_width=True):
        for key in st.session_state.appliances:
            st.session_state.appliances[key]['on'] = True
        st.session_state.appliances['heating']['temp'] = 20
        st.session_state.appliances['washing_machine']['on'] = False
        st.session_state.appliances['oven']['on'] = False
        st.session_state.appliances['computer']['on'] = False
        st.success("Tout réinitialisé !")
        st.rerun()

# Section 7: RÉSUMÉ FINANCIER
st.markdown('<div class="section-title">💰 RÉSUMÉ FINANCIER</div>', unsafe_allow_html=True)

finance_cols = st.columns(3)

with finance_cols[0]:
    with st.container():
        st.metric("📅 COÛT JOURNALIER", 
                 f"€{(current_power * 24.0 * 0.18 / 1000.0):.2f}",
                 f"{current_power:,.0f} W × 24h × €0.18/kWh")

with finance_cols[1]:
    with st.container():
        st.metric("📊 ÉCONOMIES POTENTIELLES",
                 f"€{monthly_savings_eur:.0f}/mois",
                 f"Jusqu'à €{annual_savings_eur:.0f} par an")

with finance_cols[2]:
    with st.container():
        st.metric("📈 TENDANCE MENSUELLE",
                 f"{avg_daily_kwh * 30.0:.0f} kWh",
                 "Variation: -2% vs mois dernier")

# Pied de page
st.markdown("---")
current_time = datetime.now().strftime("%d/%m/%Y %H:%M")

col_footer1, col_footer2, col_footer3 = st.columns(3)

with col_footer1:
    st.markdown("**🕒 Dernière mise à jour**")
    st.markdown(current_time)

with col_footer2:
    st.markdown("**⚡ Puissance actuelle**")
    st.markdown(f"{current_power:,.0f} W")

with col_footer3:
    st.markdown("**💰 Coût estimé aujourd'hui**")
    st.markdown(f"€{(current_power * 24.0 * 0.18 / 1000.0):.2f}")

st.markdown("---")
st.markdown("**SMART HOME ENERGY HUB** • Votre compagnon énergétique intelligent")
st.caption("Application mise à jour automatiquement toutes les 30 secondes")

# Notification
if current_power > 3000.0 and notifications:
    st.toast("⚠️ CONSOMMATION ÉLEVÉE DÉTECTÉE", icon="⚡")
    st.toast("💡 Essayez de désactiver quelques appareils", icon="💡")