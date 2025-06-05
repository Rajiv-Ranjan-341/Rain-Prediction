import streamlit as st
import numpy as np
import pickle
import plotly.express as px
from PIL import Image

# --- Page Configuration ---
st.set_page_config(
    page_title="🌦️ WeatherSense Pro",
    page_icon="⛅",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background-image: url(https://www.pixelstalk.net/wp-content/uploads/images1/HD-Blur-Pictures-Download.jpg);
        background-size: cover;
        font-family: 'Arial', sans-serif;
    }
    .title-text {
        color: #2a52be;
        text-align: center;
        font-size: 2.8rem !important;
        margin-bottom: 0.5rem !important;
    }
    .subheader {
        color: #4a6fa5 !important;
        text-align: center !important;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #2a52be;
    }
    .rain-alert {
        background: linear-gradient(135deg, #b3e0ff 0%, #66b3ff 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,66,128,0.15);
    }
    .sunny-alert {
        background: linear-gradient(135deg, #fff9b3 0%, #ffdf66 100%);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(204,153,0,0.15);
    }
    .stButton>button {
        background: linear-gradient(135deg, #2a52be 0%, #4a6fa5 100%);
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        width: 100%;
        padding: 12px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(42,82,190,0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load Model ---
@st.cache_resource
def load_model():
    try:
        with open('rain_prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Model files not found. Please train the model first.")
        st.stop()

model, scaler = load_model()

# --- Header Section ---
st.markdown('<h1 class="title-text">🌦️ WeatherSense Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Advanced Rain Prediction with AI Insights</p>', unsafe_allow_html=True)

# --- Input Section ---
with st.expander("📝 Enter Weather Parameters", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("🌡️ Temperature (°C)", -20.0, 40.0, 15.0, 0.5,
                        help="Current average temperature")
        humidity = st.slider("💧 Humidity (%)", 0, 100, 65,
                           help="Relative humidity percentage")
    with col2:
        pressure = st.slider("🌬️ Pressure (hPa)", 950.0, 1050.0, 1013.0, 0.1,
                           help="Atmospheric pressure at sea level")
        wind = st.slider("🍃 Wind Speed (km/h)", 0.0, 50.0, 10.0, 0.5,
                        help="Average wind speed")
    sunshine = st.slider("☀️ Sunshine (hours)", 0.0, 24.0, 5.0, 0.1,
                        help="Daily sunshine duration")

# --- Prediction Logic ---
if st.button("🔮 Predict Weather Conditions", use_container_width=True, type="primary"):
    features = np.array([[temp, humidity, pressure, wind, sunshine]])
    features_scaled = scaler.transform(features)
    proba = model.predict_proba(features_scaled)[0][1] * 100
    prediction = model.predict(features_scaled)[0]

    # --- Prediction Result ---
    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="rain-alert">
            <div style="display: flex; align-items: center; gap: 15px;">
                <h2 style="color: #0066cc; margin: 0;">☔ Rain Expected</h2>
                <div style="font-size: 1.5rem; background: #0066cc; color: white; padding: 0.2rem 1rem; border-radius: 20px;">
                    {proba:.0f}% chance
                </div>
            </div>
            <p style="margin-top: 0.8rem;">Higher probability of precipitation today. Consider carrying an umbrella.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sunny-alert">
            <div style="display: flex; align-items: center; gap: 15px;">
                <h2 style="color: #cc8800; margin: 0;">🌤️ Clear Skies</h2>
                <div style="font-size: 1.5rem; background: #cc8800; color: white; padding: 0.2rem 1rem; border-radius: 20px;">
                    {proba:.0f}% chance
                </div>
            </div>
            <p style="margin-top: 0.8rem;">Low probability of rain. Enjoy outdoor activities!</p>
        </div>
        """, unsafe_allow_html=True)

    # --- Enhanced Weather Insights ---
    st.markdown("---")
    st.subheader("🌍 Detailed Weather Analysis")
    
    # 1. Weather Gauges
    st.markdown("### 📊 Current Conditions")
    def create_gauge(value, title, min_val, max_val, color, unit):
        fig = px.pie(
            values=[value - min_val, max_val - value],
            names=["", ""],
            hole=0.7,
            title=f"{title}<br><span style='font-size:0.8em'>{value}{unit}</span>"
        )
        fig.update_traces(
            marker_colors=[color, "lightgray"],
            textinfo="none",
            hoverinfo="none",
            rotation=90
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=60, b=30),
            annotations=[dict(
                text=f"{value}{unit}",
                x=0.5, y=0.5,
                font_size=24,
                showarrow=False,
                font_color=color
            )]
        )
        return fig

    cols = st.columns(4)
    with cols[0]:
        st.plotly_chart(
            create_gauge(temp, "Temperature", -20, 40, "#FF6B6B", "°C"),
            use_container_width=True
        )
    with cols[1]:
        st.plotly_chart(
            create_gauge(humidity, "Humidity", 0, 100, "#4ECDC4", "%"),
            use_container_width=True
        )
    with cols[2]:
        st.plotly_chart(
            create_gauge(wind, "Wind Speed", 0, 50, "#45B7D1", "km/h"),
            use_container_width=True
        )
    with cols[3]:
        st.plotly_chart(
            create_gauge(sunshine, "Sunshine", 0, 12, "#FFD166", "hrs"),
            use_container_width=True
        )

    # 2. Weather Impact Analysis
    st.markdown("### 🔍 How Each Factor Affects Rain")
    impact_data = {
        "Factor": ["Temperature", "Humidity", "Wind", "Pressure", "Sunshine"],
        "Effect": [
            "Colder temps increase rain chance",
            ">70% humidity favors precipitation",
            "Strong winds may bring storms",
            "Lower pressure often means rain",
            "Less sunshine = higher rain risk"
        ],
        "Current Value": [
            f"{temp}°C",
            f"{humidity}%",
            f"{wind} km/h",
            f"{pressure} hPa",
            f"{sunshine} hrs"
        ],
        "Ideal Range": [
            "10-25°C (less rain)",
            "30-60% (optimal)",
            "5-15 km/h (stable)",
            "1010-1020 hPa (fair)",
            "6+ hrs (sunny)"
        ]
    }
    st.dataframe(
        impact_data,
        column_config={
            "Factor": st.column_config.TextColumn("Weather Factor"),
            "Effect": st.column_config.TextColumn("Impact on Rain"),
            "Current Value": st.column_config.TextColumn("Your Input"),
            "Ideal Range": st.column_config.TextColumn("Optimal Range")
        },
        hide_index=True,
        use_container_width=True
    )

    # 3. Personalized Recommendations
    st.markdown("### 💡 Smart Weather Advisory")
    if prediction == 1:
        st.markdown("""
        <div style="background-color: #e6f2ff; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #0066cc; margin-top: 0;">🌧️ Rain Preparedness Guide</h4>
        </div>
        """, unsafe_allow_html=True)
        
        rain_tips = {
            "👕 Dressing Smart": [
                "• Waterproof jacket with sealed seams",
                "• Quick-dry pants or waterproof overtrousers",
                "• Water-resistant footwear with good traction"
            ],
            "🚗 Travel Advisory": [
                "• Add 20-30% extra travel time for wet roads",
                "• Check tire tread depth before driving",
                "• Use headlights even during daylight rain"
            ],
            "🏠 Home Preparation": [
                "• Clear storm drains near your property",
                "• Secure outdoor furniture and decorations",
                "• Check basement sump pump if available"
            ],
            "🌿 Outdoor Activities": [
                "• Postpone non-essential gardening",
                "• Reschedule picnics or outdoor events",
                "• Indoor exercise alternatives recommended"
            ],
            "⚠️ Special Considerations": [
                "• Watch for localized flooding in low areas",
                "• Have flashlights ready in case of power outages",
                "• Keep pets indoors during heavy downpours"
            ]
        }
        
        for category, items in rain_tips.items():
            with st.expander(f"{category}", expanded=True):
                for item in items:
                    st.markdown(f"<div style='margin-left: 20px;'>{item}</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 10px; background-color: #f0f8ff; border-radius: 8px;">
            <p style="font-size: 0.9em; color: #0066cc;"><b>Pro Tip:</b> Keep a compact umbrella in your bag and waterproof covers for electronics.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background-color: #fff8e6; border-radius: 10px; padding: 15px; margin-bottom: 20px;">
            <h4 style="color: #cc8800; margin-top: 0;">☀️ Sunny Day Optimization</h4>
        </div>
        """, unsafe_allow_html=True)
        
        sunny_tips = {
            "👗 Perfect Outfits": [
                "• Light, breathable fabrics like cotton or linen",
                "• Wide-brimmed hat for sun protection",
                "• UV-protection sunglasses (polarized recommended)"
            ],
            "🚴 Ideal Activities": [
                "• Early morning or late afternoon hikes",
                "• Open-air markets or street fairs",
                "• Outdoor sports with proper hydration"
            ],
            "🏡 Home Opportunities": [
                "• Wash windows for maximum sunlight",
                "• Line-dry laundry for freshness",
                "• Solar panel cleaning for efficiency"
            ],
            "🌻 Gardening Guide": [
                "• Plant sun-loving annuals",
                "• Water plants early to reduce evaporation",
                "• Apply mulch to retain soil moisture"
            ],
            "🌡️ Heat Management": [
                "• Use fans to circulate cool morning air",
                "• Close curtains on sun-facing windows",
                "• Stay hydrated with electrolyte balance"
            ]
        }
        
        for category, items in sunny_tips.items():
            with st.expander(f"{category}", expanded=True):
                for item in items:
                    st.markdown(f"<div style='margin-left: 20px;'>{item}</div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style="margin-top: 15px; padding: 10px; background-color: #fffaf0; border-radius: 8px;">
            <p style="font-size: 0.9em; color: #cc8800;"><b>Pro Tip:</b> Apply sunscreen 30 minutes before going outside and reapply every 2 hours.</p>
        </div>
        """, unsafe_allow_html=True)

    # Add UV Index recommendation if sunny
    if prediction == 0 and sunshine > 4:
        uv_index = min(10, round((sunshine/12)*8 + (temp/40)*2))
        uv_color = "#FF9900" if uv_index < 6 else "#FF3300"
        
        st.markdown(f"""
        <div style="margin-top: 20px; padding: 15px; background-color: #fff5e6; border-radius: 10px; border-left: 5px solid {uv_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {uv_color};">☀️ UV Index Estimate: {uv_index}/10</h4>
                <small>{'Moderate' if uv_index < 6 else 'High'} Risk</small>
            </div>
            <p style="margin-bottom: 5px;">{'Sun protection recommended' if uv_index < 6 else 'Extra protection required'}</p>
            <div style="height: 10px; background: linear-gradient(to right, #4CAF50 0%, #FFEB3B 50%, #FF9800 75%, #FF3300 100%); border-radius: 5px;">
                <div style="width: {uv_index*10}%; height: 100%; background-color: transparent; border-right: 2px solid black;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 4. Historical Comparison (Simulated Data)
    st.markdown("### 📅 Compared to Seasonal Averages")
    avg_data = {
        "Metric": ["Temperature", "Humidity", "Wind", "Sunshine"],
        "Today": [temp, humidity, wind, sunshine],
        "January Average": [2.1, 85, 12.3, 2.8],
        "July Average": [19.5, 72, 10.1, 7.6]
    }
    fig = px.bar(
        avg_data,
        x="Metric",
        y=["Today", "January Average", "July Average"],
        barmode="group",
        color_discrete_map={
            "Today": "#2A52BE",
            "January Average": "#7F7F7F",
            "July Average": "#D3D3D3"
        },
        labels={"value": "Value", "variable": "Period"},
        height=400
    )
    fig.update_layout(
        legend_title_text=None,
        yaxis_title="Value",
        xaxis_title=None
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<center><b>Made with ❤️ Rajiv Ranjan </b></center>",
    unsafe_allow_html=True
)
