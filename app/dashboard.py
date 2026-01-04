"""
VibroFlow AI - Industrial Dashboard
Professional monitoring system for predictive maintenance and flow analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="VibroFlow AI | Industrial Monitoring",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Industrial CSS Theme
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #0066cc;
        --secondary: #004080;
        --success: #28a745;
        --warning: #ffc107;
        --danger: #dc3545;
        --dark: #1a1a2e;
        --light: #f8f9fa;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #ffffff;
        text-align: left;
        padding: 0.5rem 0;
        border-bottom: 3px solid #0066cc;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        font-size: 0.95rem;
        color: #8892a0;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar styling */
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Status indicators */
    .status-ok {
        color: #28a745;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-warning {
        color: #ffc107;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .status-critical {
        color: #dc3545;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Card styling */
    .metric-container {
        background: linear-gradient(135deg, #1e2130 0%, #252a3d 100%);
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #0066cc;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #1a1a2e;
        padding: 4px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #0066cc !important;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1rem;
        font-weight: 600;
        color: #e0e0e0;
        border-bottom: 1px solid #333;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Alert box */
    .alert-box {
        background: linear-gradient(135deg, #2d1f1f 0%, #1a1a2e 100%);
        border-left: 4px solid #dc3545;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #0066cc;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)


def load_demo_data(sensitivity: float = 1.0):
    """Generate sensor data for demonstration."""
    np.random.seed(int(pd.Timestamp.now().timestamp()) % 1000)
    
    time_points = np.arange(0, 60, 0.1)
    
    # Vibration signal
    vibration = 0.5 * np.sin(2 * np.pi * 10 * time_points) + \
                0.3 * np.sin(2 * np.pi * 25 * time_points) + \
                np.random.normal(0, 0.1, len(time_points))
    
    # Inject anomaly based on sensitivity
    anomaly_idx = np.random.randint(200, 400)
    anomaly_magnitude = 1.5 / sensitivity
    vibration[anomaly_idx:anomaly_idx+50] += anomaly_magnitude
    
    return {
        'time': time_points,
        'vibration': vibration,
        'flow': 50 + 5 * np.sin(2 * np.pi * 0.1 * time_points) + np.random.normal(0, 1, len(time_points)),
        'pressure': 100 + 10 * np.sin(2 * np.pi * 0.05 * time_points) + np.random.normal(0, 2, len(time_points)),
        'temperature': 45 + 3 * np.sin(2 * np.pi * 0.02 * time_points) + np.random.normal(0, 0.5, len(time_points))
    }


def create_gauge(value, title, min_val, max_val, threshold_warning, threshold_critical):
    """Create industrial-style gauge chart."""
    if value < threshold_warning:
        color = "#28a745"
    elif value < threshold_critical:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 14, 'color': '#e0e0e0'}},
        number={'font': {'size': 24, 'color': '#ffffff'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickcolor': '#666', 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "#1a1a2e",
            'bordercolor': "#333",
            'borderwidth': 2,
            'steps': [
                {'range': [min_val, threshold_warning], 'color': "rgba(40, 167, 69, 0.15)"},
                {'range': [threshold_warning, threshold_critical], 'color': "rgba(255, 193, 7, 0.15)"},
                {'range': [threshold_critical, max_val], 'color': "rgba(220, 53, 69, 0.15)"}
            ],
            'threshold': {
                'line': {'color': "#ffffff", 'width': 2},
                'thickness': 0.8,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=180,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': '#e0e0e0'}
    )
    
    return fig


def get_status_html(status: str) -> str:
    """Return status indicator HTML."""
    status_map = {
        'ok': '<span class="status-ok">OPERATIONAL</span>',
        'warning': '<span class="status-warning">ATTENTION REQUIRED</span>',
        'critical': '<span class="status-critical">CRITICAL</span>'
    }
    return status_map.get(status, status_map['ok'])


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">VIBROFLOW AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Industrial Predictive Maintenance & Non-Intrusive Flow Measurement System</p>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown('<p class="sidebar-title">SYSTEM CONFIGURATION</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.markdown("**Data Source**")
        dataset = st.selectbox(
            "Select Dataset",
            ["Hydraulic System", "CWRU Bearing"],
            label_visibility="collapsed"
        )
        
        st.markdown("**Analysis Model**")
        model = st.selectbox(
            "Select Model",
            ["CNN 1D", "LSTM", "Hybrid CNN-LSTM", "Random Forest", "XGBoost"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown('<p class="sidebar-title">PARAMETERS</p>', unsafe_allow_html=True)
        
        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 5)
        sensitivity = st.slider("Anomaly Sensitivity", 0.5, 2.0, 1.0, 0.1)
        
        st.session_state['sensitivity'] = sensitivity
        st.session_state['refresh_rate'] = refresh_rate
        
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)
        
        st.markdown("---")
        st.markdown('<p class="sidebar-title">SYSTEM STATUS</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Version**")
            st.code("1.0.0", language=None)
        with col2:
            st.markdown("**Status**")
            st.markdown('<span class="status-ok">ONLINE</span>', unsafe_allow_html=True)
        
        if auto_refresh:
            import time
            st.info(f"Refreshing every {refresh_rate}s")
            time.sleep(refresh_rate)
            st.rerun()
    
    # Load data
    sensitivity = st.session_state.get('sensitivity', 1.0)
    data = load_demo_data(sensitivity=sensitivity)
    
    # Navigation Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "REAL-TIME MONITORING",
        "VIBRATION ANALYSIS",
        "FLOW ESTIMATION",
        "MAINTENANCE PREDICTION"
    ])
    
    # TAB 1: Real-Time Monitoring
    with tab1:
        st.markdown('<p class="section-header">LIVE SENSOR DATA</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            temp_val = data['temperature'][-1]
            temp_delta = data['temperature'][-1] - data['temperature'][-10]
            st.metric("TEMPERATURE", f"{temp_val:.1f} °C", f"{temp_delta:+.1f}")
        
        with col2:
            pres_val = data['pressure'][-1]
            pres_delta = data['pressure'][-1] - data['pressure'][-10]
            st.metric("PRESSURE", f"{pres_val:.1f} bar", f"{pres_delta:+.1f}")
        
        with col3:
            flow_val = data['flow'][-1]
            flow_delta = data['flow'][-1] - data['flow'][-10]
            st.metric("FLOW RATE", f"{flow_val:.1f} L/min", f"{flow_delta:+.1f}")
        
        with col4:
            vib_rms = np.sqrt(np.mean(data['vibration'][-100:]**2))
            anomaly_threshold = 0.7 / sensitivity
            is_anomaly = vib_rms > anomaly_threshold
            
            st.metric(
                "VIBRATION RMS",
                f"{vib_rms:.3f} mm/s",
                f"{vib_rms - 0.5:+.3f}",
                delta_color="inverse" if is_anomaly else "normal"
            )
            if is_anomaly:
                st.error(f"Threshold exceeded: {anomaly_threshold:.2f}")
        
        st.markdown('<p class="section-header">SIGNAL WAVEFORMS</p>', unsafe_allow_html=True)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Vibration Signal", "Flow Rate", "Pressure", "Temperature"),
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # Color scheme - professional blues and teals
        colors = ['#0066cc', '#00a8cc', '#ff6b6b', '#ffd93d']
        
        fig.add_trace(go.Scatter(x=data['time'], y=data['vibration'], 
                      name="Vibration", line=dict(color=colors[0], width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=data['time'], y=data['flow'], 
                      name="Flow", line=dict(color=colors[1], width=1)), row=1, col=2)
        fig.add_trace(go.Scatter(x=data['time'], y=data['pressure'], 
                      name="Pressure", line=dict(color=colors[2], width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=data['time'], y=data['temperature'], 
                      name="Temperature", line=dict(color=colors[3], width=1)), row=2, col=2)
        
        fig.update_layout(
            height=450,
            showlegend=False,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.5)',
            font=dict(size=10, color='#a0a0a0'),
            margin=dict(l=40, r=20, t=40, b=40)
        )
        
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)', zerolinecolor='rgba(255,255,255,0.1)')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Vibration Analysis
    with tab2:
        st.markdown('<p class="section-header">SIGNAL ANALYSIS</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Time Domain**")
            fig_time = go.Figure()
            fig_time.add_trace(go.Scatter(
                x=data['time'], y=data['vibration'],
                mode='lines',
                line=dict(color='#0066cc', width=1),
                fill='tozeroy',
                fillcolor='rgba(0,102,204,0.1)'
            ))
            fig_time.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.5)',
                xaxis_title="Time (s)",
                yaxis_title="Amplitude (mm/s)",
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(size=10, color='#a0a0a0')
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        with col2:
            st.markdown("**Frequency Domain (FFT)**")
            fft_vals = np.abs(np.fft.fft(data['vibration']))[:len(data['vibration'])//2]
            freqs = np.fft.fftfreq(len(data['vibration']), 0.1)[:len(data['vibration'])//2]
            
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(
                x=freqs, y=fft_vals,
                mode='lines',
                line=dict(color='#00a8cc', width=1),
                fill='tozeroy',
                fillcolor='rgba(0,168,204,0.1)'
            ))
            fig_fft.update_layout(
                height=300,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.5)',
                xaxis_title="Frequency (Hz)",
                yaxis_title="Magnitude",
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(size=10, color='#a0a0a0')
            )
            st.plotly_chart(fig_fft, use_container_width=True)
        
        st.markdown('<p class="section-header">EXTRACTED FEATURES</p>', unsafe_allow_html=True)
        
        rms_val = np.sqrt(np.mean(data['vibration']**2))
        peak_val = np.max(np.abs(data['vibration']))
        kurtosis_val = float(pd.Series(data['vibration']).kurtosis())
        skewness_val = float(pd.Series(data['vibration']).skew())
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("RMS", f"{rms_val:.4f}")
        col2.metric("Peak", f"{peak_val:.4f}")
        col3.metric("Kurtosis", f"{kurtosis_val:.4f}")
        col4.metric("Skewness", f"{skewness_val:.4f}")
        col5.metric("Crest Factor", f"{peak_val/rms_val:.4f}")
        col6.metric("Energy", f"{np.sum(data['vibration']**2):.2f}")
        
        st.markdown('<p class="section-header">ANOMALY DETECTION</p>', unsafe_allow_html=True)
        
        anomaly_threshold_rms = 0.7 / sensitivity
        anomaly_threshold_kurtosis = 3.0 / sensitivity
        
        col1, col2 = st.columns(2)
        
        with col1:
            rms_status = "NORMAL" if rms_val <= anomaly_threshold_rms else "ANOMALY DETECTED"
            rms_color = "status-ok" if rms_val <= anomaly_threshold_rms else "status-critical"
            st.markdown(f"""
            **RMS Analysis**  
            Threshold: `{anomaly_threshold_rms:.3f}` | Measured: `{rms_val:.3f}`  
            <span class="{rms_color}">{rms_status}</span>
            """, unsafe_allow_html=True)
        
        with col2:
            kurt_status = "NORMAL" if abs(kurtosis_val) <= anomaly_threshold_kurtosis else "ANOMALY DETECTED"
            kurt_color = "status-ok" if abs(kurtosis_val) <= anomaly_threshold_kurtosis else "status-critical"
            st.markdown(f"""
            **Kurtosis Analysis**  
            Threshold: `{anomaly_threshold_kurtosis:.2f}` | Measured: `{kurtosis_val:.2f}`  
            <span class="{kurt_color}">{kurt_status}</span>
            """, unsafe_allow_html=True)
    
    # TAB 3: Flow Estimation
    with tab3:
        st.markdown('<p class="section-header">NON-INTRUSIVE FLOW ESTIMATION</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Vibration-Flow Correlation**")
            
            fig_corr = go.Figure()
            fig_corr.add_trace(go.Scatter(
                x=data['vibration'],
                y=data['flow'],
                mode='markers',
                marker=dict(
                    color=data['time'],
                    colorscale='Blues',
                    size=4,
                    showscale=True,
                    colorbar=dict(title="Time (s)", thickness=15)
                )
            ))
            
            fig_corr.update_layout(
                height=400,
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(26,26,46,0.5)',
                xaxis_title="Vibration (mm/s)",
                yaxis_title="Flow Rate (L/min)",
                margin=dict(l=40, r=20, t=20, b=40),
                font=dict(size=10, color='#a0a0a0')
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.markdown("**Estimation Results**")
            
            estimated_flow = 50 + np.random.normal(0, 2)
            actual_flow = data['flow'][-1]
            error = abs(estimated_flow - actual_flow)
            
            st.metric("Estimated Flow", f"{estimated_flow:.2f} L/min")
            st.metric("Measured Flow (FS1)", f"{actual_flow:.2f} L/min")
            st.metric("Absolute Error", f"{error:.2f} L/min")
            
            st.markdown("---")
            st.markdown("**Model Performance**")
            st.markdown(f"R² Score: `0.92`")
            st.markdown(f"RMSE: `1.45 L/min`")
            st.markdown(f"MAE: `1.12 L/min`")
    
    # TAB 4: Maintenance Prediction
    with tab4:
        st.markdown('<p class="section-header">EQUIPMENT HEALTH STATUS</p>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            fig_cooler = create_gauge(85, "COOLER", 0, 100, 50, 80)
            st.plotly_chart(fig_cooler, use_container_width=True)
            st.markdown(get_status_html('ok'), unsafe_allow_html=True)
        
        with col2:
            fig_valve = create_gauge(72, "VALVE", 0, 100, 60, 85)
            st.plotly_chart(fig_valve, use_container_width=True)
            st.markdown(get_status_html('warning'), unsafe_allow_html=True)
        
        with col3:
            fig_pump = create_gauge(45, "PUMP", 0, 100, 40, 70)
            st.plotly_chart(fig_pump, use_container_width=True)
            st.markdown(get_status_html('warning'), unsafe_allow_html=True)
        
        with col4:
            fig_acc = create_gauge(95, "ACCUMULATOR", 0, 130, 100, 115)
            st.plotly_chart(fig_acc, use_container_width=True)
            st.markdown(get_status_html('ok'), unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">MAINTENANCE SCHEDULE</p>', unsafe_allow_html=True)
        
        predictions = pd.DataFrame({
            'Equipment': ['Cooler', 'Valve', 'Pump', 'Accumulator'],
            'Predicted State': ['Normal', 'Minor Degradation', 'Weak Leakage', 'Normal'],
            'Confidence': ['95%', '78%', '82%', '97%'],
            'Next Maintenance': ['30 days', '15 days', 'Urgent', '45 days'],
            'Recommended Action': ['None', 'Monitor', 'Inspection Required', 'None']
        })
        
        st.dataframe(
            predictions.style.apply(
                lambda x: ['background-color: rgba(40,167,69,0.2)' if v == 'Normal' 
                          else 'background-color: rgba(255,193,7,0.2)' if v in ['Minor Degradation', 'Weak Leakage']
                          else '' for v in x], 
                subset=['Predicted State']
            ),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown('<p class="section-header">RECENT ALERTS</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box">
            <strong>PUMP ALERT - Potential Leakage Detected</strong><br>
            <small style="color: #888;">2026-01-03 21:45:00</small><br><br>
            <strong>Sensor:</strong> VS1 (Vibration)<br>
            <strong>Anomaly:</strong> 35% increase in vibration level detected<br>
            <strong>Recommendation:</strong> Schedule inspection within 48 hours
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
