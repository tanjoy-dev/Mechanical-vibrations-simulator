import streamlit as st
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import BytesIO

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="VibraSim Pro | Mechanical Vibrations Simulator",
    page_icon="🏗️",
    layout="wide",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main {
        background: #0e1117;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .sidebar .sidebar-content {
        background: #1a1c24;
    }
    h1, h2, h3 {
        color: #00d4ff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #00d4ff;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("assets/iconvibsem.png", width=150)
    except:
        st.write("🏗️")
with col2:
    st.title("VibraSim Pro")
    st.subheader("Advanced Mechanical Vibrations & Dynamics Simulator")

st.markdown("---")

# --- SIDEBAR PARAMETERS ---
st.sidebar.header("🕹️ Simulation Control")

topic = st.sidebar.selectbox(
    "Select Vibration Type",
    ["Simple Harmonic Motion", "Free Damped Vibrations", "Forced Damped Vibrations"]
)

with st.sidebar.expander("📝 Physical Parameters", expanded=True):
    mass = st.slider("Mass (m) [kg]", 0.1, 50.0, 5.0, 0.1)
    stiffness = st.slider("Spring Constant (k) [N/m]", 1.0, 1000.0, 500.0, 10.0)
    
    # Calculate derived natural frequency
    wn = np.sqrt(stiffness / mass)
    fn = wn / (2 * np.pi)
    
    st.info(f"Natural Frequency (fₙ): {fn:.2f} Hz")

if topic != "Simple Harmonic Motion":
    zeta = st.sidebar.slider("Damping Ratio (ζ)", 0.0, 1.5, 0.1, 0.01)
else:
    zeta = 0.0

if topic == "Forced Damped Vibrations":
    f0 = st.sidebar.slider("Driving Force (F₀) [N]", 0.0, 100.0, 10.0, 1.0)
    fd = st.sidebar.slider("Driving Frequency (f_d) [Hz]", 0.1, 10.0, 2.0, 0.1)
    wd = 2 * np.pi * fd
    amplitude = 0.0 # Not used for steady-state forced
    v0 = 0.0
else:
    amplitude = st.sidebar.slider("Initial Displacement (x₀) [m]", -5.0, 5.0, 1.0, 0.1)
    v0 = st.sidebar.slider("Initial Velocity (v₀) [m/s]", -10.0, 10.0, 0.0, 0.1)

# --- PHYSICS CALCULATIONS ---
t = np.linspace(0, 10, 1000)

def calculate_motion():
    if topic == "Simple Harmonic Motion":
        # x(t) = x0 cos(wn*t) + (v0/wn) sin(wn*t)
        x = amplitude * np.cos(wn * t) + (v0 / wn) * np.sin(wn * t)
        v = -amplitude * wn * np.sin(wn * t) + v0 * np.cos(wn * t)
        a = - (wn**2) * x
        return x, v, a
    
    elif topic == "Free Damped Vibrations":
        # General Case for x(0)=x0, v(0)=v0
        if zeta < 1: # Underdamped
            wd_damped = wn * np.sqrt(1 - zeta**2)
            c1 = amplitude
            c2 = (v0 + zeta * wn * amplitude) / wd_damped
            envelope = np.exp(-zeta * wn * t)
            x = envelope * (c1 * np.cos(wd_damped * t) + c2 * np.sin(wd_damped * t))
            v = -zeta * wn * x + envelope * (-c1 * wd_damped * np.sin(wd_damped * t) + c2 * wd_damped * np.cos(wd_damped * t))
            a = -(wn**2) * x - 2 * zeta * wn * v
        elif zeta == 1: # Critically damped
            c1 = amplitude
            c2 = v0 + wn * amplitude
            x = (c1 + c2 * t) * np.exp(-wn * t)
            v = (c2 - wn * (c1 + c2 * t)) * np.exp(-wn * t)
            a = (-2 * wn * c2 + (wn**2) * (c1 + c2 * t)) * np.exp(-wn * t)
        else: # Overdamped
            gamma = wn * np.sqrt(zeta**2 - 1)
            # x = e^(-zeta*wn*t) * (c1*e^(gamma*t) + c2*e^(-gamma*t))
            c1 = (v0 + (zeta * wn + gamma) * amplitude) / (2 * gamma)
            c2 = amplitude - c1
            x = np.exp(-zeta * wn * t) * (c1 * np.exp(gamma * t) + c2 * np.exp(-gamma * t))
            v = -(zeta * wn - gamma) * c1 * np.exp(-(zeta * wn - gamma) * t) - (zeta * wn + gamma) * c2 * np.exp(-(zeta * wn + gamma) * t)
            a = -(2 * zeta * wn) * v - (wn**2) * x
        return x, v, a

    elif topic == "Forced Damped Vibrations":
        # Steady state solution: x(t) = X cos(wd*t - phi)
        # Ratio r = wd / wn
        r = wd / wn
        # Resonance protection: ensure denominator is not zero
        denom = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
        X = (f0 / stiffness) / np.maximum(denom, 1e-4) # Cap at 10000x magnification
        phi = np.arctan2(2 * zeta * r, 1 - r**2)
        
        x = X * np.cos(wd * t - phi)
        v = -X * wd * np.sin(wd * t - phi)
        a = -X * (wd**2) * np.cos(wd * t - phi)
        return x, v, a

x, v, a = calculate_motion()

# --- MAIN DASHBOARD ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Motion Analysis", "🔄 Phase Space", "📈 Frequency Response", "📚 Theory"])

with tab1:
    st.subheader("Time Domain Analysis")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=x, name="Displacement (m)", line=dict(color='#00d4ff', width=3)))
    fig.add_trace(go.Scatter(x=t, y=v, name="Velocity (m/s)", line=dict(color='#ff4b4b', width=2, dash='dot')))
    fig.add_trace(go.Scatter(x=t, y=a, name="Acceleration (m/s²)", line=dict(color='#00ff00', width=1, dash='dash')))
    
    fig.update_layout(
        template="plotly_dark",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- ANIMATION SECTION ---
    with st.expander("🎥 Live System Animation", expanded=True):
        # Create zigzag spring function
        def get_spring_path(y_mass, y_support=2.5, n_turns=12, width=0.2):
            y_points = np.linspace(y_support, y_mass, n_turns * 2 + 1)
            x_points = np.zeros_like(y_points)
            x_points[1:-1:4] = width
            x_points[3:-1:4] = -width
            return x_points, y_points

        # Create animation frames - High density to prevent aliasing
        # We use a 2.0s window instead of 5s to show high-freq details better
        anim_t = np.linspace(0, 2.0, 120)
        
        # Re-calculate x for animation subset using current formulas
        if topic == "Simple Harmonic Motion":
            anim_x = amplitude * np.cos(wn * anim_t) + (v0 / wn) * np.sin(wn * anim_t)
        elif topic == "Free Damped Vibrations":
            if zeta < 1:
                wd_damped = wn * np.sqrt(1 - zeta**2)
                c1, c2 = amplitude, (v0 + zeta * wn * amplitude) / wd_damped
                anim_x = np.exp(-zeta * wn * anim_t) * (c1 * np.cos(wd_damped * anim_t) + c2 * np.sin(wd_damped * anim_t))
            elif zeta == 1:
                c1, c2 = amplitude, v0 + wn * amplitude
                anim_x = (c1 + c2 * anim_t) * np.exp(-wn * anim_t)
            else:
                gamma = wn * np.sqrt(zeta**2 - 1)
                c1 = (v0 + (zeta * wn + gamma) * amplitude) / (2 * gamma)
                c2 = amplitude - c1
                anim_x = np.exp(-zeta * wn * anim_t) * (c1 * np.exp(gamma * anim_t) + c2 * np.exp(-gamma * anim_t))
        else:
            r = wd / wn
            denom = np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
            # Physical cap to prevent astronomical values that freeze the plot
            X = (f0 / stiffness) / np.maximum(denom, 1e-4)
            if X > 10.0:
                st.warning("⚠️ RESONANCE: Displacement capped at 10m for simulation stability.")
                X = 10.0
            phi = np.arctan2(2 * zeta * r, 1 - r**2)
            anim_x = X * np.cos(wd * anim_t - phi)

        # Build Frames
        frames = []
        for val in anim_x:
            sx, sy = get_spring_path(val)
            frames.append(go.Frame(data=[go.Scatter(x=[0], y=[val]), go.Scatter(x=sx, y=sy)]))

        # Initial Spring Path
        init_sx, init_sy = get_spring_path(anim_x[0])

        # Dynamic Y-axis Scaling with safety bounds
        max_amp = np.max(np.abs(anim_x)) if len(anim_x) > 0 else 1.0
        # Use a minimum range but allow up to 15m
        y_limit = max(3.0, min(15.0, max_amp * 1.5))

        fig_anim = go.Figure(
            data=[
                go.Scatter(x=[0], y=[anim_x[0]], mode="markers", marker=dict(size=40, color="#00d4ff", symbol="square"), name="Mass"),
                go.Scatter(x=init_sx, y=init_sy, mode="lines", line=dict(color="white", width=2), name="Spring")
            ],
            layout=go.Layout(
                xaxis=dict(range=[-1, 1], autorange=False, visible=False),
                yaxis=dict(range=[-y_limit, y_limit], autorange=False, visible=False),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(label="▶️ Play", method="animate", args=[None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}]),
                        dict(label="⏸️ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
                    ]
                )],
                template="plotly_dark",
                height=400,
                margin=dict(l=0, r=0, t=20, b=0)
            ),
            frames=frames
        )
        st.plotly_chart(fig_anim, use_container_width=True)

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Max Displacement", f"{np.max(np.abs(x)):.3f} m")
    m2.metric("Max Velocity", f"{np.max(np.abs(v)):.3f} m/s")
    m3.metric("Max Acceleration", f"{np.max(np.abs(a)):.3f} m/s²")
    m4.metric("Damping Ratio", f"{zeta:.2f}")

with tab2:
    st.subheader("Phase Space Trajectory")
    col_a, col_b = st.columns([2, 1])
    with col_a:
        fig_3d = go.Figure(data=[go.Scatter3d(x=t, y=x, z=v, mode='lines', line=dict(color=t, colorscale='Viridis', width=5))])
        fig_3d.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title='Time (s)',
                yaxis_title='Displacement (m)',
                zaxis_title='Velocity (m/s)'
            ),
            margin=dict(l=0, r=0, b=0, t=0)
        )
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_b:
        st.info("**Insight:** The phase space plot shows the relationship between state variables. A closed loop indicates sustained oscillation, while a spiral inward indicates damping.")
        
        # 2D Phase Portrait
        fig_2d_phase = go.Figure()
        fig_2d_phase.add_trace(go.Scatter(x=x, y=v, mode='lines', line=dict(color='#00d4ff')))
        fig_2d_phase.update_layout(template="plotly_dark", title="2D Phase Portrait (x vs v)", xaxis_title="Displacement (m)", yaxis_title="Velocity (m/s)")
        st.plotly_chart(fig_2d_phase, use_container_width=True)

with tab3:
    st.subheader("Frequency Response curves")
    if topic == "Forced Damped Vibrations":
        r_range = np.linspace(0, 3, 500)
        # Magnification Factor
        M = 1 / np.sqrt((1 - r_range**2)**2 + (2 * zeta * r_range)**2)
        # Transmissibility
        TR = np.sqrt(1 + (2 * zeta * r_range)**2) / np.sqrt((1 - r_range**2)**2 + (2 * zeta * r_range)**2)
        
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(x=r_range, y=M, name="Magnification Factor (M)", line=dict(color='#00d4ff')))
        fig_res.add_trace(go.Scatter(x=r_range, y=TR, name="Transmissibility (TR)", line=dict(color='#ffaa00')))
        
        # Mark current operating point
        current_r = wd / wn
        current_M = 1 / np.sqrt((1 - current_r**2)**2 + (2 * zeta * current_r)**2)
        fig_res.add_trace(go.Scatter(x=[current_r], y=[current_M], name="Current State", mode="markers", marker=dict(size=12, color="white")))

        fig_res.update_layout(template="plotly_dark", xaxis_title="Frequency Ratio (ω/ωₙ)", yaxis_title="Ratio")
        st.plotly_chart(fig_res, use_container_width=True)
        
        st.write(f"**Current Frequency Ratio (r):** {current_r:.2f}")
        if 0.9 < current_r < 1.1:
            st.warning("⚠️ SYSTEM NEAR RESONANCE")
    else:
        st.info("Frequency response analysis is available for Forced Vibrations.")

with tab4:
    st.markdown("""
    ### Fundamental Equations
    
    The motion of a single degree of freedom (SDOF) system is governed by:
    
    $$m\\ddot{x} + c\\dot{x} + kx = F(t)$$
    
    Where:
    - $m$ is the mass
    - $c$ is the damping coefficient ($c = 2\\zeta \\sqrt{mk}$)
    - $k$ is the stiffness
    
    #### Solutions:
    1. **SHM:** $x(t) = A \\cos(\\omega_n t)$
    2. **Free Damped:** $x(t) = A e^{-\\zeta \\omega_n t} \\cos(\\omega_d t)$
    3. **Forced:** $x(t) = X \\cos(\\omega_d t - \\phi)$
    """)

# --- DATA EXPORT ---
st.sidebar.markdown("---")
df = pd.DataFrame({"Time (s)": t, "Displacement (m)": x, "Velocity (m/s)": v, "Acceleration (m/s^2)": a})
csv = df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "📥 Download Data (CSV)",
    csv,
    "simulation_data.csv",
    "text/csv",
    key='download-csv'
)

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.info("Developed for Mechanical Engineering Simulations")
st.sidebar.markdown("""
Made by:
- **Ekamveer Singh** (12303013)
- **Tanveer Singh Sidhu** (12303026)
- **Gurvinder Singh** (12303006)
""")
