import streamlit as st
import streamlit.components.v1 as components
import json
from agent import LeaseAgent
from data_prep import DataPrepper  # KEEP THIS - Needed to populate Pinecone with embeddings
import time
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Tower Lease Agent",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== CUSTOM PROFESSIONAL CSS ==========
css = """
<style>
    /* ── Import professional font ── */
    /* ============================================================
   Tower Lease Agent — v2 CSS
   Fully dual-mode (light + dark). Paste inside the
   st.markdown("<style> ... </style>", unsafe_allow_html=True)
   block at the top of app.py.
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Tokens ── */
:root {
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --shadow-sm: 0 1px 4px rgba(0,0,0,0.07);
    --shadow-md: 0 4px 14px rgba(0,0,0,0.10);
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ────────────────────────────────────────────
   SIDEBAR  —  deep navy, consistent both modes
   ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1B2A3B !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

[data-testid="stSidebar"] * {
    color: #A8C0D0 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li {
    color: #7A9BAE !important;
    font-size: 0.82rem !important;
    line-height: 1.7 !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #C8DFF0 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1px;
}

[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: #3E6A84 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.07) !important;
    margin: 0.5rem 0 !important;
}

/* Sidebar buttons — ghost style on dark navy */
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #8AAABE !important;
    border-radius: var(--radius-md) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: all 0.18s ease !important;
    text-align: left !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(45,106,159,0.18) !important;
    border-color: rgba(45,106,159,0.35) !important;
    color: #C8E0F0 !important;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #2D5F88 !important;
    border-color: #2D5F88 !important;
    color: #E0F0FF !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: #3870A0 !important;
    border-color: #3870A0 !important;
}

/* Sidebar success message */
[data-testid="stSidebar"] .stSuccess {
    background: rgba(52,168,110,0.1) !important;
    border: 1px solid rgba(52,168,110,0.22) !important;
    border-left: 3px solid #34A86E !important;
    color: #4CAF82 !important;
    border-radius: var(--radius-md) !important;
}

[data-testid="stSidebar"] .stError {
    background: rgba(180,60,60,0.1) !important;
    border: 1px solid rgba(180,60,60,0.22) !important;
    border-left: 3px solid #B43C3C !important;
    border-radius: var(--radius-md) !important;
}

/* ────────────────────────────────────────────
   MAIN AREA  —  adapts via prefers-color-scheme
   ──────────────────────────────────────────── */

/* Light mode main */
@media (prefers-color-scheme: light) {
    .stApp { background-color: #F4F6F9 !important; }
    
    .stTextArea textarea {
        background: #FFFFFF !important;
        border: 1.5px solid #D2DCE6 !important;
        color: #1C2B3A !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameContainer"] {
        background: #FFFFFF !important;
    }
    
    details { background: #FFFFFF !important; }
    details summary { background: #F0F4F8 !important; color: #1C2B3A !important; }
    
    .stCodeBlock { background: #F0F4F8 !important; }
    
    [data-testid="stMetricValue"] { color: #2D5F88 !important; }
    
    h1, h2, h3 { color: #1C2B3A !important; }
}

/* Dark mode main */
@media (prefers-color-scheme: dark) {
    .stApp { background-color: #141B24 !important; }
    
    .stTextArea textarea {
        background: #1A2434 !important;
        border: 1.5px solid #243040 !important;
        color: #C0D0DF !important;
    }
    
    .stDataFrame [data-testid="StyledDataFrameContainer"] {
        background: #1A2434 !important;
    }
    
    details { background: #1A2434 !important; }
    details summary { background: #1E2C3C !important; color: #B0C8DC !important; }
    
    .stCodeBlock { background: #1A2434 !important; }
    
    [data-testid="stMetricValue"] { color: #5A9DC4 !important; }
    
    h1, h2, h3 { color: #C0D0DF !important; }
}

/* ── Common main area styles (mode-agnostic) ── */

.main-header {
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -0.3px;
    margin-bottom: 0.2rem;
}

.sub-header {
    font-size: 0.92rem;
    margin-bottom: 1.5rem;
    border-left: 3px solid #2D6A9F;
    padding-left: 0.9rem;
    border-radius: 0;
}

/* ── Status badges ── */
.status-approved {
    background: rgba(52,140,90,0.09);
    border: 1px solid rgba(52,140,90,0.28);
    color: #267A50;
    padding: 0.55rem 1.1rem;
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.3px;
    display: inline-block;
}

.status-rejected {
    background: rgba(160,60,60,0.08);
    border: 1px solid rgba(160,60,60,0.25);
    color: #9A3535;
    padding: 0.55rem 1.1rem;
    border-radius: var(--radius-md);
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.3px;
    display: inline-block;
}

/* ── Info box ── */
.info-box {
    background: rgba(45,106,159,0.07);
    border-left: 3px solid #2D6A9F;
    padding: 0.9rem 1.1rem;
    border-radius: 0;
    margin: 0.75rem 0;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #2D6A9F !important;
    border: none !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    border-radius: var(--radius-md) !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.18s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button[kind="primary"]:hover {
    background: #3878B4 !important;
    transform: translateY(-1px);
    box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"]:disabled {
    opacity: 0.4;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Secondary buttons (main area only) ── */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    transform: translateY(-1px);
}

/* ── Textarea ── */
.stTextArea textarea {
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}

.stTextArea textarea:focus {
    border-color: #2D6A9F !important;
    box-shadow: 0 0 0 3px rgba(45,106,159,0.12) !important;
}

/* ── Expanders ── */
details {
    border: 1px solid rgba(100,130,160,0.18) !important;
    border-radius: var(--radius-md) !important;
    margin: 0.5rem 0 !important;
}

details summary {
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    border-radius: var(--radius-md) !important;
    cursor: pointer;
    border-bottom: 1px solid rgba(100,130,160,0.12);
}

details[open] summary {
    border-bottom-left-radius: 0 !important;
    border-bottom-right-radius: 0 !important;
}

/* ── Alerts ── */
.stAlert {
    border-radius: var(--radius-md) !important;
    border-left: 3px solid #2D6A9F !important;
}

.stSuccess { border-left-color: #34A86E !important; }
.stError   { border-left-color: #B43C3C !important; }
.stWarning { border-left-color: #A07830 !important; }

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid rgba(100,130,160,0.18) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ── Dividers ── */
hr {
    border-color: rgba(100,130,160,0.18) !important;
    margin: 1.5rem 0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { border-radius: 4px; }
::-webkit-scrollbar-thumb { background: rgba(100,130,160,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #2D6A9F; }
</style>
"""
components.html(css, height=0)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_request' not in st.session_state:
    st.session_state.current_request = ""

# Sidebar
with st.sidebar:
    st.markdown("## 📡 Tower Lease Agent")
    st.markdown("---")
    
    # Initialization section
    st.markdown("### 🔧 System Setup")
    
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            with st.spinner("Initializing system..."):
                try:
                    # Run data preparation (checks Pinecone, adds only missing embeddings)
                    with st.spinner("📊 Preparing data and generating embeddings..."):
                        prepper = DataPrepper()
                        prepper.run()
                    
                    # Initialize the agent
                    with st.spinner("🤖 Initializing AI agent..."):
                        st.session_state.agent = LeaseAgent()
                    
                    st.session_state.initialized = True
                    st.success("✅ System initialized successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Initialization failed: {str(e)}")
                    st.info("Please check your API keys and file paths")
    else:
        st.success("✅ System is ready")
        if st.button("🔄 Reinitialize System", use_container_width=True):
            st.session_state.initialized = False
            st.session_state.agent = None
            st.session_state.conversation_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Example requests
    st.markdown("### 📋 Example Requests")
    
    examples = [
        "Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters on Tower TWR-101.",
        "Operator Etisalat wants to install a 30kg microwave dish at 50 meters on Tower TWR-101.",
        "Operator Verizon requests to mount a 20kg antenna on Tower TWR-102 at 15 meters height.",
        "Operator Vodafone needs to place a 10kg 4G antenna at 35 meters on Tower TWR-104.",
        "Operator Du wants to install a 25kg 5G unit at 20 meters on Tower TWR-107."
    ]
    
    for i, example in enumerate(examples):
        if st.button(f"📝 Example {i+1}", key=f"ex_{i}", use_container_width=True):
            st.session_state.current_request = example
            st.rerun()
    
    st.markdown("---")
    
    # Info section
    with st.expander("ℹ️ About"):
        st.markdown("""
        **How it works:**
        1. Extracts request details using AI (Mistral)
        2. Checks tower availability & capacity
        3. Validates regional policies via semantic search (Pinecone)
        4. Returns structured judgment
        
        **Policies:**
        - DXB-North: Max height 45m
        - SHJ-South: Max weight 15kg
        - SHJ-Coastal: Max weight 25kg
        """)

# Main area
col_title, col_icon = st.columns([6, 1])
with col_title:
    st.markdown('<div class="main-header">🤖 Autonomous Tower Lease Agent</div>', unsafe_allow_html=True)
with col_icon:
    st.markdown("")

st.markdown('<div class="sub-header">AI-powered agent for vetting telecom infrastructure lease requests</div>', unsafe_allow_html=True)

# Input section
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📝 Lease Request")
    user_input = st.text_area(
        "Enter your lease request:",
        value=st.session_state.current_request,
        height=150,
        placeholder="Example: Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters on Tower TWR-101.",
        disabled=not st.session_state.initialized,
        label_visibility="collapsed"
    )

with col2:
    st.markdown("### 🎯 Actions")
    if st.button("🚀 Process Request", type="primary", use_container_width=True, disabled=not st.session_state.initialized or not user_input):
        with st.spinner("🤔 Analyzing lease request..."):
            # Process the request
            result = st.session_state.agent.process_request(user_input)
            
            # Add to history
            st.session_state.conversation_history.append({
                "timestamp": datetime.now(),
                "request": user_input,
                "result": result
            })
            
            st.success("✅ Request processed!")
            time.sleep(0.5)
            st.rerun()
    
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

# Results section
if st.session_state.conversation_history:
    st.markdown("---")
    st.markdown("### 📊 Processing Results")
    
    # Latest result first
    for idx, item in enumerate(reversed(st.session_state.conversation_history)):
        result = item['result']
        
        # Status card
        if result['status'] == 'APPROVED':
            st.markdown(f"""
            <div class="status-approved">
            ✅ <strong>STATUS: {result['status']}</strong>
            </div>
            """, unsafe_allow_html=True)
        elif result['status'] == 'REJECTED':
            st.markdown(f"""
            <div class="status-rejected">
            ❌ <strong>STATUS: {result['status']}</strong>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="status-rejected">
            ⚠️ <strong>STATUS: {result['status']}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Summary
        st.markdown(f"**📌 Summary:** {result.get('summary', 'No summary available')}")
        
        # Reasons
        if result.get('reasons'):
            with st.expander("📋 View Details", expanded=True):
                st.markdown("**🔍 Decision Reasons:**")
                for reason in result['reasons']:
                    st.write(f"• {reason}")
                
                # Technical details
                if result.get('details'):
                    st.markdown("**📊 Technical Assessment:**")
                    
                    details = result['details']
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        st.markdown("**Request Details:**")
                        req = details.get('request', {})
                        st.write(f"• Operator: {req.get('operator', 'N/A')}")
                        st.write(f"• Tower: {req.get('tower_id', 'N/A')}")
                        st.write(f"• Equipment: {req.get('equipment', 'N/A')}")
                        st.write(f"• Weight: {req.get('weight_kg', 0)} kg")
                        st.write(f"• Height: {req.get('height_meters', 0)} m")
                    
                    with col_b:
                        st.markdown("**Tower Assessment:**")
                        tower = details.get('tower_assessment', {})
                        if tower:
                            st.write(f"• Region: {tower.get('region', 'N/A')}")
                            st.write(f"• Current Load: {tower.get('current_load_kg', 0)} kg")
                            st.write(f"• Max Capacity: {tower.get('max_capacity_kg', 0)} kg")
                            st.write(f"• New Total: {tower.get('new_total_kg', 0)} kg")
                            st.write(f"• Within Capacity: {'✅' if tower.get('within_capacity') else '❌'}")
                        else:
                            st.write("• No tower assessment available")
                    
                    # Policy assessment
                    policy = details.get('policy_assessment', {})
                    if policy:
                        st.markdown("**📜 Policy Assessment:**")
                        applicable_rule = policy.get('applicable_rule') or policy.get('policy_rule')
                        if applicable_rule:
                            st.write(f"• Applicable Rule: {str(applicable_rule)[:200]}...")
                        else:
                            st.write("• Applicable Rule: Not found")
                        st.write(f"• Compliant: {'✅' if policy.get('compliant') or policy.get('is_compliant') else '❌'}")
                        if policy.get('violations'):
                            st.write("• Violations:")
                            for v in policy['violations']:
                                st.write(f"  - {v}")
        
        st.markdown("---")

# Display current data status (if initialized)
if st.session_state.initialized:
    with st.expander("📡 View Tower Inventory"):
        try:
            with open('data/towers_inventory.json', 'r') as f:
                towers = json.load(f)
            st.dataframe(towers, use_container_width=True)
        except Exception as e:
            st.error(f"Could not load tower inventory: {e}")
    
    with st.expander("📜 View Regional Policies"):
        try:
            with open('data/regional_policies.txt', 'r') as f:
                policies = f.read()
            st.code(policies, language='text')
        except Exception as e:
            st.error(f"Could not load policies: {e}")
else:
    st.info("👈 Please initialize the system from the sidebar to start processing lease requests")

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the example requests in the sidebar to test the system")