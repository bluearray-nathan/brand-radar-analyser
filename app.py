import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from urllib.parse import urlparse
import re

# 1. Config & Setup
st.set_page_config(page_title="Brand Radar Sentiment Auditor", layout="wide")

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API Key not found in Secrets. Please add it to Settings > Secrets.")

st.title("🤖 AI Response Sentiment Auditor")
st.markdown("Upload a Brand Radar export to automatically extract sentiment, attributes, and top influencing URLs.")

if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'suggested_tags' not in st.session_state:
    st.session_state.suggested_tags = None

# 2. Input Fields
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name *", placeholder="e.g. OutSystems")
with col2:
    competitors_input = st.text_input("Competitors (comma separated)", placeholder="e.g. Mendix, PowerApps, Appian")

col3, col4 = st.columns(2)
with col3:
    preferred_tags = st.text_input("Preferred Attribute Tags (Optional)", placeholder="e.g. Agentic AI, High Cost, Scalability")
with col4:
    max_attributes = st.number_input("Max Attributes per Brand", min_value=1, max_value=10, value=2, step=1)

# --- Tag Suggestion Function ---
def suggest_tags(df_sample, target_client, comp_list):
    sample_text = "\n---\n".join(df_sample['AI Overview'].dropna().astype(str).tolist())
    prompt = f"Analyze the following text about {target_client} and {comp_list}. Identify 5-10 common attribute tags (max 2 words each). Return ONLY a comma-separated list."
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": f"{prompt}\n\nData:\n{sample_text}"}])
        return response.choices[0].message.content.strip()
    except: return "Error"

# 3. AI Processing Function (REINFORCED INSTRUCTIONS)
def analyze_response(text, target_client, comp_list, pref_tags, max_attrs):
    if not text or len(str(text)) < 5:
        return " | ".join(["Not Mentioned | N/A"] * (1 + len(comp_list)))
    
    format_parts = ["ClientSentiment | ClientAttributes"]
    for i in range(len(comp_list)):
        format_parts.append(f"Comp{i+1}Sentiment | Comp{i+1}Attributes")
    format_str = " | ".join(format_parts)
    expected_items = (len(comp_list) + 1) * 2
    
    tag_logic = f"Map attributes to these Preferred Tags if similar: [{pref_tags}]." if pref_tags else "Normalize into broad industry terms."
    comp_prompts = "\n".join([f"    - '{c}': Sentiment? Attributes?" for c in comp_list])

    prompt = f"""
    Analyze the following AI-generated text. 
    
    RULES FOR EXTRACTION:
    1. SENTIMENT: If a brand is listed as an 'Industry Leader', 'Enterprise-Grade', or 'Top Platform', treat this as POSITIVE. Only use Neutral if the mention is purely functional with no status attached.
    2. ATTRIBUTES: If a brand is categorized (e.g., 'Business Process Automation' or 'Complex Scaling'), use those categories as attributes.
    3. BRAND MATCHING: Recognize variations (e.g., 'Power apps' = 'PowerApps').
    4. {tag_logic}
    
    You MUST evaluate the client and EVERY SINGLE competitor. Do not skip any:
    1. Client: '{target_client}'
    2. Competitors:
{comp_prompts}
    
    Response text: "{text}"
    
    Return ONLY in this exact pipe-separated format (exactly {expected_items} values):
    {format_str}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a senior brand analyst. Extract sentiment and attributes precisely."},
                      {"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except:
        return " | ".join(["Error | Error"] * (1 + len(comp_list)))

# --- Helper Functions ---
def get_domain(url):
    try: return urlparse(str(url)).netloc.replace('www.', '')
    except: return str(url)

def extract_all_domains(text):
    if pd.isna(text) or str(text).strip() == '': return ""
    urls = re.findall(r'(https?://[^\s,]+)', str(text))
    domains = list(set([get_domain(u) for u in urls if get_domain(u)]))
    return ", ".join(domains)

# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file and client_name:
    df = pd.read_csv(uploaded_file)
    country_col = df.columns[0]
    unique_countries = ["All"] + sorted(df[country_col].dropna().astype(str).str.lower().unique().tolist())
    selected_country = st.selectbox(f"Select Country", options=unique_countries)
    
    if selected_country != "All":
        df = df[df[country_col].astype(str).str.lower() == selected_country]
    
    if st.button("💡 Suggest Tags"):
        st.session_state.suggested_tags = suggest_tags(df.head(15), client_name, competitors_input)
    if st.session_state.suggested_tags:
        st.code(st.session_state.suggested_tags)

    if st.button(f"🚀 Run Audit"):
        process_df = df.head(5).copy() if sample_mode else df.copy()
        comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
        
        progress_bar = st.progress(0)
        parsed_results = []
        for i, row in enumerate(process_df['AI Overview']):
            raw_output = analyze_response(row, client_name, comp_list, preferred_tags, max_attributes)
            parts = [p.strip() for p in raw_output.split("|")]
            row_dict = {f"{client_name} Sentiment": parts[0] if len(parts)>0 else "Unknown",
                        f"{client_name} Attributes": parts[1] if len(parts)>1 else "N/A"}
            for idx, comp in enumerate(comp_list):
                base_idx = 2 + (idx * 2)
                row_dict[f"{comp} Sentiment"] = parts[base_idx] if base_idx < len(parts) else "Unknown"
                row_dict[f"{comp} Attributes"] = parts[base_idx + 1] if (base_idx + 1) < len(parts) else "N/A"
            parsed_results.append(row_dict)
            progress_bar.progress(int((i + 1) / len(process_df) * 100))

        results_df = pd.DataFrame(parsed_results)
        process_df['Extracted Domains'] = process_df['Link URL'].apply(extract_all_domains)
        st.session_state.output_df = pd.concat([process_df.reset_index(drop=True), results_df], axis=1)

    if st.session_state.output_df is not None:
        st.markdown(f"### 📊 Analysis for {client_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.bar(st.session_state.output_df[f"{client_name} Sentiment"].value_counts().reset_index(), x='index', y=f'{client_name} Sentiment', title="Sentiment"), use_container_width=True)
        with c2:
            attr_list = st.session_state.output_df[f"{client_name} Attributes"].str.split(',').explode().str.strip()
            st.plotly_chart(px.bar(attr_list[attr_list!='N/A'].value_counts().head(10).reset_index(), x='count', y='index', orientation='h', title="Top Attributes"), use_container_width=True)
        
        st.markdown("### 🌐 Sources")
        urls = st.session_state.output_df['Link URL'].dropna().str.split(r'[\s,]+').explode().str.strip()
        valid_urls = urls[urls.str.startswith('http', na=False)]
        st.download_button("📥 Download Master CSV", st.session_state.output_df.to_csv(index=False).encode('utf-8'), "audit.csv", "text/csv")
        st.dataframe(st.session_state.output_df)
