import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from urllib.parse import urlparse
import re
import concurrent.futures

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

def suggest_tags(df_sample, target_client, comp_list):
    sample_text = "\n---\n".join(df_sample['AI Overview'].dropna().astype(str).tolist())
    comps = ", ".join(comp_list) if comp_list else "its competitors"
    prompt = f"Analyze the following sample of AI-generated responses about the brand '{target_client}' and {comps}. Identify the 5 to 10 most common attributes (pros, cons, or features) mentioned. Return ONLY a comma-separated list. Keep each attribute to a maximum of 2 words.\n\nSample Data:\n{sample_text}"
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.5)
        return response.choices[0].message.content.strip()
    except:
        return "Error generating suggestions."

def analyze_response(text, target_client, comp_list, pref_tags, max_attrs, mentions_list):
    if not text or len(str(text)) < 5:
        return " | ".join(["Not Mentioned | N/A"] * (1 + len(comp_list)))
    
    format_parts = ["ClientSentiment | ClientAttributes"]
    for i in range(len(comp_list)):
        format_parts.append(f"Comp{i+1}Sentiment | Comp{i+1}Attributes")
    format_str = " | ".join(format_parts)
    
    expected_items = (len(comp_list) + 1) * 2
    tag_logic = f"CRITICAL: Map attributes to these Preferred Tags if semantically similar: [{pref_tags}]. If no preferred tags fit, create a new broad industry term." if pref_tags else "Normalize into broad industry terms."
    comp_prompts = "\n".join([f"    - '{c}': Sentiment? Attributes?" for c in comp_list])

    prompt = f"""
    Analyze the following AI-generated text. 
    IMPORTANT: Recognize brand name variations.
    
    For each brand mentioned, extract:
    A. Sentiment: (Positive, Neutral, Negative, or Not Mentioned)
    B. Attributes: Extract up to {max_attrs} short labels (max 2 words each) summarizing key pros/cons, separated by commas. 
    {tag_logic}
    If no specific attributes are mentioned, output 'N/A'.
    
    RULES FOR EXTRACTION:
    1. SENTIMENT: If a brand is listed as an 'Industry Leader', 'Enterprise-Grade', or 'Top Platform', treat this as POSITIVE. 
    2. KNOWN MENTIONS: The system confirms these brands are definitely in the text: [{mentions_list}]. You MUST find them and evaluate their sentiment.
    
    You MUST evaluate the client and EVERY SINGLE competitor. Do not skip any:
    1. Client: '{target_client}'
    2. Competitors:
{comp_prompts}
    
    Response text: "{text}"
    
    Return ONLY in this exact pipe-separated format. 
    CRITICAL: You MUST output exactly {expected_items} pipe-separated values:
    {format_str}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {"role": "system", "content": "You are a senior brand analyst. Extract sentiment and attributes precisely according to the format provided."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            seed=42
        )
        return response.choices[0].message.content.strip()
    except:
        return " | ".join(["Error | Error"] * (1 + len(comp_list)))

def extract_all_domains(text):
    if pd.isna(text) or str(text).strip() == '': return ""
    urls = re.findall(r'(https?://[^\s,]+)', str(text))
    domains = []
    for u in urls:
        try:
            d = urlparse(u).netloc.replace('www.', '')
            if d and d not in domains: domains.append(d)
        except: pass
    return ", ".join(domains)

def get_single_domain(url):
    try: return urlparse(str(url)).netloc.replace('www.', '')
    except: return str(url)

# --- Threaded Processing Function ---
def process_single_row(i, row_text, mentions_val, client_name, comp_list, preferred_tags, max_attributes):
    raw_output = analyze_response(row_text, client_name, comp_list, preferred_tags, max_attributes, mentions_val)
    parts = [p.strip() for p in raw_output.split("|")]
    
    row_dict = {
        f"{client_name} Sentiment": parts[0] if len(parts) > 0 else "Unknown",
        f"{client_name} Attributes": parts[1] if len(parts) > 1 else "N/A"
    }
    for idx, comp in enumerate(comp_list):
        base_idx = 2 + (idx * 2)
        row_dict[f"{comp} Sentiment"] = parts[base_idx] if base_idx < len(parts) else "Unknown"
        row_dict[f"{comp} Attributes"] = parts[base_idx + 1] if (base_idx + 1) < len(parts) else "N/A"
    return i, row_dict

# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file and client_name:
    df = pd.read_csv(uploaded_file)
    if 'AI Overview' not in df.columns or 'Link URL' not in df.columns:
        st.error("CSV must contain columns: 'AI Overview' and 'Link URL'")
    else:
        country_col = df.columns[0]
        unique_countries = ["All"] + sorted(df[country_col].dropna().astype(str).str.lower().unique().tolist())
        selected_country = st.selectbox(f"Select Country (Filtering by '{country_col}')", options=unique_countries)
        if selected_country != "All":
            df = df[df[country_col].astype(str).str.lower() == selected_country]
            
        st.info(f"Rows ready to process: {len(df)}")
        
        if len(df) > 0:
            if st.button("💡 Suggest Tags from Data"):
                with st.spinner("Scanning data for common themes..."):
                    comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
                    st.session_state.suggested_tags = suggest_tags(df.head(15), client_name, comp_list)
            if st.session_state.suggested_tags:
                st.code(st.session_state.suggested_tags, language="text")

        st.divider()

        if st.button(f"🚀 Run Audit for {client_name}"):
            process_df = df.head(5).copy() if sample_mode else df.copy()
            comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Pre-allocate list for results so they stay in the correct order
            parsed_results = [None] * len(process_df)
            completed = 0
            total_rows = len(process_df)
            
            # TURBO MODE: 10 rows at a time
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = {}
                for i, row in enumerate(process_df['AI Overview']):
                    mentions_val = str(process_df['Mentions'].iloc[i]) if 'Mentions' in process_df.columns and pd.notna(process_df['Mentions'].iloc[i]) else ""
                    future = executor.submit(process_single_row, i, row, mentions_val, client_name, comp_list, preferred_tags, max_attributes)
                    futures[future] = i
                    
                for future in concurrent.futures.as_completed(futures):
                    idx, row_dict = future.result()
                    parsed_results[idx] = row_dict
                    completed += 1
                    pct = int((completed / total_rows) * 100)
                    progress_bar.progress(pct)
                    status_text.text(f"Processed {completed} of {total_rows} rows...")

            results_df = pd.DataFrame(parsed_results)
            process_df['Extracted Domains'] = process_df['Link URL'].apply(extract_all_domains)
            base_df = process_df.reset_index(drop=True)
            st.session_state.output_df = pd.concat([base_df, results_df], axis=1)
            st.success("Turbo Audit complete!")

        # --- Display Results (Unchanged) ---
        if st.session_state.output_df is not None:
            comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
            
            st.markdown(f"### 📊 Brand Positioning: {client_name}")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(px.bar(st.session_state.output_df[f"{client_name} Sentiment"].value_counts().reset_index(), x='count', y=f'{client_name} Sentiment', title="Sentiment", orientation='h'), use_container_width=True)
            with c2:
                attr_list = st.session_state.output_df[f"{client_name} Attributes"].dropna().astype(str).str.split(',').explode().str.strip()
                st.plotly_chart(px.bar(attr_list[attr_list != 'N/A'].value_counts().head(10).reset_index(), x='count', y='index', orientation='h', title="Top Attributes"), use_container_width=True)

            if comp_list:
                st.markdown("### 📊 Competitor Brand Positioning")
                c_comp1, c_comp2 = st.columns(2)
                with c_comp1:
                    comp_sent_cols = [f"{c} Sentiment" for c in comp_list]
                    if all(col in st.session_state.output_df.columns for col in comp_sent_cols):
                        melted_sent = st.session_state.output_df.melt(value_vars=comp_sent_cols, var_name='Competitor', value_name='Sentiment')
                        melted_sent['Competitor'] = melted_sent['Competitor'].str.replace(' Sentiment', '')
                        st.plotly_chart(px.bar(melted_sent.groupby(['Competitor', 'Sentiment']).size().reset_index(name='Mentions'), x='Competitor', y='Mentions', color='Sentiment', barmode='group'), use_container_width=True)
                with c_comp2:
                    comp_attr_data = [{'Competitor': c, 'Attribute': attr} for c in comp_list if f"{c} Attributes" in st.session_state.output_df.columns for attr in st.session_state.output_df[f"{c} Attributes"].dropna().astype(str).str.split(',').explode().str.strip() if attr != 'N/A' and attr != '']
                    if comp_attr_data:
                        comp_attr_df = pd.DataFrame(comp_attr_data)
                        top_attrs = comp_attr_df['Attribute'].value_counts().head(10).index.tolist()
                        st.plotly_chart(px.bar(comp_attr_df[comp_attr_df['Attribute'].isin(top_attrs)].groupby(['Attribute', 'Competitor']).size().reset_index(name='Freq'), y='Attribute', x='Freq', color='Competitor', orientation='h', barmode='stack'), use_container_width=True)

            st.markdown("### 🌐 Top Influencing Sources")
            c3, c4 = st.columns(2)
            valid_urls = st.session_state.output_df['Link URL'].dropna().astype(str).str.split(r'[\s,]+').explode().str.strip()
            valid_urls = valid_urls[valid_urls.str.startswith('http', na=False)]
            with c3:
                st.plotly_chart(px.bar(valid_urls.value_counts().reset_index().head(10), x='count', y='index', orientation='h', title="Top Exact URLs"), use_container_width=True)
            with c4:
                st.plotly_chart(px.bar(valid_urls.apply(get_single_domain).value_counts().reset_index().head(10), x='count', y='index', orientation='h', title="Top Domains"), use_container_width=True)

            st.dataframe(st.session_state.output_df)
            st.download_button("📥 Download Master CSV", st.session_state.output_df.to_csv(index=False).encode('utf-8'), f"{client_name.lower()}_master_audit.csv", "text/csv")
