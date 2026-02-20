import streamlit as st
import pandas as pd
import openai

# 1. Config & Setup
st.set_page_config(page_title="Brand Radar Sentiment Auditor", layout="wide")

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API Key not found in Secrets. Please add it to Settings > Secrets.")

st.title("🤖 AI Response Sentiment Auditor")
st.markdown("Upload a Brand Radar export to automatically extract sentiment and key attributes.")

# Initialize Session State
if 'output_df' not in st.session_state:
    st.session_state.output_df = None

# 2. Input Fields
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name *", placeholder="e.g. OutSystems")
with col2:
    competitors_input = st.text_input("Competitors (comma separated)", placeholder="e.g. Mendix, PowerApps")

col3, col4 = st.columns(2)
with col3:
    preferred_tags = st.text_input(
        "Preferred Attribute Tags (Optional)", 
        placeholder="e.g. Agentic AI, High Cost, Scalability",
        help="The AI will try to map synonyms to these exact tags for cleaner reporting."
    )
with col4:
    max_attributes = st.number_input(
        "Max Attributes per Brand", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1,
        help="Choose how many attribute labels the AI should extract for each brand."
    )

# 3. AI Processing Function
def analyze_response(text, target_client, comp_list, pref_tags, max_attrs):
    """Extracts sentiment and attributes, mapping to preferred tags when possible."""
    if not text or len(str(text)) < 5:
        return " | ".join(["Not Mentioned | N/A"] * (1 + len(comp_list)))
    
    comp_prompts = "\n".join([f"    - '{c}': Sentiment? Attributes?" for c in comp_list])
    
    format_parts = ["ClientSentiment | ClientAttributes"]
    for i in range(len(comp_list)):
        format_parts.append(f"Comp{i+1}Sentiment | Comp{i+1}Attributes")
    format_str = " | ".join(format_parts)
    
    # --- UPDATED PROMPT LOGIC ---
    tag_logic = f"CRITICAL: You must map attributes to these Preferred Tags if they are semantically similar: [{pref_tags}]. If no preferred tags fit, create a new broad industry term." if pref_tags else "Normalize into broad industry terms."
    
    prompt = f"""
    Analyze the following AI-generated text. 
    IMPORTANT: You must recognize variations in spacing or typos as the same brand (e.g., 'Power apps' = 'PowerApps').
    
    For each brand mentioned, extract:
    A. Sentiment: (Positive, Neutral, Negative, or Not Mentioned)
    B. Attributes: Extract up to {max_attrs} short labels (max 2 words each) summarizing their key pros/cons, separated by commas. 
    {tag_logic}
    If no specific attributes are mentioned, output 'N/A'.
    
    1. Check the client '{target_client}'.
    2. Check the following competitors:
{comp_prompts}
    
    Response text: "{text}"
    
    Return ONLY in this exact pipe-separated format:
    {format_str}
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return " | ".join(["Error | Error"] * (1 + len(comp_list)))

# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
st.sidebar.divider()
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file is None and st.session_state.output_df is not None:
    st.session_state.output_df = None

if uploaded_file and client_name:
    df = pd.read_csv(uploaded_file)
    
    if 'AI Overview' not in df.columns or 'Link URL' not in df.columns:
        st.error("CSV must contain columns: 'AI Overview' and 'Link URL'")
    else:
        country_col = df.columns[0]
        unique_countries = ["All"] + sorted(df[country_col].dropna().astype(str).str.lower().unique().tolist())
        
        st.markdown("### Filter Data")
        selected_country = st.selectbox(f"Select Country (Filtering by '{country_col}')", options=unique_countries)
        
        if selected_country != "All":
            df = df[df[country_col].astype(str).str.lower() == selected_country]
            
        st.info(f"Rows ready to process: {len(df)}")
        st.divider()

        if st.button(f"🚀 Run Audit for {client_name}"):
            if len(df) == 0:
                st.warning("No data left to process after filtering.")
            else:
                process_df = df.head(5).copy() if sample_mode else df.copy()
                comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                parsed_results = []
                
                # Processing Loop
                for i, row in enumerate(process_df['AI Overview']):
                    raw_output = analyze_response(row, client_name, comp_list, preferred_tags, max_attributes)
                    
                    parts = [p.strip() for p in raw_output.split("|")]
                    row_dict = {
                        f"{client_name} Sentiment": parts[0] if len(parts) > 0 else "Unknown",
                        f"{client_name} Attributes": parts[1] if len(parts) > 1 else "N/A"
                    }
                    
                    for idx, comp in enumerate(comp_list):
                        base_idx = 2 + (idx * 2)
                        row_dict[f"{comp} Sentiment"] = parts[base_idx] if base_idx < len(parts) else "Unknown"
                        row_dict[f"{comp} Attributes"] = parts[base_idx + 1] if (base_idx + 1) < len(parts) else "N/A"
                        
                    parsed_results.append(row_dict)
                    
                    pct = int((i + 1) / len(process_df) * 100)
                    progress_bar.progress(pct)
                    status_text.text(f"Analyzing row {i+1} of {len(process_df)}...")

                results_df = pd.DataFrame(parsed_results)
                base_df = process_df.reset_index(drop=True)
                st.session_state.output_df = pd.concat([base_df, results_df], axis=1)
                
                st.success("Audit complete! Preview the data below:")

        if st.session_state.output_df is not None:
            st.dataframe(st.session_state.output_df)
            csv = st.session_state.output_df.to_csv(index=False).encode('utf-8')
            file_country_tag = selected_country if selected_country != "All" else "all_countries"
            
            st.download_button(
                label="📥 Download CSV", 
                data=csv, 
                file_name=f"{client_name.lower().replace(' ', '_')}_{file_country_tag}_audit.csv", 
                mime="text/csv"
            )
elif uploaded_file and not client_name:
    st.warning("Please enter a Client Name before running the audit.")
else:
    st.info("Waiting for CSV upload and Client Name...")
