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

# Initialize Session State
if 'output_df' not in st.session_state:
    st.session_state.output_df = None
if 'suggested_tags' not in st.session_state:
    st.session_state.suggested_tags = None

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

# --- Tag Suggestion Function ---
def suggest_tags(df_sample, target_client, comp_list):
    sample_text = "\n---\n".join(df_sample['AI Overview'].dropna().astype(str).tolist())
    comps = ", ".join(comp_list) if comp_list else "its competitors"
    
    prompt = f"""
    Analyze the following sample of AI-generated responses about the brand '{target_client}' and {comps}.
    Identify the 5 to 10 most common attributes (pros, cons, or features) mentioned across these texts.
    
    Return ONLY a comma-separated list of these attributes. 
    Keep each attribute to a maximum of 2 words (e.g., High Cost, Scalability, Agentic AI, Legacy UI).
    
    Sample Data:
    {sample_text}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error generating suggestions."

# 3. AI Processing Function
def analyze_response(text, target_client, comp_list, pref_tags, max_attrs):
    if not text or len(str(text)) < 5:
        return " | ".join(["Not Mentioned | N/A"] * (1 + len(comp_list)))
    
    comp_prompts = "\n".join([f"    - '{c}': Sentiment? Attributes?" for c in comp_list])
    
    format_parts = ["ClientSentiment | ClientAttributes"]
    for i in range(len(comp_list)):
        format_parts.append(f"Comp{i+1}Sentiment | Comp{i+1}Attributes")
    format_str = " | ".join(format_parts)
    
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

# --- Helper Functions for URLs and Domains ---
def extract_all_domains(text):
    """Finds all URLs in a text block and returns a comma-separated list of their root domains."""
    if pd.isna(text) or str(text).strip() == '':
        return ""
    urls = re.findall(r'(https?://[^\s,]+)', str(text))
    domains = []
    for u in urls:
        try:
            d = urlparse(u).netloc.replace('www.', '')
            if d and d not in domains: # Keep them unique
                domains.append(d)
        except:
            pass
    return ", ".join(domains)

def get_single_domain(url):
    """Helper to get domain from a single URL string for the ranking chart."""
    try:
        return urlparse(str(url)).netloc.replace('www.', '')
    except:
        return str(url)

# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
st.sidebar.divider()
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file is None:
    st.session_state.output_df = None
    st.session_state.suggested_tags = None

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
        
        if len(df) > 0:
            if st.button("💡 Suggest Tags from Data"):
                with st.spinner("Scanning data for common themes..."):
                    comp_list = [c.strip() for c in competitors_input.split(",")] if competitors_input else []
                    sample_df = df.head(15)
                    st.session_state.suggested_tags = suggest_tags(sample_df, client_name, comp_list)
            
            if st.session_state.suggested_tags:
                st.success("Here are the most common themes found in the data. Copy and paste your favorites into the 'Preferred Attribute Tags' box above!")
                st.code(st.session_state.suggested_tags, language="text")

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
                
                # Extract clean domains for the Master CSV
                process_df['Extracted Domains'] = process_df['Link URL'].apply(extract_all_domains)
                
                base_df = process_df.reset_index(drop=True)
                st.session_state.output_df = pd.concat([base_df, results_df], axis=1)
                
                st.success("Audit complete! Visualizations and data below:")

        # --- Display Results & Charts ---
        if st.session_state.output_df is not None:
            file_country_tag = selected_country if selected_country != "All" else "all_countries"
            
            st.markdown(f"### 📊 Brand Positioning for {client_name}")
            c1, c2 = st.columns(2)
            
            with c1:
                sentiment_col = f"{client_name} Sentiment"
                sent_counts = st.session_state.output_df[sentiment_col].value_counts().reset_index()
                sent_counts.columns = ['Sentiment', 'Mentions']
                fig_sent = px.bar(sent_counts, x='Sentiment', y='Mentions', title="Sentiment Distribution",
                                  color='Sentiment', color_discrete_map={'Positive':'#2ecc71', 'Neutral':'#95a5a6', 'Negative':'#e74c3c', 'Not Mentioned':'#34495e', 'Unknown':'#34495e'})
                st.plotly_chart(fig_sent, use_container_width=True)
                
            with c2:
                attr_col = f"{client_name} Attributes"
                all_attrs = st.session_state.output_df[attr_col].dropna().astype(str)
                attr_list = all_attrs[all_attrs != 'N/A'].str.split(',').explode().str.strip()
                attr_list = attr_list[attr_list != ''] 
                
                if not attr_list.empty:
                    attr_counts = attr_list.value_counts().head(10).reset_index()
                    attr_counts.columns = ['Attribute', 'Frequency']
                    fig_attrs = px.bar(attr_counts, x='Frequency', y='Attribute', orientation='h', title="Top Attributes Identified")
                    fig_attrs.update_layout(yaxis={'categoryorder':'total ascending'}) 
                    st.plotly_chart(fig_attrs, use_container_width=True)
                else:
                    st.info("Not enough attribute data to generate a chart.")

            # --- Source Influence Extraction (Exploding Multiple URLs) ---
            st.markdown("### 🌐 Top Influencing Sources")
            c3, c4 = st.columns(2)
            
            # Step 1: Find all URLs and explode them into a giant flat list
            raw_urls = st.session_state.output_df['Link URL'].dropna().astype(str)
            # Split by whitespace or commas, then explode
            exploded_urls = raw_urls.str.split(r'[\s,]+').explode().str.strip()
            # Filter out any non-URL junk
            valid_urls = exploded_urls[exploded_urls.str.startswith('http', na=False)]
            
            with c3:
                # Top Exact URLs Chart
                url_counts = valid_urls.value_counts().reset_index()
                url_counts.columns = ['URL', 'Citations']
                fig_urls = px.bar(url_counts.head(10), x='Citations', y='URL', orientation='h', title="Top Exact URLs Cited")
                fig_urls.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_urls, use_container_width=True)
                
                csv_urls = url_counts.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Top URLs CSV", csv_urls, f"{client_name.lower()}_{file_country_tag}_top_urls.csv", "text/csv")
                
            with c4:
                # Top Root Domains Chart
                exploded_domains = valid_urls.apply(get_single_domain)
                exploded_domains = exploded_domains[exploded_domains != '']
                domain_counts = exploded_domains.value_counts().reset_index()
                domain_counts.columns = ['Domain', 'Citations']
                
                fig_domains = px.bar(domain_counts.head(10), x='Citations', y='Domain', orientation='h', title="Top Root Domains")
                fig_domains.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_domains, use_container_width=True)
                
                csv_domains = domain_counts.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Top Domains CSV", csv_domains, f"{client_name.lower()}_{file_country_tag}_top_domains.csv", "text/csv")

            # Display Data Table
            st.divider()
            st.markdown("### 📄 Processed Master Data")
            st.dataframe(st.session_state.output_df)
            
            csv_main = st.session_state.output_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Master Audited CSV", 
                data=csv_main, 
                file_name=f"{client_name.lower().replace(' ', '_')}_{file_country_tag}_master_audit.csv", 
                mime="text/csv"
            )
elif uploaded_file and not client_name:
    st.warning("Please enter a Client Name before running the audit.")
else:
    st.info("Waiting for CSV upload and Client Name...")
