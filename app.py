import streamlit as st
import pandas as pd
import openai
import tldextract
import plotly.express as px

# 1. Config & Sidebar Setup
st.set_page_config(page_title="OutSystems Brand Auditor", layout="wide")

# Retrieve API Key from Streamlit Secrets
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API Key not found in Secrets. Please add it to Settings > Secrets.")

st.title("🤖 OutSystems: AI Positioning Auditor")
st.markdown("""
Upload a **Brand Radar export** to automatically analyze sentiment, brand positioning (Agentic AI vs. Low-Code), 
and see which websites are influencing the AI's responses.
""")

# 2. Utility Functions
def get_domain(url):
    """Extracts clean domain (e.g., bbc.co.uk) from long URLs."""
    if pd.isna(url) or str(url).strip() == "":
        return "Unknown/Direct"
    ext = tldextract.extract(str(url))
    return f"{ext.domain}.{ext.suffix}"

def analyze_response(text):
    """Tags sentiment and positioning using GPT-4o-mini."""
    if not text or len(str(text)) < 5:
        return "Neutral | Unknown"
        
    prompt = f"""
    Analyze the following AI-generated response about the brand 'OutSystems'.
    Categorize it strictly into these two fields:
    1. Sentiment: Positive, Neutral, or Negative.
    2. Positioning: 'Agentic AI' (if focused on AI/Agents), 'Low-Code' (if focused on dev efficiency), or 'Enterprise/Other'.
    
    Response: "{text}"
    
    Return ONLY in this exact format: Sentiment | Positioning
    Do not include any other text, markdown, or explanation.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # High speed, lowest cost
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error | {str(e)}"

# 3. File Uploader & Options
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
st.sidebar.divider()
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)
st.sidebar.info("Sample Mode saves API costs while testing.")

if uploaded_file:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Check for required columns
    required_cols = ['AI Overview', 'Link URL']
    if not all(col in df.columns for col in required_cols):
        st.error(f"CSV must contain columns: {required_cols}")
    else:
        if st.button("🚀 Run Brand Audit"):
            # Prep data
            process_df = df.head(5).copy() if sample_mode else df.copy()
            
            # Progress Bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # --- UPDATED PROCESSING BLOCK (Safety Parsing) ---
            results_sentiment = []
            results_positioning = []
            
            for i, row in enumerate(process_df['AI Overview']):
                raw_output = analyze_response(row)
                
                # Safety Parse: Ensure we always have two exact values to assign
                if "|" in raw_output:
                    parts = raw_output.split("|")
                    s = parts[0].strip() if len(parts) > 0 else "Neutral"
                    p = parts[1].strip() if len(parts) > 1 else "Unknown"
                else:
                    # Fallback if API returns weird formatting
                    s, p = "Neutral", "Unknown"
                
                results_sentiment.append(s)
                results_positioning.append(p)
                
                # Update progress
                pct = int((i + 1) / len(process_df) * 100)
                progress_bar.progress(pct)
                status_text.text(f"Analyzing row {i+1} of {len(process_df)}...")

            # Assign directly to columns to avoid the pandas ValueError mismatch entirely
            process_df['Sentiment'] = results_sentiment
            process_df['Positioning'] = results_positioning
            process_df['Source Domain'] = process_df['Link URL'].apply(get_domain)
            # --- END OF UPDATED BLOCK ---

            # --- 📊 DASHBOARD ---
            st.divider()
            
            # Row 1: Key Metrics
            m1, m2, m3 = st.columns(3)
            pos_pct = (process_df['Sentiment'].str.contains('Positive', na=False)).mean() * 100
            ai_pct = (process_df['Positioning'].str.contains('Agentic AI', na=False)).mean() * 100
            
            m1.metric("Positive Sentiment", f"{pos_pct:.1f}%")
            m2.metric("Agentic AI Positioning", f"{ai_pct:.1f}%")
            m3.metric("Data Rows Analyzed", len(process_df))

            # Row 2: Visualizations
            c1, c2 = st.columns(2)
            
            with c1:
                # Positioning Chart
                fig_pos = px.pie(process_df, names='Positioning', title="Brand Positioning (AI vs Low-Code)",
                                color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_pos, use_container_width=True)
                
            with c2:
                # Source Share Chart
                top_sources = process_df['Source Domain'].value_counts().reset_index().head(10)
                fig_src = px.bar(top_sources, x='count', y='Source Domain', orientation='h', 
                                 title="Top Influencing Domains", labels={'count':'Mentions'})
                st.plotly_chart(fig_src, use_container_width=True)

            # Download Result
            st.divider()
            st.subheader("📥 Download Audited Data")
            csv = process_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download CSV", csv, "outsystems_audit_results.csv", "text/csv")
            
            st.success("Audit complete! You can now download the tagged file for the report.")

else:
    st.info("Waiting for CSV upload in the sidebar...")
