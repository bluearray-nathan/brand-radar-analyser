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
st.markdown("Upload a Brand Radar export to automatically extract sentiment for your specific client and their competitors.")

# 2. Input Fields
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name", placeholder="e.g. OutSystems")
with col2:
    competitors = st.text_input("Competitors (comma separated)", placeholder="e.g. Mendix, PowerApps")

# 3. AI Processing Function
def analyze_response(text, target_client, target_competitors):
    """Extracts sentiment for the specific client and competitors."""
    if not text or len(str(text)) < 5:
        return "Not Mentioned | Not Mentioned"
        
    prompt = f"""
    Analyze the following AI-generated text. 
    1. Check if the client '{target_client}' is mentioned. What is the sentiment towards them? (Reply strictly with: Positive, Neutral, Negative, or Not Mentioned)
    2. Check if any of these competitors '{target_competitors}' are mentioned. What is the overall sentiment towards them? (Reply strictly with: Positive, Neutral, Negative, or Not Mentioned)
    
    Response text: "{text}"
    
    Return ONLY in this exact format: Client Sentiment | Competitor Sentiment
    Example: Positive | Negative
    Example: Not Mentioned | Positive
    Do not include any other text, reasoning, or markdown.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return "Error | Error"

# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
st.sidebar.divider()
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file and client_name:
    df = pd.read_csv(uploaded_file)
    
    if 'AI Overview' not in df.columns or 'Link URL' not in df.columns:
        st.error("CSV must contain columns: 'AI Overview' and 'Link URL'")
    else:
        if st.button(f"🚀 Run Audit for {client_name}"):
            process_df = df.head(5).copy() if sample_mode else df.copy()
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            client_sentiments = []
            competitor_sentiments = []
            
            # Processing Loop
            for i, row in enumerate(process_df['AI Overview']):
                comp_names = competitors if competitors else "None"
                raw_output = analyze_response(row, client_name, comp_names)
                
                # Safety Parse
                if "|" in raw_output:
                    parts = raw_output.split("|")
                    c_sent = parts[0].strip() if len(parts) > 0 else "Unknown"
                    comp_sent = parts[1].strip() if len(parts) > 1 else "Unknown"
                else:
                    c_sent, comp_sent = "Unknown", "Unknown"
                
                client_sentiments.append(c_sent)
                competitor_sentiments.append(comp_sent)
                
                pct = int((i + 1) / len(process_df) * 100)
                progress_bar.progress(pct)
                status_text.text(f"Analyzing row {i+1} of {len(process_df)}...")

            # Assign to DataFrame
            process_df[f'{client_name} Sentiment'] = client_sentiments
            process_df['Competitor Sentiment'] = competitor_sentiments
            
            # Clean up the output dataframe to just show what matters
            output_df = process_df[['AI Overview', 'Link URL', f'{client_name} Sentiment', 'Competitor Sentiment']]

            st.divider()
            st.success("Audit complete! Preview the data below:")
            st.dataframe(output_df)

            # Download Button
            csv = output_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download CSV", 
                csv, 
                f"{client_name.lower().replace(' ', '_')}_sentiment_audit.csv", 
                "text/csv"
            )

elif uploaded_file and not client_name:
    st.warning("Please enter a Client Name before running the audit.")
else:
    st.info("Waiting for CSV upload and Client Name...")
