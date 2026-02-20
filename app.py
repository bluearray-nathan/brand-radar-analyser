import streamlit as st
import pandas as pd
import openai
import plotly.express as px

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
    """Scans a sample of the data to suggest common attribute tags."""
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
    """Extracts sentiment and attributes, mapping to preferred tags when possible."""
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
    B. Attributes: Extract up to {max_attrs} short
