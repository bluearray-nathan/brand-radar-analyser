import streamlit as st
import pandas as pd
import openai
import plotly.express as px
from urllib.parse import urlparse
import re

# 1. Config & Setup
st.set_page_config(page_title="Brand Radar Sentiment Auditor", layout="wide")

# ---- OpenAI client (Streamlit Cloud) ----
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except Exception:
    st.error("OpenAI API Key not found in Secrets. Please add it to Settings > Secrets.")
    st.stop()

st.title("🤖 AI Response Sentiment Auditor")
st.markdown("Upload a Brand Radar export to automatically extract sentiment, attributes, and top influencing URLs.")

# (Optional) quick debug info in sidebar
with st.sidebar.expander("Debug", expanded=False):
    st.write("openai sdk version:", getattr(openai, "__version__", "unknown"))

# Initialize Session State
if "output_df" not in st.session_state:
    st.session_state.output_df = None
if "suggested_tags" not in st.session_state:
    st.session_state.suggested_tags = None

# 2. Input Fields
col1, col2 = st.columns(2)
with col1:
    client_name = st.text_input("Client Name *", placeholder="e.g. OutSystems")
with col2:
    competitors_input = st.text_input("Competitors (comma separated)", placeholder="e.g. Mendix, PowerApps, Appian")

col3, col4 = st.columns(2)
with col3:
    preferred_tags = st.text_input(
        "Preferred Attribute Tags (Optional)",
        placeholder="e.g. Agentic AI, High Cost, Scalability",
        help="The AI will try to map synonyms to these exact tags for cleaner reporting.",
    )
with col4:
    max_attributes = st.number_input(
        "Max Attributes per Brand",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Choose how many attribute labels the AI should extract for each brand.",
    )

# -------- Helpers --------
def clean_comp_list(raw: str) -> list[str]:
    if not raw:
        return []
    return [c.strip() for c in raw.split(",") if c and c.strip()]


def extract_all_domains(text):
    if pd.isna(text) or str(text).strip() == "":
        return ""
    urls = re.findall(r"(https?://[^\s,]+)", str(text))
    domains = []
    for u in urls:
        try:
            d = urlparse(u).netloc.replace("www.", "")
            if d and d not in domains:
                domains.append(d)
        except Exception:
            pass
    return ", ".join(domains)


def get_single_domain(url):
    try:
        return urlparse(str(url)).netloc.replace("www.", "")
    except Exception:
        return str(url)


def safe_pipe_parts(raw_output: str, expected_items: int) -> list[str]:
    """
    Ensures we always return exactly expected_items parts.
    Pads with 'Unknown' if too short, truncates if too long.
    """
    parts = [p.strip() for p in str(raw_output).split("|")]
    if len(parts) != expected_items:
        parts = (parts + ["Unknown"] * expected_items)[:expected_items]
    return parts


# -------- OpenAI wrappers (Responses API) --------
def call_model_text(model: str, system: str | None, user: str, temperature: float = 0.0) -> str:
    """
    Uses the modern Responses API for best compatibility with newer models.
    Returns plain text (or raises).
    """
    input_payload = []
    if system:
        input_payload.append({"role": "system", "content": system})
    input_payload.append({"role": "user", "content": user})

    resp = client.responses.create(
        model=model,
        input=input_payload,
        temperature=temperature,
    )
    # In the modern SDK, output_text is the easiest way to get the final text.
    return (resp.output_text or "").strip()


# --- Tag Suggestion Function ---
def suggest_tags(df_sample, target_client, comp_list):
    """Scans a sample of the data to suggest common attribute tags."""
    sample_text = "\n---\n".join(df_sample["AI Overview"].dropna().astype(str).tolist())
    comps = ", ".join(comp_list) if comp_list else "its competitors"

    prompt = f"""
Analyze the following sample of AI-generated responses about the brand '{target_client}' and {comps}.
Identify the 5 to 10 most common attributes (pros, cons, or features) mentioned across these texts.

Return ONLY a comma-separated list of these attributes.
Keep each attribute to a maximum of 2 words (e.g., High Cost, Scalability, Agentic AI, Legacy UI).

Sample Data:
{sample_text}
""".strip()

    try:
        # Using gpt-4o-mini here is totally fine; swap to gpt-5-mini if you prefer.
        return call_model_text(
            model="gpt-4o-mini",
            system=None,
            user=prompt,
            temperature=0.5,
        )
    except Exception:
        return "Error generating suggestions."


# 3. AI Processing Function
def analyze_response(text, target_client, comp_list, pref_tags, max_attrs, mentions_list):
    """Extracts sentiment and attributes with deterministic-ish output (temperature=0)."""
    if not text or len(str(text)) < 5:
        return " | ".join(["Not Mentioned | N/A"] * (1 + len(comp_list)))

    format_parts = ["ClientSentiment | ClientAttributes"]
    for i in range(len(comp_list)):
        format_parts.append(f"Comp{i+1}Sentiment | Comp{i+1}Attributes")
    format_str = " | ".join(format_parts)

    expected_items = (len(comp_list) + 1) * 2
    tag_logic = (
        f"CRITICAL: You must map attributes to these Preferred Tags if they are semantically similar: [{pref_tags}]. "
        f"If no preferred tags fit, create a new broad industry term."
        if pref_tags
        else "Normalize into broad industry terms."
    )
    comp_prompts = "\n".join([f"    - '{c}': Sentiment? Attributes?" for c in comp_list])

    prompt = f"""
Analyze the following AI-generated text.
IMPORTANT: Recognize brand name variations (e.g., 'Power apps' = 'PowerApps').

For each brand mentioned, extract:
A. Sentiment: (Positive, Neutral, Negative, or Not Mentioned)
B. Attributes: Extract up to {max_attrs} short labels (max 2 words each) summarizing their key pros/cons, separated by commas.
{tag_logic}
If no specific attributes are mentioned, output 'N/A'.

RULES FOR EXTRACTION:
1. SENTIMENT: If a brand is listed as an 'Industry Leader', 'Enterprise-Grade', or 'Top Platform', treat this as POSITIVE.
2. KNOWN MENTIONS: The system confirms these brands are definitely in the text: [{mentions_list}]. You MUST find them and evaluate their sentiment (look for typos).

You MUST evaluate the client and EVERY SINGLE competitor in this list. Do not skip any:
1. Client: '{target_client}'
2. Competitors:
{comp_prompts}

Response text: "{text}"

Return ONLY in this exact pipe-separated format.
CRITICAL: You MUST output exactly {expected_items} pipe-separated values. Do not skip the brands at the end of the list:
{format_str}

Do not include any other text, reasoning, or markdown.
""".strip()

    try:
        # GPT-5 mini via Responses API
        return call_model_text(
            model="gpt-5-mini",
            system="You are a senior brand analyst. Extract sentiment and attributes precisely according to the format provided.",
            user=prompt,
            temperature=0.0,
        )
    except Exception:
        return " | ".join(["Error | Error"] * (1 + len(comp_list)))


# 4. File Uploader & Execution
uploaded_file = st.sidebar.file_uploader("Upload Brand Radar CSV", type="csv")
st.sidebar.divider()
sample_mode = st.sidebar.checkbox("Sample Mode (Analyze first 5 rows only)", value=True)

if uploaded_file is None:
    st.session_state.output_df = None
    st.session_state.suggested_tags = None

if uploaded_file and client_name:
    df = pd.read_csv(uploaded_file)

    if "AI Overview" not in df.columns or "Link URL" not in df.columns:
        st.error("CSV must contain columns: 'AI Overview' and 'Link URL'")
    else:
        country_col = df.columns[0]
        unique_countries = ["All"] + sorted(
            df[country_col].dropna().astype(str).str.lower().unique().tolist()
        )

        st.markdown("### Filter Data")
        selected_country = st.selectbox(
            f"Select Country (Filtering by '{country_col}')", options=unique_countries
        )

        if selected_country != "All":
            df = df[df[country_col].astype(str).str.lower() == selected_country]

        st.info(f"Rows ready to process: {len(df)}")

        if len(df) > 0:
            if st.button("💡 Suggest Tags from Data"):
                with st.spinner("Scanning data for common themes..."):
                    comp_list = clean_comp_list(competitors_input)
                    sample_df = df.head(15)
                    st.session_state.suggested_tags = suggest_tags(sample_df, client_name, comp_list)

            if st.session_state.suggested_tags:
                st.success("Suggested Themes:")
                st.code(st.session_state.suggested_tags, language="text")

        st.divider()

        if st.button(f"🚀 Run Audit for {client_name}"):
            if len(df) == 0:
                st.warning("No data left to process after filtering.")
            else:
                process_df = df.head(5).copy() if sample_mode else df.copy()
                comp_list = clean_comp_list(competitors_input)

                progress_bar = st.progress(0.0)
                status_text = st.empty()
                parsed_results = []

                expected_items = (len(comp_list) + 1) * 2

                for i, row in enumerate(process_df["AI Overview"]):
                    # Mentions column is optional (avoid KeyError)
                    mentions_val = ""
                    if "Mentions" in process_df.columns:
                        v = process_df["Mentions"].iloc[i]
                        mentions_val = "" if pd.isna(v) else str(v)

                    raw_output = analyze_response(
                        row,
                        client_name,
                        comp_list,
                        preferred_tags,
                        max_attributes,
                        mentions_val,
                    )

                    parts = safe_pipe_parts(raw_output, expected_items)

                    # Map Client Data
                    row_dict = {
                        f"{client_name} Sentiment": parts[0],
                        f"{client_name} Attributes": parts[1],
                    }

                    # Map Competitor Data
                    for idx, comp in enumerate(comp_list):
                        base_idx = 2 + (idx * 2)
                        row_dict[f"{comp} Sentiment"] = parts[base_idx]
                        row_dict[f"{comp} Attributes"] = parts[base_idx + 1]

                    parsed_results.append(row_dict)

                    progress_bar.progress((i + 1) / len(process_df))
                    status_text.text(f"Analyzing row {i+1} of {len(process_df)}...")

                results_df = pd.DataFrame(parsed_results)
                process_df["Extracted Domains"] = process_df["Link URL"].apply(extract_all_domains)
                base_df = process_df.reset_index(drop=True)
                st.session_state.output_df = pd.concat([base_df, results_df], axis=1)
                st.success("Audit complete!")

        # --- Display Results ---
        if st.session_state.output_df is not None:
            comp_list = clean_comp_list(competitors_input)

            # --- 1. Client Positioning Charts ---
            st.markdown(f"### 📊 Brand Positioning: {client_name}")
            c1, c2 = st.columns(2)

            with c1:
                sentiment_col = f"{client_name} Sentiment"
                sent_counts = st.session_state.output_df[sentiment_col].value_counts().reset_index()
                sent_counts.columns = ["Sentiment", "Mentions"]
                fig_sent = px.bar(
                    sent_counts,
                    x="Sentiment",
                    y="Mentions",
                    title=f"{client_name} Sentiment Distribution",
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#2ecc71",
                        "Neutral": "#95a5a6",
                        "Negative": "#e74c3c",
                        "Not Mentioned": "#34495e",
                        "Unknown": "#34495e",
                    },
                )
                st.plotly_chart(fig_sent, use_container_width=True)

            with c2:
                attr_col = f"{client_name} Attributes"
                all_attrs = st.session_state.output_df[attr_col].dropna().astype(str)
                attr_list = all_attrs[all_attrs != "N/A"].str.split(",").explode().str.strip()
                attr_list = attr_list[attr_list != ""]

                if not attr_list.empty:
                    attr_counts = attr_list.value_counts().head(10).reset_index()
                    attr_counts.columns = ["Attribute", "Frequency"]
                    fig_attrs = px.bar(
                        attr_counts,
                        x="Frequency",
                        y="Attribute",
                        orientation="h",
                        title=f"Top Attributes: {client_name}",
                    )
                    fig_attrs.update_layout(yaxis={"categoryorder": "total ascending"})
                    st.plotly_chart(fig_attrs, use_container_width=True)

            # --- 2. Competitor Positioning Charts ---
            if comp_list:
                st.divider()
                st.markdown("### 📊 Competitor Brand Positioning")
                c_comp1, c_comp2 = st.columns(2)

                with c_comp1:
                    comp_sent_cols = [f"{c} Sentiment" for c in comp_list]
                    if all(col in st.session_state.output_df.columns for col in comp_sent_cols):
                        melted_sent = st.session_state.output_df.melt(
                            value_vars=comp_sent_cols,
                            var_name="Competitor",
                            value_name="Sentiment",
                        )
                        melted_sent["Competitor"] = melted_sent["Competitor"].str.replace(" Sentiment", "")

                        comp_sent_counts = (
                            melted_sent.groupby(["Competitor", "Sentiment"])
                            .size()
                            .reset_index(name="Mentions")
                        )

                        fig_comp_sent = px.bar(
                            comp_sent_counts,
                            x="Competitor",
                            y="Mentions",
                            color="Sentiment",
                            title="Competitor Sentiment Comparison",
                            barmode="group",
                            color_discrete_map={
                                "Positive": "#2ecc71",
                                "Neutral": "#95a5a6",
                                "Negative": "#e74c3c",
                                "Not Mentioned": "#34495e",
                                "Unknown": "#34495e",
                            },
                        )
                        st.plotly_chart(fig_comp_sent, use_container_width=True)

                with c_comp2:
                    comp_attr_data = []
                    for c in comp_list:
                        colname = f"{c} Attributes"
                        if colname in st.session_state.output_df.columns:
                            attrs = st.session_state.output_df[colname].dropna().astype(str)
                            a_list = attrs[attrs != "N/A"].str.split(",").explode().str.strip()
                            a_list = a_list[a_list != ""]
                            for attr in a_list:
                                comp_attr_data.append({"Competitor": c, "Attribute": attr})

                    if comp_attr_data:
                        comp_attr_df = pd.DataFrame(comp_attr_data)
                        top_attrs = comp_attr_df["Attribute"].value_counts().head(10).index.tolist()
                        filtered_attrs = comp_attr_df[comp_attr_df["Attribute"].isin(top_attrs)]

                        attr_counts = (
                            filtered_attrs.groupby(["Attribute", "Competitor"])
                            .size()
                            .reset_index(name="Frequency")
                        )

                        fig_comp_attrs = px.bar(
                            attr_counts,
                            y="Attribute",
                            x="Frequency",
                            color="Competitor",
                            orientation="h",
                            title="Top Competitor Attributes (Grouped)",
                            barmode="stack",
                        )
                        fig_comp_attrs.update_layout(yaxis={"categoryorder": "total ascending"})
                        st.plotly_chart(fig_comp_attrs, use_container_width=True)
                    else:
                        st.info("Not enough attribute data for competitors to generate a chart.")

            # --- 3. Source Influence Extraction ---
            st.divider()
            st.markdown("### 🌐 Top Influencing Sources")
            c3, c4 = st.columns(2)
            raw_urls = st.session_state.output_df["Link URL"].dropna().astype(str)
            exploded_urls = raw_urls.str.split(r"[\s,]+").explode().str.strip()
            valid_urls = exploded_urls[exploded_urls.str.startswith("http", na=False)]

            with c3:
                url_counts = valid_urls.value_counts().reset_index()
                url_counts.columns = ["URL", "Citations"]
                st.plotly_chart(
                    px.bar(
                        url_counts.head(10),
                        x="Citations",
                        y="URL",
                        orientation="h",
                        title="Top Exact URLs",
                    ),
                    use_container_width=True,
                )
                st.download_button(
                    "📥 Download URLs",
                    url_counts.to_csv(index=False).encode("utf-8"),
                    f"{client_name.lower()}_urls.csv",
                    "text/csv",
                )

            with c4:
                domain_counts = valid_urls.apply(get_single_domain).value_counts().reset_index()
                domain_counts.columns = ["Domain", "Citations"]
                st.plotly_chart(
                    px.bar(
                        domain_counts.head(10),
                        x="Citations",
                        y="Domain",
                        orientation="h",
                        title="Top Domains",
                    ),
                    use_container_width=True,
                )
                st.download_button(
                    "📥 Download Domains",
                    domain_counts.to_csv(index=False).encode("utf-8"),
                    f"{client_name.lower()}_domains.csv",
                    "text/csv",
                )

            st.divider()
            st.dataframe(st.session_state.output_df)
            st.download_button(
                "📥 Download Master CSV",
                st.session_state.output_df.to_csv(index=False).encode("utf-8"),
                f"{client_name.lower()}_master_audit.csv",
                "text/csv",
            )

elif uploaded_file and not client_name:
    st.warning("Please enter a Client Name.")
else:
    st.info("Waiting for CSV upload and Client Name...")
