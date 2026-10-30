import re
import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(
    page_title="CineSage | Movie Information Extractor",
    page_icon="🎬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
.stApp {
    background: radial-gradient(circle at 8% 0%, rgba(130,83,52,.16), transparent 30%),
                radial-gradient(circle at 95% 12%, rgba(86,104,126,.14), transparent 28%),
                #101114;
    color: #f4f0e9;
    font-family: 'DM Sans', sans-serif;
}
.block-container { max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem; }
.hero {
    padding: 2.2rem 2.4rem; border: 1px solid rgba(226,194,153,.22);
    border-radius: 24px; background: linear-gradient(120deg, #2f2722, #1b1e24);
    box-shadow: 0 18px 55px rgba(0,0,0,.22); margin-bottom: 1.5rem;
}
.eyebrow { color: #d6a875; text-transform: uppercase; letter-spacing: .22em;
    font-size: .72rem; font-weight: 700; margin-bottom: .65rem; }
.hero h1 { font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2.2rem, 5vw, 3.7rem); line-height: 1.08;
    color: #fff8ef; margin: 0 0 .7rem 0; }
.hero p { color: #c7c0b7; font-size: 1.02rem; margin: 0; max-width: 720px; }
.section-label { color: #d6a875; text-transform: uppercase; letter-spacing: .16em;
    font-size: .75rem; font-weight: 700; margin: .5rem 0 .65rem 0; }
div[data-testid="stTextArea"] textarea {
    background: #191b20; color: #f5f0e8; border: 1px solid #414047;
    border-radius: 14px; line-height: 1.65;
}
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
    background: linear-gradient(100deg, #c18b57, #e0b982); color: #171411;
    border: 0; border-radius: 12px; padding: .72rem 1.4rem; font-weight: 700;
    min-height: 3rem;
}
.stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
    background: #edc999; color: #171411; border: 0;
}
.report-card {
    background: rgba(25,27,32,.92);
    border: 1px solid rgba(214,168,117,.24);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    min-height: 100%;
}
.report-label {
    color: #d6a875;
    text-transform: uppercase;
    letter-spacing: .12em;
    font-size: .70rem;
    font-weight: 700;
    margin-bottom: .45rem;
}
.report-value { color: #f4f0e9; line-height: 1.65; font-size: .98rem; }
.summary-card {
    background: linear-gradient(120deg, rgba(64,48,36,.75), rgba(29,32,38,.95));
    border: 1px solid rgba(214,168,117,.34);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-top: 1rem;
}
.muted { color: #aaa49c; font-size: .9rem; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">Your AI-powered movie desk</div>
  <h1>CineSage</h1>
  <p>Extract movie details, creative credits, story information, themes and a concise summary from a paragraph.</p>
</div>
""", unsafe_allow_html=True)

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

SYSTEM_PROMPT = """You are an expert movie information extractor.
Extract information ONLY from the paragraph below.

Return these fields in this exact order, each on its own line:
Movie Name:
Release Year:
Genre:
Director:
Writers:
Main Cast:
Story Setting:
Main Plot:
Key Themes:
Quick Summary:

Rules:
1. Do not invent information.
2. If a detail is missing, write "Not mentioned".
3. Keep the output clear and concise.
4. Do not include explanations outside the requested fields.
5. Put each field on a separate line.
6. For Main Cast, list names separated by commas.
7. Quick Summary must be 2-3 sentences.
"""

HUMAN_PROMPT = """Extract information from this paragraph:
{paragraph}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

@st.cache_resource(show_spinner="Loading TinyLlama locally...")
def load_model():
    generator = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=model_id,
        max_new_tokens=350,
        do_sample=False,
        return_full_text=False,
        device=-1,
    )
    return HuggingFacePipeline(pipeline=generator)

FIELDS = [
    "Movie Name", "Release Year", "Genre", "Director", "Writers",
    "Main Cast", "Story Setting", "Main Plot", "Key Themes", "Quick Summary"
]

def parse_movie_response(raw_text):
    """Split model output into fields, even if it puts several labels on one line."""
    text = raw_text.replace("\r", "\n")

    # Ignore any prompt/instruction echo before the first expected field.
    first_field = re.search(r"\bMovie\s*Name\s*:", text, flags=re.IGNORECASE)
    if first_field:
        text = text[first_field.start():]

    # Split before every known label, including labels generated on the same line.
    labels_pattern = "|".join(
        re.escape(label).replace(r"\ ", r"\s*") for label in FIELDS
    )
    parts = re.split(
        rf"(?i)(?=\b(?:{labels_pattern})\s*:)",
        text
    )

    result = {}
    for part in parts:
        match = re.match(r"(?is)^\s*(" + labels_pattern + r")\s*:\s*(.*)$", part.strip())
        if match:
            label = re.sub(r"\s+", " ", match.group(1)).title()
            value = re.sub(r"\s+", " ", match.group(2)).strip()
            # Remove stray chat-template tokens if the model emits any.
            value = value.replace("</s>", "").replace("<|assistant|>", "").strip()
            result[label] = value

    return result

st.markdown('<div class="section-label">01 / Movie description</div>', unsafe_allow_html=True)

with st.form("movie_extraction_form"):
    paragraph = st.text_area(
        "Paste your paragraph",
        height=210,
        placeholder="Paste a movie description here...",
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button(
        "✦  Extract movie information",
        use_container_width=True,
    )

if submitted:
    if not paragraph.strip():
        st.warning("Please paste a movie paragraph before extracting information.")
    else:
        try:
            llm = load_model()
            prompt_messages = prompt.format_messages(paragraph=paragraph)

            # Format messages using TinyLlama's own chat template.
            chat_messages = []
            for message in prompt_messages:
                role = "system" if message.type == "system" else "user"
                chat_messages.append({"role": role, "content": message.content})

            tokenizer = llm.pipeline.tokenizer
            formatted_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            with st.spinner("CineSage is extracting the movie information..."):
                response = llm.invoke(formatted_prompt).strip()

            st.session_state["movie_result"] = response
            st.session_state["movie_fields"] = parse_movie_response(response)
            st.session_state["movie_input"] = paragraph

        except Exception as error:
            st.error(f"Could not generate the movie report: {error}")

if st.session_state.get("movie_result"):
    st.markdown("---")
    st.markdown('<div class="section-label">02 / Extraction result</div>', unsafe_allow_html=True)
    st.subheader("🎬 Movie information report")

    fields = st.session_state.get("movie_fields", {})
    if fields:
        # Display metadata as compact cards in two columns.
        display_fields = [f for f in FIELDS if f != "Quick Summary"]
        for start in range(0, len(display_fields), 2):
            cols = st.columns(2)
            for col, field in zip(cols, display_fields[start:start + 2]):
                value = fields.get(field, "Not mentioned")
                with col:
                    st.markdown(
                        f'<div class="report-card"><div class="report-label">{field}</div>'
                        f'<div class="report-value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )
            st.write("")

        summary = fields.get("Quick Summary", "Not generated.")
        st.markdown(
            f'<div class="summary-card"><div class="report-label">Quick Summary</div>'
            f'<div class="report-value">{summary}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.warning("The model did not return the expected field labels. Raw response is shown below.")
        st.text(st.session_state["movie_result"])

    st.download_button(
        "Download result as TXT",
        data=st.session_state["movie_result"],
        file_name="cinesage_movie_report.txt",
        mime="text/plain",
    )

st.markdown(
    '<p class="muted" style="text-align:center; margin-top:3rem;">CineSage · Local inference powered by TinyLlama</p>',
    unsafe_allow_html=True,
)
