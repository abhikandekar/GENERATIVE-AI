import streamlit as st
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.set_page_config(
    page_title="Hindi Translator",
    page_icon="🌐",
    layout="centered",
)

st.title("🌐 Hindi Translator")
st.caption("Powered by TinyLlama running locally")

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
SYSTEM_PROMPT = (
    "You are a helpful Hindi translator. Translate the user's latest text "
    "into Hindi. Return only the translation, with no explanation or labels."
)


@st.cache_resource
def load_model():
    generator = pipeline(
        "text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        max_new_tokens=120,
        do_sample=False,
        return_full_text=False,
        device=-1,
    )
    return HuggingFacePipeline(pipeline=generator)


llm = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content=SYSTEM_PROMPT)]

# Show conversation history
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue

    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

prompt = st.chat_input("Enter text to translate into Hindi...")

if prompt:
    if prompt.strip() == "0":
        st.stop()

    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert LangChain messages into TinyLlama's native chat format.
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            chat_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            chat_messages.append({"role": "assistant", "content": message.content})

    tokenizer = llm.pipeline.tokenizer
    formatted_prompt = tokenizer.apply_chat_template(
        chat_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    with st.chat_message("assistant"):
        with st.spinner("Translating..."):
            response = llm.invoke(formatted_prompt).strip()
            st.markdown(response)

    st.session_state.messages.append(AIMessage(content=response))