from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 1. Load the local model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    max_new_tokens=120,
    do_sample=True,
    temperature=0.3,
    return_full_text=False,
    device=-1,
)

llm = HuggingFacePipeline(pipeline=generator)

# 2. Define a reusable prompt
template = """<|user|>
You are a helpful AI tutor.
Answer the question clearly in simple English.
If possible, give one example.

Question: {question}
</s>
<|assistant|>
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template,
)

# 3. Connect the prompt and model
chain = prompt | llm

# 4. Ask questions interactively
while True:
    question = input("\nAsk a question (or type 'exit'): ").strip()

    if question.lower() == "exit":
        print("Goodbye!")
        break

    if not question:
        print("Please enter a question.")
        continue

    answer = chain.invoke({"question": question})
    print("\nAI Answer:\n", answer)
