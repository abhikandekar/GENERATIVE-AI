from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# Step 1: Load your local Hugging Face model
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

# Step 2: Create a reusable prompt template
template = """
<|user|>
You are a helpful GenAI tutor.
Explain the following topic in simple English.
Give one practical example.

Topic: {topic}
</s>
<|assistant|>
"""

prompt = PromptTemplate(
    input_variables=["topic"],
    template=template,
)

# Step 3: Create a chain
chain = prompt | llm

# Step 4: Run the chain with dynamic input
result = chain.invoke({"topic": "Retrieval-Augmented Generation (RAG)"})

print("Generated Answer:")
print(result)
