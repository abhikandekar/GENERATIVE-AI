from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# 1. Load the local model
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

generator = pipeline(
    "text-generation",
    model=model_id,
    tokenizer=model_id,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.3,
    return_full_text=False,
    device=-1,
)

llm = HuggingFacePipeline(pipeline=generator)


prompt = ChatPromptTemplate.from_messages([
("system",
"""
You are an expert movie information extractor.

Extract information ONLY from the paragraph below.

Return the following details:

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
Write a concise 2-3 sentence summary of the movie.

Rules:
1. Do not invent information.
2. If a detail is missing, write "Not mentioned".
3. Keep the output clear and concise.
4. Do not include explanations outside the requested fields.
"""),

("human",
"""Extract information from this paragraph
Paragraph: {paragraph}
""")
]
)

para = input("Provide paragraph: ")

final_prompt = prompt.format(paragraph = para)

response = llm.invoke(final_prompt)

print("\nMovie Information:\n")
print(response)
