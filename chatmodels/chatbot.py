from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


messages = [
    SystemMessage(content= "You are a helpful cricket expert.")
]

print("------------Welcome type 0 to exit-----------")

while True:
    
    prompt = input("You: ")
    
    messages.append(HumanMessage(content=prompt))
    
    if prompt == "0":
        break
    
    response = llm.invoke(messages)
    
    messages.append(AIMessage(content=response))
    
    print("Bot:", response)
    
print(messages)
