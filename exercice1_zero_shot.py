"""
Exercise 1 - Zero-Shot Prompting

Chosen technique: Zero-Shot
Justification: no labeled example reviews are available, and sentiment
classification of a review is a task a pretrained LLM already handles well
without examples. Zero-Shot also keeps the prompt short, which matters when
classifying thousands of reviews per day.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2:latest", temperature=0.7)

template = PromptTemplate.from_template(
    "Classify the sentiment of the following customer review as "
    "POSITIVE, NEGATIVE, or NEUTRAL. Respond with only one word.\n\n"
    "Review: {review}\n"
    "Sentiment:"
)

review = "The delivery was two days late and the box arrived crushed, but the product itself works great."
prompt = template.invoke({"review": review})

response = llm.invoke(prompt)
print(response.content)
