"""
Exercise 1 - Zero-Shot Prompting

Chosen technique: Zero-Shot
Justification: no labeled example reviews are available, and sentiment
classification of a review is a task a pretrained LLM already handles well
without examples. Zero-Shot also keeps the prompt short, which matters when
classifying thousands of reviews per day.
"""

from langchain_ollama import ChatOllama       # local Ollama chat model wrapper
from langchain_core.prompts import PromptTemplate  # builds a reusable prompt with placeholders

# Local model served by Ollama. temperature=0.7 keeps some variability,
# fine for sentiment tagging where there's no single "creative" answer needed.
llm = ChatOllama(model="llama3.2:latest", temperature=0.7)

# Zero-Shot prompt: no examples are given, only the instruction + the review
# to classify. The instruction constrains the output to one of 3 words so
# the response stays easy to parse downstream.
template = PromptTemplate.from_template(
    "Classify the sentiment of the following customer review as "
    "POSITIVE, NEGATIVE, or NEUTRAL. Respond with only one word.\n\n"
    "Review: {review}\n"
    "Sentiment:"
)

# Example review chosen for this run (mixed sentiment: late delivery vs good product).
review = "The delivery was two days late and the box arrived crushed, but the product itself works great."

# Fill the {review} placeholder -> produces a PromptValue LangChain can send to the model.
prompt = template.invoke({"review": review})

# Send the prompt to the model and print only the generated text (AIMessage.content).
response = llm.invoke(prompt)
print(response.content)
