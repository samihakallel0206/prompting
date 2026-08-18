"""
Exercise 2 - Few-Shot Prompting

Why Few-Shot is preferable to Zero-Shot here:
The manager already has labeled examples, and the category set is fixed
(Electronics, Clothing, Home, Books, Other). Showing a few input -> output
examples anchors both the output format (a single category word) and the
decision boundary between look-alike classes (e.g. "wireless headphones"
could plausibly look like Electronics or Home without an anchor example).
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Lower temperature than Ex1: classification into a fixed category list
# should be as deterministic/consistent as possible.
llm = ChatOllama(model="llama3.2:latest", temperature=0.3)

# Few-Shot prompt: the instruction is followed by 3 worked
# Description -> Category examples (one per category style) before the
# real description to classify. The examples anchor both the allowed
# output format (single category word) and the decision boundaries.
template = PromptTemplate.from_template(
    "Classify each product description into exactly one category: "
    "Electronics, Clothing, Home, Books, Other.\n\n"
    "Description: Men's slim-fit denim jacket\nCategory: Clothing\n\n"
    "Description: 500-page hardcover fantasy novel\nCategory: Books\n\n"
    "Description: Non-stick 12-inch frying pan\nCategory: Home\n\n"
    "Description: {description}\nCategory:"
)

# Product description to classify (deliberately ambiguous: could look like
# Electronics or Home without the few-shot examples above).
description = "Wireless noise-canceling headphones"
prompt = template.invoke({"description": description})

response = llm.invoke(prompt)
print(response.content)  # expected: Electronics
