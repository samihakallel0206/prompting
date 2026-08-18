"""
Exercise 3 - Chain-of-Thought Prompting

Why each step of the CoT is necessary:
- Step 1 forces the model to restate the known data. This catches a common
  failure mode where the model ignores that train A already has a 1-hour
  head start (08:00 -> 09:00) before train B even departs.
- Step 2 forces the model to commit to the right approach (closing speed of
  two objects moving toward each other) before doing any arithmetic, instead
  of guessing an answer and rationalizing it afterward.
- Step 3 isolates the arithmetic in its own block, so a wrong number is easy
  to spot and fix independently of the reasoning logic.
- Step 4 produces a single, clearly delimited final answer line, which is
  what a downstream program would parse - separating the reasoning trace
  from the answer is what makes CoT output usable in a pipeline.

Ground truth check: at 09:00 train A has already covered 120 km, leaving
150 km between the trains. Closing speed = 120 + 90 = 210 km/h.
Time to meet = 150 / 210 ~= 0.714 h ~= 42 min 51 s after 09:00
-> the trains cross at approximately 09:43.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

llm = ChatOllama(model="llama3.2:latest", temperature=0)

template = PromptTemplate.from_template(
    "Solve the following problem by reasoning step by step. "
    "Show each step explicitly, then give the final answer on its own line "
    "prefixed with 'Final answer:'.\n\n"
    "Problem: {problem}\n\n"
    "Step 1 - Identify the known data (departure times, speeds, distance):\n"
    "Step 2 - Determine what operation(s) are needed to find when the trains meet:\n"
    "Step 3 - Perform the calculation:\n"
    "Step 4 - State the final answer (exact crossing time):\n"
)

problem = (
    "A train leaves Tunis at 08:00 at 120 km/h. Another train leaves Sfax at "
    "09:00 at 90 km/h heading towards Tunis. The distance between Tunis and "
    "Sfax is 270 km. At what time do the two trains cross each other?"
)

prompt = template.invoke({"problem": problem})
response = llm.invoke(prompt)
print(response.content)
