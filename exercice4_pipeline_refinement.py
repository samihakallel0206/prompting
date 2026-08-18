"""
Exercise 4 - Complete Pipeline & Refinement

Part A - Chosen technique(s) and justification:
Role-Based Prompting combined with a structured Prompt Template using
dynamic variables. A role ("experienced professor / instructional designer")
controls tone and expertise level; the CONTEXT / TASK / CONSTRAINTS / FORMAT
/ EXAMPLES structure removes ambiguity about length, audience-appropriateness,
and output shape - the two biggest failure points of a bare one-line prompt.

ROLE -> CONTEXT -> TASK -> CONSTRAINTS -> FORMAT -> EXAMPLES formula:
- ROLE: expert professor in {subject}
- CONTEXT: writing for an EdTech platform, read by {target_audience} with
  varying prior knowledge
- TASK: write a course summary adapted to that audience
- CONSTRAINTS: max 150 words, no unexplained jargon for non-experts,
  audience-appropriate tone
- FORMAT: Title / Summary / Key topics (3 bullets)
- EXAMPLES: one worked example shown to anchor style and format

What I improved between Iteration 1 and Iteration 2:
v1 leaves tone, length, structure, and audience-adaptation entirely to the
model's guess. v2 adds a role (controls expertise/voice), explicit context
and constraints (word limit, jargon rule), a fixed output format, and a
worked example - each added element removes one axis of ambiguity, which is
what typically pushes output quality from a rough baseline toward a much
more consistent, on-target result.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

# Moderate temperature: some wording variety is fine for a course summary,
# but it shouldn't drift as freely as pure creative writing would.
llm = ChatOllama(model="llama3.2:latest", temperature=0.5)

# Iteration 1 - Baseline: vague, no role, no format constraints.
# Left deliberately underspecified to contrast with v2 below.
template_v1 = PromptTemplate.from_template(
    "Summarize the university course on {subject} for {target_audience}."
)

# Iteration 2 - Improved: ROLE -> CONTEXT -> TASK -> CONSTRAINTS -> FORMAT -> EXAMPLES.
# Each labeled block below removes one axis of ambiguity left open by v1
# (who is speaking, who reads it, how long, what shape the output takes).
template_v2 = PromptTemplate.from_template(
    "ROLE: You are an experienced university professor and instructional designer "
    "specialized in {subject}.\n\n"
    "CONTEXT: You are writing a short course summary for an EdTech platform. "
    "It will be read by {target_audience}, who may have varying levels of prior "
    "knowledge of {subject}.\n\n"
    "TASK: Write a concise, engaging summary of a university-level course on "
    "{subject}, adapted to the background and expectations of {target_audience}.\n\n"
    "CONSTRAINTS:\n"
    "- Maximum 150 words\n"
    "- Avoid unexplained jargon if the audience is not expert\n"
    "- Use a tone appropriate to the audience\n\n"
    "FORMAT:\n"
    "Title: <course title>\n"
    "Summary: <150-word summary>\n"
    "Key topics: <3 bullet points>\n\n"
    "EXAMPLE:\n"
    "Title: Introduction to Machine Learning\n"
    "Summary: This course introduces beginners to the core ideas behind machine "
    "learning, from linear regression to neural networks, with hands-on Python labs...\n"
    "Key topics:\n- Supervised vs unsupervised learning\n- Model evaluation\n"
    "- Practical Python tools\n\n"
    "Now generate the summary for:\n"
    "Subject: {subject}\n"
    "Target audience: {target_audience}"
)

# Dynamic variables shared by both templates; change these to regenerate
# summaries for a different course/audience without touching the templates.
variables = {
    "subject": "Data Structures and Algorithms",
    "target_audience": "first-year computer science students",
}

# template_v1 only references {subject}; the unused {target_audience} key
# is simply ignored, so the same `variables` dict works for both templates.
prompt_v1 = template_v1.invoke(variables)
prompt_v2 = template_v2.invoke(variables)

print("--- V1 (baseline) ---")
print(llm.invoke(prompt_v1).content)

print("\n--- V2 (improved) ---")
print(llm.invoke(prompt_v2).content)

# Part C - Streaming (Bonus): instead of waiting for the full v2 response,
# print each token/chunk as soon as the model produces it.
print("\n--- V2 Streaming ---")
for chunk in llm.stream(prompt_v2):
    print(chunk.content, end="", flush=True)
