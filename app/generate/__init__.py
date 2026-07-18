"""Generate stage: LLM invocation, answer assembly, and citations.

Modules:
- llm.py      : call the LLM with a grounded prompt
- answer.py   : orchestrate context -> LLM -> answer
- citation.py : map the answer back to its source chunks
"""
