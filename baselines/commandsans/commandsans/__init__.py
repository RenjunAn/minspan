"""CommandSans baseline: task-agnostic sanitization of tool outputs.

A token classifier (XLM-RoBERTa) tags every whitespace-delimited word of a
tool output as INSTRUCTION or DATA; contiguous INSTRUCTION spans are deleted
before the output reaches the agent. Unlike MinSpan, the classifier sees only
the tool output — no user-task conditioning.
"""

from commandsans.sanitize import Sanitizer, SanitizeResult

__all__ = ["Sanitizer", "SanitizeResult"]
