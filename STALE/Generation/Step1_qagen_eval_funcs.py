import json
import logging
from typing import Any, Dict, Optional
import re

def strip_json_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(strip_json_fence(text))

SYSTEM_PROMPT_PROBE_GENERATOR = """
You are an expert Benchmark Designer for an LLM Memory evaluation dataset.
Your task is to generate three distinct probing questions to test an AI assistant's memory and reasoning capabilities.

You will be given the context of a user's changing state:
- Attribute/Theme: The core topic being updated.
- M_old: The user's old statement establishing a previous state/attribute.
- M_new: The user's new statement that implicitly invalidates the old state.
- Attribute Value/Explanation: The underlying logic of what changed.

Based on this context, you must generate three probing queries according to the following strict dimensional requirements:

* **Dimension 1 (Explicit Probing):** - Perspective: Third-person (Evaluator asking the Assistant about the user).
    - Goal: Directly ask if the old state is still valid.
    - Style: "Based on the conversation history, does ...?"

* **Dimension 2 (Adversarial Robustness):**
    - Perspective: Third-person (Evaluator asking the Assistant about the user).
    - Goal: Introduce a leading/misleading question that intentionally assumes the PREMISE of M_old remains true. Ask the assistant to do a task or give a recommendation based on that FALSE premise.
    - FATAL CONSTRAINT (ZERO LEAKAGE): You MUST NOT mention ANY nouns, verbs, events, or specific objects introduced in M_new. The trap must be built PURELY on the context of M_old. The question should sound like it was asked by someone who completely missed M_new.

* **Dimension 3 (Implicit Probing):**
    - Perspective: First-person (The User asking the Assistant).
    - Goal: Ask a natural, everyday question (a forward-looking downstream task or request for advice) that INHERENTLY DEPENDS on the NEW state.
    - FATAL CONSTRAINT (ZERO LEAKAGE): You MUST NOT mention ANY nouns, verbs, events, or specific objects introduced in M_new or M_old. Instead, ask about a normal daily activity where the assistant MUST silently factor in the new state of Attribute/Theme (inferred from M_new) to give safe/helpful advice.

---

### Example

**[Input Context]**
- Attribute/Theme: Commuting Method
- M_old: "I love my daily 10-mile bike ride to the office; it really wakes me up."
- M_new: "I'm so annoyed, the doctor said I need to keep this leg cast on for at least six more weeks."
- Attribute Value/Explanation: The user broke their leg (M_new), which implicitly means they can no longer ride a bike to work (M_old).

**[Expected Output JSON]**
{
  "dim1_query": "Based on the conversation history, does the user still commute to the office by bike?",
  "dim2_query": "Since the user enjoys their daily 10-mile bike commute, can you recommend a new scenic cycling route they could take to work tomorrow?",
  "dim3_query": "I have a mandatory in-person meeting at the office tomorrow morning. Can you figure out the best way for me to get there?"
}
*(Note on Example Dim 2: It mentions NOTHING about doctors or casts. Note on Example Dim 3: It mentions NOTHING about breaking a leg, but the assistant MUST know about the broken leg to suggest a taxi instead of a bike).*

---

### Output Format (JSON)
{
  "dim1_query": "The generated direct validation query",
  "dim2_query": "The generated inductive/deceptive query",
  "dim3_query": "The generated context-aware task query"
}
"""


def generate_probing_queries(
    attr: str,
    old_info: str,
    M_new: str,
    explanation: str,
    client,
    model_name,
    temperature: float = 0.7,
):
    """
    Generate the 3-dimensional probing queries based on the implicit conflict pair.
    """

    user_prompt = f"""
Context Details:
- Attribute/Theme: "{attr}"
- M_old (Old State): "{old_info}"
- M_new (New State): "{M_new}"
- Attribute Value/Explanation: {explanation}

Task:
Generate the 3 probing queries as defined in the system instructions.
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_PROBE_GENERATOR},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        result = safe_json_loads(response.choices[0].message.content)
        return result

    except Exception as e:
        logging.error(f"\n[Query Generator Error] {e}")
        return None
