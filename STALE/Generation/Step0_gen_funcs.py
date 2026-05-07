import json
import logging
import re
from typing import Any, Dict, Optional

def attr_list2str(attr):
    if not isinstance(attr, (list, tuple)):
        raise TypeError("attr must be list or tuple")
    if not all(isinstance(x, str) for x in attr):
        raise ValueError("all elements in attr must be strings")
    return ".".join(attr)

def strip_json_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(strip_json_fence(text))

SYSTEM_PROMPT_SCENARIO_OLD = """
You are a Context Architect for a "Slice of Life" logic benchmark.

Now, we are talking about some user-specific attributes that describe and formulate the current state of a user.

First, imagine and generate a brief description of a hypothetical person,
and your main task is to generate a REALISTIC, EXPLICIT user scenario and an
old information statement (M_old), GIVEN an explicit user-specific attribute.

Your task is to:
- Interpret the given attribute as a concrete dimension of the user's state (the main theme of M_old)
- Ground it into a realistic life situation (give it a value)
- Generate an old information statement (M_old) that is a STABLE, NATURAL, SUBSTANTIAL, and STATE-DEPENDENT projection of that attribute.
- Add natural details to prevent the statement from sounding rigid or robotic, but STRICTLY AVOID mentioning, implying, or entangling any other possible distinct user-specific attributes
- Avoid fleeting or momentary states.

Output Format (JSON):
{
  "person_description": "Brief description of the generated person"
  "context_scenario": "Brief description of the real-life situation",
  "old_info": "A natural sentence spoken by the user"
  "user-specific attribute value": "The generated value/description about the given user-specific attribute, be brief"
}
"""

def generate_scenarioandoldinfo(attribute, client, model_name, history = False):
    """
    Generate a realistic life scenario and old information (M_old),
    conditioned on an explicitly provided dependency.
    """
    major,sub = attribute[0],attribute[1]
    history_warning = ""
    if history:
        history_warning = (
            "\n**AVOID** generating scenarios similar to the following:\n- "
            + "\n- ".join(history[-2:])
        )

    user_prompt = f"""
    **User-specific attribute (structured)**:
    - Primary dimension: {major}
    - Sub-dimension: {sub if sub else "N/A"}

    **Task**:
    1. Imagine a realistic human life situation where refers to this user-specific attribute.
    2. Write a natural, stable user utterance (M_old) that relies on it.

    {history_warning}
    """

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SCENARIO_OLD},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            response_format={"type": "json_object"}
        )

        result = safe_json_loads(response.choices[0].message.content)

        return {
            "attribute": attribute,
            "person_description": result["person_description"],
            "scenario": result["context_scenario"],
            "old_info": result["old_info"],
            "dependency": result["user-specific attribute value"]
        }

    except Exception as e:
        logging.error(f"\n[Scenario Generation Error] {e}")
        return None
    
SYSTEM_PROMPT_EVALUATOR_T1 = """
You are an expert strict reviewer for a "Slice of Life" logic benchmark.
Your task is to evaluate a pair of user statements (M_old and M_new) designed to form a "Type I: Co-referential Implicit Conflict".

You need to evaluate the pair based on THREE strict criteria:

1. Independent Plausibility:
   - Are both M_old and M_new natural, realistic statements in real life?
   - They must not sound too absurd in isolation.

2. State Conflict:
   - Assuming M_new is spoken by the SAME user after a certain time gap, does M_new makes the situation in M_old no longer feasible?
   - M_new must clearly mention a new value/description of the attribute that is strictly incompatible with the one established in M_old.

3. Implicit Constraints (Type I Compliance):
   - NO explicit linguistic negation (phrases like "don't", "instead of").
   - M_new must NOT explicitly mention the name of the underlying attribute.
   - M_new must NOT explicitly mention the surface text, objects, or scenario of M_old.
   
### Output Format (JSON)
{
  "plausibility": {
    "pass": true/false,
    "reasoning": "brief explanation"
  },
  "state_conflict": {
    "pass": true/false,
    "reasoning": "brief explanation"
  },
  "implicit_constraints": {
    "pass": true/false,
    "reasoning": "check for explicit negations or overlapping vocabulary"
  }
}
"""

def evaluate_type1_conflict(
    old_info: str,
    M_new: str,
    attr,
    client,
    model_name: str,
    temperature: float = 0.1
):
    """
    Evaluate if the generated M_new successfully forms a Type I implicit conflict with old_info.
    """
    attr_str = attr_list2str(attr)
    user_prompt = f"""
Please evaluate the following pair:
- M_old: "{old_info}"
- M_new: "{M_new}"
- attribute: "{attr_str}"
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR_T1},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        result = safe_json_loads(response.choices[0].message.content)
        
        return result

    except Exception as e:
        logging.error(f"\n[Evaluator Error] {e}")
        return None

SYSTEM_PROMPT_ATTACKER_T1 = """
You are a Context Architect and Logic Attacker for a "Slice of Life" benchmark.

Your goal is to generate a NEW user statement (M_new) that creates a
CO-REFERENTIAL IMPLICIT CONFLICT with a given old statement (M_old).

Definition Reminder:
- A CO-REFERENTIAL IMPLICIT CONFLICT arises when both M_old and M_new rely on
  the SAME underlying user-specific attribute, but induce mutually incompatible
  values for that attribute.
- There must be no explicit linguistic negation between M_old and M_new.

You will be given:
- scenario: the background where M_old occurs
- M_old: an old statement
- user-specific attribute
- dependency: the old value/description of the given user-specific attribute

Your task:

First, think up a new realistic value/description for the given user-specific attribute.
Then Produce a new user statement (M_new) that:
- occurs after the value of the user-specific attribute has already changed into the new one,
- must NOT explicitly mention the name of the attribute, but clearly mention the new value/description of that attribute,
- must NOT explicitly mention any other aspects, related objects, or scenarios from M_old,
- must be grounded in a completely new scenario without any explicit linguistic negation of M_old,
- sounds like a normal continuation of life events.

Constraints:
- M_new should be plausible in isolation.
- Avoid sudden, extreme, or fantastical events.

The attack must be IMPLICIT and GROUNDED in everyday life.
The time_gap between m_old and m_new can span days, months, or even years; don't be afraid to make it long.

### Output Format (JSON)
{
  "M_new": "...",
  "explanation": "the updated value/description of user-specific attribute",
  "time_gap": "reasonable elapsed time"
}
"""

def generate_attacker_mnew_t1(
    person_description: str,
    scenario: str,
    old_info: str,
    attribute,
    dependency: str,
    client,
    model_name: str,
    eval_client,
    eval_model_name: str,
    temperature: float = 0.7,
    max_retries: int = 3
):
    """
    Generate M_new that implicitly invalidates the given dependency.
    Utilizes a feedback loop to try up to `max_retries` times if the evaluator rejects it.
    """
    feedback_history = ""
    last_result = None
    last_eval = None
    attr_str = attr_list2str(attribute)
    for attempt in range(max_retries):
        user_prompt = f"""
Context:
- scenario: {scenario}
- M_old: {old_info}
- user-specific attribute: {attr_str}
- dependency: {dependency}
"""
        if feedback_history:
            user_prompt += f"\n[WARNING: Previous Attempts Failed!]\n{feedback_history}\nTask:\nProduce a NEW user statement (M_new) that fixes the issues identified by the evaluator above."
        else:
            user_prompt += "\nTask:\nProduce a new user statement (M_new)"

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ATTACKER_T1},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            result = safe_json_loads(response.choices[0].message.content)
            M_new = result.get("M_new", "")
            last_result = result

            eval_result = evaluate_type1_conflict(old_info, M_new, attribute, eval_client, eval_model_name)
            last_eval = eval_result
            if not eval_result:
                continue
            
            p_pass = eval_result.get("plausibility", {}).get("pass", False)
            s_pass = eval_result.get("state_conflict", {}).get("pass", False)
            i_pass = eval_result.get("implicit_constraints", {}).get("pass", False)
            
            overall_pass = p_pass and s_pass and i_pass

            if overall_pass:
                return {
                    'attribute': attribute,
                    "dependency": dependency,
                    "person_description": person_description,
                    "scenario": scenario,
                    "old_info": old_info,
                    "M_new": result["M_new"],
                    "explanation": result["explanation"],
                    "time_gap": result["time_gap"],
                    "gen_eval": eval_result,
                    "attempts_needed": attempt + 1
                }
            
            feedback_str = f"-> Attempt {attempt + 1} Generated M_new: '{M_new}'\n-> Evaluator Feedback:\n"
            if not p_pass:
                feedback_str += f"   * [Plausibility Failed]: {eval_result['plausibility']['reasoning']}\n"
            if not s_pass:
                feedback_str += f"   * [State Conflict Failed]: {eval_result['state_conflict']['reasoning']}\n"
            if not i_pass:
                feedback_str += f"   * [Implicit Constraints Failed]: {eval_result['implicit_constraints']['reasoning']}\n"
            
            feedback_history += feedback_str + "\n"

        except Exception as e:
            logging.error(f"\n[Attacker Generation Error on attempt {attempt+1}] {e}")
            continue

    if last_result:
        return {
            'attribute': attribute,
            "dependency": dependency,
            "person_description": person_description,
            "scenario": scenario,
            "old_info": old_info,
            "M_new": last_result.get("M_new", ""),
            "explanation": last_result.get("explanation", ""),
            "time_gap": last_result.get("time_gap", ""),
            "gen_eval": last_eval,
            "attempts_needed": max_retries
        }
    return None

SYSTEM_PROMPT_EVALUATOR_T2 = """
You are an expert strict reviewer for a "Slice of Life" logic benchmark.
Your task is to evaluate a pair of user statements (M_old and M_new) designed to form a "Type II: Propagated Implicit Conflict".

You need to evaluate the pair based on THREE strict criteria:

1. Independent Plausibility:
   - Are both M_old and M_new natural, realistic statements in real life?
   - They must not sound too absurd in isolation.

2. Propagated State Conflict (A -> B Dependency):
   - M_old relies on a specific value of an attribute (let's call it Attribute B).
   - Does M_new introduce a completely different attribute/event (Attribute A)?
   - Does Attribute A causally or logically propagate to invalidate the value of Attribute B? 
   - Is there a plausible common-sense dependency (A -> B) that makes the situation in M_old no longer feasible?

3. Implicit Constraints (Type II Compliance):
   - NO explicit linguistic negation (phrases like "I don't", "instead of").
   - M_new must NOT explicitly mention or negate Attribute B.
   - M_new must NOT explicitly mention any surface text, objects, or scenario of M_old.
   - M_new must NOT explicitly mention the causal dependency chain (A -> B) or directly state the updated value of Attribute B.
   - The conflict MUST be indirect and arise from common-sense reasoning.

### Output Format (JSON)
{
  "plausibility": {
    "pass": true/false,
    "reasoning": "brief explanation"
  },
  "propagated_conflict": {
    "pass": true/false,
    "reasoning": "brief explanation"
  },
  "implicit_constraints": {
    "pass": true/false,
    "reasoning": "brief explanation"
  }
}
"""

def evaluate_type2_conflict(
    old_info: str,
    M_new: str,
    attr,
    client,
    model_name: str,
    temperature: float = 0.1
):
    """
    Evaluate if the generated M_new successfully forms a Type I implicit conflict with old_info.
    """
    attr_str = attr_list2str(attr)
    user_prompt = f"""
Please evaluate the following pair:
- M_old: "{old_info}"
- M_new: "{M_new}"
- Attribute B: "{attr_str}"
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_EVALUATOR_T2},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        result = safe_json_loads(response.choices[0].message.content)
        
        return result

    except Exception as e:
        logging.error(f"\n[Evaluator Error] {e}")
        return None
    
SYSTEM_PROMPT_ATTACKER_T2 = """
You are an award-winning mystery novelist and a master of subtext. Your signature style is planting innocuous, everyday clues in casual dialogue—clues that, upon deductive reasoning, completely shatter a previously established fact without the reader immediately realizing it.

Your goal is to generate a NEW user statement (M_new) that creates a
PROPAGATED IMPLICIT CONFLICT with a given old statement (M_old).

Definition Reminder:
- A propagated implicit conflict arises when M_new updates an attribute A,
  and through a known dependency relation A → B (encoded in common-sense
  knowledge), this update indirectly invalidates an earlier value of a
  DIFFERENT attribute B relied upon by M_old.
- M_new must NOT directly mention or contradict the attribute B.
- The conflict must emerge only through causal or logical propagation.

You will be given:
- scenario: the background where M_old occurs
- M_old: an old statement
- user-specific attribute(B)
- user-specific attribute value: the old value/description of the given user-specific attribute

Your task:
1. Identify a DIFFERENT attribute A such that A → B holds under common-sense
   or world knowledge (e.g., health → routine, employment → location,
   physical ability → transportation).
2. Think up a new realistic value/description of the attribute A that eventually causes the PROPAGATED IMPLICIT CONFLICT.
3. Produce a new user statement (M_new) that:
   - occurs after the value of attribute A has already updated,
   - explicitly or implicitly reflects the new value of attribute A,
   - must NOT mention attribute B or the changed value of B in any way,
   - must NOT explicitly mention any aspects, related objects in M_old,
   - is grounded in a completely new scenario,
   - makes the value of B implied by M_old no longer feasible after reasoning.

Constraints:
- M_new should be plausible in isolation.
- The conflict must only emerge when reasoning over user state consistency across time and common-sense knowledge. 
- M_new must NOT explicitly mention the causal dependency chain.
- Avoid sudden, extreme, or fantastical events.

The attack must be IMPLICIT and GROUNDED in everyday life.
the time_gap between m_old and m_new can span days, months, or even years; don't be afraid to make it long.

---

### Example
Bad:
M_old: It is 6 a.m.
M_new: It is now 7 a.m.

Good:
M_old: It is 6 a.m.
M_new: The sun dips below the horizon, leaving a soft glow.

---

### Output Format (JSON)
{
  "M_new": "...",
  "explanation": "Attribute A update → dependency A → B → why B implied by M_old is no longer feasible",
  "time_gap": "reasonable elapsed time"
}

"""

def generate_attacker_mnew_t2(
    person_description: str,
    scenario: str,
    old_info: str,
    attribute,
    dependency: str,
    client,
    model_name: str,
    eval_client,
    eval_model_name: str,
    temperature: float = 0.7,
    max_retries: int = 3
):
    """
    Generate m_new that implicitly invalidates the given dependency via an external factor.
    Utilizes a feedback loop to try up to `max_retries` times if the evaluator rejects it.
    """
    attr_str = attr_list2str(attribute)
    feedback_history = ""
    last_result = None
    last_eval = None

    for attempt in range(max_retries):
        user_prompt = f"""
Context:
- scenario: {scenario}
- M_old: {old_info}
- user-specific attribute: {attr_str}
- user-specific attribute value: {dependency}
"""
        if feedback_history:
            user_prompt += f"\n[WARNING: Previous Attempts Failed!]\n{feedback_history}\nTask:\nProduce a NEW user statement (M_new) that fixes the issues identified by the evaluator above. Important: M_new must NOT directly state the updated value of the user-specific attribute."
        else:
            user_prompt += "\nTask:\nProduce a new user statement (M_new)\n\nImportant:\n- M_new must NOT directly state the updated value of the user-specific attribute."

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_ATTACKER_T2},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            result = safe_json_loads(response.choices[0].message.content)
            M_new = result.get("M_new", "")
            last_result = result

            eval_result = evaluate_type2_conflict(old_info, M_new, attribute, eval_client, eval_model_name)
            last_eval = eval_result
            if not eval_result:
                continue
            
            p_pass = eval_result.get("plausibility", {}).get("pass", False)
            c_pass = eval_result.get("propagated_conflict", {}).get("pass", False)
            i_pass = eval_result.get("implicit_constraints", {}).get("pass", False)
            
            overall_pass = p_pass and c_pass and i_pass

            if overall_pass:
                return {
                    'attribute': attribute,
                    "dependency": dependency,
                    "person_description": person_description,
                    "scenario": scenario,
                    "old_info": old_info,
                    "M_new": result["M_new"],
                    "explanation": result["explanation"],
                    "time_gap": result["time_gap"],
                    "gen_eval": eval_result,
                    "attempts_needed": attempt + 1
                }
            
            feedback_str = f"-> Attempt {attempt + 1} Generated M_new: '{M_new}'\n-> Evaluator Feedback:\n"
            if not p_pass:
                feedback_str += f"   * [Plausibility Failed]: {eval_result['plausibility']['reasoning']}\n"
            if not c_pass:
                feedback_str += f"   * [Propagated Conflict Failed]: {eval_result['propagated_conflict']['reasoning']}\n"
            if not i_pass:
                feedback_str += f"   * [Implicit Constraints Failed]: {eval_result['implicit_constraints']['reasoning']}\n"
            
            feedback_history += feedback_str + "\n"

        except Exception as e:
            logging.error(f"\n[Attacker Generation Error on attempt {attempt+1}] {e}")
            continue

    if last_result:
        return {
            'attribute': attribute,
            "dependency": dependency,
            "person_description": person_description,
            "scenario": scenario,
            "old_info": old_info,
            "M_new": last_result.get("M_new", ""),
            "explanation": last_result.get("explanation", ""),
            "time_gap": last_result.get("time_gap", ""),
            "gen_eval": last_eval,
            "attempts_needed": max_retries
        }
    return None