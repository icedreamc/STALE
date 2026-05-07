from IC_gen_config import gpt_client,USER_SIDE_MODEL,ASSISTANT_SIDE_MODEL,MAX_ROUND_IN_SESSION,MAX_ROUND_RESPONSE
import logging
def chat_with_llm_user(
    messages,
    temperature=0.7,
    max_output_tokens=1024,
    model=USER_SIDE_MODEL
):
    full_response = ''
    try:
        for _ in range(MAX_ROUND_RESPONSE):
            response = gpt_client.responses.create(
                model=model,
                input=messages,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            text = response.output_text
            full_response += text

            if not response.incomplete_details:
                break

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "continue"})

        return full_response.strip()
    except Exception as e:
        logging.error(f"\n[LLM Chat Error] {e}")
        return "STOP"
    
def chat_with_llm_assistant(
    messages,
    max_output_tokens=1024,
    model=ASSISTANT_SIDE_MODEL
):
    full_response = ''
    try:
        for _ in range(MAX_ROUND_RESPONSE):
            response = gpt_client.responses.create(
                model=model,
                input=messages,
                max_output_tokens=max_output_tokens,
            )
            text = response.output_text
            full_response += text

            if not response.incomplete_details:
                break

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "continue"})

        return full_response.strip()
    except Exception as e:
        logging.error(f"\n[LLM Chat Error] {e}")
        return "STOP"

def info2query(info, scenario=""):
    scenario_prompt = f"Background Scenario: {scenario}\n" if scenario else ""
    messages = [
        {"role": "user", 
         "content": f"""
You are roleplaying a human user starting a new chat session with an AI assistant.
Your goal is to brainstorm and naturally embed a specific piece of personal information into a realistic request or conversational opening, be creative.

{scenario_prompt}
Target Information to inject: "{info}"

Requirements:
- [FIDELITY CONSTRAINT]: You MUST preserve the exact details, nuances, and ideally the original phrasing of the Target Information. Do not over-summarize, paraphrase aggressively, or lose any part of its original meaning.
- [NATURAL EMBEDDING]: Do not just state the information like a robot. "Wrap" a natural task, question, or request for advice around it. (e.g., Use the information as the REASON why you are asking for help).
- Act like a normal human user. Use casual, everyday language.
- Output ONLY the user's message. No quotes, no pleasantries like "Here is the message:".
"""}
    ]
    return chat_with_llm_user(messages)

def info2query_topic(info, scenario=""):
    scenario_prompt = f"Background Scenario: {scenario}\n" if scenario else ""
    messages = [
        {"role": "user", 
         "content": f"""
You are roleplaying a human user starting a new chat session with an AI assistant.
You have a specific piece of personal information in mind, BUT YOU MUST NOT REVEAL IT YET. 

{scenario_prompt}
Hidden Information (DO NOT reveal this yet): "{info}"

Requirements:
- Be creative.
- Start a conversation on the general TOPIC related to the hidden information, to set the stage so you can naturally bring it up later.
- The generated user message should strike up a conversation with the assistant, not just a statement.
- DO NOT mention or imply the hidden information at all.
- Output ONLY the user's message.
"""}
    ]
    return chat_with_llm_user(messages)

def assistant_side(previous_messages):
    system_prompt = """
    You are a helpful and friendly assistant. 
    Please provide your answers in natural, conversational language and avoid using bullet points or numbered lists as much as possible. 
    Keep your responses concise and avoid being overly wordy.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(previous_messages)
    response = chat_with_llm_assistant(messages)

    return response

def user_side(previous_messages, info, cur_round):
    system_prompt_1 = f"""
You are simulating the user in this ongoing conversation. 
Based on the assistant's last reply, decide what to say next.

User's underlying truth/persona: "{info}"
(Ensure your responses don't contradict this, but do not mention it in the user message you are about to generate).

Requirements:
- Act like a real human, output from the user's perspective only
- DO NOT repeat previous messages
- You could ask follow-up/related questions or discuss certain aspect
- Output ONLY the user's next message.
"""
    system_prompt_2 = f"""
You are simulating the user in this ongoing conversation. 
Based on the assistant's last reply, decide what to say next.

User's underlying truth/persona: "{info}"
(Ensure your responses don't contradict this, but do not mention it in the user message you are about to generate).

Requirements:
- Act like a real human, output from the user's perspective only
- DO NOT repeat previous messages
- If there is nothing meaningful, relevant, or natural for the user to ask or say next, OUTPUT EXACTLY: STOP
- Do not ask vague continuation questions just to keep talking
- Output ONLY the user's next message OR the word STOP.
"""
    if cur_round < (MAX_ROUND_IN_SESSION//3)*2:
        messages = [{"role": "system", "content": system_prompt_1}]
    else:
        messages = [{"role": "system", "content": system_prompt_2}]
    messages.extend(previous_messages)
    messages.append({"role": "user", "content": "Based on theongoing conversation, Now generate a simulated user's message following the system prompt and requirements"})
    response = chat_with_llm_user(messages)

    if not response or (("STOP" in response.upper()) and len(response)<10):
        return "STOP"
    return response

def user_side_infoinjection(previous_messages, info):
    system_prompt = f"""
You are simulating the user in this ongoing conversation. 
It is now time to reveal a specific piece of target information to the assistant.

Target Information to inject NOW: "{info}"

Requirements:
- [FIDELITY CONSTRAINT]: You MUST preserve the exact details, nuances, and ideally the original phrasing of the Target Information. Do not over-summarize or lose any part of its original meaning.
- Transition smoothly from the assistant's last response.
- Weave the target information naturally into your next reply by wrapping a follow-up question, a constraint, or a casual update around it.
- Output ONLY the user's next message.
"""
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(previous_messages)
    messages.append({"role": "user", "content": "Based on theongoing conversation, Now generate a simulated user's message following the system prompt and requirements"})
    response = chat_with_llm_user(messages)

    if not response or (("STOP" in response.upper()) and len(response)<10):
        return "STOP"
    return response

def info2session(info,inject_index = 0):
    if inject_index>=MAX_ROUND_IN_SESSION:
        raise ValueError('inject_index exceeds round limit!')
    while inject_index>=0:
        session = []
        cur_round = 0
        suc_flag = False
        if inject_index == 0:
            session.append({"role": "user", "content": info2query(info)})
        else:
            session.append({"role": "user", "content": info2query_topic(info)})
        session.append({"role": "assistant", "content": assistant_side(session)})
        cur_round += 1
        while cur_round < MAX_ROUND_IN_SESSION:
            if cur_round == inject_index:
                user_content = user_side_infoinjection(session,info)
            else:
                user_content = user_side(session,info,cur_round)
            if user_content == 'STOP':
                if cur_round<=inject_index:
                    inject_index -= 1
                else:
                    suc_flag = True
                break
            else:
                session.append({"role": "user", "content": user_content})
            session.append({"role": "assistant", "content": assistant_side(session)})
            cur_round += 1
        if suc_flag is True or cur_round == MAX_ROUND_IN_SESSION:
            break
        
    return session