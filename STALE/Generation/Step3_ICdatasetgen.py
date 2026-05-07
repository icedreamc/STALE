import json
import os
import random
from datetime import datetime, timedelta
import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from IC_gen_config import MODEL_FAST,NOISE_SESSION_INSERT_CONCURRENCY_LIMIT,HAYSTACK_SESSION_AMOUNT,gpt_client,get_Async_client,ASYNC_PROVIDER

TIME_FMT = "%Y-%m-%d %H:%M"
TARGET_QUERY_YEAR = 2025
MIN_MINUTES_AFTER_SNEW = int(os.getenv("MIN_MINUTES_AFTER_SNEW", "30"))
STRICTLY_INCREASING_MINUTE = os.getenv("STRICTLY_INCREASING_MINUTE", "1") != "0"
MAX_SAMPLE_ROUNDS = int(os.getenv("NOISE_SESSION_MAX_SAMPLE_ROUNDS", "300"))


def parse_dt(value: str) -> datetime:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Bad datetime: {value}")


def fmt_dt(value: datetime) -> str:
    return value.strftime(TIME_FMT)


def json_task_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful assistant. Follow the instruction and return valid JSON only."},
        {"role": "user", "content": prompt},
    ]


def shift_datetime_by_years(value: datetime, year_delta: int) -> datetime:
    try:
        return value.replace(year=value.year + year_delta)
    except ValueError:
        return value.replace(year=value.year + year_delta, day=28)


def shift_schedule_to_query_year(
    d_old: datetime,
    d_new: datetime,
    d_query: datetime,
    timestamps: List[str],
    target_year: int = TARGET_QUERY_YEAR,
) -> Tuple[datetime, datetime, datetime, List[str], int]:
    if not timestamps:
        return d_old, d_new, d_query, timestamps, 0

    last_dt = parse_dt(timestamps[-1])
    year_delta = target_year - last_dt.year
    if year_delta == 0:
        return d_old, d_new, d_query, timestamps, 0

    shifted_timestamps = [
        fmt_dt(shift_datetime_by_years(parse_dt(timestamp), year_delta))
        for timestamp in timestamps
    ]
    return (
        shift_datetime_by_years(d_old, year_delta),
        shift_datetime_by_years(d_new, year_delta),
        shift_datetime_by_years(d_query, year_delta),
        shifted_timestamps,
        year_delta,
    )


def strip_json_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def safe_json_loads(text: str) -> Dict[str, Any]:
    return json.loads(strip_json_fence(text))


def elapsed_desc(start: datetime, end: datetime) -> str:
    delta = end - start
    total_minutes = int(delta.total_seconds() // 60)
    if total_minutes < 0:
        return f"negative ({total_minutes} minutes)"

    days, rem = divmod(total_minutes, 60 * 24)
    hours, minutes = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days} days")
    if hours:
        parts.append(f"{hours} hours")
    if minutes or not parts:
        parts.append(f"{minutes} minutes")
    return ", ".join(parts)


def generate_reasonable_2027_date_pair(
    m_old_text: str,
    m_new_text: str,
    time_gap: str,
    client=None,
    model_name: str = MODEL_FAST,
) -> Tuple[datetime, datetime]:
    prompt = f"""
You are a careful temporal grounding assistant.

Given an old user fact M_old and a later updated fact M_new, generate two
plausible specific datetimes for a memory benchmark.

Requirements:
1. date_old MUST be in 2027.
2. date_new MUST be later than date_old.
3. The gap from date_old to date_new should match this annotation when possible: "{time_gap}".
4. Choose realistic month/day/time values if either fact suggests seasonality, school/work timing, holidays, weather, routines, recovery periods, deadlines, travel, or other temporal clues.
5. If no strong clue exists, choose a normal mid-year daytime date_old and apply the annotated gap.
6. Output JSON only.

M_old: "{m_old_text}"
M_new: "{m_new_text}"
Time gap annotation: "{time_gap}"

Output format:
{{
  "reasoning": "brief explanation",
  "date_old": "YYYY-MM-DD HH:MM",
  "date_new": "YYYY-MM-DD HH:MM"
}}
"""
    client = client or gpt_client
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=json_task_messages(prompt),
            response_format={"type": "json_object"},
        )
        result = safe_json_loads(response.choices[0].message.content)
        d_old = parse_dt(result["date_old"])
        d_new = parse_dt(result["date_new"])
        if d_old.year != 2027:
            raise ValueError(f"LLM returned non-2027 date: {d_old}")
        if d_new <= d_old:
            raise ValueError(f"LLM returned date_new <= date_old: {d_new} <= {d_old}")
        return d_old, d_new
    except Exception as e:
        logging.error(f"\n[Date Pair Gen Error] {e}. Fallback to generated gap dates.")
        anchor_old, anchor_new = generate_anchor_dates(time_gap)
        delta_old_new = anchor_new - anchor_old
        if delta_old_new <= timedelta(0):
            delta_old_new = timedelta(minutes=1)
        d_old = datetime(2027, 6, 15, 12, 0)
        return d_old, d_old + delta_old_new


TEMPORAL_AUDIT_SYSTEM_PROMPT = """You are auditing temporal validity for an LLM memory benchmark.
Your job is to judge whether a state update M_new is still temporally operative at query time.

Definitions:
- M_old: older user state.
- M_new: newer user state that invalidate or override M_old.
- dim1_query: explicit validation query. It succeeds only if M_new still validly negates / supersedes the old state at query time.
- dim3_query: downstream assistant task. It succeeds only if the answer should still be constrained by M_new at query time.

Important judgment rule:
- Consider temporal persistence carefully.
- Some updates are durable (e.g. moving to a new city, changing jobs, buying a house).
- Some updates are temporary or decaying (e.g. being on vacation, recovering from a short-term injury, temporary crowns, short-term logistics, one-off errands, transient access).
- Judge whether M_new is still likely to hold *at the provided query time*, not just immediately after M_new.
- Use common-sense temporal reasoning, not overly strict literalism.

If the current query time doesn't fit the requirements, propose a BETTER query time that is after S_new and makes BOTH conditions hold:
1) dim1 can still be negated by M_new,
2) dim3 should still be constrained by M_new.

Prefer the latest plausible query time that still works. Output valid JSON only.
"""


TEMPORAL_AUDIT_USER_PROMPT_TEMPLATE = """Audit this benchmark sample.

Sample fields:
uid: {uid}
M_old: {M_old}
M_new: {M_new}
explanation: {explanation}
time_gap_annotation: {time_gap}

Relevant timestamps:
S_new_timestamp: {s_new_ts}
current_query_time: {query_ts}
elapsed_from_S_new_to_query_time: {elapsed_desc}

Queries:
dim1_query: {dim1_query}
dim3_query: {dim3_query}

Return JSON with exactly these keys:
{{
  "dim1_still_negated": true or false,
  "dim1_reason": "...",
  "dim3_still_constrained": true or false,
  "dim3_reason": "...",
  "needs_change": true or false,
  "proposed_query_time": "YYYY-MM-DD HH:MM" or null,
  "proposed_time_reason": "...",
  "confidence": 0.0 to 1.0
}}

Rules:
- If both dim1_still_negated and dim3_still_constrained are true, set needs_change=false and proposed_query_time=null.
- If either is false, set needs_change=true and provide a proposed_query_time.
- proposed_query_time must be strictly after S_new_timestamp.
- Prefer the latest plausible query time that still works.
- Be concise in reasons.
- Output JSON only.
"""


def audit_query_time_for_item(
    item: dict,
    d_new: datetime,
    d_query: datetime,
    client=None,
    model_name: str = MODEL_FAST,
) -> Dict[str, Any]:
    client = client or gpt_client
    prompt = TEMPORAL_AUDIT_USER_PROMPT_TEMPLATE.format(
        uid=item.get("uid", ""),
        M_old=item.get("old_info", item.get("M_old", "")),
        M_new=item.get("M_new", ""),
        explanation=item.get("explanation", ""),
        time_gap=item.get("time_gap", ""),
        s_new_ts=fmt_dt(d_new),
        query_ts=fmt_dt(d_query),
        elapsed_desc=elapsed_desc(d_new, d_query),
        dim1_query=item.get("dim1_query", item.get("probing_queries", {}).get("dim1_query", "")),
        dim3_query=item.get("dim3_query", item.get("probing_queries", {}).get("dim3_query", "")),
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": TEMPORAL_AUDIT_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        result = safe_json_loads(response.choices[0].message.content)
        dim1_ok = bool(result.get("dim1_still_negated"))
        dim3_ok = bool(result.get("dim3_still_constrained"))
        result["needs_change"] = not (dim1_ok and dim3_ok)
        if not result["needs_change"]:
            result["proposed_query_time"] = None
        return result
    except Exception as e:
        logging.error(f"\n[Temporal Audit Error] {e}. Keeping generated query time.")
        return {
            "dim1_still_negated": True,
            "dim3_still_constrained": True,
            "needs_change": False,
            "proposed_query_time": None,
            "error": repr(e),
        }


def validate_or_fix_query_time(d_new: datetime, proposed_str: Optional[str], n_steps_after_snew: int) -> datetime:
    if proposed_str is None:
        candidate = d_new + timedelta(days=30)
    else:
        try:
            candidate = parse_dt(proposed_str)
        except ValueError:
            logging.error(f"\n[Temporal Audit Error] Bad proposed query time: {proposed_str}")
            candidate = d_new + timedelta(days=30)

    min_gap_minutes = max(MIN_MINUTES_AFTER_SNEW, n_steps_after_snew if STRICTLY_INCREASING_MINUTE else 1)
    min_allowed = d_new + timedelta(minutes=min_gap_minutes)

    if candidate < min_allowed:
        return min_allowed
    return candidate


def redistribute_timestamps_segment(
    timestamps: List[str],
    idx_new: int,
    new_last_dt: datetime,
) -> Tuple[List[str], List[int]]:
    new_ts = timestamps[:]
    start_dt = parse_dt(timestamps[idx_new])
    end_dt = new_last_dt

    total_positions_after = len(timestamps) - idx_new - 1
    if total_positions_after <= 0:
        return new_ts, []

    total_minutes = int((end_dt - start_dt).total_seconds() // 60)
    min_required = total_positions_after if STRICTLY_INCREASING_MINUTE else 1
    if total_minutes < min_required:
        end_dt = start_dt + timedelta(minutes=min_required)
        total_minutes = min_required

    changed_positions = []
    for step in range(1, total_positions_after + 1):
        offset_minutes = (step * total_minutes) // total_positions_after
        cur_dt = start_dt + timedelta(minutes=offset_minutes)
        pos = idx_new + step
        new_value = fmt_dt(cur_dt)
        if new_ts[pos] != new_value:
            changed_positions.append(pos)
            new_ts[pos] = new_value

    return new_ts, changed_positions


def build_timestamp_schedule_for_item(
    item: dict,
    idx_old: int,
    idx_new: int,
    total_amount: int = HAYSTACK_SESSION_AMOUNT,
) -> Dict[str, Any]:
    m_old_text = item.get("old_info", item.get("M_old", ""))
    m_new_text = item.get("M_new", "")

    d_old, d_new = generate_reasonable_2027_date_pair(
        m_old_text=m_old_text,
        m_new_text=m_new_text,
        time_gap=item.get("time_gap", "1 month"),
        client=gpt_client,
        model_name=MODEL_FAST,
    )

    initial_query = generate_query_latest_plausible_time(
        m_new_text=m_new_text,
        d_new=d_new,
        dim3_query=item.get("dim3_query", ""),
        client=gpt_client,
        model_name=MODEL_FAST,
    )
    n_steps_after_snew = total_amount - idx_new - 1
    d_query = validate_or_fix_query_time(d_new, fmt_dt(initial_query), n_steps_after_snew)

    audit = audit_query_time_for_item(
        item=item,
        d_new=d_new,
        d_query=d_query,
        client=gpt_client,
        model_name=MODEL_FAST,
    )
    if audit.get("needs_change"):
        d_query = validate_or_fix_query_time(d_new, audit.get("proposed_query_time"), n_steps_after_snew)

    timestamps = generate_timestamp_sequence(
        d_old=d_old,
        d_new=d_new,
        d_query=d_query,
        idx_old=idx_old,
        idx_new=idx_new,
        total_amount=total_amount,
    )
    timestamps, changed_positions = redistribute_timestamps_segment(
        timestamps=timestamps,
        idx_new=idx_new,
        new_last_dt=d_query,
    )
    d_old, d_new, d_query, timestamps, year_shift = shift_schedule_to_query_year(
        d_old=d_old,
        d_new=d_new,
        d_query=d_query,
        timestamps=timestamps,
    )

    return {
        "d_old": d_old,
        "d_new": d_new,
        "d_query": parse_dt(timestamps[-1]),
        "timestamps": timestamps,
        "temporal_audit": audit,
        "timestamp_changed_positions": changed_positions,
        "timestamp_year_shift": year_shift,
    }


async def async_check_conflict(statement: str, session: list, semaphore, async_client) -> bool:
    user_messages = [msg['content'] for msg in session if msg['role'] == 'user']
    if not user_messages:
        return False
        
    user_msgs_str = "\n- ".join(user_messages)
    
    system_prompt = """
    You are an expert strict logic and semantic reviewer. 
    You are given an Established Fact about a user, and a snippet of User Chat History.
    Your task is to determine if the User Chat History is "UNSAFE" to be used as random background noise alongside the Established Fact.

    The Chat History is UNSAFE (i.e., you must output "is_conflict": true) if it meets ANY of the following two conditions:
    1. Logical Contradiction: It CONTRADICTS, NEGATES, or INVALIDATES the Established Fact. (e.g., Fact: "User is vegetarian", Chat: "I ate a steak today" -> UNSAFE).
    2. Logical Elaboration/Supplement: It SUPPLEMENTS, ELABORATES ON, or acts as a DIRECT CONTINUATION of the Established Fact. (e.g., Fact: "The user moved recently", Chat: "I'm really enjoying the West Coast weather now" -> UNSAFE, because it acts as a contextual puzzle piece).

    Be extremely rigorous. Minor, purely coincidental topic overlaps (e.g., both mention "driving" in completely unrelated contexts) are fine, but if the Chat History makes it impossible for the Fact to be true, OR if it looks like a natural follow-up/detail of the Fact, you must flag it as a conflict.

    Output Format (JSON):
    {
    "reasoning": "Brief explanation.",
    "is_conflict": true or false
    }
    """
    
    user_prompt = f"""
    [Established Fact]: "{statement}"
    
    [User Chat History]:
    - {user_msgs_str}
    """

    async with semaphore:
        try:
            response = await async_client.chat.completions.create(
                model=MODEL_FAST,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            result = safe_json_loads(response.choices[0].message.content)
            return result.get("is_conflict", True)
        except Exception as e:
            logging.error(f"[Conflict Check Error] {e}")
            return True

def generate_anchor_dates(time_gap: str):
    
    prompt = f"""
    You need to generate two specific dates based on a time gap.
    The first date MUST be in the year 2027.
    The second date should be later than the first date, separated by this time gap: "{time_gap}".
    
    Output Format (JSON):
    {{
      "date_old": "YYYY-MM-DD",
      "date_new": "YYYY-MM-DD"
    }}
    """
    try:
        response = gpt_client.chat.completions.create(
            model=MODEL_FAST,
            messages=json_task_messages(prompt),
            response_format={"type": "json_object"}
        )
        result = safe_json_loads(response.choices[0].message.content)
        d_old = datetime.strptime(result['date_old'], "%Y-%m-%d")
        d_new = datetime.strptime(result['date_new'], "%Y-%m-%d")
        return d_old, d_new
    except Exception as e:
        logging.error(f"\n[Date Gen Error] {e}. Fallback to default dates.")
        return datetime(2027, 1, 1), datetime(2027, 6, 1)

def generate_query_latest_plausible_time(m_new_text: str, d_new: datetime, dim3_query: str, client, model_name: str):
    prompt = f"""
    You are an expert logical timeframe estimator. 
    A user established a New State on a specific date. Sometime after this date, they ask a Query.
    Your task is to determine the LATEST plausible date when this New State still reliably restricts or applies to the Query.

    New State (M_new): "{m_new_text}"
    Date of New State (d_new): {d_new.strftime('%Y-%m-%d %H:%M')}
    Subsequent Query (dim3_query): "{dim3_query}"

    Rules:
    1. If M_new has an explicit duration (e.g., "for 6 weeks"), calculate the exact expiration date.
    2. If M_new is a temporary condition (e.g., an urgent deadline), use common sense to limit the lifespan.
    3. If M_new is semi-permanent or permanent (e.g., moved to a new city, became a vegetarian, bought a car), output a date far into the future.
    4. Provide the MAXIMUM plausible timeframe, as long as logically sound.

    Output Format (JSON):
    {{
      "reasoning": "brief explanation",
      "latest_plausible_date": "YYYY-MM-DD HH:MM"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=json_task_messages(prompt),
            response_format={"type": "json_object"}
        )
        result = safe_json_loads(response.choices[0].message.content)
        d_query = datetime.strptime(result['latest_plausible_date'], "%Y-%m-%d %H:%M")
        
        if d_query <= d_new:
            return d_new + timedelta(days=30)
            
        return d_query
    except Exception as e:
        logging.error(f"\n[Timeframe Gen Error] {e}. Fallback to default +6 months.")
        return d_new + timedelta(days=180)

def generate_timestamp_sequence(d_old: datetime, d_new: datetime, d_query: datetime, idx_old: int, idx_new: int, total_amount: int):
    timestamps = [None] * total_amount
    timestamps[idx_old] = d_old
    timestamps[idx_new] = d_new
    
    delta_total = (d_new - d_old).total_seconds()
    gap_count_middle = idx_new - idx_old
    
    if gap_count_middle > 1:
        weights = [random.uniform(0.5, 1.5) for _ in range(gap_count_middle)]
        total_weight = sum(weights)
        
        current_weight_sum = 0
        for i in range(1, gap_count_middle):
            current_weight_sum += weights[i-1]
            fraction = current_weight_sum / total_weight
            timestamps[idx_old + i] = d_old + timedelta(seconds=delta_total * fraction)
            
    avg_interval = delta_total / gap_count_middle if gap_count_middle > 0 else 86400
    
    for i in range(idx_old - 1, -1, -1):
        jitter = random.uniform(0.5, 1.5)
        timestamps[i] = timestamps[i+1] - timedelta(seconds=avg_interval * jitter)
        
    temp_forward = []
    current_time = d_new
    for _ in range(idx_new + 1, total_amount):
        jitter = random.uniform(0.5, 1.5)
        current_time += timedelta(seconds=avg_interval * jitter)
        temp_forward.append(current_time)
        
    if temp_forward and temp_forward[-1] > d_query:
        gap_count_end = total_amount - 1 - idx_new
        delta_end = (d_query - d_new).total_seconds()
        
        weights = [random.uniform(0.5, 1.5) for _ in range(gap_count_end)]
        total_weight = sum(weights)
        
        current_weight_sum = 0
        for i in range(1, gap_count_end + 1):
            current_weight_sum += weights[i-1]
            fraction = current_weight_sum / total_weight
            timestamps[idx_new + i] = d_new + timedelta(seconds=delta_end * fraction)
    else:
        for i, t in enumerate(temp_forward):
            timestamps[idx_new + 1 + i] = t
            
    return [t.strftime("%Y-%m-%d %H:%M") for t in timestamps]

def session_signature(session):
    return json.dumps(session, ensure_ascii=False, sort_keys=True)


async def async_check_conflict_multi(statements, session, semaphore, async_client):
    valid_statements = [s for s in statements if isinstance(s, str) and s.strip()]
    if not valid_statements:
        return False

    tasks = [
        async_check_conflict(statement, session, semaphore, async_client)
        for statement in valid_statements
    ]
    results = await asyncio.gather(*tasks)
    return any(results)


async def fetch_safe_sessions_multi_concurrently(
    statements,
    needed_count,
    lme_pool,
    used_indices,
    semaphore,
    async_client,
    forbidden_signatures=None,
    max_rounds=MAX_SAMPLE_ROUNDS,
):
    safe_sessions = []
    forbidden_signatures = forbidden_signatures or set()
    rounds = 0

    while len(safe_sessions) < needed_count:
        rounds += 1
        if rounds > max_rounds:
            raise RuntimeError(
                f"Cannot fetch enough safe sessions. need={needed_count}, "
                f"got={len(safe_sessions)}, statements={statements}"
            )

        remaining_needs = needed_count - len(safe_sessions)
        batch_size = max(5, int(remaining_needs * 1.5))

        batch_candidates = []
        inner_try = 0
        while len(batch_candidates) < batch_size:
            inner_try += 1
            if inner_try > len(lme_pool) * 10:
                break

            idx = random.randint(0, len(lme_pool) - 1)
            if idx in used_indices:
                continue

            session = lme_pool[idx]
            sig = session_signature(session)
            if sig in forbidden_signatures:
                continue

            used_indices.add(idx)
            batch_candidates.append((idx, session, sig))

        if not batch_candidates:
            raise RuntimeError("No more candidate noise sessions can be sampled.")

        tasks = [
            async_check_conflict_multi(statements, session, semaphore, async_client)
            for _, session, _ in batch_candidates
        ]
        results = await asyncio.gather(*tasks)

        for (idx, session, sig), is_conflict in zip(batch_candidates, results):
            if (not is_conflict) and (len(safe_sessions) < needed_count):
                safe_sessions.append(session)
                forbidden_signatures.add(sig)
            else:
                used_indices.remove(idx)

    return safe_sessions


def sample_unique_noise_session(lme_pool, used_indices, forbidden_signatures):
    inner_try = 0
    while True:
        inner_try += 1
        if inner_try > len(lme_pool) * 10:
            raise RuntimeError("No more unique pre-old noise sessions can be sampled.")

        idx = random.randint(0, len(lme_pool) - 1)
        if idx in used_indices:
            continue

        session = lme_pool[idx]
        sig = session_signature(session)
        if sig in forbidden_signatures:
            continue

        used_indices.add(idx)
        forbidden_signatures.add(sig)
        return session


async def build_haystack_for_item(item, lme_pool, idx_old, idx_new):
    m_old_text = item.get("old_info", item.get("M_old", ""))
    m_new_text = item.get("M_new", "")
    
    local_async_client = get_Async_client(ASYNC_PROVIDER)

    semaphore = asyncio.Semaphore(NOISE_SESSION_INSERT_CONCURRENCY_LIMIT)

    haystack = [None] * HAYSTACK_SESSION_AMOUNT
    haystack[idx_old] = item.get("S_old", [])
    haystack[idx_new] = item.get("S_new", [])

    used_indices = set()
    forbidden_signatures = {
        session_signature(haystack[idx_old]),
        session_signature(haystack[idx_new]),
    }

    for i in range(idx_old):
        haystack[i] = sample_unique_noise_session(lme_pool, used_indices, forbidden_signatures)

    middle_slots = list(range(idx_old + 1, idx_new))
    if middle_slots:
        safe_sessions = await fetch_safe_sessions_multi_concurrently(
            statements=[m_old_text],
            needed_count=len(middle_slots),
            lme_pool=lme_pool,
            used_indices=used_indices,
            semaphore=semaphore,
            async_client=local_async_client,
            forbidden_signatures=forbidden_signatures,
        )
        for i, session in zip(middle_slots, safe_sessions):
            haystack[i] = session

    post_slots = list(range(idx_new + 1, HAYSTACK_SESSION_AMOUNT))
    if post_slots:
        safe_sessions = await fetch_safe_sessions_multi_concurrently(
            statements=[m_old_text, m_new_text],
            needed_count=len(post_slots),
            lme_pool=lme_pool,
            used_indices=used_indices,
            semaphore=semaphore,
            async_client=local_async_client,
            forbidden_signatures=forbidden_signatures,
        )
        for i, session in zip(post_slots, safe_sessions):
            haystack[i] = session
    
    await local_async_client.close()

    return haystack
