import argparse
import json
import os
from multiprocessing import Process
import math
from Step0_gen_funcs import generate_scenarioandoldinfo,generate_attacker_mnew_t1,generate_attacker_mnew_t2
from Step1_qagen_eval_funcs import generate_probing_queries
from Step2_info2session_funcs import info2session
from Step3_ICdatasetgen import build_timestamp_schedule_for_item,build_haystack_for_item
from collections import defaultdict
import logging
from random import randint
import uuid
import asyncio
import pickle as pkl

logging.basicConfig(
    filename=os.getenv("IC_GEN_LOG_FILE", "IC_gen.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

from IC_gen_config import *

def is_gen_valid(item: dict) -> bool:
    
    gen_eval = item.get("gen_eval")
    if not gen_eval:
        return False
    
    for eval_key, eval_result in gen_eval.items():
        if isinstance(eval_result, dict):
            if not eval_result.get("pass", False):
                return False
    return True


def augment_conflict_dataset(raw_data, conflict_type: str, target_amount: int = TARGET_CONFLICTPAIR_AMOUNT_PER_SEED):

    max_attempts = target_amount * 2

    grouped_data = defaultdict(list)
    for item in raw_data:
        attr = item.get("attribute", [])
        if len(attr) == 2:
            attr_key = (attr[0], attr[1])
            grouped_data[attr_key].append(item)
            
    final_valid_dataset = []
    
    for attr_key, items in grouped_data.items():

        #this is used for continue generation
        persona = [v.get("person_description", "") for v in items if "person_description" in v]
        valid_items = [item for item in items if is_gen_valid(item)]
        
        attempts = 0
        attribute_list = list(attr_key)
        
        while len(valid_items) < target_amount and attempts < max_attempts:
            attempts += 1
            
            old_res = generate_scenarioandoldinfo(
                attribute=attribute_list, 
                client=qwen_client, 
                model_name=M_OLD_GEN_MODEL, 
                history=persona if persona else False
            )
            
            if not old_res:
                continue
            
            persona.append(old_res['person_description'])

            if conflict_type == 'T1':
                new_res = generate_attacker_mnew_t1(
                    person_description=old_res['person_description'],
                    scenario=old_res['scenario'],
                    old_info=old_res['old_info'],
                    attribute=old_res['attribute'],
                    dependency=old_res['dependency'],
                    client=gpt_client,
                    model_name=M_NEW_GEN_MODEL,
                    eval_client=gpt_client,
                    eval_model_name=GEN_EVAL_MODEL
                )
            elif conflict_type == 'T2':
                new_res = generate_attacker_mnew_t2(
                    person_description=old_res['person_description'],
                    scenario=old_res['scenario'],
                    old_info=old_res['old_info'],
                    attribute=old_res['attribute'],
                    dependency=old_res['dependency'],
                    client=gpt_client,
                    model_name=M_NEW_GEN_MODEL,
                    eval_client=gpt_client,
                    eval_model_name=GEN_EVAL_MODEL
                )
            else:
                raise ValueError("conflict_type must be 'T1' or 'T2'")

            if new_res and is_gen_valid(new_res):
                valid_items.append(new_res)
                
        final_valid_dataset.extend(valid_items)

    logging.info(f"\nConflict pair generation completed. Total valid items: {len(final_valid_dataset)}")
    return final_valid_dataset

def attr_list2str(attr):

    if not isinstance(attr, (list, tuple)):
        raise TypeError("attr must be list or tuple")
    if not all(isinstance(x, str) for x in attr):
        raise ValueError("all elements in attr must be strings")
    return ".".join(attr)

def s1_generate_q(data):

    for item in data:
        attr = attr_list2str(item['attribute'])
        result = generate_probing_queries(attr,item['old_info'],item['M_new'],item['explanation'],gpt_client,QA_ANS_GEN_MODEL)
        if result:
            item.update(result)
    
    logging.info(f"\nQuery generation completed.")
    
    return data

def s2_info2session(data):
    
    for item in data:
        item['S_old'] = info2session(item['old_info'],randint(0,MAX_ROUND_IN_SESSION//2))
        item['S_new'] = info2session(item['M_new'],randint(0,MAX_ROUND_IN_SESSION//2))
    
    logging.info(f"\nSession compilation completed.")

    return data

def build_final_dataset(a_data, out_evidence_path, out_icds_path):
    with open(NOISE_DATASET_DIR, 'r', encoding='utf-8') as f:
        lme_pool = json.load(f)

    with open(out_evidence_path, 'w', encoding='utf-8') as f_ev, \
         open(out_icds_path, 'w', encoding='utf-8') as f_ic:
             
        f_ev.write("[\n")
        f_ic.write("[\n")
        
        for index, item in enumerate(a_data):
            
            uid = str(uuid.uuid4())
            item['uid'] = uid
            
            json.dump(item, f_ev, ensure_ascii=False, indent=2)
            if index < len(a_data) - 1: f_ev.write(",\n")
            
            m_old_text = item.get("old_info", "")
            m_new_text = item.get("M_new", "")
            
            idx_old = randint(0, int(HAYSTACK_SESSION_AMOUNT * 0.3) - 1)
            idx_new = randint(int(HAYSTACK_SESSION_AMOUNT * 0.6), int(HAYSTACK_SESSION_AMOUNT * 0.8) - 1)
            
            timestamp_plan = build_timestamp_schedule_for_item(
                item=item,
                idx_old=idx_old,
                idx_new=idx_new,
                total_amount=HAYSTACK_SESSION_AMOUNT,
            )
            d_query = timestamp_plan["d_query"]
            timestamps = timestamp_plan["timestamps"]

            haystack = asyncio.run(build_haystack_for_item(item, lme_pool, idx_old, idx_new))
            
            icds_entry = {
                "uid": uid,
                "M_old": m_old_text,
                "M_new": m_new_text,
                "explanation": item.get("explanation", ""),
                "time_gap": item.get("time_gap", ""),
                "query_time": d_query.strftime("%Y-%m-%d %H:%M"),
                "probing_queries": {
                    "dim1_query": item.get("dim1_query", ""),
                    "dim2_query": item.get("dim2_query", ""),
                    "dim3_query": item.get("dim3_query", "")
                },
                "relevant_session_index": [idx_old, idx_new],
                "timestamps": timestamps,
                "haystack_session": haystack
            }
            
            json.dump(icds_entry, f_ic, ensure_ascii=False, indent=2)
            if index < len(a_data) - 1: f_ic.write(",\n")
            f_ic.flush()
            
        f_ev.write("\n]")
        f_ic.write("\n]")
    logging.info("\nAssembly Complete! Files saved safely.")

def generate_full_IC_dataset(conflict_type,seed_data,desired_dataset_name,output_dir=ROOT_DATASET_DIR):
    os.makedirs(output_dir, exist_ok=True)
    with_conflict_pair = augment_conflict_dataset(seed_data,conflict_type)
    with_queries = s1_generate_q(with_conflict_pair)
    into_sessions = s2_info2session(with_queries)
    with open(f'{output_dir}/{desired_dataset_name}_checkpoint.pkl','wb') as f:
        pkl.dump(into_sessions,f)
    build_final_dataset(into_sessions,out_icds_path=f'{output_dir}/{desired_dataset_name}_MAIN.json',out_evidence_path=f'{output_dir}/{desired_dataset_name}_EVID.json')

def split_list_into_chunks(lst, n_chunks):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be greater than 0")
    chunk_size = math.ceil(len(lst) / n_chunks)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def worker_task(conflict_type, seed_chunk, base_output_name, chunk_id, output_dir):
    chunk_output_name = f"{base_output_name}_part{chunk_id}"
    logging.info(f"\n[Process {chunk_id}] Started processing {len(seed_chunk)} seeds -> Output: {chunk_output_name}")
    generate_full_IC_dataset(conflict_type, seed_chunk, chunk_output_name, output_dir=output_dir)
    logging.info(f"\n[Process {chunk_id}] Finished!")

def merge_json_files(base_output_name, suffix, num_chunks, output_dir=ROOT_DATASET_DIR):
    merged_data = []
    final_filename = f"{output_dir}/{base_output_name}{suffix}.json"
    
    for i in range(num_chunks):
        part_filename = f"{output_dir}/{base_output_name}_part{i}{suffix}.json"
        if os.path.exists(part_filename):
            with open(part_filename, 'r', encoding='utf-8') as f:
                part_data = json.load(f)
                merged_data.extend(part_data)
        else:
            logging.warning(f"Warning: Expected part file not found: {part_filename}")

    with open(final_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    logging.info(f"\nMerged {len(merged_data)} items into {final_filename}")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate a STALE dataset.")
    parser.add_argument("--seed-file", required=True, help="Path to the input seed JSON file.")
    parser.add_argument("--output-name", required=True, help="Base name for generated output files.")
    parser.add_argument("--conflict-type", required=True, choices=["T1", "T2"], help="Conflict generation type.")
    parser.add_argument("--output-dir", default=ROOT_DATASET_DIR, help="Directory for generated files.")
    parser.add_argument("--num-workers", type=int, default=TASK_CONCURRENCY_LIMIT, help="Number of worker processes.")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.seed_file,'r',encoding='utf-8') as f:
        all_seeds = json.load(f)

    chunks = split_list_into_chunks(all_seeds, args.num_workers)

    processes = []
    for i, chunk in enumerate(chunks):
        p = Process(target=worker_task, args=(args.conflict_type, chunk, args.output_name, i, args.output_dir))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
        
    logging.info("\nAll processes completed! Starting aggregation...")

    merge_json_files(args.output_name, "_MAIN", len(chunks), output_dir=args.output_dir)
    merge_json_files(args.output_name, "_EVID", len(chunks), output_dir=args.output_dir)
    
    logging.info("\nFull pipeline executed and merged successfully!")

if __name__ == "__main__":
    main()
