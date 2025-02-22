import re
import json
import difflib
import os
from collections import defaultdict
from typing import List, Dict, Tuple

def str_similarity(str1, str2):
    seq = difflib.SequenceMatcher(None, str1, str2)
    return seq.ratio()

def find_most_similar_index(str_list, target_str):
    """
    Given a list of strings and a target string, returns the index of the most similar string in the list.
    """
    # Initialize variables to keep track of the most similar string and its index
    most_similar_str = None
    most_similar_index = None
    highest_similarity = 0
    
    # Iterate through each string in the list
    for i, str in enumerate(str_list):
        # Calculate the similarity between the current string and the target string
        similarity = str_similarity(str, target_str)
        
        # If the current string is more similar than the previous most similar string, update the variables
        if similarity >= highest_similarity:
            most_similar_str = str
            most_similar_index = i
            highest_similarity = similarity

    return most_similar_index

def match_choice(text,options):
    # For HuatuoGPT-o1
    if '## Final Response\n\n' in text:
        text = text.split('## Final Response\n\n')[-1]
    
    # for strict prompt 
    matches = list(re.finditer(r"(answer is\s*?)([A-N])", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first,ans_last],1

    # non strict
    match_options = 'ABCDEFGHIJKLMN'[:len(options)]
    matches = list(re.finditer(r"([\u4e00-\u9fff]|is |是|项|\*|\W|\ |\(|为|^|'|\"|#)(?![aA] )(["+match_options+r"])(\W|[\u4e00-\u9fff]|$)", text, re.S))
    if matches:
        ans_first = matches[0].group(2)
        ans_last = matches[-1].group(2)
        return [ans_first,ans_last],1

    text = text.lower()
    opsindex = [(opt,text.rindex(options[opt].lower())) for opt in options if options[opt].lower() in text]
    if len(opsindex) > 0:
        ans_last = sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0]
        opsindex = [(opt,text.index(options[opt].lower())) for opt in options if options[opt].lower() in text]
        ans_first = sorted(opsindex,key=lambda x:x[1],reverse=True)[0][0]
        return [ans_first,ans_last],2
    else:
        oplabels = [x for x in options]
        opans = [options[x].lower() for x in options]
        ansindex = find_most_similar_index(opans,text.lower())
        return [oplabels[ansindex],oplabels[ansindex]],3

def match(prediction, ground_truth):
    for gt in ground_truth:
        matchres = re.search(r"(\W|^)("+re.escape(gt)+r")(\W|$)",prediction.lower(),re.S)
        if matchres:
            return 1
    return 0



def score(data: List[Dict], ignore_miss: bool = False) -> Tuple[Dict, List[Dict], List[Dict]]:
    """
    Score the data
    """
    res: Dict = {}
    wrong_data: List[Dict] = []
    cor_data: List[Dict] = []
    for da in data:
        if 'source' not in da:
            da['source'] = 'unknown'
        if da['source'] not in res:
            res[da['source']] = [0,0,0,0]

        output = da['output']
        ans,ans_type = match_choice(output,da['options'])
        if ignore_miss and ans_type!= 1:
            continue

        da['ans'] = ans
        da['ans_type'] = ans_type

        if ans[0].lower() == da['answer_idx'].lower():
            res[da['source']][1] += 1
            cor_data.append(da)
        else:
            wrong_data.append(da)
        
        if ans[1].lower() == da['answer_idx'].lower():
            res[da['source']][3] += 1

        res[da['source']][2] += 1

    for k in res:
        head_match_score = res[k][1] / res[k][2]
        tail_match_score = res[k][3] / res[k][2]
        if head_match_score > tail_match_score:
            res[k][0] = head_match_score
        else:
            res[k][0] = tail_match_score

    return res,wrong_data,cor_data



def get_results(res_path):
    with open(res_path) as f:
        data: List[Dict] = json.load(f) 

    res, wrong_data, cor_data =  score(data)
    
    # Logging
    print(f"*{os.path.basename(res_path)}*")
    print(json.dumps(res,indent=4))

    # save results
    with open('result_' + os.path.basename(res_path),'w') as fw:
        json.dump(res,fw,ensure_ascii=False,indent=2)

# if __name__ == "__main__":
#     get_results('output_file_path')