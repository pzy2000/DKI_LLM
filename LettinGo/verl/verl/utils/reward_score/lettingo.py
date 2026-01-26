import sys
from prompt_config import (
    RATING_SYSTEM_PROMPT,
    RATING_PREDICTOR_PROMPT,
)
import re
import requests, json
def call(system_prompt, user_prompt,seed=42):
    api = "http://0.0.0.0:8008/chat"

    payload = {
        "system": system_prompt,
        "user": user_prompt,
        "max_tokens": 20,
        "thinking": False,
        "seed":seed
    }

    res = requests.post(api, json=payload)
    text = res.json()["response"]
    return text

from typing import Dict, Optional

def extract_profile(completion_text: str) -> tuple[str, str]:
    # 先找最后一个<user_profile>，再找其后最近的</user_profile>
    user_profile = None

    user_start = completion_text.lower().rfind('<user_profile>')
    if user_start != -1:
        user_end = completion_text.lower().find('</user_profile>', user_start)
        if user_end != -1:
            user_profile = completion_text[user_start+14:user_end].strip()
        else:
            print("No closing </user_profile> tag found after last <user_profile>.")
    else:
        print("No <user_profile> tag found.")

    return user_profile

RATING_TAG_PATTERN = re.compile(
    r"<rating>\s*((?:[1-4]\.\d{2}|5\.00))\s*</rating>",
    re.DOTALL
)

def extract_rating(rating_text: str) -> tuple[float, bool]:
    matches = list(RATING_TAG_PATTERN.finditer(rating_text))
    if not matches:
        print("No valid <rating> tag found.")
        return 0.0, False

    tag_content = matches[-1].group(1).strip()
    try:
        return float(tag_content), True
    except Exception:
        return 0.0, False

def compute_reward(data_source, solution_str, ground_truth, extra_info=None):
            
    current_user_title = ground_truth["user_title"]
    current_item_title = ground_truth["item_title"]
    current_user_avg_rating = ground_truth["user_avg_rating"]
    current_item_avg_rating = ground_truth["item_avg_rating"]
    gt_rating = ground_truth["gt_rating"]
    item_description = ground_truth["item_description"]
    # print("========message info=========")
    # print(extra_info["messages"])
    # print("===========solution_str============")
    # print(solution_str)

    user_profile = extract_profile(solution_str)
   


    total_reward=0
    predicted_rating = -1 # means error

            
    # 计算格式奖励
    format_reward = 0.0
    if user_profile is None:
        print("Profile extraction failed or incomplete.....")
        format_reward = -1.0
        # 直接给负格式奖励，准确奖励为0
        total_reward = format_reward
        
    else:
        # 构造评分预测prompt
        rating_prompt = RATING_PREDICTOR_PROMPT.format(
            user_title=current_user_title,
            item_title=current_item_title,
            user_avg_rating=current_user_avg_rating,
            item_avg_rating=current_item_avg_rating,
            user_profile=user_profile,
            item_description=item_description
        )
        
        rating_text= call(
            RATING_SYSTEM_PROMPT,
            rating_prompt
        )


        predicted_rating, format_valid = extract_rating(rating_text)
          
        # 计算格式奖励
        if format_valid is False:
            format_reward = -1.0
            total_reward = format_reward
        else:
            # 计算准确奖励
            max_possible_reward = 1.0
            error = abs(predicted_rating - gt_rating)/4.0  # 归一化误差到[0,1]
            acc_reward = max_possible_reward - error
            
            # 总奖励 = 格式奖励 + 准确奖励
            total_reward = acc_reward + format_reward
    return {
        "score": total_reward,
        "solution_str": solution_str,
        "ground_truth": ground_truth,
        "predicted_rating": predicted_rating,
        "gt_rating": gt_rating
    }


