# Copyright 2023-2024 SGLang Team
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random

import requests


def gen_radix_tree(num_nodes=400, chunk_len=256):
    num0 = num_nodes // 2
    num1 = num_nodes - num0
    nodes = [{"input_ids": [37] * 117, "decode_len": 217}]
    for _ in range(num0):
        parent = random.choice(nodes)
        unique_len = random.randint(0, chunk_len)
        decode_len = random.randint(0, chunk_len)
        token_id = random.randint(0, 32000)
        child = {
            "input_ids": parent["input_ids"] + [token_id] * unique_len,
            "decode_len": decode_len,
        }
        nodes.append(child)

    while num1 > 0:
        num_branch = random.randint(1, min(num1, 10))
        parent = random.choice(nodes)
        for _ in range(num_branch):
            unique_len = random.randint(0, chunk_len)
            decode_len = random.randint(0, chunk_len)
            token_id = random.randint(0, 32000)
            child = {
                "input_ids": parent["input_ids"] + [token_id] * unique_len,
                "decode_len": decode_len,
            }
            nodes.append(child)

        num1 -= num_branch

    random.shuffle(nodes)
    return nodes


def run_radix_attention_test(base_url: str):
    nodes = gen_radix_tree()
    data = {
        "input_ids": [node["input_ids"] for node in nodes],
        "sampling_params": [
            {"max_new_tokens": node["decode_len"], "temperature": 0} for node in nodes
        ],
    }

    res = requests.post(base_url + "/generate", json=data)
    assert res.status_code == 200
