# Скрипт для подсчета метрик неопределенности на наших текстах

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import argparse
from huggingface_hub import login
from google.colab import userdata
from google.colab import drive
import pandas as pd
import gc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.spatial.distance import mahalanobis

model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)


human_train = pd.read_csv('distance_train.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def calculate_entropy(logprobs):
    entropies = []
    for s_lp in logprobs:
        entropies.append([])
        for lp in s_lp:
            mask = ~np.isinf(lp)
            entropies[-1].append(-np.sum(np.array(lp[mask]) * np.exp(lp[mask])))
    return entropies

def calculate_perplexity(token_logprobs):
    N = len(token_logprobs)
    sum_logprobs = np.sum([np.max(logprobs) for logprobs in token_logprobs])
    avg_logprobs = sum_logprobs / N
    perplexity = np.exp(-avg_logprobs)
    return perplexity

def calculate_mc_sequence_entropy(model, tokenizer, text, num_samples=5):
    total_entropy = 0
    
    for _ in range(num_samples):
        with torch.no_grad():
            inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
        logits, _ = torch.topk(logits, k=512, dim=-1)
        probs = F.softmax(logits, dim=-1)

        log_probs = np.array(torch.log(probs))

        total_entropy += np.sum([np.max(logprobs) for logprobs in log_probs])

        
    return -total_entropy / num_samples


def calculate_mahalanobis_distance(hidden_states, mean_vec, cov_matrix):
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distances = [mahalanobis(h, mean_vec, inv_cov_matrix) for h in hidden_states]
    return np.mean(distances)

def get_logits(text, model, tokenizer):
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True
            )
        
    logits = outputs.logits[0]
    hidden_states = outputs.hidden_states[-1].squeeze(0).mean(dim=0).cpu().numpy()
    logits, _ = torch.topk(logits, k=512, dim=-1)
    probs = F.softmax(logits, dim=-1)

    log_probs = torch.log(probs)

    return np.array(log_probs), hidden_states

def calculate_hidden_states_stats(train_data, model, tokenizer):
    all_hidden_states = []
    
    for text in tqdm(train_data['generation']):
        _, hidden_state = get_logits(text, model, tokenizer)
        all_hidden_states.append(hidden_state)
    
    hidden_states_array = np.array(all_hidden_states)
    mean_vec = np.mean(hidden_states_array, axis=0)
    cov_matrix = np.cov(hidden_states_array, rowvar=False)
    
    cov_matrix += 1e-6 * np.eye(cov_matrix.shape[0])
    
    return mean_vec, cov_matrix

def add_metrics_to_dataframe(balanced_data, model, tokenizer, mean_vec, cov_matrix):
    mean_entropies, perplexities = [], []
    mc_entropies, rmi_scores, md_scores = [], [], []

    for text in tqdm(balanced_data['generation']):
        log_probs, hidden_states = get_logits(text, model, tokenizer)
        mc_entropy = calculate_mc_sequence_entropy(model, tokenizer, text)
        md = calculate_mahalanobis_distance([hidden_states], mean_vec, cov_matrix)
        mc_entropies.append(mc_entropy)

        entropies = calculate_entropy(log_probs)
        mean_entropy = np.mean(entropies)
        perplexity = calculate_perplexity(log_probs)

        mean_entropies.append(mean_entropy)
        perplexities.append(perplexity)
        md_scores.append(md)

    balanced_data['mean_entropy'] = mean_entropies
    balanced_data['perplexity'] = perplexities
    balanced_data['mc_entropy'] = mc_entropies
    balanced_data['mahalanobis'] = md_scores

    return balanced_data


mean_vec, cov_matrix = calculate_hidden_states_stats(human_train, model, tokenizer)
test_data = add_metrics_to_dataframe(test, model, tokenizer, mean_vec, cov_matrix)
train_data = add_metrics_to_dataframe(train, model, tokenizer, mean_vec, cov_matrix)
test_data.to_csv('test_data.csv', index=False)
train_data.to_csv('train_data.csv', index=False)
