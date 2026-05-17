import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import requests
import time
import os


# ==================== Value Evaluation Forces Local Model ====================
# Consistent with inference_model.py but forces local mode
INFERENCE_LOCAL = True  # True = local mode (value evaluation)
API_KEY = os.getenv("API_KEY")
CALL_INTERVAL = 21  
last_call_time = 0

# Only need model path in local mode - use 0.5B small model to save VRAM
VALUE_MODEL_DIR = ""
global_value_model = None
global_tokenizer = None


def call_pi_for_probabilities(context, query):

    global last_call_time

    # Frequency control
    current_time = time.time()
    time_since_last_call = current_time - last_call_time
    if time_since_last_call < CALL_INTERVAL:
        time.sleep(CALL_INTERVAL - time_since_last_call)

    if not API_KEY:
        print("WARNING: API_KEY not set, returning dummy probabilities")
        return [-0.1] * len(query.split())  # Return dummy values to avoid crash

    url = " "
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Build evaluation prompt
    prompt = f"""Evaluate the plausibility of the following text (return only a score between 0 and 1)：

Context: {context}
Query: {query}

Reasoning Score (0-1-200):"""

    data = {
        "model": "moonshot-v1-8k",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()

        last_call_time = time.time()

        # Parse score
        score_text = result["choices"][0]["message"]["content"].strip()
        score = float(score_text) if score_text.replace('.', '').isdigit() else 0.5

        # Convert to log probability (simulated)
        return [-max(0.1, 1 - score)]  # Higher score, log prob closer to 0

    except Exception as e:
        print(f"Error calling API: {e}")
        last_call_time = time.time()
        return [-0.1]  # Return default value


def initialize_value_model():

    global global_value_model, global_tokenizer

    # ========== Ensure forced local mode for value evaluation ==========
    if not INFERENCE_LOCAL:
        print("Value model must use local mode, skipping online initialization")
        return True

    if global_value_model is not None and global_tokenizer is not None:
        return True

    try:
        # Load tokenizer
        print(f"Loading tokenizer from {VALUE_MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(VALUE_MODEL_DIR, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model - use standard method to load non-quantized model
        print(f"Loading model from {VALUE_MODEL_DIR}...")
        model = AutoModelForCausalLM.from_pretrained(
            VALUE_MODEL_DIR,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        global_value_model = model
        global_tokenizer = tokenizer
        print("Value model initialized successfully")
        return True

    except Exception as e:
        print(f"Error initializing value model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_value_model():

    global global_value_model, global_tokenizer

    if global_value_model is not None:
        del global_value_model
        global_value_model = None

    if global_tokenizer is not None:
        del global_tokenizer
        global_tokenizer = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Value model resources cleaned up")


def get_token_probabilities(text, idx, inputs=None):
    """Calculate token probabilities, value evaluation forces local model"""
    global global_value_model, global_tokenizer

    # ========== Value evaluation always uses local model ==========
    if not INFERENCE_LOCAL:
        print("Value model must use local mode, skipping online API")
        return []

    # Original local model logic
    if (global_value_model is None or global_tokenizer is None) and not initialize_value_model():
        return []

    try:
        # ... original local processing code remains unchanged ...
        if inputs is None:
            inputs = global_tokenizer(
                text, padding=True, truncation=True, max_length=512, return_tensors="pt"
            )

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()

        log_probs = []
        with torch.no_grad():
            outputs = global_value_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = outputs.logits[0]

            for pos in range(idx - 1, input_ids.shape[1] - 1):
                next_token_logits = logits[pos]
                log_probs_t = torch.log_softmax(next_token_logits, dim=-1)
                next_token_id = input_ids[0, pos + 1]
                log_prob = log_probs_t[next_token_id].item()
                log_probs.append(log_prob)

        return log_probs

    except Exception as e:
        print(f"Error calculating token probabilities: {str(e)}")
        return []


def get_query_token_probabilities(context, query):
    """Get token probabilities for query part, value evaluation forces local model"""
    global global_value_model, global_tokenizer

    # ========== Value evaluation always uses local model ==========
    if not INFERENCE_LOCAL:
        print("Value model must use local mode, skipping online API")
        return []

    # Original local model logic
    if (global_value_model is None or global_tokenizer is None) and not initialize_value_model():
        return []

    try:
        full_text = context + query
        
        # Limit max length to avoid VRAM overflow
        MAX_LENGTH = 256
        
        inputs = global_tokenizer(
            full_text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )

        context_tokens = global_tokenizer(
            context, padding=False, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
        )
        query_start_idx = min(context_tokens["input_ids"].shape[1], MAX_LENGTH - 10)

        return get_token_probabilities(full_text, query_start_idx, inputs)

    except Exception as e:
        print(f"Error in get_query_token_probabilities: {str(e)}")
        return []

# ========== Ensure no auto-initialization on import ==========
# Delete or comment out this line
# initialize_value_model()  # ❌ Delete this line