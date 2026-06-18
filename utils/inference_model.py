import torch
import transformers

import time
import requests
import logging

# Add to utils/inference_model.py
import os
USE_ONLINE = True  
API_KEY = os.getenv("API_KEY")
CALL_INTERVAL = 0  # Call interval (seconds), increase to 35s to avoid rate limiting
last_call_time = 0  # Record last call time

# Global variables for model caching
INFERENCE_LOCAL = True  # Set to True to enable local mode (for value evaluation)
INFERENCE_MODEL_DIR = "/media/m811/1.6T/m811/models/Qwen2.5-7B-Instruct"

ONLINE_API_KEY = ""  # Add API key
ONLINE_MODEL_NAME = ""  # Online model name

global_pipline = None


def initialize_model():
    global global_pipline
    if global_pipline is not None:
        return True

    if INFERENCE_LOCAL:
        
        try:
            from transformers import BitsAndBytesConfig
            # Use 8-bit quantization to reduce VRAM usage
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            pipeline = transformers.pipeline(
                "text-generation",
                model=INFERENCE_MODEL_DIR,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            global_pipline = pipeline
            print("Local model initialized successfully (8-bit quantized)")
            return True
        except Exception as e:
            print(f"8-bit init failed: {e}, fallback to bfloat16")
            try:
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=INFERENCE_MODEL_DIR,
                    torch_dtype=torch.bfloat16,
                    device_map="balanced_low_0",
                )
                global_pipline = pipeline
                print("Local model initialized successfully")
                return True
            except Exception as e2:
                print(f"Error during local model initialization: {str(e2)}")
                return False
    else:
        # In online mode, no need to initialize local model
        print("Online mode enabled")
        global_pipline = "online_mode"  # Mark as online mode
        return True


# Add online API call function - specifically for sub-question decomposition and answering
def call_api_for_subq_and_answer(query, **kwargs):

    global last_call_time

    # Check API key
    if not API_KEY:
        raise ValueError("API_KEY environment variable not set")

    url = os.environ.get("BASE_URL", "") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    # Handle different types of query input
    if isinstance(query, str):
        messages = [{"role": "user", "content": query}]
    elif isinstance(query, list) and len(query) > 0:
        # Assume query is already in message list format
        messages = query
    else:
        raise ValueError(f"Unsupported query type: {type(query)}")

    data = {
        "model": "moonshot-v1-8k",
        "messages": messages,
        "temperature": kwargs.get("temperature", 0.7),
        "max_tokens": kwargs.get("max_new_tokens", 1024)
    }

    # Implement exponential backoff retry mechanism
    max_retries = 8
    base_delay = 5  # Initial delay 5 seconds
    for attempt in range(max_retries):
        try:
            # Control call frequency - check before each attempt
            current_time = time.time()
            time_since_last_call = current_time - last_call_time
            if time_since_last_call < CALL_INTERVAL:
                sleep_time = CALL_INTERVAL - time_since_last_call
                print(f"Waiting {sleep_time:.2f}s before API call to respect rate limit")
                time.sleep(sleep_time)

            response = requests.post(url, headers=headers, json=data, timeout=60)
            
            # Check if it's a rate limit error
            if response.status_code == 429:
                print(f"Received 429 rate limit error on attempt {attempt + 1}")
                try:
                    print(f" 429 response body: {response.text}")
                except Exception:
                    pass
                if attempt < max_retries - 1:  # If not the last attempt
                    # Exponential backoff delay, max 60 seconds
                    delay = min(base_delay * (2 ** attempt), 60)  # 5s, 10s, 20s, 40s, 60s...
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    print("Max retries reached for rate limit error")
                    return []
            
            response.raise_for_status()
            result = response.json()

            # Update last call time
            last_call_time = time.time()

            # Return unified format
            content = result["choices"][0]["message"]["content"]
            return [{"generated_text": [{"role": "assistant", "content": content}]}]

        except requests.exceptions.RequestException as e:
            if 'response' in locals() and response.status_code == 429:
                print(f"Rate limit error (429): {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                    continue
            else:
                print(f"Request error: {e}")
            last_call_time = time.time()
            if attempt >= max_retries - 1:  # Last attempt failed
                return []
        except Exception as e:
            print(f"Error calling API: {e}")
            last_call_time = time.time()
            if attempt >= max_retries - 1:  # Last attempt failed
                return []
    
    return []

def local_inference_model(
    query,
    max_length=2048,
    truncation=True,
    do_sample=False,
    max_new_tokens=1024,
    temperature=0.7,
    num_return_sequences=1,
):
    """Local inference using cached model."""
    global global_pipline
    assert global_pipline is not None, "Model not initialized"

    if INFERENCE_LOCAL:
        return get_local_response_llama(
            query,
            global_pipline,
            max_length=max_length,
            truncation=truncation,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
    else:
        return call_api_for_subq_and_answer(
            query,
            max_length=max_length,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )


def get_subq_and_answer_response(
    prompt,
    temperature=0.7,
    max_tokens=2048,
    seed=170,
    max_length=2048,
    truncation=True,
    do_sample=True,
    max_new_tokens=1024,
    num_return_sequences=1,
):
    """Response retrieval function specifically for sub-question decomposition and answering"""
    response = []
    cnt = 2
    while not response and cnt:
        if USE_ONLINE:
 
            response = call_api_for_subq_and_answer(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                max_length=max_length,
                truncation=truncation,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                num_return_sequences=num_return_sequences,
            )
        else:
            # Use local model for sub-question decomposition and answering
            response = local_inference_model(
                prompt,
                max_length=max_length,
                truncation=truncation,
                do_sample=do_sample,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
        cnt -= 1
    if not response:
        print("Failed to obtain subq and answer response")
        return []
    return response


def get_value_response(
    prompt,
    temperature=0.7,
    max_tokens=2048,
    seed=170,
    max_length=2048,
    truncation=True,
    do_sample=True,
    max_new_tokens=1024,
    num_return_sequences=1,
):
    """Response retrieval function specifically for value evaluation - always use local model"""
    response = []
    cnt = 2
    while not response and cnt:
        # Value evaluation always uses local model
        response = local_inference_model(
            prompt,
            max_length=max_length,
            truncation=truncation,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        cnt -= 1
    if not response:
        print("Failed to obtain value response")
        return []
    return response


def get_response(
    prompt,
    temperature=0.7,
    max_tokens=2048,
    seed=170,
    max_length=2048,
    truncation=True,
    do_sample=True,
    max_new_tokens=1024,
    num_return_sequences=1,
):
    """General response retrieval function - decide which model to use based on context"""
    # By default use sub-question decomposition and answering function (may be online model)
    return get_subq_and_answer_response(
        prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        max_length=max_length,
        truncation=truncation,
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        num_return_sequences=num_return_sequences,
    )


def get_local_response_llama(
    query,
    pipeline,
    max_length=2048,
    truncation=True,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=False,
    num_return_sequences=1,
):
    """
    Generate response using Llama model with flexible configuration.

    Args:
        query (str): Input query or prompt
        pipeline: HuggingFace pipeline
        max_length (int): Maximum sequence length
        truncation (bool): Whether to truncate long sequences
        max_new_tokens (int): Maximum new tokens to generate
        temperature (float): Sampling temperature
        do_sample (bool): Whether to use sampling
        num_return_sequences (int): Number of different sequences to generate

    Returns:
        List[str]: Generated responses
    """
    try:
        # Prepare input for model
        message = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": query},
        ]

        if USE_ONLINE:
            # Use online API model
            outputs = call_api_for_subq_and_answer(message,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temperature)
        else:
            # Original local model logic
            outputs = pipeline(
                message,
                max_length=max_length,
                truncation=truncation,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
            )

        # Process outputs
        responses = []
        for output in outputs:
            response = output["generated_text"][-1]["content"]
            if query in str(response):
                response = str(response).replace(query, "").strip()
            responses.append(response)

        return responses

    except Exception as e:
        print(f"Error generating response: {e}")
        return []


def cleanup_model():
    """Cleanup model resources."""
    global global_pipline
    if global_pipline is None:
        del global_pipline
        global_pipline = None
    torch.cuda.empty_cache()
    global_pipline = None
    print("Model resources cleaned up")

# Initialize the model
initialize_model()
