"""
Alisha02 - Free Claude Code on HuggingFace Spaces
A simplified proxy that routes Claude API requests to free providers
"""

import os
import json
import Gradio
from collections.abc import AsyncGenerator
from typing import Union

# Get API keys from environment
HF_TOKEN = os.getenv("HF_TOKEN", "")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# Model configuration
MODELS = {
    "opus": os.getenv("MODEL_OPUS", "nvidia_nim/meta/llama-3.1-70b-instruct"),
    "sonnet": os.getenv("MODEL_SONNET", "nvidia_nim/meta/llama-3.1-8b-instruct"),
    "haiku": os.getenv("MODEL_HAIKU", "nvidia_nim/meta/llama-3.1-8b-instruct"),
    "default": os.getenv("MODEL", "nvidia_nim/meta/llama-3.1-8b-instruct"),
}

def get_model(model: str) -> str:
    """Get the model URL based on model name"""
    model_lower = model.lower() if model else "default"
    if "opus" in model_lower:
        return MODELS["opus"]
    elif "sonnet" in model_lower:
        return MODELS["sonnet"]
    elif "haiku" in model_lower:
        return MODELS["haiku"]
    return MODELS["default"]

async def call_nvidia_nim(messages: list, model: str) -> str:
    """Call NVIDIA NIM API"""
    if not NVIDIA_API_KEY:
        return "Error: NVIDIA_API_KEY not configured"
    
    import httpx
    model_url = get_model(model)
    
    # Convert messages to NVIDIA format
    nvim_messages = []
    for msg in messages:
        nvim_messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"https://integrate.api.nvidia.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {NVIDIA_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model_url,
                    "messages": nvim_messages,
                    "temperature": 0.7,
                    "max_tokens": 4096,
                    "stream": False
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"

async def call_openrouter(messages: list, model: str) -> str:
    """Call OpenRouter API"""
    if not OPENROUTER_API_KEY:
        return "Error: OPENROUTER_API_KEY not configured"
    
    import httpx
    model_url = get_model(model)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://huggingface.co/spaces/alish02",
                    "X-Title": "Alisha02"
                },
                json={
                    "model": model_url,
                    "messages": messages
                }
            )
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

async def call_huggingface_inference(messages: list, model: str = "meta-llama/Llama-3.1-70B-Instruct") -> str:
    """Call HuggingFace Inference API"""
    if not HF_TOKEN:
        return "Error: HF_TOKEN not configured"
    
    import httpx
    
    # Convert to HF format
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                json={"inputs": prompt, "parameters": {"max_new_tokens": 512}}
            )
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    return data[0].get("generated_text", "")
                return str(data)
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error: {str(e)}"

async def chat_completion(messages: list, provider: str = "nvidia", model: str = "") -> str:
    """Main chat completion function"""
    if provider == "nvidia":
        return await call_nvidia_nim(messages, model)
    elif provider == "openrouter":
        return await call_openrouter(messages, model)
    elif provider == "huggingface":
        return await call_huggingface_inference(messages, model)
    else:
        return "Error: Unknown provider"

# Gradio Interface
def generate_response(message: str, history: list, provider: str, model: str) -> tuple:
    """Generate response using Gradio interface"""
    # Build messages from history
    messages = [{"role": "system", "content": "You are Alisha, a helpful AI assistant."}]
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    
    import asyncio
    try:
        response = asyncio.run(chat_completion(messages, provider, model))
    except RuntimeError:
        # If already in async context, run directly
        import httpx
        loop = asyncio.get_event_loop()
        response = loop.run_until_complete(chat_completion(messages, provider, model))
    
    return response

# Create Gradio interface
with gr.Blocks(title="Alisha02") as demo:
    gr.Markdown("# 🤖 Alisha02 - Free AI Chat")
    
    with gr.Row():
        provider_dropdown = gr.Dropdown(
            label="Provider",
            choices=["nvidia", "openrouter", "huggingface"],
            value="huggingface"
        )
        model_dropdown = gr.Dropdown(
            label="Model",
            choices=[
                "meta-llama/Llama-3.1-70B-Instruct",
                "meta-llama/Llama-3.1-8B-Instruct", 
                "Qwen/Qwen2-72B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2"
            ],
            value="meta-llama/Llama-3.1-8B-Instruct"
        )
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Message", placeholder="Type your message...")
    
    with gr.Row():
        submit_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    def respond(message, history, provider, model):
        if not message:
            yield history, ""
            return
        
        response = generate_response(message, history, provider, model)
        history.append((message, response))
        yield history, ""
    
    submit_btn.click(respond, [msg, chatbot, provider_dropdown, model_dropdown], [chatbot, msg])
    msg.submit(respond, [msg, chatbot, provider_dropdown, model_dropdown], [chatbot, msg])
    clear_btn.click(lambda: (None, ""), outputs=[chatbot, msg])

# Demo app launch
app = demo

if __name__ == "__main__":
    demo.launch()