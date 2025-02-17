from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

# Authenticate with Hugging Face
login(token="your_huggingface_token")

# Load DeepSeek-R1-Distill-Llama-70B
print("Loading DeepSeek-R1-Distill-Llama-70B...")
tokenizer_70b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", use_auth_token=True)
model_70b = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", use_auth_token=True)
print("DeepSeek-R1-Distill-Llama-70B loaded successfully!")

# Load DeepSeek-R1-Distill-Qwen-14B
print("Loading DeepSeek-R1-Distill-Qwen-14B...")
tokenizer_14b = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", use_auth_token=True)
model_14b = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", use_auth_token=True)
print("DeepSeek-R1-Distill-Qwen-14B loaded successfully!")
