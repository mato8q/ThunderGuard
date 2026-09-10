"""
Configuration file for JBShield.
"""

# Data paths — ThunderGuard datasets (data_thunderguard/)
# Switch DATASET to run a different attack: zulu | base64 | pair | autodan | drattack
DATASET = "zulu"

path_harmful_test        = f"data_thunderguard/{DATASET}_harmful_test.csv"
path_harmless_test       = "data_thunderguard/xstest_harmless_test.csv"
path_harmful_calibration = "data_thunderguard/harmful_calibration.csv"
path_harmless_calibration = "data_thunderguard/harmless_calibration.csv"

# Original JBShield data (kept for reference)
path_harmful  = "data/harmful.csv"
path_harmless = "data/harmless.csv"

# Model paths — use HuggingFace Hub IDs (uses local cache if already downloaded)
model_paths = {
    "mistral":              "mistralai/Mistral-7B-Instruct-v0.2",
    "llama-2":              "meta-llama/Llama-2-7b-chat-hf",
    "vicuna-7b":            "lmsys/vicuna-7b-v1.5",
    "vicuna-13b":           "lmsys/vicuna-13b-v1.5",
    "llama-3":              "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistral-sorry-bench":  "./models/ft-mistral-7b-instruct-v0.2-sorry-bench-202406",
}