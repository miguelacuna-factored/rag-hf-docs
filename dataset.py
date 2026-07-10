from datasets import load_dataset
from dotenv import load_dotenv
from os import getenv

load_dotenv()
hf_token = getenv("HF_TOKEN")

# This single line downloads the data from that repository and loads it into memory
dataset = load_dataset("m-ric/huggingface_doc", split="train") # train or test or validation
