"""Raw dataset loading only.

This module owns the one call that pulls the Hugging Face docs dataset down
from the Hub. Every other stage of the pipeline gets rows through
`load_raw_dataset` instead of calling `load_dataset` itself, so there is a
single place to change if the dataset name/split/token handling ever changes.
"""

from datasets import Dataset, load_dataset
from dotenv import load_dotenv


def load_raw_dataset() -> Dataset:
    """Load the Hugging Face docs dataset (train split) from the Hub."""
    load_dotenv()
    return load_dataset("m-ric/huggingface_doc", split="train")
