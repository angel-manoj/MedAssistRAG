from datasets import load_dataset

dataset = load_dataset("pubmed_qa", "pqa_labeled")

dataset.save_to_disk("../../data/raw/pubmedqa")