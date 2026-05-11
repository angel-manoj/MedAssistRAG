import argparse
from collections import Counter
from pathlib import Path
import re
import sys

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, total=None):
        return iterable


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_PATH = BASE_DIR / "data" / "qa_dataset_evaluation.csv"

RESULTS_PATH = BASE_DIR / "evaluation_results.csv"

PLOT_PATH = BASE_DIR / "evaluation_plot.png"

SIMILARITY_MODEL_NAME = "BAAI/bge-base-en-v1.5"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# =========================================================
# DATA LOADING
# =========================================================

def load_clean_data():

    df = pd.read_csv(DATA_PATH)

    return df.reset_index(drop=True)


def normalize(text):

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", "", text)

    return text.strip()


# =========================================================
# METRICS
# =========================================================

def compute_f1(pred, truth):

    pred_tokens = normalize(pred).split()
    truth_tokens = normalize(truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)

    num_same = sum(common.values())

    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(pred, truth):

    return int(normalize(pred) == normalize(truth))


def is_hallucinated(answer, context):

    answer_words = set(normalize(answer).split())
    context_words = set(normalize(context).split())

    overlap = len(answer_words & context_words)

    return int(overlap < 5)


def compute_bleu(pred, truth):

    from nltk.translate.bleu_score import sentence_bleu

    return sentence_bleu(
        [normalize(truth).split()],
        normalize(pred).split()
    )


# =========================================================
# SEMANTIC SIMILARITY
# =========================================================

def get_similarity_model():

    from sentence_transformers import SentenceTransformer

    if not hasattr(get_similarity_model, "_model"):

        print("\nLoading semantic similarity model...")

        get_similarity_model._model = SentenceTransformer(
            SIMILARITY_MODEL_NAME
        )

    return get_similarity_model._model


def semantic_similarity_batch(preds, truths):

    from sentence_transformers import util

    sim_model = get_similarity_model()

    emb1 = sim_model.encode(
        preds,
        convert_to_tensor=True
    )

    emb2 = sim_model.encode(
        truths,
        convert_to_tensor=True
    )

    scores = util.cos_sim(emb1, emb2)

    return [
        float(scores[i][i])
        for i in range(len(preds))
    ]


# =========================================================
# FULL RAG
# =========================================================

def build_rag_pipeline():

    from src.pipeline.rag_pipeline import RAGPipeline

    return RAGPipeline()


def evaluate_rag(sample_size=None, random_state=42):

    print("\n=================================================")
    print("Evaluating: FULL RAG")
    print("=================================================")

    rag = build_rag_pipeline()

    df = load_clean_data()

    if sample_size is not None:

        sample_n = min(sample_size, len(df))

        df = df.sample(
            n=sample_n,
            random_state=random_state
        ).reset_index(drop=True)

    preds = []
    truths = []
    contexts = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]

        ground_truth = row["answer"]

        result = rag.query(question)

        preds.append(result["answer"])

        truths.append(ground_truth)

        contexts.append(
            " ".join(result["sources"])
        )

    f1_scores = [
        compute_f1(p, t)
        for p, t in zip(preds, truths)
    ]

    em_scores = [
        exact_match(p, t)
        for p, t in zip(preds, truths)
    ]

    hallucinations = [
        is_hallucinated(p, c)
        for p, c in zip(preds, contexts)
    ]

    bleu_scores = [
        compute_bleu(p, t)
        for p, t in zip(preds, truths)
    ]

    similarities = semantic_similarity_batch(
        preds,
        truths
    )

    return {
        "Model": "Full RAG",
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Exact Match": sum(em_scores) / len(em_scores),
        "Hallucination Rate": sum(hallucinations) / len(hallucinations),
        "Semantic Similarity": sum(similarities) / len(similarities),
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
    }


# =========================================================
# GENERATOR ONLY
# =========================================================

generator_model = None


def generate_without_retrieval(question):

    global generator_model

    if generator_model is None:

        print("\n=================================================")
        print("Loading Generator ONLY model...")
        print("=================================================")

        from transformers import pipeline

        generator_model = pipeline(
            "text2text-generation",
            model="facebook/bart-large-cnn",
            device=0
        )

        print("\nGenerator model loaded!")

    prompt = f"""
    Answer the following healthcare question clearly and accurately.

    Question:
    {question}

    Answer:
    """

    result = generator_model(
        prompt,
        max_length=256,
        do_sample=False
    )

    return result[0]["generated_text"]


def evaluate_generator_only(sample_size=None, random_state=42):

    print("\n=================================================")
    print("Evaluating: GENERATOR ONLY")
    print("=================================================")

    df = load_clean_data()

    if sample_size is not None:

        sample_n = min(sample_size, len(df))

        df = df.sample(
            n=sample_n,
            random_state=random_state
        ).reset_index(drop=True)

    preds = []
    truths = []

    for _, row in tqdm(df.iterrows(), total=len(df)):

        question = row["question"]

        ground_truth = row["answer"]

        pred = generate_without_retrieval(question)

        preds.append(pred)

        truths.append(ground_truth)

    f1_scores = [
        compute_f1(p, t)
        for p, t in zip(preds, truths)
    ]

    em_scores = [
        exact_match(p, t)
        for p, t in zip(preds, truths)
    ]

    bleu_scores = [
        compute_bleu(p, t)
        for p, t in zip(preds, truths)
    ]

    similarities = semantic_similarity_batch(
        preds,
        truths
    )

    return {
        "Model": "Generator Only",
        "F1 Score": sum(f1_scores) / len(f1_scores),
        "Exact Match": sum(em_scores) / len(em_scores),
        "Hallucination Rate": None,
        "Semantic Similarity": sum(similarities) / len(similarities),
        "BLEU Score": sum(bleu_scores) / len(bleu_scores),
    }


# =========================================================
# PLOT RESULTS
# =========================================================

def plot_results(df):

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    metrics = [
        "F1 Score",
        "Exact Match",
        "Hallucination Rate",
        "Semantic Similarity",
        "BLEU Score",
    ]

    ax = df.set_index("Model")[metrics].plot(
        kind="bar"
    )

    ax.set_title("RAG vs Generator Only")

    ax.set_ylabel("Score")

    ax.set_xlabel("")

    plt.xticks(rotation=0)

    plt.tight_layout()

    plt.savefig(PLOT_PATH)

    plt.close()


# =========================================================
# SAVE RESULTS
# =========================================================

def save_results(results):

    df_results = pd.DataFrame(results)

    df_results.to_csv(
        RESULTS_PATH,
        index=False
    )

    plot_results(df_results)

    return df_results


# =========================================================
# MAIN
# =========================================================

def main(sample_size=None, random_state=42):

    #rag_result = evaluate_rag(
    #    sample_size=sample_size,
    #    random_state=random_state
    #)

    generator_result = evaluate_generator_only(
        sample_size=sample_size,
        random_state=random_state
    )

    df_results = save_results([
       # rag_result,
        generator_result
    ])

    print("\n=================================================")
    print("FINAL RESULTS")
    print("=================================================\n")

    print(df_results)

    print(f"\nResults saved to:")
    print(RESULTS_PATH)

    print(f"\nPlot saved to:")
    print(PLOT_PATH)

    return df_results


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Compare Full RAG vs Generator Only"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    main(
        sample_size=args.sample_size,
        random_state=args.random_state
    )