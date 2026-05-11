import numpy as np
import pandas as pd
from retriever import Retriever


class RetrieverEvaluator:

    def __init__(self, retriever):
        self.retriever = retriever

    def reciprocal_rank(self, retrieved_items, ground_truth):
        """
        Reciprocal Rank:
        RR = 1 / rank of first correct result
        """

        for rank, item in enumerate(retrieved_items, start=1):

            if item.strip().lower() == ground_truth.strip().lower():
                return 1.0 / rank

        return 0.0

    def hit_rate_at_k(self, retrieved_items, ground_truth, k):
        """
        Hit Rate @ K

        Returns:
            1 -> if correct result found in top-k
            0 -> otherwise
        """

        top_k = retrieved_items[:k]

        for item in top_k:

            if item.strip().lower() == ground_truth.strip().lower():
                return 1

        return 0

    def precision_at_k(self, retrieved_items, ground_truth, k):
        """
        Precision @ K

        Precision = relevant retrieved / total retrieved
        """

        top_k = retrieved_items[:k]

        relevant = 0

        for item in top_k:

            if item.strip().lower() == ground_truth.strip().lower():
                relevant += 1

        return relevant / k

    def average_precision(self, retrieved_items, ground_truth):
        """
        Average Precision (AP)

        Since we usually have 1 relevant document,
        AP simplifies nicely.
        """

        relevant_found = 0
        precision_sum = 0

        for rank, item in enumerate(retrieved_items, start=1):

            if item.strip().lower() == ground_truth.strip().lower():

                relevant_found += 1

                precision = relevant_found / rank

                precision_sum += precision

        if relevant_found == 0:
            return 0.0

        return precision_sum / relevant_found

    def dcg_at_k(self, retrieved_items, ground_truth, k):
        """
        DCG@K
        """

        dcg = 0.0

        for i, item in enumerate(retrieved_items[:k]):

            rel = 1 if item.strip().lower() == ground_truth.strip().lower() else 0

            dcg += rel / np.log2(i + 2)

        return dcg

    def ndcg_at_k(self, retrieved_items, ground_truth, k):
        """
        nDCG@K
        """

        dcg = self.dcg_at_k(retrieved_items, ground_truth, k)

        ideal_retrieved = [ground_truth]

        idcg = self.dcg_at_k(ideal_retrieved, ground_truth, 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def evaluate(self, test_df, k_values=[1, 3, 5]):

        metrics = {}

        # Store metric scores
        recall_scores = {k: [] for k in k_values}
        precision_scores = {k: [] for k in k_values}
        hit_scores = {k: [] for k in k_values}
        ndcg_scores = {k: [] for k in k_values}

        mrr_scores = []
        map_scores = []

        total = len(test_df)

        print("\nStarting Evaluation...\n")

        for idx, row in test_df.iterrows():

            query = row["question"]

            # IMPORTANT:
            # We compare questions because retriever
            # is retrieving based on question embeddings
            ground_truth = row["question"]

            results = self.retriever.search(
                query,
                top_k=max(k_values)
            )

            retrieved_questions = results["question"].tolist()

            # -----------------------------
            # Debug Printing
            # -----------------------------
            print("\n====================================")
            print(f"Query: {query}")

            print("\nRetrieved Questions:")

            for rank, q in enumerate(retrieved_questions, start=1):

                print(f"{rank}. {q}")

            # -----------------------------
            # Metrics @ K
            # -----------------------------
            for k in k_values:

                # Recall@K
                hit = self.hit_rate_at_k(
                    retrieved_questions,
                    ground_truth,
                    k
                )

                recall_scores[k].append(hit)

                # HitRate@K
                hit_scores[k].append(hit)

                # Precision@K
                precision = self.precision_at_k(
                    retrieved_questions,
                    ground_truth,
                    k
                )

                precision_scores[k].append(precision)

                # nDCG@K
                ndcg = self.ndcg_at_k(
                    retrieved_questions,
                    ground_truth,
                    k
                )

                ndcg_scores[k].append(ndcg)

            # -----------------------------
            # MRR
            # -----------------------------
            rr = self.reciprocal_rank(
                retrieved_questions,
                ground_truth
            )

            mrr_scores.append(rr)

            # -----------------------------
            # MAP
            # -----------------------------
            ap = self.average_precision(
                retrieved_questions,
                ground_truth
            )

            map_scores.append(ap)

            print(f"\nProcessed {idx + 1}/{total}")

        # ==================================
        # Final Aggregated Metrics
        # ==================================

        for k in k_values:

            metrics[f"Recall@{k}"] = np.mean(recall_scores[k])

            metrics[f"Precision@{k}"] = np.mean(precision_scores[k])

            metrics[f"HitRate@{k}"] = np.mean(hit_scores[k])

            metrics[f"nDCG@{k}"] = np.mean(ndcg_scores[k])

        metrics["MRR"] = np.mean(mrr_scores)

        metrics["MAP"] = np.mean(map_scores)

        return metrics


if __name__ == "__main__":

    print("Loading Retriever...\n")

    retriever = Retriever()

    print("\nLoading Test Dataset...\n")

    # Your test CSV
    # Must contain:
    # question, answer
    test_df = pd.read_csv("test_dataset.csv")

    print(f"Loaded {len(test_df)} test samples")

    evaluator = RetrieverEvaluator(retriever)

    metrics = evaluator.evaluate(test_df)

    # ==================================
    # Final Results
    # ==================================

    print("\n\n========== FINAL RESULTS ==========\n")

    for metric, value in metrics.items():

        print(f"{metric}: {value:.4f}")