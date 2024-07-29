import argparse
import os
os.environ['RWKV_JIT_ON'] = '0'
os.environ['RWKV_T_MAX'] = '4096'
os.environ['RWKV_FLOAT_MODE'] = 'bf16'
precision = "bf16"
os.environ['RWKV_HEAD_SIZE_A'] = '64'
os.environ['RWKV_T_MAX'] = '4096'
os.environ["RWKV_MY_TESTING"]='x060'
os.environ['RWKV_CTXLEN'] = '4096'

from sentence_transformers.evaluation import  SimilarityFunction
import logging
from sklearn.metrics.pairwise import paired_cosine_distances, paired_euclidean_distances, paired_manhattan_distances
from scipy.stats import pearsonr, spearmanr
import numpy as np
from typing import List, Literal, Optional


logger = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader

class EmbeddingSimilarityEvaluator:
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.
    """

    def __init__(
        self,
        data_loader : DataLoader,
        main_similarity: SimilarityFunction = None,
        name: str = "",
        write_csv: bool = True,
        precision: Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]] = None,
    ):
        """
        Constructs an evaluator based for the dataset

        The labels need to indicate the similarity between the sentences.

        :param sentences1:  List with the first sentence in a pair
        :param sentences2: List with the second sentence in a pair
        :param scores: Similarity score between sentences1[i] and sentences2[i]
        :param write_csv: Write results to a CSV file
        :param precision: The precision to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or
            "ubinary". Defaults to None.
        """
        self.data_loader = data_loader
        self.write_csv = write_csv
        self.precision = precision


        self.main_similarity = main_similarity
        self.name = name


        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
            "cosine_pearson",
            "cosine_spearman",
            "euclidean_pearson",
            "euclidean_spearman",
            "manhattan_pearson",
            "manhattan_spearman",
            "dot_pearson",
            "dot_spearman",
        ]


    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("EmbeddingSimilarityEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)
        device = model.device
        #all matrix values
        all_pearson_cosine = []
        all_spearman_cosine = []
        all_pearson_manhattan = []
        all_spearman_manhattan = []
        all_pearson_euclidean = []
        all_spearman_euclidean = []
        all_pearson_dot = []
        all_spearman_dot = []
        for batch in self.data_loader:
            sentences1 = batch['sentence1'].to(device)
            sentences2 = batch['sentence2'].to(device)
            scores = batch['scores'].to(device)
            with torch.no_grad():
                embeddings1 = model.forward(sentences1).float()
                
                embeddings2 = model.forward(sentences2).float() 
            #convert embeddings1,embeddings2 from pytorch tensor to numpy array
            embeddings1 = embeddings1.cpu().numpy()
            embeddings2 = embeddings2.cpu().numpy()

            # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
            if self.precision == "binary":
                embeddings1 = (embeddings1 + 128).astype(np.uint8)
                embeddings2 = (embeddings2 + 128).astype(np.uint8)
            if self.precision in ("ubinary", "binary"):
                embeddings1 = np.unpackbits(embeddings1, axis=1)
                embeddings2 = np.unpackbits(embeddings2, axis=1)

            labels = scores.cpu().numpy()

            cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))
            manhattan_distances = -paired_manhattan_distances(embeddings1, embeddings2)
            euclidean_distances = -paired_euclidean_distances(embeddings1, embeddings2)
            dot_products = [np.dot(emb1, emb2) for emb1, emb2 in zip(embeddings1, embeddings2)]

            eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
            eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

            eval_pearson_manhattan, _ = pearsonr(labels, manhattan_distances)
            eval_spearman_manhattan, _ = spearmanr(labels, manhattan_distances)

            eval_pearson_euclidean, _ = pearsonr(labels, euclidean_distances)
            eval_spearman_euclidean, _ = spearmanr(labels, euclidean_distances)

            eval_pearson_dot, _ = pearsonr(labels, dot_products)
            eval_spearman_dot, _ = spearmanr(labels, dot_products)
            #append all values
            all_pearson_cosine.append(eval_pearson_cosine)
            all_spearman_cosine.append(eval_spearman_cosine)
            all_pearson_manhattan.append(eval_pearson_manhattan)
            all_spearman_manhattan.append(eval_spearman_manhattan)
            all_pearson_euclidean.append(eval_pearson_euclidean)
            all_spearman_euclidean.append(eval_spearman_euclidean)
            all_pearson_dot.append(eval_pearson_dot)
            all_spearman_dot.append(eval_spearman_dot)

            logger.info(
                "Cosine-Similarity :\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_cosine, eval_spearman_cosine)
            )
            logger.info(
                "Manhattan-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                    eval_pearson_manhattan, eval_spearman_manhattan
                )
            )
            logger.info(
                "Euclidean-Distance:\tPearson: {:.4f}\tSpearman: {:.4f}".format(
                    eval_pearson_euclidean, eval_spearman_euclidean
                )
            )
            logger.info(
                "Dot-Product-Similarity:\tPearson: {:.4f}\tSpearman: {:.4f}".format(eval_pearson_dot, eval_spearman_dot)
            )

        
        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if os.path.exists(output_path) is False:
                os.makedirs(output_path)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                        np.mean(all_pearson_cosine),
                        np.mean(all_spearman_cosine),
                        np.mean(all_pearson_euclidean),
                        np.mean(all_spearman_euclidean),
                        np.mean(all_pearson_manhattan),
                        np.mean(all_spearman_manhattan),
                        np.mean(all_pearson_dot),
                        np.mean(all_spearman_dot),
                    ]
                )


        if self.main_similarity == SimilarityFunction.COSINE:
            return np.mean(all_spearman_cosine)
        elif self.main_similarity == SimilarityFunction.EUCLIDEAN:
            return np.mean(all_spearman_euclidean)
        elif self.main_similarity == SimilarityFunction.MANHATTAN:
            return np.mean(all_spearman_manhattan)
        elif self.main_similarity == SimilarityFunction.DOT_PRODUCT:
            return np.mean(all_spearman_dot)
        elif self.main_similarity is None:
            return np.mean([all_spearman_cosine, all_spearman_manhattan, all_spearman_euclidean, all_spearman_dot])
        else:
            raise ValueError("Unknown main_similarity value")
