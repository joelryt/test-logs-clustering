import string
import re
import os
from ast import literal_eval
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models import Word2Vec


def preprocess_text(text: str) -> list:
    """Preprocesses text by removing raw newline characters,
    punctuation marks, digits and stopwords,
    and converting all characters to lowercase.
    Underscores (_) are left in the text.

    :param text: string that will be preprocessed
    :return: list of preprocessed words/tokens
    """
    all_stopwords = stopwords.words("english")
    remove_new_lines = re.compile(r"\\n")
    text = remove_new_lines.sub("", text)
    removed_chars = string.punctuation + string.digits
    removed_chars = removed_chars.replace("_", "")
    clean_chars = [char.lower() for char in text if char not in removed_chars]
    cleaned_string = "".join(clean_chars)
    cleaned_tokens = [token for token in cleaned_string.split() if token not in all_stopwords]
    return cleaned_tokens


def preprocess_text_stem(text: str, stemmer: PorterStemmer) -> str:
    """Preprocesses text using stemming.

    :param text: string that will be preprocessed
    :param stemmer: stemmer to be used for stemming
    :return: preprocessed text as a string
    """
    if not isinstance(text, str):
        return None
    cleaned_tokens = preprocess_text(text)
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]
    stemmed_text = " ".join(stemmed_tokens)
    return stemmed_text


def preprocess_log_line(log_line: str, method: int = 1, stemmer=None) -> str:
    """Preprocesses a log line.

    :param log_line: line of log that will be preprocessed
    :param method: which preprocessing method is used for log line preprocessing, 1 or 2
        method 1: basic text preprocessing with stemming, underscores (_) are left in the logs
        method 2: text preprocessing adjusted for log messages:
            - no stopword removal.
            - removes punctuation marks, except for '_', '/', ':', '-' and '.' marks
            - other special characters, extra whitespaces and newline characters are removed
            - all characters are converted to lowercase
            - no lemmatization or stemming
    :param stemmer: stemmer to be used for stemming if using method 1
    :return: preprocessed log line as a string
    """
    if method == 1 and not stemmer:
        raise TypeError("Stemmer cannot be None if preprocessing method 1 is used.")
    if not isinstance(log_line, str):
        return log_line

    if method == 1:
        return preprocess_text_stem(log_line, stemmer)
    if method == 2:
        regex = r"(?u)\b\w[\w|:|/|-|.]+\b"
        log_line = log_line.lower()
        cleaned_tokens = re.findall(regex, log_line)
        cleaned_log_line = " ".join(cleaned_tokens)
        return cleaned_log_line
    raise ValueError("Invalid method specified, must be 1 or 2.")


def get_test_case_id_from_filename(filename: str, split_string: str) -> str:
    """Parses test case id from log filename.

    :param filename: filename or full path of the file to be parsed
    :param split_string: string that will be used to split the filename,
        the first part before the specified string will be considered as the test case id
    :return: test case id as a string
    """
    filename = os.path.split(filename)[-1]
    test_case_id = filename.split(split_string)[0]
    return test_case_id


def concatenate_logs(logs: list) -> str:
    """Concatenates a list of logs to one string.

    :param logs: list of logs to concatenate
    :return: concatenated logs as a string or None if there's nothing to concatenate
    """
    concatenated = ""
    for log in logs:
        if not log:
            continue
        if len(concatenated) == 0:
            concatenated += log
        else:
            concatenated += " " + log
    return concatenated if concatenated else None


def preprocess_logs(directory: str, df: pd.DataFrame = None, method: int = 1, stemmer=None) -> pd.DataFrame:
    """Preprocesses all the logs inside the directory.
    Returns the results as a dataframe.

    :param directory: directory from which all .parquet files are processed
    :param df: dataframe where the preprocessing results will be appended instead
        of creating a new one, optional
    :param method: which preprocessing method is used for log line preprocessing, 1 or 2
        method 1: basic text preprocessing with stemming
        method 2: text preprocessing adjusted for log messages without stemming
    :param stemmer: stemmer to be used in the preprocessing if using preprocessing method 1
    :return: dataframe with the results
    """
    result = {
        "test_case_id": [],
        "result": [],
        "concatenated_raw_log": [],
        "concatenated_preprocessed_log_message": [],
        "all_logging_entities": [],
    }

    if df is not None:
        try:
            result["test_case_id"] = df["test_case_id"].tolist()
            result["result"] = df["result"].tolist()
            result["concatenated_raw_log"] = df["concatenated_raw_log"].tolist()
            result["concatenated_preprocessed_log_message"] = df["concatenated_preprocessed_log_message"].tolist()
            result["all_logging_entities"] = df["all_logging_entities"].tolist()
        except KeyError as error:
            print(f"Cannot append to given dataframe. KeyError: {error}")
            return df

    print(f"Preprocessing files from {directory}...")
    num_of_files = len([name for name in os.listdir(directory) if name.endswith(".parquet")])
    with tqdm(total=num_of_files) as pbar:
        for filename in os.scandir(directory):
            if filename.path.endswith(".parquet"):
                log_df = pd.read_parquet(filename.path)
                if len(log_df) > 0 and "resultId" in log_df.columns:
                    test_case_id = log_df.iloc[0]["resultId"]
                else:
                    split_string = [
                        string for string in ["_bts_log_collector", "_ContainerLogs"] if string in filename.path
                    ][0]
                    test_case_id = get_test_case_id_from_filename(filename.path, split_string)
                if test_case_id not in result["test_case_id"]:
                    result["test_case_id"].append(test_case_id)
                    result["result"].append("FAILED" if "failed_cases" in directory else "PASSED")

                    result["concatenated_raw_log"].append(concatenate_logs(log_df["raw_log"].tolist()))
                    result["concatenated_preprocessed_log_message"].append(
                        concatenate_logs(
                            log_df["parsed_log_line"].apply(lambda x: preprocess_log_line(x, method, stemmer))
                        )
                    )
                    result["all_logging_entities"].append(concatenate_logs(log_df["logging_entity"].tolist()))
                pbar.update()
    return pd.DataFrame(result)


def get_weighted_word_embedding_vector(string: str, word2vec_model: Word2Vec, tfidf_model: TfidfVectorizer) -> np.array:
    """Calculates TF-IDF weighted Word2Vec word embedding vector for a string.

    :param string: string for which the vector will be calculated for
    :param word2vec_model: trained Word2Vec model from which the word embeddings are fetched
    :param tfidf_model: trained TF-IDF model from which the TF-IDF weights are fetched
    :return: weighted word embedding vector as a numpy array
    """
    weighted_word_embeddings = []
    tfidf_weights = tfidf_model.transform([string]).toarray()[0]
    for token in word_tokenize(string):
        if token in tfidf_model.vocabulary_.keys() and token in word2vec_model.wv:
            word_embedding = word2vec_model.wv[token]
            weight = tfidf_weights[np.where(tfidf_model.get_feature_names_out() == token)[0][0]]
            weighted_word_embeddings.append(word_embedding * weight)
    if weighted_word_embeddings:
        vector = np.asarray(weighted_word_embeddings, dtype=np.float32).sum(axis=0)
    else:
        vector = np.zeros(word2vec_model.vector_size, dtype=np.float32)
    return vector


def get_line_idf_features(
    directory: str,
    template_miner,
    template_miner_exclude_templates=None,
) -> pd.DataFrame:
    """Creates feature vectors based on log event templates for all parquet
    files inside the directory.

    :param directory: directory containing the log files that will be processed
    :param template_miner: trained drain3 template miner that contains the different
        log event templates that should be found from the log files
    :param template_miner_exclude_templates: trained drain3 template miner that contains
        log event templates that should be excluded from the feature vectors
    :return: dataframe that contains the feature vectors for each log file on 'feature_vector' column
        and corresponding result ids on 'resultId' column
    """

    def _match_templates(df: pd.DataFrame, vector_length) -> np.array:
        """Matches log lines with event templates and creates a feature vector for the log.
        Also collects the corresponding logging entities for the matched log lines.

        :param df: pandas dataframe that contains the log file, the processed log line
            should be in 'parsed_log_line' column
        :param vector_length: length of the feature vector
        :return: returns a tuple containing a feature vector based on which log event templates
            are found from the log and a list of the corresponding logging entities and log messages
        """
        nonlocal template_miner
        nonlocal template_miner_exclude_templates
        feature_vector = np.zeros(vector_length)
        entities = []
        log_messages = []
        for line, entity in zip(df["parsed_log_line"], df["logging_entity"]):
            template_match = template_miner.match(line)
            excluded_template_match = (
                template_miner_exclude_templates.match(line) if template_miner_exclude_templates else None
            )
            if template_match and not excluded_template_match:
                log_messages.append(line)
                cluster_id = template_match.cluster_id
                feature_vector[cluster_id - 1] = 1
                entities.append(entity)
        return feature_vector, entities, log_messages

    num_of_files = len([name for name in os.listdir(directory) if name.endswith(".parquet")])
    result_ids = []
    feature_vectors = []
    template_logging_entities = []
    log_messages = []
    vector_length = len(template_miner.drain.clusters)
    with tqdm(total=num_of_files) as pbar:
        for filename in os.scandir(directory):
            if filename.path.endswith(".parquet"):
                df = pd.read_parquet(filename.path)
                if len(df) > 0:
                    if "resultId" in df.columns:
                        result_ids.append(df.iloc[0]["resultId"])
                    else:
                        # Container logs do not have the result id as a column
                        result_ids.append(get_test_case_id_from_filename(filename.path, "_ContainerLogs"))
                    feature_vector, entities, messages = _match_templates(df, vector_length)
                    feature_vectors.append(feature_vector)
                    template_logging_entities.append(entities)
                    log_messages.append(messages)
                pbar.update()
    feature_vectors = np.asarray(feature_vectors)
    # Calculate line-IDF values
    feature_vectors = (
        feature_vectors * np.log(feature_vectors.shape[0] / (np.sum(feature_vectors, axis=0) + 1))
    ).tolist()  # +1 in denominator to avoid zero divisions
    return pd.DataFrame(
        {
            "resultId": result_ids,
            "feature_vector": feature_vectors,
            "log_template_logging_entities": template_logging_entities,
            "log_messages": log_messages,
        }
    )


def plot_silhouette_scores(
    feature_vectors: list,
    min_k: int = 1,
    max_k: int = 15,
    legend_labels: list = None,
    save_path: str = None,
):
    """Fits k-means clustering with number of clusters varying from min_k to max_k (default 1 to 15)
    and calculates and plots the silhouette score for each number of cluster.
    Silhouette score cannot be calculated for single cluster, so the minimum number of clusters
    is always two even if the parameter value is one.

    :param feature_vectors: list of array-like feature vectors that are used to fit the clustering algorithm
    :param min_k: minimum number of cluster to be used
    :param max_k: maximum number of clusters to be used
    :param legend_labels: list of legend labels for each feature vector
    :param save_path: file path where the resulting figure will be saved
    """
    plt.figure(figsize=(10, 6))  # figsize=(8, 5))
    if np.isscalar(feature_vectors[0]):
        feature_vectors = [feature_vectors]
    max_values = []
    max_inds = []
    for feature_vector in feature_vectors:
        metric_scores = []
        if min_k == 1:
            min_k = 2
        with tqdm(total=max_k + 1 - min_k) as pbar:
            for k in range(min_k, max_k + 1):
                kmeans = KMeans(n_clusters=k, n_init="auto", random_state=1)
                clusters = kmeans.fit_predict(feature_vector)
                metric_scores.append(silhouette_score(feature_vector, clusters, metric="euclidean"))
                pbar.update()
        max_values.append(np.max(metric_scores))
        max_inds.append(np.argmax(metric_scores) + min_k)
        plt.plot(range(min_k, max_k + 1), metric_scores)
    plt.legend(legend_labels)
    # Mark maximum silhouette scores with black circles
    plt.scatter(max_inds, max_values, marker="o", color="k")
    plt.xlabel("Number of clusters", fontsize=14)
    ylabel = "Silhouette score"
    plt.ylabel(ylabel, fontsize=14)
    if save_path:
        plt.savefig(save_path)
    plt.show()
    # Print number of clusters where max silhouette scores are achieved
    print("Number of clusters with max silhouette scores:", max_inds)


def parse_failed_verdicts(verification_detailed_reason: str) -> str:
    """Parses failed partial verdicts from all partial verdicts in the verification detailed reason.

    :param verification_detailed_reason: verification detailed reason as a string
    :return: returns a list of the failed partial verdicts as a string
        or the original verification detailed reason without parsing if it only contains a single failed verdict
    """
    reason = literal_eval(verification_detailed_reason)
    if isinstance(reason, list):
        failed_verdicts = []
        for verdict_rule in reason:
            for partial_verdict in verdict_rule["partial_verdicts"]:
                if partial_verdict["verdict"] == "FAILED":
                    failed_verdicts.append(partial_verdict)
        return str(failed_verdicts)
    return verification_detailed_reason
