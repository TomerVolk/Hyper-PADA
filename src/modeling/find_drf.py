#####################################################################################
# this code is partially taken from https://github.com/yftah89/PBLM-Domain-Adaptation
#####################################################################################
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mutual_info_score
import pandas as pd
import pickle
import nagisa
from nltk.stem import SnowballStemmer
from typing import Dict, List, Tuple


def get_top_nmi(x, target):
    mis = []
    length = x.shape[1]

    for i in range(length):
        temp = mutual_info_score(x[:, i], target)
        mis.append((temp, i))
    top_mi = sorted(mis, reverse=True)[:1000]
    return top_mi


def get_counts(x, i):
    return sum(x[:, i])


def find_drfs(base_model) -> Dict[str, List]:
    """find DRF from each source domain with mutual information
    Args:
        base_model: the model on which to rely for the tokenizer and read methods
    Returns:
    a dictionary whose keys are the domains and the values are the list of DRFs from that domain
   """
    h_params = base_model.h_params
    src = h_params.source_domains
    try:
        tgt = h_params.target_domain
        with open(f"{h_params.results_dir_path}/DRFs to {tgt}.pkl", "rb") as f:
            drfs = pickle.load(f)
        return drfs
    except FileNotFoundError:
        pass

    drfs = {}
    drf_data = pd.DataFrame()
    src_domains = src
    for domain in src_domains:
        cur_drfs, cur_drf_data = find_drfs_single_domain(base_model, domain)
        drfs[domain] = cur_drfs
        cur_drf_data["Source"] = [domain] * len(cur_drfs)
        drf_data = pd.concat([drf_data, cur_drf_data])
        pass
    tgt = h_params.target_domain
    drf_data.to_csv(f"{h_params.results_dir_path}/MI DRFs to {tgt}.csv")
    with open(f"{h_params.results_dir_path}/DRFs to {tgt}.pkl", "wb") as f:
        pickle.dump(drfs, f)
    return drfs


def check_stems(word: str, language: str) -> str:
    """
    removes words whose stems were already taken to increase variance in the DRF set
    Args:
        word: the current word to check
        language: the language in which the stemming would happen

    Returns: the stem

    """
    if language == "Japanese" or language is None:
        return word
    stemmer = SnowballStemmer(language.lower())
    stem = stemmer.stem(word)
    return stem


def find_drfs_single_domain(base_model, current_domain: str) -> Tuple[List[str], pd.DataFrame]:
    """find DRFs from a single source domain with mutual information
    Args:
        base_model: the model on which to rely for the tokenizer and read methods
        current_domain: the current domain from which we are looking to find DRFs
    Returns:
    the list of DRFs from that domain
   """
    h_params = base_model.h_params
    source_domains = h_params.source_domains
    print(f"Finding DRFs from {current_domain}")
    all_sentences, domain_labels, target_sentences, source_sentences = [], [], [], []
    lang_to_full = {"de": "German", "en": "English", "jp": "Japanese", "fr": "French"}
    if base_model.task_name == 'sentiment':
        language = lang_to_full[current_domain.split('-')[0]]
    elif 'sentiment language' in base_model.task_name:
        lang = base_model.task_name.split('language')[1].strip()
        language = lang_to_full[lang]
    else:
        language = "English"
    for domain in source_domains:
        if "sentiment" == base_model.task_name and domain.split("-")[0] != current_domain.split("-")[0]:
            # using the text from a single language only
            continue
        cur_file_path = h_params.data_dir + domain + f"/train.review"
        cur_sentences = base_model.read_single_file_txt(cur_file_path)
        if language == 'Japanese':
            # Japanese words aren't divided by spaces, and require a more complex split from nagisa
            for idx, sen in enumerate(cur_sentences):
                sen = nagisa.tagging(sen).words
                sen = " ".join(sen)
                cur_sentences[idx] = sen
        dom_label = int(domain == current_domain)
        domain_labels += [dom_label] * len(cur_sentences)
        all_sentences += cur_sentences
        if dom_label == 1:
            source_sentences += cur_sentences
        else:
            target_sentences += cur_sentences
    src_count = h_params.drf_src_count
    dest_count = h_params.drf_tgt_count
    stop_words = base_model.get_stop_words()

    vectorizer = CountVectorizer(min_df=5, binary=True)

    x_2_train = vectorizer.fit_transform(all_sentences).toarray()
    vectorizer_source = CountVectorizer(min_df=src_count, binary=True)
    x_2_train_source = vectorizer_source.fit_transform(source_sentences).toarray()
    vectorizer_rest = CountVectorizer(min_df=dest_count, binary=True)
    x_2_train_rest = vectorizer_rest.fit_transform(target_sentences).toarray()
    # get a sorted list of DRFs with respect to the MI with the label
    mi_sorted = get_top_nmi(x_2_train, domain_labels)

    drfs_tokens = []
    drfs_data = {}
    drf_stems = set()

    for mi, word_index in mi_sorted:
        word = vectorizer.get_feature_names()[word_index]
        if word.lower() == current_domain.lower() or word.lower() in stop_words:
            continue
        if len(word) < 3 or word.isnumeric() or (len(word) == 3 and any(char.isdigit() for char in word)):
            continue

        stem = check_stems(word=word, language=language)
        if stem in drf_stems:
            continue
        s_count = get_counts(x_2_train_source, vectorizer_source.get_feature_names().index(
            word)) if word in vectorizer_source.get_feature_names() else 0
        t_count = get_counts(x_2_train_rest, vectorizer_rest.get_feature_names().index(
            word)) if word in vectorizer_rest.get_feature_names() else 0

        if s_count > 0 and float(t_count) / s_count <= 1.5:
            if s_count > 0.5 * len(source_sentences):
                continue
            drfs_tokens.append(word)
            drf_stems.add(stem)
            drfs_data[word] = {"MI Domain": mi, "rest / cur domain ratio": float(t_count) / s_count}

        if len(drfs_data) >= h_params.num_drfs:
            break

    drfs_data = pd.DataFrame.from_dict(drfs_data, orient="index")
    return drfs_tokens, drfs_data
