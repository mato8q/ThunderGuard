import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

from config import model_paths
from config import path_harmful_test, path_harmless_test, path_harmful_calibration, path_harmless_calibration
from utils import load_model, load_ori_prompts, get_jailbreak_prompts
from utils import get_sentence_embeddings
from utils import interpret_difference_matrix
from utils import cosine_similarity



def find_critical_layer(embeddings1, embeddings2):
    '''
    Find the layer with the minimum average cosine similarity between the two sets of embeddings
    
    Args:
    - embeddings1: first set of embeddings
    - embeddings2: second set of embeddings
    - visualize: whether to visualize the results

    Returns:
    - cosine_similarities: list of average cosine similarities for each layer
    - seleced_layer_index: index of the selected layer
    '''
    num_layers = len(embeddings1)
    cosine_similarities = []
    seleced_layer_index = 0
    min_cosine = 1

    # if the number of embeddings in two sets are not equal, truncate the longer one.
    if len(embeddings1[0]) != len(embeddings2[0]):
        min_len = min(len(embeddings1[0]), len(embeddings2[0]))
        embeddings1 = [emb[:min_len] for emb in embeddings1]
        embeddings2 = [emb[:min_len] for emb in embeddings2]

    for layer_index in range(num_layers):
        layer_embeddings1 = torch.stack(embeddings1[layer_index])
        layer_embeddings2 = torch.stack(embeddings2[layer_index])
        
        layer_cosine = []

        # Calculate the cosine similarity between each pair of embeddings
        for emb1 in layer_embeddings1:
            for emb2 in layer_embeddings2:
                cos_sim = cosine_similarity(emb1, emb2)
                layer_cosine.append(cos_sim.item())
        
        # Calculate the average cosine similarity for the layer
        avg_cosine = sum(layer_cosine) / len(layer_cosine)
        cosine_similarities.append(avg_cosine)
        if avg_cosine < min_cosine:
            min_cosine = avg_cosine
            seleced_layer_index = layer_index

    return cosine_similarities, seleced_layer_index


def get_thershold(scores1, scores2):
    '''
    Get the optimal threshold for the given scores

    Args:
    - scores1: first set of scores (eg [0.2, 0.3, 0.4, 0.2, 0.5])
    - scores2: second set of scores (eg [0.8, 0.9, 0.4, 0.6, 1.0])

    Returns:
    - optimal_threshold: optimal threshold to distinguish the two sets of scores (eg 0.6)
    '''
    scores1 = np.array(scores1)
    scores2 = np.array(scores2)

    scores = np.concatenate((scores1, scores2))
    labels = np.array([1] * len(scores1) + [0] * len(scores2))
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Find the optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # If the optimal threshold is not finite, set it to the average of the min and max scores
    if not np.isfinite(optimal_threshold):
        optimal_threshold = (np.min(scores1) + np.max(scores2)) / 2

    return optimal_threshold


def find_optimal_threshold(model, tokenizer, calibration_embeddings1, calibration_embeddings2, base_calibration_embedding, calibration_vector):
    '''
    Find the optimal threshold to distinguish the two sets of embeddings

    Args:
    - model: model to get embeddings from
    - tokenizer: tokenizer to use for encoding prompts
    - calibration_embeddings1: first set of embeddings
    - calibration_embeddings2: second set of embeddings
    - calibration_embedding: base embedding to compare against
    - calibration_vector: base vector to compare against
    - layer_index: index of the layer to find the optimal threshold for

    Returns:
    - thershold: optimal threshold to distinguish the two sets of embeddings
    '''
    v_cs1 = []
    v_cs2 = []

    for embed1, embed2 in zip(
        calibration_embeddings1,
        calibration_embeddings2,
    ):
        v1, _ = interpret_difference_matrix(
            model,
            tokenizer,
            embed1,
            base_calibration_embedding,
            return_tokens=False,
        )
        v2, _ = interpret_difference_matrix(
            model,
            tokenizer,
            embed2,
            base_calibration_embedding,
            return_tokens=False,
        )
        v_cs1.append(cosine_similarity(v1, calibration_vector).item())
        v_cs2.append(cosine_similarity(v2, calibration_vector).item())

    thershold = get_thershold(v_cs1, v_cs2)

    return thershold


def detection_judge(model, tokenizer, embeddings1, calibration_embedding, calibration_vector, threshold):
    results = []
    for embed in embeddings1:
        vec, _ = interpret_difference_matrix(
            model,
            tokenizer,
            embed,
            calibration_embedding,
            return_tokens=False,
        )
        if cosine_similarity(vec, calibration_vector).item() >= threshold:
            results.append(1.0)
        else:
            results.append(0.0)
    return results


def detection(model_name, update_vectors=False):
    # Load model
    model, tokenizer = load_model(model_name, model_paths)

    # Load data
    # harmful_prompts, harmless_prompts = load_ori_prompts(path_harmful, path_harmless)
    _, harmless_prompts_test = load_ori_prompts(path_harmful_test, path_harmless_test)
    harmful_prompts_calibration, harmless_prompts_calibration = load_ori_prompts(path_harmful_calibration, path_harmless_calibration)
    jailbreaks = ["ijp", "gcg", "saa", "autodan", "pair", "drattack", "puzzler", "zulu", "base64"]
    jailbreak_prompts_calibration = get_jailbreak_prompts(model_name, jailbreaks, split="calibration")
    jailbreak_prompts_test = get_jailbreak_prompts(model_name, jailbreaks, split="test")

    # Remove for potential data leakage
    # # Get embdddings for prompts
    # print("Get embeddings for harmful and harmless prompts...")
    # harmful_embeddings = get_sentence_embeddings(harmful_prompts, model, model_name, tokenizer)
    # harmless_embeddings = get_sentence_embeddings(harmless_prompts, model, model_name, tokenizer)
    # # Mean embeddings for harmful and harmless prompts
    # mean_harmful_embedding = []
    # mean_harmless_embedding = []
    # for i in range(len(harmful_embeddings)):
    #     mean_harmful_embedding.append(torch.mean(torch.stack(harmful_embeddings[i]), dim=0))
    #     mean_harmless_embedding.append(torch.mean(torch.stack(harmless_embeddings[i]), dim=0))

    # Embeddings for calibration prompts
    print("Get embeddings for calibration prompts...")
    calibration_harmless_embeddings = get_sentence_embeddings(harmless_prompts_calibration, model, model_name, tokenizer)
    calibration_harmful_embeddings = get_sentence_embeddings(harmful_prompts_calibration, model, model_name, tokenizer)
    # Mean embeddings for harmful and harmless prompts
    mean_harmful_embedding = []
    mean_harmless_embedding = []
    for i in range(len(calibration_harmless_embeddings)):
        mean_harmful_embedding.append(torch.mean(torch.stack(calibration_harmful_embeddings[i]), dim=0))
        mean_harmless_embedding.append(torch.mean(torch.stack(calibration_harmless_embeddings[i]), dim=0))
    if update_vectors:
        # Save mean embeddings for harmful and harmless prompts when the first time to run this script
        torch.save(mean_harmful_embedding, './vectors/{}/mean_harmful_embedding.pt'.format(model_name))
        torch.save(mean_harmless_embedding, './vectors/{}/mean_harmless_embedding.pt'.format(model_name))

    calibration_gcg_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['gcg'], model, model_name, tokenizer)
    calibration_puzzler_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['puzzler'], model, model_name, tokenizer)
    calibration_saa_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['saa'], model, model_name, tokenizer)
    calibration_autodan_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['autodan'], model, model_name, tokenizer)
    calibration_drattack_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['drattack'], model, model_name, tokenizer)
    calibration_pair_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['pair'], model, model_name, tokenizer)
    calibration_ijp_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['ijp'], model, model_name, tokenizer)
    calibration_base64_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['base64'], model, model_name, tokenizer)
    calibration_zulu_embeddings = get_sentence_embeddings(jailbreak_prompts_calibration['zulu'], model, model_name, tokenizer)

    # Embeddings for test prompts
    print("Get embeddings for test prompts...")
    test_harmless_embeddings = get_sentence_embeddings(harmless_prompts_test, model, model_name, tokenizer)
    # test_harmful_embeddings = get_sentence_embeddings(harmful_prompts_test, model, model_name, tokenizer)

    test_gcg_embeddings = get_sentence_embeddings(jailbreak_prompts_test['gcg'], model, model_name, tokenizer)
    test_puzzler_embeddings = get_sentence_embeddings(jailbreak_prompts_test['puzzler'], model, model_name, tokenizer)
    test_saa_embeddings = get_sentence_embeddings(jailbreak_prompts_test['saa'], model, model_name, tokenizer)
    test_autodan_embeddings = get_sentence_embeddings(jailbreak_prompts_test['autodan'], model, model_name, tokenizer)
    test_drattack_embeddings = get_sentence_embeddings(jailbreak_prompts_test['drattack'], model, model_name, tokenizer)
    test_pair_embeddings = get_sentence_embeddings(jailbreak_prompts_test['pair'], model, model_name, tokenizer)
    test_ijp_embeddings = get_sentence_embeddings(jailbreak_prompts_test['ijp'], model, model_name, tokenizer)
    test_base64_embeddings = get_sentence_embeddings(jailbreak_prompts_test['base64'], model, model_name, tokenizer)
    test_zulu_embeddings = get_sentence_embeddings(jailbreak_prompts_test['zulu'], model, model_name, tokenizer)


    # Find the critical layers
    _, seleced_safety_layer_index = find_critical_layer(calibration_harmful_embeddings, calibration_harmless_embeddings)
    print("Selected layer index for toxic concept detection: {}".format(seleced_safety_layer_index))
    _, seleced_jailbreak_layer_index_gcg = find_critical_layer(calibration_gcg_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_puzzler = find_critical_layer(calibration_puzzler_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_saa = find_critical_layer(calibration_saa_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_autodan = find_critical_layer(calibration_autodan_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_drattack = find_critical_layer(calibration_drattack_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_pair = find_critical_layer(calibration_pair_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_ijp = find_critical_layer(calibration_ijp_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_base64 = find_critical_layer(calibration_base64_embeddings, calibration_harmful_embeddings)
    _, seleced_jailbreak_layer_index_zulu = find_critical_layer(calibration_zulu_embeddings, calibration_harmful_embeddings)
    print("Selected layer index for gcg jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_gcg))
    print("Selected layer index for puzzler jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_puzzler))
    print("Selected layer index for saa jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_saa))
    print("Selected layer index for autodan jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_autodan))
    print("Selected layer index for drattack jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_drattack))
    print("Selected layer index for pair jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_pair))
    print("Selected layer index for ijp jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_ijp))
    print("Selected layer index for base64 jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_base64))
    print("Selected layer index for zulu jailbreak concept detection: {}".format(seleced_jailbreak_layer_index_zulu))
    

    # Get calibration vectors and thersholds
    print("Get calibration vectors and thersholds...")
    calibration_safety_vector, delta_safety = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_harmful_embeddings[seleced_safety_layer_index],
        calibration_harmless_embeddings[seleced_safety_layer_index],
        return_tokens=False,
    )

    calibration_jailbreak_vector_gcg, delta_jailbreak_gcg = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_gcg_embeddings[seleced_jailbreak_layer_index_gcg],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_gcg],
        return_tokens=False,
    )
    delta_jailbreak_gcg = delta_jailbreak_gcg * -1

    calibration_jailbreak_vector_puzzler, delta_jailbreak_puzzler = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_puzzler_embeddings[seleced_jailbreak_layer_index_puzzler],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_puzzler],
        return_tokens=False,
    )
    delta_jailbreak_puzzler = delta_jailbreak_puzzler * -1

    calibration_jailbreak_vector_saa, delta_jailbreak_saa = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_saa_embeddings[seleced_jailbreak_layer_index_saa],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_saa],
        return_tokens=False,
    )
    delta_jailbreak_saa = delta_jailbreak_saa * -1

    calibration_jailbreak_vector_autodan, delta_jailbreak_autodan = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_autodan_embeddings[seleced_jailbreak_layer_index_autodan],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_autodan],
        return_tokens=False,
    )
    delta_jailbreak_autodan = delta_jailbreak_autodan * -1

    calibration_jailbreak_vector_drattack, delta_jailbreak_drattack = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_drattack_embeddings[seleced_jailbreak_layer_index_drattack],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_drattack],
        return_tokens=False,
    )
    delta_jailbreak_drattack = delta_jailbreak_drattack * -1

    calibration_jailbreak_vector_pair, delta_jailbreak_pair = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_pair_embeddings[seleced_jailbreak_layer_index_pair],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_pair],
        return_tokens=False,
    )
    delta_jailbreak_pair = delta_jailbreak_pair * -1

    calibration_jailbreak_vector_ijp, delta_jailbreak_ijp = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_ijp_embeddings[seleced_jailbreak_layer_index_ijp],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_ijp],
        return_tokens=False,
    )
    delta_jailbreak_ijp = delta_jailbreak_ijp * -1

    calibration_jailbreak_vector_base64, delta_jailbreak_base64 = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_base64_embeddings[seleced_jailbreak_layer_index_base64],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_base64],
        return_tokens=False,
    )
    delta_jailbreak_base64 = delta_jailbreak_base64 * -1

    calibration_jailbreak_vector_zulu, delta_jailbreak_zulu = interpret_difference_matrix(
        model,
        tokenizer,
        calibration_zulu_embeddings[seleced_jailbreak_layer_index_zulu],
        calibration_harmful_embeddings[seleced_jailbreak_layer_index_zulu],
        return_tokens=False,
    )
    delta_jailbreak_zulu = delta_jailbreak_zulu * -1


    calibration_embeddings = [
        calibration_ijp_embeddings,
        calibration_gcg_embeddings,
        calibration_saa_embeddings,
        calibration_autodan_embeddings,
        calibration_pair_embeddings,
        calibration_drattack_embeddings,
        calibration_puzzler_embeddings,
        calibration_zulu_embeddings,
        calibration_base64_embeddings,
    ]
    test_embeddings = [
        test_ijp_embeddings,
        test_gcg_embeddings,
        test_saa_embeddings,
        test_autodan_embeddings,
        test_pair_embeddings,
        test_drattack_embeddings,
        test_puzzler_embeddings,
        test_zulu_embeddings,
        test_base64_embeddings,
    ]
    calibration_jailbreak_vectors = [
        calibration_jailbreak_vector_ijp,
        calibration_jailbreak_vector_gcg,
        calibration_jailbreak_vector_saa,
        calibration_jailbreak_vector_autodan,
        calibration_jailbreak_vector_pair,
        calibration_jailbreak_vector_drattack,
        calibration_jailbreak_vector_puzzler,
        calibration_jailbreak_vector_zulu,
        calibration_jailbreak_vector_base64,
    ]
    seleced_jailbreak_layer_indexs = [
        seleced_jailbreak_layer_index_ijp,
        seleced_jailbreak_layer_index_gcg,
        seleced_jailbreak_layer_index_saa,
        seleced_jailbreak_layer_index_autodan,
        seleced_jailbreak_layer_index_pair,
        seleced_jailbreak_layer_index_drattack,
        seleced_jailbreak_layer_index_puzzler,
        seleced_jailbreak_layer_index_zulu,
        seleced_jailbreak_layer_index_base64,
    ]

    # Evaluate the jailbreak detection
    print("Evaluate the jailbreak detection...")
    ## For transfer attack
    # for idx_calibration in tqdm(range(len(jailbreaks))):
    #     for idx_test in range(len(jailbreaks)):
    for idx_calibration in tqdm(range(len(jailbreaks))):
        idx_test = idx_calibration
        print("Calibration data from : ", jailbreaks[idx_calibration], "- Test data from : ", jailbreaks[idx_test])
        calibration_embedding = calibration_embeddings[idx_calibration]
        test_embedding = test_embeddings[idx_test]
        labels_jb = []
        labels_harmless = []
        # Find the optimal threshold for the jailbreak detection
        thershold_safety = find_optimal_threshold(
            model,
            tokenizer,
            calibration_embedding[seleced_safety_layer_index],
            calibration_harmless_embeddings[seleced_safety_layer_index],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
        )
        thershold_jailbreak = find_optimal_threshold(
            model,
            tokenizer,
            calibration_embedding[seleced_jailbreak_layer_indexs[idx_calibration]],
            calibration_harmful_embeddings[seleced_jailbreak_layer_indexs[idx_calibration]],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[idx_calibration]],
            calibration_jailbreak_vectors[idx_calibration],
        )
        if update_vectors:
            # Save thersholds for mitigation when the first time to run this script
            if idx_calibration == idx_test:
                torch.save(thershold_safety, './vectors/{}/thershold_safety_{}.pt'.format(model_name, jailbreaks[idx_calibration]))
                torch.save(thershold_jailbreak, './vectors/{}/thershold_jailbreak_{}.pt'.format(model_name, jailbreaks[idx_calibration]))
        # Detect the jailbreak prompts
        print("Num of test jailbreak prompts: ", len(test_embedding[seleced_safety_layer_index]))
        results_safety = detection_judge(
            model,
            tokenizer,
            test_embedding[seleced_safety_layer_index],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
            thershold_safety,
        )
        results_jailbreak = detection_judge(
            model,
            tokenizer,
            test_embedding[seleced_jailbreak_layer_indexs[idx_calibration]],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[idx_calibration]],
            calibration_jailbreak_vectors[idx_calibration],
            thershold_jailbreak,
        )
        # Detect the harmless prompts
        print("Num of test harmless prompts: ", len(test_harmless_embeddings[seleced_safety_layer_index][:len(test_embedding[seleced_safety_layer_index])]))
        results_harmless_safety = detection_judge(
            model,
            tokenizer,
            test_harmless_embeddings[seleced_safety_layer_index][:len(test_embedding[seleced_safety_layer_index])],
            mean_harmless_embedding[seleced_safety_layer_index],
            calibration_safety_vector,
            thershold_safety,
        )
        results_harmless_jailbreak = detection_judge(
            model,
            tokenizer,
            test_harmless_embeddings[seleced_jailbreak_layer_indexs[idx_calibration]][:len(test_embedding[seleced_jailbreak_layer_indexs[idx_calibration]])],
            mean_harmful_embedding[seleced_jailbreak_layer_indexs[idx_calibration]],
            calibration_jailbreak_vectors[idx_calibration],
            thershold_jailbreak,
        )
        # If result_safety and result_jailbreak are all 1.0, this prompt is judged as jailbreak
        for result_safety, result_jailbreak in zip(results_safety, results_jailbreak):
            if result_safety == 1.0 and result_jailbreak == 1.0:
                labels_jb.append(1.0)
            else:
                labels_jb.append(0.0)
        for result_safety, result_jailbreak in zip(results_harmless_safety, results_harmless_jailbreak):
            if result_safety == 1.0 and result_jailbreak == 1.0:
                labels_harmless.append(1.0)
            else:
                labels_harmless.append(0.0)
        tp = sum(labels_jb)
        fp = sum(labels_harmless)
        fn = len(labels_jb) - tp
        tn = len(labels_harmless) - fp
        if tp + fp != 0 and tp != 0:    
            accuracy = (tp + tn) / (len(labels_jb) + len(labels_harmless))
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f1 = 2 * precision * recall / (precision + recall)
        else:
            accuracy = (tp + tn) / (len(labels_jb) + len(labels_harmless))
            f1 = 0
        print("Accuracy: {}".format(accuracy), " | F1 score: {}".format(f1))

        if update_vectors:
            # Save vectors for mitigation when the first time to run this script
            layer_indexs = [seleced_safety_layer_index, seleced_jailbreak_layer_index_gcg, seleced_jailbreak_layer_index_puzzler, seleced_jailbreak_layer_index_saa, seleced_jailbreak_layer_index_autodan, seleced_jailbreak_layer_index_drattack, seleced_jailbreak_layer_index_pair, seleced_jailbreak_layer_index_ijp, seleced_jailbreak_layer_index_base64, seleced_jailbreak_layer_index_zulu]
            torch.save(layer_indexs, './vectors/{}/layer_indexs.pt'.format(model_name))    
        
            torch.save(delta_jailbreak_gcg, './vectors/{}/delta_jailbreak_gcg.pt'.format(model_name))
            torch.save(delta_jailbreak_puzzler, './vectors/{}/delta_jailbreak_puzzler.pt'.format(model_name))
            torch.save(delta_jailbreak_saa, './vectors/{}/delta_jailbreak_saa.pt'.format(model_name))
            torch.save(delta_jailbreak_autodan, './vectors/{}/delta_jailbreak_autodan.pt'.format(model_name))
            torch.save(delta_jailbreak_drattack, './vectors/{}/delta_jailbreak_drattack.pt'.format(model_name))
            torch.save(delta_jailbreak_pair, './vectors/{}/delta_jailbreak_pair.pt'.format(model_name))
            torch.save(delta_jailbreak_ijp, './vectors/{}/delta_jailbreak_ijp.pt'.format(model_name))
            torch.save(delta_jailbreak_base64, './vectors/{}/delta_jailbreak_base64.pt'.format(model_name))
            torch.save(delta_jailbreak_zulu, './vectors/{}/delta_jailbreak_zulu.pt'.format(model_name))
            torch.save(delta_safety, './vectors/{}/delta_safety.pt'.format(model_name))
        
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_gcg], './vectors/{}/calibration_harmful_embedding_gcg.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_puzzler], './vectors/{}/calibration_harmful_embedding_puzzler.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_saa], './vectors/{}/calibration_harmful_embedding_saa.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_autodan], './vectors/{}/calibration_harmful_embedding_autodan.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_drattack], './vectors/{}/calibration_harmful_embedding_drattack.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_pair], './vectors/{}/calibration_harmful_embedding_pair.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_ijp], './vectors/{}/calibration_harmful_embedding_ijp.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_base64], './vectors/{}/calibration_harmful_embedding_base64.pt'.format(model_name))
            torch.save(calibration_harmful_embeddings[seleced_jailbreak_layer_index_zulu], './vectors/{}/calibration_harmful_embedding_zulu.pt'.format(model_name))
            torch.save(calibration_harmless_embeddings[seleced_safety_layer_index], './vectors/{}/calibration_harmless_embedding.pt'.format(model_name))
        
            torch.save(calibration_safety_vector, './vectors/{}/calibration_safety_vector.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_gcg, './vectors/{}/calibration_jailbreak_vector_gcg.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_puzzler, './vectors/{}/calibration_jailbreak_vector_puzzler.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_saa, './vectors/{}/calibration_jailbreak_vector_saa.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_autodan, './vectors/{}/calibration_jailbreak_vector_autodan.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_drattack, './vectors/{}/calibration_jailbreak_vector_drattack.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_pair, './vectors/{}/calibration_jailbreak_vector_pair.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_ijp, './vectors/{}/calibration_jailbreak_vector_ijp.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_base64, './vectors/{}/calibration_jailbreak_vector_base64.pt'.format(model_name))
            torch.save(calibration_jailbreak_vector_zulu, './vectors/{}/calibration_jailbreak_vector_zulu.pt'.format(model_name))



if __name__ == '__main__':
    # Get parameters
    parser = argparse.ArgumentParser(description='JBShield-D')
    parser.add_argument('--model', type=str, help='Taregt model')

    args = parser.parse_args()
    model_name = args.model

    # Run this script to evaluate the detection performance of JBShield-D
    detection(model_name)

# An example for run this script to evaluate JBShield-D on the Mistral model
# python detection.py --model mistral
