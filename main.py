import argparse
import os
os.environ["TRANSFORMERS_CACHE"] = "/home/yavuz/yavuz/.cache/huggingface/hub"
os.environ["OPENAI_API_KEY"] = "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"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3,4,5"
import pathlib
import pickle
from lib2to3.pgen2.tokenize import tokenize
import datasets
import evaluate
import numpy as np
import torch
from tqdm import tqdm
import wandb    
from transformers import AutoModelForCausalLM, AutoTokenizer,AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

import random
import csv
import copy
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers import SentenceTransformer 
import IPython

import numpy as np
from scipy.special import softmax

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import csv
from litellm import completion
import argparse
from datasets import load_dataset


def model_respond_decoder_only(model, tokenizer, question, context, problem_description, icl_example, mapping=None):
    inverse_mapping = {v: k for k, v in mapping.items()}

    
    prompt = DEFAULT_TEMPLATE.format(problem_description=problem_description, icl_example=icl_example, context=context, question=question, option_a=inverse_mapping['A'], option_b=inverse_mapping['B'])

    vocab = ['A', 'B']
    log_probs_dict = {}
    for i in range(2):
        input_text = prompt  + ' ' + vocab[i]
        input_ids = tokenizer.encode(input_text, return_tensors='pt', add_special_tokens=False).to(model.device)#think about the special tokens
        logits = model(input_ids).logits[0,:-1,:] # remove the last token  
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs = torch.gather(logprobs, dim=1, index = input_ids[:,1:].view(-1, 1))#logprobs for each token in the generated text
        
        log_probs_dict[vocab[i]] = logprobs.sum().item()
        #log_probs_dict[vocab[i]] = logprobs[-1].item()
      
        
    return log_probs_dict


def model_respond_encoder_decoder(model, tokenizer, question, context, problem_description, icl_example, mapping=None):

    inverse_mapping = {v: k for k, v in mapping.items()}
   
   
    prompt = DEFAULT_TEMPLATE.format(problem_description=problem_description, icl_example=icl_example, context=context, question=question, option_a=inverse_mapping['A'], option_b=inverse_mapping['B'])

    vocab = ['A', 'B']
    log_probs_dict = {}
    for i in range(2):
        input_text_encoder = prompt #icl example will be in the encoder
        input_text_decoder = ' ' + vocab[i] 
        input_ids_encoder = tokenizer.encode(input_text_encoder, return_tensors='pt', add_special_tokens=False).to(model.device)
        input_ids_decoder = tokenizer.encode(input_text_decoder, return_tensors='pt', add_special_tokens=True).to(model.device)

        input_ids_decoder = model._shift_right(input_ids_decoder)
     
        logits = model(input_ids_encoder, decoder_input_ids=input_ids_decoder).logits[0,:-1,:] # remove the last token  
        logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
        logprobs = torch.gather(logprobs, dim=1, index = input_ids_decoder[:,1:].view(-1, 1))#logprobs for each token in the generated text

        log_probs_dict[vocab[i]] = logprobs.sum().item()
        #log_probs_dict[vocab[i]] = logprobs[-1].item()
        
    return log_probs_dict

def model_respond_api_based(model, tokenizer, question, context, problem_description, icl_example, mapping=None):
    inverse_mapping = {v: k for k, v in mapping.items()}
    problem_description = "Aşağıdaki soruları cevapla. Sadece cevap olarak A veya B yazman lazım. "
    prompt = DEFAULT_TEMPLATE.format(problem_description=problem_description, icl_example=icl_example, context=context, question=question, option_a=inverse_mapping['A'], option_b=inverse_mapping['B'])

    vocab = ['A', 'B']

    response = completion(
            model=model,
            messages=[{"content": prompt, "role": "user"}],
            seed=args.seed,
            max_tokens=1
        )
    generated_text = response.choices[0].message['content']
    print(generated_text)
    if generated_text.strip().lower() == 'a.' or generated_text.strip().lower() == 'a':
        return {'A': 1.0, 'B': 0.0}
    else:
        return {'A': 0.0, 'B': 1.0}

   
    


def model_respond(model, tokenizer, question, context, problem_description, icl_example, mapping=None):
    if args.model_name in model_dict['encoder_decoder']:
        return model_respond_encoder_decoder(model, tokenizer, question, context, problem_description, icl_example, mapping)
    if args.model_name in model_dict['decoder_only']:
        return model_respond_decoder_only(model, tokenizer, question, context, problem_description, icl_example, mapping)
    if args.model_name in model_dict['api_based']:
        return model_respond_api_based(model, tokenizer, question, context, problem_description, icl_example, mapping)

def evaluate_model(model, tokenizer, dataset, problem_description, icl_example, option_order='random', eval_method='permutation'):
    total_a = 0
    total_b = 0
    correct = 0
    total = 0
    shifted_count = 0
    speaker_count = 0
    all_predictions = []
    for sample in tqdm(dataset):
        name_dict = random.choice(name_list)
        #mapping = {'Ahmet': 'A', 'Mehmet':'B'}
        speaker = name_dict['SPEAKER']
        other_person  = name_dict['NAMENull']
        if option_order == 'random':
            if np.random.rand() > 0.5:
                mapping = {speaker: 'A', other_person:'B'}
            else:
                mapping = {speaker: 'B', other_person:'A'}
        if option_order == 'AB':
            mapping = {speaker: 'A', other_person:'B'}
        if option_order == 'BA':
            mapping = {speaker: 'B', other_person:'A'}

        inverse_mapping = {v: k for k, v in mapping.items()}

        context = sample['context']
        question = sample['question']
        sentence = sample['sentence']

        context = context.replace("SPEAKER", speaker)
        context = context.replace("NAMENull", other_person)
        context = context.replace('NAME-acc', name_dict['NAME-acc'])
        context = context.replace('NAME-dat', name_dict['NAME-dat'])
        context = context.replace('NAME-gen', name_dict['NAME-gen'])
        context = context.replace('NAME-com', name_dict['NAME-com'])

        question = question.replace("SPEAKER", speaker)
        question = question.replace("NAMENull", other_person)
        question = question.replace('NAME-acc', name_dict['NAME-acc'])
        question = question.replace('NAME-dat', name_dict['NAME-dat'])
        question = question.replace('NAME-gen', name_dict['NAME-gen'])
        question = question.replace('NAME-com', name_dict['NAME-com'])  

        sentence = sentence.replace("SPEAKER", speaker)
        sentence = sentence.replace("NAMENull", other_person)
        sentence = sentence.replace('NAME-acc', name_dict['NAME-acc'])
        sentence =  sentence.replace('NAME-dat', name_dict['NAME-dat'])
        sentence =  sentence.replace('NAME-gen', name_dict['NAME-gen'])
        sentence =  sentence.replace('NAME-com', name_dict['NAME-com'])
        
        
        pred_stats = {}

        for key  in sample.keys():
            pred_stats[key] = sample[key]

        pred_stats.pop('Unnamed: 0')
        pred_stats['idx'] = sample['Unnamed: 0']

        pred_stats['gender'] = name_dict['Gender']
        pred_stats['speaker'] = name_dict['SPEAKER']
        pred_stats['other_person'] = name_dict['NAMENull']


        if sample['ground_truth'].strip() == 'speaker'.strip():
            ground_truth = speaker
        else:
            ground_truth = other_person

        if eval_method == 'fixed' or args.model_name in model_dict['api_based']:
            logprob_dict = model_respond(model, tokenizer, question, context, problem_description, icl_example, mapping=mapping)
            if logprob_dict['A'] > logprob_dict['B']:
                prediction = inverse_mapping['A']
            else:
                prediction = inverse_mapping['B']

        if eval_method == 'permutation' and args.model_name not in model_dict['api_based']:
            scores = { speaker: 0, other_person: 0}
            logprob_dict = model_respond(model, tokenizer, question, context, problem_description, icl_example, mapping=mapping)
            scores[inverse_mapping['A']] += logprob_dict['A']
            scores[inverse_mapping['B']] += logprob_dict['B']

            #change mapping
            new_mapping = {speaker: mapping[other_person], other_person: mapping[speaker]}
            new_inverse_mapping = {v: k for k, v in new_mapping.items()}
            logprob_dict = model_respond(model, tokenizer, question, context, problem_description, icl_example, mapping=new_mapping)
            scores[new_inverse_mapping['A']] += logprob_dict['A']
            scores[new_inverse_mapping['B']] += logprob_dict['B']


            if scores[speaker] > scores[other_person]:
                prediction = speaker
            else:
                prediction = other_person


        #prediction = model_respond_encoder_decoder(model, tokenizer, sample, problem_description, icl_example, mapping=mapping)
        #print(prediction)
        

        if mapping[prediction] == 'A':
            total_a += 1
        else:
            total_b += 1
        #print('prediction:', prediction, 'ground_truth:', ground_truth)
        if prediction == ground_truth:
            pred_stats['correct'] = True
            correct += 1
        else:
            pred_stats['correct'] = False
        
        if prediction == speaker:
            speaker_count += 1
            pred_stats['model_pred'] = 'speaker'
        else:
            shifted_count += 1
            pred_stats['model_pred'] = 'shifted'

        pred_stats['model_name'] = args.model_name
        total += 1
        all_predictions.append(pred_stats)

    return {'accuracy': correct/total, 'total_a': total_a, 'total_b': total_b, 'total_questions': total, 'predictions': all_predictions, 'shifted_count': shifted_count, 'speaker_count': speaker_count}

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for the given default values")
    
    # Adding arguments with default values as specified in the Args class
    parser.add_argument("--num_examples", type=int, default=5, help="Number of ICL examplaes")
    parser.add_argument("--option_order", type=str, default='random', help="Order of options processing")
    parser.add_argument("--eval_method", type=str, default='permutation', help="Evaluation method")
    parser.add_argument("--model_name", type=str, default='gpt-4', help="Name of the model to use")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument("--dataset", type=str, default='data.csv', help="Path to the dataset file")
    
    # Parsing arguments
    args = parser.parse_args()
    return args

# This will parse the command line arguments when the script is run
if __name__ == "__main__":
    args = get_args()
    print(args)
    model_dict = {}
    model_dict['encoder_decoder'] = ['CohereForAI/aya-101']
    model_dict['decoder_only'] = ['TURKCELL/Turkcell-LLM-7b-v1', 'Trendyol/Trendyol-LLM-7b-base-v0.1']
    model_dict['api_based'] = ['gpt-3.5-turbo', 'gpt-4','gpt-4-turbo']
    # List to hold dictionaries
    name_list = []

    # Replace 'yourfile.csv' with the path to your CSV file
    with open('name_list.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            row.pop('Row')
            name_list.append(row)
    dataset = load_dataset("csv", data_files=args.dataset)

    if args.num_examples > 0:
        rate  = args.num_examples/len(dataset['train'])
        dataset = dataset['train'].train_test_split(test_size=rate, seed=args.seed)
        dataset_icl = dataset['test']
        dataset = dataset['train']
    else:
        dataset = dataset['train']
        dataset_icl = None

    # gpu_no = args.gpu_no
    print(torch.cuda.is_available())
    # device = torch.device("cuda:" + str(gpu_no) if torch.cuda.is_available() else "cpu")
    seed_value = args.seed
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if args.model_name in model_dict['encoder_decoder']:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name,torch_dtype=torch.float16, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.model_name in model_dict['decoder_only']:
        model = AutoModelForCausalLM.from_pretrained(args.model_name,torch_dtype=torch.float16, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    if args.model_name in model_dict['api_based']:
        model = args.model_name
        tokenizer = None

    DEFAULT_TEMPLATE = """{problem_description}{icl_example}Soru: {context} {question} 

    Seçenekler:
    A. {option_a} 
    B. {option_b} 

    Doğru cevap:"""

    problem_description = ""
    icl_example = ""
    for i in range(args.num_examples):
        #pick a random name_dict
        name_dict = random.choice(name_list)
        #mapping = {'Ahmet': 'A', 'Mehmet':'B'}
        speaker = name_dict['SPEAKER']
        other_person  = name_dict['NAMENull']
        if np.random.rand() > 0.5:
            mapping = {speaker: 'A', other_person:'B'}
            inverse_mapping = {v: k for k, v in mapping.items()}
        else:
            mapping = {speaker: 'B', other_person:'A'}
            inverse_mapping = {v: k for k, v in mapping.items()}

        idx = i
        if dataset_icl[idx]['ground_truth'] == 'speaker':
            ground_truth = mapping[speaker]
        else:
            ground_truth = mapping[other_person]

        context = dataset_icl[idx]['context']
        question = dataset_icl[idx]['question']
        sentence = dataset_icl[idx]['sentence']
        #put actual names to the content, question and sentence
        context = context.replace("SPEAKER", speaker)
        context = context.replace("NAMENull", other_person)
        context = context.replace('NAME-acc', name_dict['NAME-acc'])
        context = context.replace('NAME-dat', name_dict['NAME-dat'])
        context = context.replace('NAME-gen', name_dict['NAME-gen'])
        context = context.replace('NAME-com', name_dict['NAME-com'])

        question = question.replace("SPEAKER", speaker)
        question = question.replace("NAMENull", other_person)
        question = question.replace('NAME-acc', name_dict['NAME-acc'])
        question = question.replace('NAME-dat', name_dict['NAME-dat'])
        question = question.replace('NAME-gen', name_dict['NAME-gen'])
        question = question.replace('NAME-com', name_dict['NAME-com'])  

        sentence = sentence.replace("SPEAKER", speaker)
        sentence = sentence.replace("NAMENull", other_person)
        sentence = sentence.replace('NAME-acc', name_dict['NAME-acc'])
        sentence =  sentence.replace('NAME-dat', name_dict['NAME-dat'])
        sentence =  sentence.replace('NAME-gen', name_dict['NAME-gen'])
        sentence =  sentence.replace('NAME-com', name_dict['NAME-com'])

        icl_example += f'''Soru: {context} {sentence} 

    {question} 

    Seçenekler:
    A. {inverse_mapping["A"]} 
    B. {inverse_mapping["B"]}

    Doğru cevap: {ground_truth} 

    '''

    output = evaluate_model(model, tokenizer, dataset, problem_description,icl_example=icl_example, option_order='random', eval_method = 'permutation')

    dict_list = output['predictions']

    dict_list.sort(key=lambda x: x['idx'])
    csv_file = f'results/{args.model_name}_{args.num_examples}_{args.option_order}_{args.eval_method}_{args.seed}_output.csv'

    with open(csv_file, mode='w', newline='') as file:
        # Creating a writer object specifying the field names
        writer = csv.DictWriter(file, fieldnames=dict_list[0].keys())
        
        # Writing the header (column names)
        writer.writeheader()
        
        # Writing data
        for data in dict_list:
            writer.writerow(data)

    print(f'Data has been written to {csv_file}')

