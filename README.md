# Do LLMs Recognize Me, When I is Not Me: Assessment of LLMs' Understanding of Turkish Indexical Pronouns in Indexical Shift Contexts - ACL 2024 SIGTURK Workshop

[Read the paper](https://aclanthology.org/2024.sigturk-1.5/)

This repository contains the code and resources for the paper titled **"Do LLMs Recognize Me, When I is Not Me: Assessment of LLMs' Understanding of Turkish Indexical Pronouns in Indexical Shift Contexts"**, presented at the ACL 2024 SIGTURK Workshop.

## Overview

This project investigates how well Large Language Models (LLMs) understand Turkish indexical pronouns in contexts where their reference shifts (indexical shift contexts). Indexical pronouns like "ben" (I) and "sen" (you) can change meaning depending on the syntactic and semantic context, especially in embedded clauses with certain verbs or predicates. Understanding these nuances is crucial for accurate natural language understanding and machine translation.

## Models Evaluated

The following LLMs were evaluated in this study:

- **OpenAI's GPT-3.5 Turbo**
- **OpenAI's GPT-4**
- **TURKCELL/Turkcell-LLM-7b-v1**
- **Trendyol/Trendyol-LLM-7b-base-v0.1**
- **CohereForAI/aya-101**

## Run Experiments

To reproduce the experiments from the paper, run the following commands:

```shell
python main.py --model_name gpt-3.5-turbo --num_examples 5 --option_order random --eval_method permutation 
python main.py --model_name gpt-4 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name TURKCELL/Turkcell-LLM-7b-v1 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name Trendyol/Trendyol-LLM-7b-base-v0.1 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name CohereForAI/aya-101 --num_examples 5 --option_order random --eval_method permutation
```

### Parameters

- `--model_name`: Specifies the name of the LLM to evaluate.
- `--num_examples`: Number of examples to use in the evaluation.
- `--option_order`: Order of options; can be 'random' or 'fixed'.
- `--eval_method`: Evaluation method; 'permutation' is used to assess the models across different permutations of the options.

## Dataset

The dataset consists of Turkish sentences containing indexical pronouns within indexical shift contexts. These sentences are crafted to test the models' ability to interpret pronouns whose reference can shift depending on the context.

## Results

The detailed results and analysis can be found in the [paper](https://aclanthology.org/2024.sigturk-1.5/). The findings highlight the strengths and weaknesses of current LLMs in handling complex linguistic phenomena in Turkish.

## Reference

If you use this code or data in your research, please cite the paper:

```bibtex
@inproceedings{oguz-etal-2024-llms,
    title = "Do {LLM}s Recognize Me, When {I} is Not Me: Assessment of {LLM}s Understanding of {T}urkish Indexical Pronouns in Indexical Shift Contexts",
    author = "O{\u{g}}uz, Metehan  and
      Ciftci, Yusuf  and
      Bakman, Yavuz Faruk",
    editor = {Ataman, Duygu  and
      Derin, Mehmet Oguz  and
      Ivanova, Sardana  and
      K{\"o}ksal, Abdullatif  and
      S{\"a}lev{\"a}, Jonne  and
      Zeyrek, Deniz},
    booktitle = "Proceedings of the First Workshop on Natural Language Processing for Turkic Languages (SIGTURK 2024)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.sigturk-1.5",
    pages = "53--61",
}
```

