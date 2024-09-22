# Do LLMs Recognize me, When I is not me: Assessment of LLMs Understanding of Turkish Indexical Pronouns in Indexical Shift Contexts - ACL 2024 SIGTURK Workshop

Read the paper: https://aclanthology.org/2024.sigturk-1.5/

## Run Experiments

To run the experiments:

```shell
python main.py --model_name gpt-3.5-turbo --num_examples 5 --option_order random --eval_method permutation 
python main.py --model_name gpt-4 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name TURKCELL/Turkcell-LLM-7b-v1 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name Trendyol/Trendyol-LLM-7b-base-v0.1 --num_examples 5 --option_order random --eval_method permutation
python main.py --model_name CohereForAI/aya-101 --num_examples 5 --option_order random --eval_method permutation
```

## Reference

```shell
@inproceedings{oguz-etal-2024-llms,
    title = "Do {LLM}s Recognize me, When {I} is not me: Assessment of {LLM}s Understanding of {T}urkish Indexical Pronouns in Indexical Shift Contexts",
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

