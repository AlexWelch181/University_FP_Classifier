import sys

import jsonlines
import re

import spacy
from nltk.tokenize import sent_tokenize

import pandas as pd

CHATGPT = "chatgpt"
HUMAN = "human"
DATA_FILE = 'all.jsonl'

nlp = spacy.load('en_core_web_sm')
stop_words = nlp.Defaults.stop_words
nlp_stop_words = list(stop_words)
labels = []
data_list = []
questions_count = 0
human_count = 0
gpt_count = 0
human_ans_count = 0
gpt_ans_count = 0
wantedSections = ['wiki_csai', 'open_qa', 'reddit_eli5', 'finance', 'medicine']

try:
    limit = sys.argv[1]
except IndexError:
    limit = 600000

with jsonlines.open(DATA_FILE, 'r') as reader:
    for data in reader:
        if data['source'] in wantedSections:
            questions_count += 1
            if human_count + gpt_count > limit:
                break
            for ans in data['human_answers']:
                human_ans_count += 1
                sentences = sent_tokenize(ans)
                for sentence in sentences:
                    sentence = sentence.strip('\n').lower()
                    human_ans = re.sub(r'[^\w\s]', '', sentence)
                    human_ans = re.sub(r'url_[0-9]', '', human_ans)
                    human_ans = re.sub(r'\s+', ' ', human_ans)
                    lemma_text = human_ans
                    if len(lemma_text.split()) < 3:
                        continue
                    data_list.append(lemma_text)
                    labels.append(HUMAN)
                    human_count += 1

            for ans in data['chatgpt_answers']:
                gpt_ans_count += 1
                sentences = sent_tokenize(ans)
                for sentence in sentences:
                    if 'AI language model' in sentence:
                        continue
                    sentence = sentence.strip('\n').lower()
                    chatgpt_ans = re.sub(r'[^\w\s]', '', sentence)
                    chatgpt_ans = re.sub(r'url_[0-9]', '', chatgpt_ans)
                    chatgpt_ans = re.sub(r'\s+', ' ', chatgpt_ans)
                    lemma_text = chatgpt_ans
                    if len(lemma_text.split()) < 3:
                        continue
                    data_list.append(lemma_text)
                    labels.append(CHATGPT)
                    gpt_count += 1

    print(f'Questions = {questions_count}')
    print(f'Human responses = {human_ans_count}')
    print(f'ChatGPT responses = {gpt_ans_count}')
    print(f'Human sentences = {human_count}')
    print(f'ChatGPT sentences = {gpt_count}')
    print(f'ratio = {human_count / gpt_count} : 1')
    print(f'total sentences = {human_count + gpt_count}')

data_frame = pd.DataFrame({'data': data_list, 'label': labels})
data_frame.to_csv('labeled_data.csv', index=False)
