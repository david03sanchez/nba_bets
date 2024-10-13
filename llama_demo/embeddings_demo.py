import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from scipy.spatial.distance import cosine
import numpy as np

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(input_text):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

    # Tokenize sentences
    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings.cpu().squeeze(0).numpy()


#create and embed questions
question1 = 'Which year did we land on the moon?'
question2 = 'Which year was the iPhone released?'
question3 = 'Which year was the E=m2 formula published?'
embedding_q1 = embed_text(question1)
embedding_q2 = embed_text(question2)
embedding_q3 = embed_text(question3)

#create base dataframe
embeddings_df = pd.DataFrame({'text': [question1, question2, question3]})
embeddings_df['embedding'] = [embedding_q1, embedding_q2, embedding_q3]
embeddings_df.index = ['Question 1', 'Question 2', 'Question 3']

#MC answers
answer1 = '1969'
answer2 = '2024'
answer3 = '2007'
answer4 = '1905'

#embed MC answers
answer1_embeddings = embed_text(answer1)
answer2_embeddings = embed_text(answer2)
answer3_embeddings = embed_text(answer3)
answer4_embeddings = embed_text(answer4)


#add MC answers & answer embeddings to embeddings_df
embeddings_df.loc['Answer 1'] = [answer1, answer1_embeddings]
embeddings_df.loc['Answer 2'] = [answer2, answer2_embeddings]
embeddings_df.loc['Answer 3'] = [answer3, answer3_embeddings]
embeddings_df.loc['Answer 4'] = [answer4, answer4_embeddings]
embeddings_df['similarity'] = np.nan


#compute similarity for chosen question
question_num = 2
for r in range(3,embeddings_df.shape[0]):
    # r=1
    question = embeddings_df['embedding'].iloc[question_num]
    answer = embeddings_df['embedding'].iloc[r]
    similarity_score = 1 - cosine(question, answer)
    embeddings_df['similarity'].iloc[r] = similarity_score

