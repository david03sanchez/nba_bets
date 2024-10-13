import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from scipy.spatial.distance import cosine
import requests
from bs4 import BeautifulSoup

def fetch_wikipedia_article(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch the page. Status code: {response.status_code}")

    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    content = ' '.join([p.get_text() for p in paragraphs])
    return content


def chunk_text(text, chunk_size=50, overlap=10):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(' '.join(chunk))
        if i + chunk_size >= len(words):
            break
    return chunks
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(input_text):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(device)

    # Tokenize sentences and get embddings
    encoded_input = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


#fetch wikipedia article then apply the chunking function to the article string
url = "https://en.wikipedia.org/wiki/Super_Bowl_LVIII"
article_content = fetch_wikipedia_article(url)
wiki_chunks = chunk_text(article_content, chunk_size=100, overlap=10)

#generate embeddings for each chunk of text
wiki_embeddings = embed_text(wiki_chunks)
wiki_df = pd.DataFrame(wiki_chunks,columns=['chunked_text'])
wiki_df['embedding'] = wiki_embeddings.tolist()

#question for rag and generate embeddings using the sentence_transformers model
question_for_rag = 'Which teams played and who won the 2024 Super Bowl? Who received the openning kickoff? And who was Taylor Swift Rooting For'
question_embedding = embed_text(question_for_rag)
question_embedding = question_embedding.tolist()

#calculate the similarity score between the question and each chunk of text
wiki_df['cosine_sim'] = wiki_df['embedding'].apply(lambda x: 1 - cosine(x, question_embedding[0]))

#take the top 10 most "similar" chunks of text to the question
wiki_rag_context = wiki_df.sort_values('cosine_sim', ascending=False).iloc[0:10]['chunked_text'].str.cat(sep=' ')
#%%

#load llama model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("unsloth/llama-3-8b-Instruct-bnb-4bit",cache_dir ='/data/huggingface/')

def ask_llama(text):
    print('Asking llama...')
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)

    # Generate the response from the model
    with torch.no_grad():
        outputs = model.generate(input_ids, max_length=4000)

    # Decode the response back to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_df = pd.DataFrame({'question': [text], 'response': [response]})
    return response_df


llama_prompt_no_context = f'''
You are a helpful llama that answers questions. Keep your answer to a few sentences at most.

Question:
{question_for_rag}
'''
no_rag = ask_llama(llama_prompt_no_context)
print(no_rag['response'].iloc[0])

llama_prompt_context = f'''
You are a helpful llama that answers questions. Keep your answer to a few sentences at most.

Given the context below:

Context:
"""
{wiki_rag_context}
"""

Question:
{question_for_rag}

'''
rag = ask_llama(llama_prompt_context)
#extract answer from response:
print(rag['response'].iloc[0])

#%%

#wrapping the rag query and llama request in a function
def llama_wiki_rag(query,wiki_url):
    # query = 'Who won the 2024 superbowl? What attraced siginificant national attention prior to the game?'
    # wiki_url = 'https://en.wikipedia.org/wiki/Super_Bowl_LVIII'
    article_content = fetch_wikipedia_article(wiki_url)
    wiki_chunks = chunk_text(article_content, chunk_size=100, overlap=50)
    wiki_embeddings = embed_text(wiki_chunks)
    wiki_df = pd.DataFrame(wiki_chunks,columns=['chunked_text'])
    wiki_df['embedding'] = wiki_embeddings.tolist()

    query_text = query
    query_embeddings = embed_text(query_text)
    query_embeddings = query_embeddings.tolist()

    wiki_df['cosine_sim'] = wiki_df['embedding'].apply(lambda x: 1 - cosine(x, query_embeddings[0]))
    wiki_rag_context = wiki_df.sort_values('cosine_sim', ascending=False).iloc[0:10]['chunked_text'].str.cat(sep=' ')

    llama_prompt_context = f'''
    You are a helpful llama that answers questions. Keep your answer to a few sentences at most.

    Given the context below:

    Context:
    """
    {wiki_rag_context}
    """

    Question:
    {query_text}

    '''
    rag = ask_llama(llama_prompt_context)
    return rag['response'].iloc[0]

response = llama_wiki_rag('Give me an overview and summary of this page. Include details that '\
                          'a client would want to know.','https://aws.amazon.com/generative-ai/')
print(response)
