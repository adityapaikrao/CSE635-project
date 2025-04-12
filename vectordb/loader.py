import uuid
from sentence_transformers import SentenceTransformer
from spacy.lang.en import English
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Union, Any


def get_sentences(text):
    """
    Return list of sentences given a body of text
    """

    nlp = English()
    nlp.add_pipe('sentencizer')
    
    clean_text = text.replace("\n", " ").strip()
    
    sentences = [str(sentence) for sentence in nlp(clean_text).sents]

    return sentences


def create_documents(json_file):
    """
    Returns a list of documents given the raw JSON file
    """
    documents = []
    for site, jsons_list in json_file.items():
        for json_content in jsons_list:
            for url, text in json_content.items():
                if not text:
                    continue
                yield {
                    "id":str(uuid.uuid4()),
                    "site":site, 
                    "url":url,
                    "sentences":get_sentences(text)
                }
                

def combine_sentences(sentences: list[str], buffer_size: int) -> list[dict[str, Union[str, int]]]:
    sentences = [{'sentence': x, 'index': i} for i, x in enumerate(sentences)]

    for i in range(len(sentences)):

        # Create a string that will hold the sentences which are joined
        combined_sentence = ''

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]['sentence'] + ' '

        # Add the current sentence
        combined_sentence += sentences[i]['sentence']

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += ' ' + sentences[j]['sentence']

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]['combined_sentence'] = combined_sentence

    return sentences

def calculate_cosine_distances(sentences: List[dict[str, Union[str, int]]]) -> str:
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]['combined_sentence_embedding']
        embedding_next = sentences[i + 1]['combined_sentence_embedding']

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]['distance_to_next'] = distance

    # Optionally handle the last sentence
    sentences[-1]['distance_to_next'] = 1  # or a default value

    return distances, sentences

def chunk_sentences(doc_text_sentences: List[str], buffer_size: int = 1,
                    breakpoint_percentile_threshold: int = 80) -> list[str]:
    """

    :param breakpoint_percentile_threshold:
    :param doc_text_sentences:
    :param buffer_size:
    :return:
    """
    combined_sentences = combine_sentences(doc_text_sentences, buffer_size)

    # Embed sentences
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 
    embeddings = model.encode(sentences=[x['combined_sentence'] for x in combined_sentences])

    for i, sentence in enumerate(combined_sentences):
        sentence['combined_sentence_embedding'] = embeddings[i]
    # Calculate semantic distances
    distances, sentences = calculate_cosine_distances(combined_sentences)

    # find breakpoint
    breakpoint_distance_threshold = np.percentile(distances, breakpoint_percentile_threshold)
    indices_above_thresh = [i for i, x in enumerate(distances) if x > breakpoint_distance_threshold]

    # Create chunks
    start_index = 0
    chunks = []
    for index in indices_above_thresh:
        end_index = index
        group = sentences[start_index:end_index + 1]
        combined_text = ' '.join([str(d['sentence']) for d in group])
        chunks.append(combined_text)
        start_index = index + 1

    # Handle the last group, if any sentences remain
    if start_index < len(sentences):
        combined_text = ' '.join([str(d['sentence']) for d in sentences[start_index:]])
        chunks.append(combined_text)

    return chunks

def get_embeddings(chunked_list: List[str]) -> Union[object, list]:
    """

    :param chunked_list:
    :return:

    """
    model = SentenceTransformer('neuml/pubmedbert-base-embeddings')
    embeddings = model.encode(sentences=[x for x in chunked_list])

    # chunked_dict = {}
    # for i in range(len(chunked_list)):
    #     chunked_dict[chunked_list[i]] = embeddings[i]

    return embeddings.tolist()