import numpy as np
import spacy

def is_lower_triangular(matrix):
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(i+1, cols):
            if matrix[i, j] != 0:
                return False
    return True

def Tokens2Sentences(text, tokens, tokenizer, inputs):
    """
    Map tokens to sentences using spaCy.
    
    Args:
        text: Input text
        tokens: List of tokens
        tokenizer: Tokenizer for the model
        inputs: Tokenized inputs
        
    Returns:
        token_sentence_mapping: List mapping each token to its sentence index
        sentences: List of sentences
        sentence_token_indices: Dictionary mapping sentence indices to lists of token indices
    """
    nlp = spacy.load("en_core_web_sm")
    
    doc = nlp(text)
    sentences = list(doc.sents)
    
    sentence_spans = [(sent.start_char, sent.end_char) for sent in sentences]
    
    token_sentence_mapping = []
    
    token_texts = []
    for token_id in inputs.input_ids[0].cpu().numpy():
        token_text = tokenizer.decode([token_id]).strip()
        token_texts.append(token_text)
    
    current_pos = 0
    for token, token_text in zip(tokens, token_texts):
        if token.startswith('<') and token.endswith('>'):
            token_sentence_mapping.append(-1)
            continue
        
        if token_text:
            pos = text.find(token_text, current_pos) if token_text in text[current_pos:] else -1
            
            if pos >= 0:
                sent_idx = -1
                for i, (start, end) in enumerate(sentence_spans):
                    if start <= pos < end:
                        sent_idx = i
                        break
                
                if sent_idx >= 0:
                    token_sentence_mapping.append(sent_idx)
                    current_pos = pos + len(token_text)
                else:
                    token_sentence_mapping.append(token_sentence_mapping[-1] if token_sentence_mapping and token_sentence_mapping[-1] != -1 else 0)
            else:
                token_sentence_mapping.append(token_sentence_mapping[-1] if token_sentence_mapping and token_sentence_mapping[-1] != -1 else 0)
        else:
            token_sentence_mapping.append(token_sentence_mapping[-1] if token_sentence_mapping and token_sentence_mapping[-1] != -1 else 0)
    
    sentence_token_indices = {}
    for token_idx, sent_idx in enumerate(token_sentence_mapping):
        if sent_idx != -1:  
            if sent_idx not in sentence_token_indices:
                sentence_token_indices[sent_idx] = []
            sentence_token_indices[sent_idx].append(token_idx)
    
    return token_sentence_mapping, [sent.text for sent in sentences], sentence_token_indices


def compute_sentence_attention(token_attention, token_sentence_mapping, num_sentences):
    """
    Compute sentence-level attention matrix from token-level attention.
    
    Args:
        token_attention: Token-level attention matrix
        token_sentence_mapping: List mapping each token to its sentence index
        num_sentences: Number of sentences
        
    Returns:
        Sentence-level attention matrix
    """
    sentence_attention = np.zeros((num_sentences, num_sentences))
    sentence_token_counts = np.zeros((num_sentences, num_sentences))
    
    for i, sent_i in enumerate(token_sentence_mapping):
        if sent_i == -1: 
            continue
        for j, sent_j in enumerate(token_sentence_mapping):
            if sent_j == -1: 
                continue
            sentence_attention[sent_i, sent_j] += token_attention[i, j]
            sentence_token_counts[sent_i, sent_j] += 1
    
    sentence_token_counts = np.maximum(sentence_token_counts, 1)
    sentence_attention = sentence_attention / sentence_token_counts
    
    return sentence_attention

def delta_row(A, i, j):
    """
    Calculate the row difference: A_i,j - A_i-1,j
    
    Args:
        A: Attention matrix
        i: Row index
        j: Column index
        
    Returns:
        Row difference
    """
    if i > 0:
        return A[i, j] - A[i-1, j]
    return A[i, j] 

def delta_col(A, i, j): 
    """
    Calculate the column difference: A_i,j - A_i,j-1
    
    Args:
        A: Attention matrix
        i: Row index
        j: Column index
        
    Returns:
        Column difference
    """
    if j > 0:
        return A[i, j] - A[i, j-1]
    return A[i, j]  


def visualize_segmentation(text, sentences, breakpoints):
    """
    Notebook Only
    Visualize the text segmentation with breakpoints highlighted.
    
    Args:
        text: Original text
        sentences: List of sentences
        breakpoints: List of sentence indices used as breakpoints
    """
    from IPython.display import HTML, display
    
    html = "<h3>WADSeg Text</h3>"
    html += "<div style='line-height: 1.5; max-width: 800px;'>"
    
    for i, sentence in enumerate(sentences):
        if i in breakpoints:
            html += "<hr style='border-top: 2px dashed #ff5555;'>"
            html += f"<span style='background-color: #ffdddd; padding: 2px;'>{sentence}</span> "
        else:
            html += f"{sentence} "
    
    html += "</div>"
    
    display(HTML(html))
    
    print("\nSegmented Text:")
    chunk_start = 0
    for bp in breakpoints:
        print(f"\n--- Segment ---")
        print(" ".join(sentences[chunk_start:bp]))
        chunk_start = bp
    
    print(f"\n--- Segment ---")
    print(" ".join(sentences[chunk_start:]))