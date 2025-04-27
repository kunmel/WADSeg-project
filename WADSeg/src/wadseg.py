import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils
import spacy

def TokenLevel_Attention_Heatmap_ALL_Heads(tokenizer,
    model,
    device,
    text,
    target_layers = [0, 8, 12, 16, 20]
    ):
  
  inputs = tokenizer(text, return_tensors="pt").to(device)
  
  with torch.no_grad():
      outputs = model(**inputs, output_attentions=True)
  
  attentions = outputs.attentions 
  
  num_heads = attentions[0].shape[1]
  print(f"Number of attention heads: {num_heads}")
  

  figures = []
  
  for layer_idx in target_layers:
      layer_attention = attentions[layer_idx].detach().cpu().numpy()[0]  
      
      grid_size = int(np.ceil(np.sqrt(num_heads)))
      
      fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
      fig.suptitle(f"Layer {layer_idx} - All Attention Heads", fontsize=16)
      
      axes = axes.flatten()
      
      for head_idx in range(num_heads):
          if head_idx < len(axes):  
              head_attention = layer_attention[head_idx]
              
              is_lower = utils.is_lower_triangular(head_attention)
              
              ax = axes[head_idx]
              sns.heatmap(head_attention * 20000, ax=ax, cmap="viridis", cbar=False, vmax=100)
              ax.set_title(f"Head {head_idx}" + (" (Lower Triangular)" if is_lower else ""))
              
              ax.set_xticks([])
              ax.set_yticks([])
              
              ax.set_aspect('equal', adjustable='box')
      
      for i in range(num_heads, len(axes)):
          axes[i].axis('off')
      
      plt.tight_layout(rect=[0, 0, 1, 0.96]) 
      
      figures.append(fig)
  
  return figures

def TokenLevel_Attention_Heatmap_Meaning_Heads(tokenizer,
    model,
    device,
    text,
    target_layers = [0, 8, 12, 16, 20]
    ):

  inputs = tokenizer(text, return_tensors="pt").to(device)

  with torch.no_grad():
      outputs = model(**inputs, output_attentions=True)

  attentions = outputs.attentions  

  num_heads = attentions[0].shape[1]
  print(f"Number of attention heads: {num_heads}")

  for layer_idx in target_layers:
      layer_attention = attentions[layer_idx].detach().cpu().numpy()[0] 

  fig, axes = plt.subplots(len(target_layers), 1, figsize=(6, 6 * len(target_layers)))
  if len(target_layers) == 1:
      axes = [axes]  

  for idx, layer_idx in enumerate(target_layers):
      layer_attention = attentions[layer_idx].detach().cpu().numpy()

      mean_attention = np.mean(layer_attention, axis=1)[0] 
      is_lower = utils.is_lower_triangular(mean_attention)

      ax = axes[idx]
      sns.heatmap(mean_attention * 20000, ax=ax, cmap="viridis", vmax = 100)

      ax.set_title(f"Layer {layer_idx} Mean Attention Pattern" + (" (Lower Triangular)" if is_lower else ""))
      ax.set_xlabel("Key Tokens")
      ax.set_ylabel("Query Tokens")

      ax.set_aspect('equal', adjustable='box')

  plt.tight_layout()
  return plt



def score_1(sentence_attention, i, s_=2):
    """
    Calculate Score1 for sentence i using the formula:
    Score1(i) = -∑(j=i-s_ to i-1) Δrow(Attn, i, j) + ∑(k=i+1 to i+2*s_) Δcol(Attn, k, i)
    
    Args:
        sentence_attention: Sentence-level attention matrix
        i: Sentence index
        s_: Parameter for the range of consideration
        
    Returns:
        Score1 value
    """
    score = 0.0
    num_sentences = sentence_attention.shape[0]
    
    for j in range(max(0, i-s_), i):
        score -= utils.delta_row(sentence_attention, i, j)
    
    for k in range(i+1, min(i+2*s_+1, num_sentences)):
        score += utils.delta_col(sentence_attention, k, i)
    
    return score

def score_2(sentence_attention, i, s_=2):
    """
    Calculate Score2 for sentence i using the formula:
    Score2(i) = ∑(k=i+1 to i+2*s_) Attn_k,i-1
    
    Args:
        sentence_attention: Sentence-level attention matrix
        i: Sentence index
        s_: Parameter for the range of consideration
        
    Returns:
        Score2 value
    """
    score = 0.0
    num_sentences = sentence_attention.shape[0]
    
    if i <= 0:
        return 0.0
    
    for k in range(i+1, min(i+2*s_+1, num_sentences)):
        score += sentence_attention[k, i-1]
    
    return score

def calculate_sentence_scores(text, tokenizer, model, device, layer_idx, theta=0.5, s_=2):
    """
    Calculate Score1 and Score2 for each sentence in the text.
    
    Args:
        text: Input text
        tokenizer: Tokenizer for the model
        model: The transformer model
        device: Device to run the model on
        layer_idx: Array of layer index to use for attention
        theta: Weight parameter for Score2
        s_: Parameter for the range of consideration
        
    Returns:
        sentences: List of sentences
        scores: Dictionary with Score1, Score2, and combined Score for each sentence in each layer
        sentence_attention: List of sentence-level attention matrices for each layer
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        
    token_sentence_mapping, sentences, _ = utils.Tokens2Sentences(text, tokens, tokenizer, inputs)
    num_sentences = len(sentences)
    
    sentence_attention = []
    scores = {}
    
    for i, layer in enumerate(layer_idx):
        token_attention = outputs.attentions[layer].detach().cpu().numpy()
        
        mean_token_attention = np.mean(token_attention, axis=1)[0] 
        
        layer_attention = utils.compute_sentence_attention(mean_token_attention, token_sentence_mapping, num_sentences)
        sentence_attention.append(layer_attention)
        
        layer_scores = {}
        for j in range(num_sentences):
            score1 = score_1(layer_attention, j, s_)
            score2 = score_2(layer_attention, j, s_)
            combined_score = score1 - theta * score2
            
            layer_scores[j] = {
                'score1': score1,
                'score2': score2,
                'score': combined_score
            }
        
        scores[layer] = layer_scores
    
    return sentences, scores, sentence_attention

def SentenceLevel_Attention_Heatmap_Meaning_Heads(sentence_attention, sentences, layer_idx):
    """
    Visualize the sentence-level attention matrices for multiple layers.
    
    Args:
        sentence_attention: List of sentence-level attention matrices
        sentences: List of sentences
        layer_idx: List of layer indices corresponding to the attention matrices
    """
    num_layers = len(sentence_attention)
    fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6))
    
    if num_layers == 1:
        axes = [axes]
    
    for i, attn_matrix in enumerate(sentence_attention):
        ax = axes[i]
        sns.heatmap(attn_matrix*20000, cmap="viridis", vmax=100, ax=ax,
                    xticklabels=[f"S{i+1}" for i in range(len(sentences))], 
                    yticklabels=[f"S{i+1}" for i in range(len(sentences))])
        ax.set_title(f"Layer {layer_idx[i]} Sentence-Level Attention")
        ax.set_xlabel("Key Sentences")
        ax.set_ylabel("Query Sentences")
    
    plt.tight_layout()
    return plt

def wadseg(text, tokenizer, model, device, layer_idx=0, theta=0.5, s_=2, N=3, min_chunk_size=3, max_chunk_size=None):
    """
    Segment text using the WADSeg algorithm based on attention patterns.
    
    Args:
        text: Input text to segment
        tokenizer: Tokenizer for the model
        model: The transformer model
        device: Device to run the model on
        layer_idx: Layer index to use for attention
        theta: Weight parameter for Score2
        s_: Parameter for the range of consideration
        N: Number of top-scoring sentences to consider as breakpoints
        min_chunk_size: Minimum number of sentences in a chunk
        max_chunk_size: Maximum number of sentences in a chunk (if None, no limit)
        
    Returns:
        chunks: List of text segments
        breakpoints: List of sentence indices used as breakpoints
        scentences: List of sentences in the original text
    """
    sentences, scores, sentence_attention = calculate_sentence_scores(
        text, tokenizer, model, device, [layer_idx], theta, s_
    )
    
    layer_scores = scores[layer_idx]
    
    sentence_scores = [layer_scores[i]['score'] for i in range(len(sentences))]
    
    attn_matrix = sentence_attention[0] 
    
    candidate_indices = list(range(1, len(sentences) - 1))
    candidate_indices.sort(key=lambda i: sentence_scores[i], reverse=True)
    top_candidates = candidate_indices[:min(N, len(candidate_indices))]
    
    top_candidates.sort()
    
    breakpoints = []
    
    current_chunk_start = 0
    
    for i in top_candidates:
        if i - current_chunk_start < min_chunk_size:
            continue
            
        if max_chunk_size and i - current_chunk_start > max_chunk_size:
            for j in range(current_chunk_start + min_chunk_size, i):
                if j in top_candidates:
                    breakpoints.append(j)
                    current_chunk_start = j
                    break
            
            if current_chunk_start + max_chunk_size < i:
                forced_break = current_chunk_start + max_chunk_size
                breakpoints.append(forced_break)
                current_chunk_start = forced_break
        
        if i > 0 and i < len(sentences) - 1:
            attn_prev = attn_matrix[i, i-1]  
            attn_next = attn_matrix[i+1, i]  
            
            C = max(attn_prev, attn_next) - (3/2) * min(attn_prev, attn_next)
            
            if C < 0:
                breakpoints.append(i)
                continue
        
        breakpoints.append(i)
        current_chunk_start = i

    chunks = []
    chunk_start = 0
    
    for i, bp in enumerate(breakpoints):
        if i > 0 and bp == breakpoints[i-1] + 1:
            pass
        else:
            chunk_text = " ".join(sentences[chunk_start:bp])
            chunks.append(chunk_text)
            chunk_start = bp
    
    if chunk_start < len(sentences):
        chunk_text = " ".join(sentences[chunk_start:])
        chunks.append(chunk_text)
    
    return chunks, breakpoints, sentences


