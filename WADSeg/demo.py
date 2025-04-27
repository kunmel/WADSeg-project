import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from WADSeg.src import wadseg



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)
model.to(device)


text = ""

target_layers = [0,8,12,16,20]
# Heatmaps of token attention in all layers
plt_token_allH = wadseg.TokenLevel_Attention_Heatmap_ALL_Heads(tokenizer = tokenizer,
                                           model = model,
                                           device = device,
                                           text = text,
                                           target_layers = target_layers
                                           )
for i in range(len(target_layers)):
  plt_token_allH[i].show()


# Heatmaps of token attention meaning among all layers
plt_token = wadseg.TokenLevel_Attention_Heatmap_Meaning_Heads(tokenizer = tokenizer,
                                           model = model,
                                           device = device,
                                           text = text,
                                           target_layers = target_layers
                                           )
plt_token.show()

# Segmentation show
sentences, scores, sentence_attention = wadseg.calculate_sentence_scores(text, tokenizer, model, device, layer_idx = target_layers)
plt_sentence = wadseg.SentenceLevel_Attention_Heatmap_Meaning_Heads(sentence_attention, sentences, layer_idx = target_layers)
plt_sentence.show()
chunks, breakpoints, sentences = wadseg.wadseg(text, tokenizer, model, device, layer_idx=23, theta=0.5, s_=2, N=3, min_chunk_size=3, max_chunk_size=None)
