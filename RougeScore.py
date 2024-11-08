# from datasets import load_dataset, load_metric
from datasets import load_dataset
from evaluate import load #check later
import json
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# rouge = load_metric("rouge")
rouge = load("rouge")
# dataset_test = load_dataset("ccdv/cnn_dailymail", "3.0.0")["test"]
dataset_test = load_dataset("abisee/cnn_dailymail", "3.0.0")["test"]

model = GPT2LMHeadModel.from_pretrained("summary_loop46")
#model.to('cuda')
tokenizer = GPT2TokenizerFast.from_pretrained("summary_loop46")

json_array = []

data_size = 5
for i in range(0,len(dataset_test[:data_size])):
  d = dataset_test[i]
  tokenized_document = tokenizer([d['article']], max_length=300, truncation=True, return_tensors="pt")["input_ids"]
  #tokenized_document = tokenizer([document], max_length=300, truncation=True, return_tensors="pt")["input_ids"].cuda()
  input_shape = tokenized_document.shape
  outputs = model.generate(tokenized_document, do_sample=False, max_length=500, num_beams=4, num_return_sequences=4, no_repeat_ngram_size=6, return_dict_in_generate=True, output_scores=True)
  #candidate_sequences = outputs.sequences[:, input_shape[1]:] # Remove the encoded text, keep only the summary
  #candidate_scores = outputs.sequences_scores.tolist()
  #summary = tokenizer.decode(candidate_sequences[0])
  summary = tokenizer.decode(outputs[0])
  json_array.append({"id": d["id"], "summary_loop_gen": summary[:summary.index("END")]})
json_file = open("summary_loop46_test.json","w");
json.dump(json_array, json_file)
json_file.close()

#from datasets import load_dataset, load_metric
#import json
with open("summary_loop46_test.json", "r") as f:
    summary_loop_gens = json.load(f)
#rouge = load_metric("rouge")
#dataset_test = load_dataset("ccdv/cnn_dailymail", "3.0.0")["test"]
id2summary_loop = {d["id"]: d["summary_loop_gen"] for d in summary_loop_gens}
candidates, references = [], []
for i in range(0,len(dataset_test[:data_size])):
    d = dataset_test[i]
    references.append(d["highlights"])
    candidates.append(id2summary_loop[d["id"]])
print(len(references), len(candidates))
print(rouge.compute(predictions=candidates, references=references))


