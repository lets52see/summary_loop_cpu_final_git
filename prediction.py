from torch.utils.data import DataLoader, RandomSampler
from utils_dataset import SQLDataset, HDF5Dataset
import torch, os, time, argparse, numpy as np
from transformers.optimization import AdamW
from model_generator import GeneTransformer
import utils_misc, utils_tokenizer
from utils_logplot import LogPlot
from datetime import datetime

#from model_coverage import KeywordCoverage
#from model_guardrails import PatternPenalty, LengthPenalty, RepeatPenalty

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="./model/model.bin", help="path to model file")
parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
parser.add_argument("--max_output_length", type=int, default=25, help="Maximum output length. Saves time if the sequences are short.")

args = parser.parse_args()
if args.device == "cuda":
    freer_gpu = str(utils_misc.get_freer_gpu())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""+str(freer_gpu)

models_folder = "./models/"

summarizer = GeneTransformer(max_output_length=args.max_output_length, device=args.device, tokenizer_type='gpt2') #starter_model?
print("Summarizer loaded")

document = "High tides are forecast to batter the country this weekend as a storm sweeping in from the Atlantic brings 80mph gale force winds and coastal flooding. Devon, Somerset, the Bristol Channel, the south Wales coast and the English Channel are all expected to be affected, with flood warnings issued across much of the country as it braces itself for a wet and windy weekend coupled with 50ft high tides. The Environment Agency has warned of potential flooding across dozens of coastal areas, with the towns of Clevedon, Chepstow and Sheepway directly in the firing line. And many more seaside towns and villages were on sandbag alert with huge waves starting to roll in this morning and likely to continue until Monday. Scroll down for video . Surfers take advantage of the Severn Bore at Over, Gloucestershire this morning. The rare phenomenon occurs when tidal surges create surfing waves through the estuary . The huge swell, which has the Bristol Channel particularly badly, created perfect longboarding conditions . The massive sea swell created a 'five star' Severn Bore - which may only occur once a year - in Minsterworth, Gloucestershire . A group of surfers paddle into the water in anticipation of the Severn Bore waves at Over, Gloucestershire . The surfers, all wearing thick wetsuits and armed with longboards, prepare for the extremely rare surfing phenomenom . An early riser manages to catch a wave all to himself up the River Severn in Gloucestershire this morning . The huge swell was a boon for surfers, who bravely entered the freezing water in a bid to catch near-perfect waves . A single sea kayaker (right) joins ten surfers on a wave, when the Severn Bore conditions peaked this morning . High tides batter Lynmouth Harbour, Devon, this morning as flood warnings"
map_location = torch.device(summarizer.device) ##sa
# summarizer.model.load_state_dict(torch.load(args.model, map_location='cpu'))
summarizer.model.load_state_dict(torch.load(args.model, map_location=map_location))##sa
summary = summarizer.decode([document])
print(summary)

