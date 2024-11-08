# python db_create.py cnn_dailymail_train_abi.db train  
from datasets import load_dataset
import sys
# dataset_test = load_dataset("ccdv/cnn_dailymail", "3.0.0") #tb 
from datasets import load_dataset
dataset_test = load_dataset("abisee/cnn_dailymail", "3.0.0")
data = dataset_test[sys.argv[2]] # "train"
import sqlite3, os 
db_file =  sys.argv[1] #"cnn_dailymail_train.db"
if os.path.isfile(db_file):
	os.remove(db_file)
conn = sqlite3.connect(db_file,detect_types=sqlite3.PARSE_DECLTYPES)
conn.row_factory = sqlite3.Row
c = conn.cursor()
sql_create = "CREATE TABLE articles (id INTEGER PRIMARY KEY AUTOINCREMENT, article TEXT NOT NULL, highlights TEXT);"
c.execute(sql_create)
conn.commit()
sql_insert = "INSERT INTO articles (article, highlights) VALUES (?, ?)"
for a in data:
	c.execute(sql_insert, (a['article'], a['highlights']))
conn.commit()
