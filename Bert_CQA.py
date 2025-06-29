# Here is Bert model with keras 
# I tried to train it but i got high val loss 
# i don't know if it works or not as i train it on my cpu core i7-6700
# But what I know that it Should work

from keras.api.layers import MultiHeadAttention, Embedding, Dense, Dropout, LayerNormalization, TextVectorization
from keras.api.models import Model
from keras.api.losses import SparseCategoricalCrossentropy
from keras.api.optimizers import Adam
from keras.api.callbacks import ModelCheckpoint, Callback
from keras.api.utils import to_categorical
import tensorflow as tf
import numpy as np
from datasets import load_dataset
# Name: keras
# Version: 3.6.0
# Name: datasets
# Version: 3.2.0
# === Transformer Block ===
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.multi_head = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.multi_head(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

# === BERT-like Model ===
class Bert(Model):
    def __init__(self, max_seq_len, embed_dim, vocab_size, num_heads, ff_dim, num_layers=1):
        super().__init__()
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_embedding = Embedding(input_dim=max_seq_len, output_dim=embed_dim)
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        self.start_dense = Dense(max_seq_len)
        self.end_dense = Dense(max_seq_len)

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        pos = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.pos_embedding(pos)
        x = self.embedding(inputs) + pos_emb
        
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, training=training)
        
        start_logits = self.start_dense(x)
        end_logits = self.end_dense(x)
        
        start_logits = tf.reduce_mean(start_logits, axis=1)
        end_logits = tf.reduce_mean(end_logits, axis=1)
        
        return start_logits, end_logits

# === Callback for Best Model Saving ===
class BestModelCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = logs.get('accuracy')
        loss = logs.get('loss')
        if acc >= 0.95 and loss <= 0.1:
            model.save(f"best_model_{acc:.2f}_acc_{loss:.2f}_loss.keras")

# === Data Preparation ===
def prepare_data(squad , tokenizer , max_seq_len):
    inputs  , ans_start , ans_end = [] , [] , []
    largest_seq = 0

    for item in squad:
        context = item["context"]
        question = item["question"]
        answer_start = item["answers"]["answer_start"][0]
        answer_text  = item["answers"]["text"][0]
        #print(answer_text)
        if largest_seq <= len(question) + len(context):
            largest_seq = len(question) + len(context)
        
        question = tokenizer(question).numpy()
        question = [item for item in question]
        context = tokenizer(context).numpy()
        context = [item for item in context]
        context_before = item["context"]
        answer_start = context_before[:answer_start]
        answer_start = tokenizer(answer_start).numpy()
        answer_start = [item for item in answer_start]
        answer_start = len(answer_start) 
        answer_end = tokenizer(answer_text).numpy()
        answer_end = [item for item in answer_end]
        answer_end = len(answer_end) + answer_start +1 
        answer_text = tokenizer(answer_text).numpy()
        answer_text = [item for item in answer_text]
        
        #print(context)
        #print(question)
        X = question + context 


        answer_start = answer_start + len(question) 
        answer_end = answer_start + len(answer_text)
        #print(len(X))
        #print(answer_start)
        #print(answer_end)
        #print("X")
        #print(X)
        #print("X:answer_start")
        #print(X[:answer_start])
        #print("X BET")
        #print(X[answer_start:answer_end])
        #print("done")
        #print(answer_start)
        #print(answer_end)
        #userinput = input("")
        # Convert to tensors
        #X = tokenizer(X).numpy()
        #X = np.array(X)
        #text = [item for item in X]
        #text = text[answer_start:answer_end]
        #text = [tokenizer.get_vocabulary()[tk] for tk in text]
        #print(text)
        X = X[:max_seq_len] + [0] * (max_seq_len - len(X))
        inputs.append(X)
        ans_start.append(answer_start)
        ans_end.append(answer_end)
    


    return np.array(inputs) , np.array(ans_start) , np.array(ans_end) 
    
# === Answer Decoding ===
def decode_answer(vectorizer, input_tokens, start_pos, end_pos):
    vocab = vectorizer.get_vocabulary()
    answer_tokens = input_tokens[start_pos:end_pos+1]
    return ''.join([vocab[token] for token in answer_tokens if token != 0])

# === Answer Prediction ===
def predict_answer(model, vectorizer, context, question, max_seq_len):
    input_text = question + " [SEP] " + context
    input_tokens = vectorizer(input_text).numpy()
    input_tokens = input_tokens[:max_seq_len]
    input_tokens = np.pad(input_tokens, (0, max_seq_len - len(input_tokens)), constant_values=0)

    input_tokens_tensor = tf.constant(input_tokens)[tf.newaxis, :]
    start_logits, end_logits = model(input_tokens_tensor, training=False)

    start_pos = tf.argmax(start_logits, axis=-1).numpy()[0]
    end_pos = tf.argmax(end_logits, axis=-1).numpy()[0]
    return decode_answer(vectorizer, input_tokens, start_pos, end_pos)

# === Main Execution ===
max_seq_len = 400
embed_dim = 16
num_heads = 2
ff_dim = 512
num_layers = 1
vocab_size = 100000

# Load dataset
squad = load_dataset("squad", split="train[:100]")

# Initialize TextVectorization
vectorizer = TextVectorization()
texts = [item['context'] + " " + item['question'] for item in squad]
vectorizer.adapt(texts)

# Prepare data
X, y_start, y_end = prepare_data(squad, vectorizer, max_seq_len)
X = tf.convert_to_tensor(X)
#y_start = to_categorical(y_start, num_classes=max_seq_len)
#y_end = to_categorical(y_end, num_classes=max_seq_len)

# Instantiate model
model = Bert(max_seq_len=max_seq_len, embed_dim=embed_dim, vocab_size=vocab_size, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers)
model.compile(optimizer=Adam(), 
              loss=[SparseCategoricalCrossentropy(from_logits=True), SparseCategoricalCrossentropy(from_logits=True)], 
              metrics=['accuracy' , 'accuracy'])

# Train model
model.fit(X, [y_start, y_end], epochs=40, batch_size=16, verbose=1, callbacks=[BestModelCallback()] , validation_split=0.01)

# Test Example
example = squad[0]
context = example['context']
question = example['question']
true_answer = example['answers']['text'][0]
predicted_answer = predict_answer(model, vectorizer, context, question, max_seq_len)

print("\nTest Result:")
print(f"Question: {question}")
print(f"True Answer: {true_answer}")
print(f"Predicted Answer: {predicted_answer}")
