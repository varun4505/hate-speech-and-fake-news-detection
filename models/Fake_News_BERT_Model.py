!pip install transformers torch pycaret pandas matplotlib scikit-learn -q

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from transformers import AutoModel, BertTokenizerFast, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# %% [markdown]
# ## Load Dataset

# %%
true_data = pd.read_csv("a1_True.csv")
fake_data = pd.read_csv("a2_Fake.csv")

true_data["Target"] = "True"
fake_data["Target"] = "Fake"

label_map = {"Fake": 1, "True": 0}
true_data["label"] = true_data["Target"].replace(label_map)
fake_data["label"] = fake_data["Target"].replace(label_map)

data = pd.concat([true_data, fake_data]).sample(frac=1, random_state=2018).reset_index(drop=True)

print(data.shape)
data.head()

# %%
label_size = [data["label"].sum(), len(data["label"]) - data["label"].sum()]
plt.pie(
    label_size,
    explode=[0.1, 0.1],
    colors=["firebrick", "navy"],
    startangle=90,
    shadow=True,
    labels=["Fake", "True"],
    autopct="%1.1f%%",
)

# %% [markdown]
# ## Train/Validation/Test Split

# %%
train_text, temp_text, train_labels, temp_labels = train_test_split(
    data["title"],
    data["label"],
    random_state=2018,
    test_size=0.3,
    stratify=data["label"],
)

val_text, test_text, val_labels, test_labels = train_test_split(
    temp_text,
    temp_labels,
    random_state=2018,
    test_size=0.5,
    stratify=temp_labels,
)

# %% [markdown]
# ## Load BERT and Tokenizer

# %%
bert = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# %% [markdown]
# ## Tokenization

# %%
MAX_LENGTH = 15

def encode_texts(texts):
    return tokenizer.batch_encode_plus(
        texts,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

tokens_train = encode_texts(train_text.tolist())
tokens_val = encode_texts(val_text.tolist())
tokens_test = encode_texts(test_text.tolist())

train_seq = tokens_train["input_ids"]
train_mask = tokens_train["attention_mask"]
train_y = torch.tensor(train_labels.tolist(), dtype=torch.long)

val_seq = tokens_val["input_ids"]
val_mask = tokens_val["attention_mask"]
val_y = torch.tensor(val_labels.tolist(), dtype=torch.long)

test_seq = tokens_test["input_ids"]
test_mask = tokens_test["attention_mask"]
test_y = torch.tensor(test_labels.tolist(), dtype=torch.long)

# %% [markdown]
# ## DataLoader Setup

# %%
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)
val_data = TensorDataset(val_seq, val_mask, val_y)

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

# %% [markdown]
# ## Define Model

# %%
class BERT_Arch(nn.Module):
    def __init__(self, bert_model):
        super(BERT_Arch, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        output = self.bert(sent_id, attention_mask=mask)
        cls_hs = output.pooler_output
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.log_softmax(x)

model = BERT_Arch(bert)
model = model.to(device)

# Freeze BERT encoder
for param in bert.parameters():
    param.requires_grad = False

# %% [markdown]
# ## Optimizer and Loss

# %%
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.NLLLoss()
epochs = 2

# %% [markdown]
# ## Training and Evaluation

# %%
def train_epoch():
    model.train()
    total_loss = 0

    for step, batch in enumerate(train_dataloader):
        sent_id, mask, labels = [b.to(device) for b in batch]
        optimizer.zero_grad()
        preds = model(sent_id, mask)
        loss = criterion(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

def evaluate_epoch():
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            sent_id, mask, labels = [b.to(device) for b in batch]
            preds = model(sent_id, mask)
            loss = criterion(preds, labels)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

# %% [markdown]
# ## Run Training

# %%
best_valid_loss = float("inf")
train_losses, valid_losses = [], []

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train_epoch()
    valid_loss = evaluate_epoch()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "bert_fakenews_best.pt")

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Training Loss: {train_loss:.3f} | Validation Loss: {valid_loss:.3f}")

# %% [markdown]
# ## Model Evaluation

# %%
model.load_state_dict(torch.load("bert_fakenews_best.pt"))
model.eval()

with torch.no_grad():
    sent_id, mask = test_seq.to(device), test_mask.to(device)
    preds = model(sent_id, mask)
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(classification_report(test_y, preds, target_names=["True", "Fake"]))

# %%
ConfusionMatrixDisplay.from_predictions(test_y, preds)

# %% [markdown]
# ## Unseen Predictions

# %%
unseen_news_text = [
    "Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing",
    "WATCH: George W. Bush Calls Out Trump For Supporting White Supremacy",
    "U.S. lawmakers question businessman at 2016 Trump Tower meeting: sources",
    "Trump administration issues new rules on U.S. visa waivers"
]

tokens_unseen = encode_texts(unseen_news_text)
unseen_seq = tokens_unseen["input_ids"].to(device)
unseen_mask = tokens_unseen["attention_mask"].to(device)

with torch.no_grad():
    preds = model(unseen_seq, unseen_mask)
    preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis=1)
print(preds)