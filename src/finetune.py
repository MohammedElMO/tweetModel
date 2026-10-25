"""Part E: fine-tuning BERT, single-task (lam=0) or multi-task sarcasm + sentiment (lam>0)."""
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from .preprocess import clean_plm, remove_artifacts
from .evaluate import report, best_threshold, set_seed, Timer

SENT = {"NEG": 0, "NEU": 1, "POS": 2}
ARABERT = "aubmindlab/bert-base-arabertv02-twitter"
MARBERT = "UBC-NLP/MARBERTv2"


class MultiTask(nn.Module):
    """Shared BERT encoder + a sarcasm head (2 classes) + a sentiment head (3 classes)."""

    def __init__(self, name):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.sarc, self.sent = nn.Linear(h, 2), nn.Linear(h, 3)

    def forward(self, ids, mask):
        cls = self.drop(
            self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.sarc(cls), self.sent(cls)


def encode(name, df, max_length=64):
    """Tokenize with the model's own tokenizer; AraBERT gets its preprocessor, MARBERT raw text."""
    prep = clean_plm if "arabert" in name else remove_artifacts
    tok = AutoTokenizer.from_pretrained(name)
    e = tok([prep(t) for t in df.tweet], padding="max_length", truncation=True,
            max_length=max_length, return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


def finetune(name, short, train, val, test, lam=0.0, seed=0,
             epochs=4, bs=32, lr=2e-5, device="cuda"):
    set_seed(seed)
    ids_tr, m_tr = encode(name, train)
    ids_va, m_va = encode(name, val)
    ids_te, m_te = encode(name, test)
    y_s = torch.tensor(train.y.values)
    y_t = torch.tensor(train.sentiment.map(SENT).values)
    loader = DataLoader(TensorDataset(ids_tr, m_tr, y_s,
                        y_t), batch_size=bs, shuffle=True)

    model = MultiTask(name).to(device)
    w = torch.tensor(len(train) / (2 * np.bincount(train.y)),
                     dtype=torch.float, device=device)
    ce_sarc, ce_sent = nn.CrossEntropyLoss(weight=w), nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    scaler = torch.amp.GradScaler("cuda")

    @torch.no_grad()
    def probs(ids, m):
        model.eval()
        out = []
        for k in range(0, len(ids), 128):
            with torch.autocast("cuda", dtype=torch.float16):
                ls, _ = model(ids[k:k + 128].to(device),
                              m[k:k + 128].to(device))
            out.append(torch.softmax(ls.float(), -1)[:, 1].cpu())
        return torch.cat(out).numpy()

    best, state = -1, None
    with Timer() as t:
        for ep in range(epochs):
            model.train()
            for a, b, c, d in loader:
                a, b, c, d = a.to(device), b.to(
                    device), c.to(device), d.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    ls, lt = model(a, b)
                    loss = ce_sarc(ls.float(), c) + lam * \
                        ce_sent(lt.float(), d)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
            _, f1 = best_threshold(val.y.values, probs(ids_va, m_va))
            print(f"{short} lam={lam} seed={seed} epoch {ep + 1}  val F1 = {f1:.4f}")
            if f1 > best:
                best = f1
                state = {k: v.detach().cpu().clone()
                         for k, v in model.state_dict().items()}
            else:
                break                                    # early stopping
    model.load_state_dict(state)

    tag = f"E2_{short}_lam{lam}" if lam > 0 else f"E1_{short}"
    report(tag, val.y.values, probs(ids_va, m_va), test.y.values, probs(ids_te, m_te),
           seed=seed, prep=short, train_time=t.s,
           n_params=sum(p.numel() for p in model.parameters()))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return best
