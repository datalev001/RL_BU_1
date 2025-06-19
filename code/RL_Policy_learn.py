import pandas as pd
import numpy as np

# Read data from CSV
csv_file = 'layout_data_pg.csv'
df = pd.read_csv(csv_file)

if 'advantage' not in df.columns:
    baseline = df['reward'].mean()  # or df['click'].mean()
    df['advantage'] = df['reward'] - baseline
else:
    baseline = df['reward'].mean()

# Prepare data for policy gradient
sources = df['source'].unique().tolist()
devices = df['device'].unique().tolist()
layouts = df['layout'].unique().tolist()

src_map = {s:i for i,s in enumerate(sources)}
dev_map = {d:i for i,d in enumerate(devices)}
lay_map = {l:i for i,l in enumerate(layouts)}

X, A, ADV = [], [], []
for _, row in df.iterrows():
    src_oh = np.eye(len(sources))[src_map[row['source']]]
    dev_oh = np.eye(len(devices))[dev_map[row['device']]]
    state = np.concatenate([src_oh, dev_oh, [row['ctr_30d']]])
    X.append(state)
    A.append(lay_map[row['layout']])
    ADV.append(row['advantage'])
X = np.array(X)
A = np.array(A)
ADV = np.array(ADV)

n_samples = len(df)

# Initialize linear policy parameters
input_dim = X.shape[1]
output_dim = len(layouts)
W = np.random.randn(input_dim, output_dim) * 0.01
b = np.zeros(output_dim)
lr = 0.05

# Training loop (REINFORCE)
for epoch in range(1, 11):
    idx = np.random.permutation(n_samples)
    total_loss = 0.0
    for i in idx:
        x = X[i]
        a = A[i]
        adv = ADV[i]
        logits = x.dot(W) + b
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / exp_logits.sum()
        loss = -adv * np.log(probs[a] + 1e-8)
        total_loss += loss
        grad_logits = -adv * (np.eye(output_dim)[a] - probs)
        W -= lr * np.outer(x, grad_logits)
        b -= lr * grad_logits
    avg_loss = total_loss / n_samples
    print(f"Epoch {epoch}/10 | Avg Loss: {avg_loss:.4f}")

# Evaluation: expected CTR under learned policy
def get_prob_click(row, layout):
    base = row['ctr_30d']
    if row['device']=='mobile' and layout=='B': base += 0.2
    elif row['device']=='desktop' and layout=='A': base += 0.15
    elif row['device']=='tablet' and layout=='C': base += 0.1
    return min(base, 1.0)

total_ctr = 0.0
for i, row in df.iterrows():
    x = X[i]
    logits = x.dot(W) + b
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    chosen = np.argmax(probs)
    layout = layouts[chosen]
    total_ctr += get_prob_click(row, layout)
avg_ctr = total_ctr / n_samples

print(f"\nBaseline CTR: {baseline:.3f}")
print(f"Policy-Gradient Expected CTR: {avg_ctr:.3f}")