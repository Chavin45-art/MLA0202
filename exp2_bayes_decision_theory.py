"""
Experiment 2: Bayes Decision Theory
Dataset: SMS Spam Collection Dataset
Task: Implement Bayes' Theorem to calculate the posterior probability for a
      given message and classify it as Spam or Ham (Not Spam).

Note: A representative labelled sample (in the same two-column
"label \t message" format as the original UCI SMS Spam Collection
Dataset) is used here so the full Naive-Bayes / Bayes'-theorem pipeline
(tokenisation -> word likelihoods -> posterior -> classification) can be
demonstrated end-to-end and reproduced by students on any machine.
"""

import re
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

# ----------------------------------------------------------------------
# 1. Dataset (label, message) - SMS Spam Collection style corpus
# ----------------------------------------------------------------------
data = [
    ("ham", "Ok lar... Joking wif u oni..."),
    ("ham", "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight"),
    ("ham", "Ok i thk i got it. Then u wan me 2 come now or later?"),
    ("ham", "I HAVE A DATE ON SUNDAY WITH WILL!!"),
    ("ham", "As per your request 'Melle Melle' has been set as your callertune"),
    ("ham", "Nah I don't think he goes to usf, he lives around here though"),
    ("ham", "Even my brother is not like to speak with me. They treat me like aids patent."),
    ("ham", "I HAVE A DATE ON SUNDAY WITH WILL!!"),
    ("ham", "Sorry, I'll call later in meeting"),
    ("ham", "Can you send me the notes from class today"),
    ("ham", "Are you coming to the party tonight? Let me know"),
    ("ham", "Happy birthday to you, have a wonderful day"),
    ("ham", "Let's meet for lunch tomorrow at the usual place"),
    ("ham", "I will call you back after my meeting is over"),
    ("ham", "Please review the report and send your feedback"),
    ("ham", "Thanks for helping me move last weekend, really appreciate it"),
    ("ham", "See you at the gym later this evening"),
    ("ham", "Mom said dinner will be ready by eight"),
    ("ham", "Can we reschedule our call to tomorrow morning"),
    ("ham", "The movie was great, we should watch the sequel soon"),
    ("spam", "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry"),
    ("spam", "WINNER!! As a valued network customer you have been selected to receive a 900 prize reward! Call now"),
    ("spam", "Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free"),
    ("spam", "SIX chances to win CASH! From 100 to 20,000 pounds txt CSH11 and send to 87575"),
    ("spam", "URGENT! You have won a 1 week FREE membership in our 100,000 Prize Jackpot! Txt WORD to 81010"),
    ("spam", "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message"),
    ("spam", "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077"),
    ("spam", "Congratulations! You have won a free cruise to the Bahamas, call now to claim your prize"),
    ("spam", "You have been selected to receive a FREE cash prize of 5000 pounds, text CLAIM to 88888"),
    ("spam", "URGENT! Your mobile number has won 2000 pounds in our lucky draw, call now to claim"),
    ("spam", "Get a FREE ringtone and win cash prizes, txt WIN to 80086 now"),
    ("spam", "Claim your free holiday voucher now, call 09061701444 to claim your prize"),
    ("spam", "You have won a guaranteed 1000 cash or a prize, to claim call 09050000928"),
    ("spam", "FreeMsg: Text CASH to 86688 and win up to 500 pounds every week"),
    ("spam", "Reply WIN to this free message to claim your cash prize of 10000 pounds now"),
    ("spam", "Congratulations, you have been selected for a free gift voucher worth 500 pounds, call now"),
    ("spam", "Urgent! Call this number now to claim your free prize before it expires"),
    ("spam", "You have won a brand new car, text CAR to 12345 to claim your prize now"),
    ("spam", "Get cash now! No credit check required, apply free at our website today"),
    ("spam", "Win a free iPhone today, just click the link and register now"),
]

df = pd.DataFrame(data, columns=["label", "message"])

print("=" * 70)
print("EXPERIMENT 2: BAYES DECISION THEORY - SMS Spam Collection Dataset")
print("=" * 70)
print(f"\nTotal messages: {len(df)}")
print(df["label"].value_counts())


# ----------------------------------------------------------------------
# 2. Text preprocessing (tokenisation)
# ----------------------------------------------------------------------
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens


df["tokens"] = df["message"].apply(tokenize)

# ----------------------------------------------------------------------
# 3. Prior probabilities  P(Spam), P(Ham)
# ----------------------------------------------------------------------
n_total = len(df)
n_spam = (df["label"] == "spam").sum()
n_ham = (df["label"] == "ham").sum()

p_spam = n_spam / n_total
p_ham = n_ham / n_total

print("\n--- Step 1: Prior Probabilities ---")
print(f"P(Spam) = {n_spam}/{n_total} = {p_spam:.4f}")
print(f"P(Ham)  = {n_ham}/{n_total}  = {p_ham:.4f}")

# ----------------------------------------------------------------------
# 4. Build word frequency tables (Multinomial Naive Bayes with
#    Laplace/add-1 smoothing)
# ----------------------------------------------------------------------
spam_words = Counter()
ham_words = Counter()

for _, row in df.iterrows():
    if row["label"] == "spam":
        spam_words.update(row["tokens"])
    else:
        ham_words.update(row["tokens"])

vocab = set(spam_words.keys()) | set(ham_words.keys())
vocab_size = len(vocab)

total_spam_words = sum(spam_words.values())
total_ham_words = sum(ham_words.values())

print(f"\nVocabulary size: {vocab_size}")
print(f"Total words in Spam messages: {total_spam_words}")
print(f"Total words in Ham messages: {total_ham_words}")


def word_likelihood(word, cls_word_counts, total_cls_words, vocab_size):
    """P(word | class) with Laplace (add-1) smoothing."""
    count = cls_word_counts.get(word, 0)
    return (count + 1) / (total_cls_words + vocab_size)


def classify_message(message, verbose=True):
    tokens = tokenize(message)

    log_p_spam = math.log(p_spam)
    log_p_ham = math.log(p_ham)

    detail = []
    for w in tokens:
        p_w_spam = word_likelihood(w, spam_words, total_spam_words, vocab_size)
        p_w_ham = word_likelihood(w, ham_words, total_ham_words, vocab_size)
        log_p_spam += math.log(p_w_spam)
        log_p_ham += math.log(p_w_ham)
        detail.append((w, p_w_spam, p_w_ham))

    # Convert back from log-space to a normalised posterior probability
    max_log = max(log_p_spam, log_p_ham)
    spam_unnorm = math.exp(log_p_spam - max_log)
    ham_unnorm = math.exp(log_p_ham - max_log)
    total = spam_unnorm + ham_unnorm

    post_spam = spam_unnorm / total
    post_ham = ham_unnorm / total

    predicted = "spam" if post_spam > post_ham else "ham"

    if verbose:
        print(f"\nMessage: \"{message}\"")
        print(f"Tokens: {tokens}")
        print("\nPer-word likelihoods P(word|Spam) vs P(word|Ham):")
        for w, ps, ph in detail:
            print(f"   {w:12s}  P(w|Spam)={ps:.5f}   P(w|Ham)={ph:.5f}")
        print(f"\nlog P(Spam) + sum(log P(word|Spam)) = {log_p_spam:.4f}")
        print(f"log P(Ham)  + sum(log P(word|Ham))  = {log_p_ham:.4f}")
        print(f"\nPosterior P(Spam | message) = {post_spam:.6f}")
        print(f"Posterior P(Ham  | message) = {post_ham:.6f}")
        print(f"==> CLASSIFIED AS: {predicted.upper()}")

    return predicted, post_spam, post_ham


# ----------------------------------------------------------------------
# 5. Classify new/unseen test messages
# ----------------------------------------------------------------------
test_messages = [
    "Congratulations! You have won a free prize, call now to claim your cash reward",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT! Claim your free cash prize now by texting WIN to 12345",
    "Can you call me back when you get a chance",
]

print("\n" + "=" * 70)
print("Step 2 & 3: Word Likelihoods and Posterior Probability Calculation")
print("=" * 70)

results = []
for msg in test_messages:
    pred, ps, ph = classify_message(msg)
    results.append({"message": msg, "predicted": pred, "P(Spam|msg)": ps, "P(Ham|msg)": ph})
    print("-" * 70)

results_df = pd.DataFrame(results)

# ----------------------------------------------------------------------
# 6. Evaluate accuracy on the training corpus itself (sanity check)
# ----------------------------------------------------------------------
correct = 0
for _, row in df.iterrows():
    pred, _, _ = classify_message(row["message"], verbose=False)
    if pred == row["label"]:
        correct += 1
train_accuracy = correct / len(df)
print(f"\nSanity-check accuracy on training corpus: {correct}/{len(df)} = {train_accuracy:.4f}")

# ----------------------------------------------------------------------
# 7. Save results
# ----------------------------------------------------------------------
with open("/home/claude/lab/results/exp2_results.txt", "w") as f:
    f.write("EXPERIMENT 2 RESULTS: BAYES DECISION THEORY (SMS Spam Classification)\n\n")
    f.write(f"Total messages: {n_total} (Spam={n_spam}, Ham={n_ham})\n")
    f.write(f"P(Spam) = {p_spam:.4f}, P(Ham) = {p_ham:.4f}\n\n")
    for r in results:
        f.write(f"Message: {r['message']}\n")
        f.write(f"  P(Spam|msg) = {r['P(Spam|msg)']:.6f}\n")
        f.write(f"  P(Ham|msg)  = {r['P(Ham|msg)']:.6f}\n")
        f.write(f"  Predicted: {r['predicted'].upper()}\n\n")
    f.write(f"Training-corpus sanity-check accuracy: {train_accuracy:.4f}\n")

results_df.to_csv("/home/claude/lab/results/exp2_predictions.csv", index=False)

# ----------------------------------------------------------------------
# 8. Visualisation
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(["Ham", "Spam"], [p_ham, p_spam], color=["#5cb85c", "#d9534f"])
axes[0].set_title("Prior Probabilities: P(Ham) vs P(Spam)")
axes[0].set_ylabel("Probability")
for i, v in enumerate([p_ham, p_spam]):
    axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

x = range(len(results_df))
axes[1].bar([i - 0.2 for i in x], results_df["P(Spam|msg)"], width=0.4, label="P(Spam|msg)", color="#d9534f")
axes[1].bar([i + 0.2 for i in x], results_df["P(Ham|msg)"], width=0.4, label="P(Ham|msg)", color="#5cb85c")
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([f"Msg{i+1}" for i in x])
axes[1].set_title("Posterior Probabilities for Test Messages")
axes[1].legend()

plt.tight_layout()
plt.savefig("/home/claude/lab/results/exp2_posteriors.png", dpi=150)
plt.close()

print("\nResults, predictions CSV, and plots saved to /home/claude/lab/results/")
