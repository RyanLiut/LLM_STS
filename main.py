'''
To test the vector similarity for a sentence pair.
Datasets
  - STS-B
'''
#%%
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/liuzhu')
from SOX.utilize import repre_extractor, get_cos_similairty

def get_data(path, name):
    if name == "STS-B":
        df = pd.read_csv(path, sep="\t", header=None,usecols=[0, 1, 2, 3, 4,5,6])
        df.columns = ['domain', 'type', 'year', 'id', 'score', 'sent1', 'sent2']
        assert len(df) == 1379, print(len(df))
        # print(df.head(5))
        rows_with_na = df.isna().any(axis=1)
        print("包含 NaN 值的行:")
        print(df[rows_with_na])

        sent1_list = df['sent1'].tolist()
        sent2_list = df['sent2'].tolist()
        score = df['score'].tolist()
        id = df['id'].tolist()
        # assert len(set(id)) == len(id), print(len(set(id)), len(id))

        return sent1_list, sent2_list, score, id



# if __name__ == "__main__":
#%%
path = "/home/liuzhu/datasets/STS-B/original/sts-test.tsv"
name = "STS-B"
sent1_list, sent2_list, score, id = get_data(path, name)
# for i in range(5):
#     print(sent1_list[i], sent2_list[i], score[i], id[i])

X = repre_extractor('/home/liuzhu/models/llama-main/llama-2-7b-hf', model_name="Llama")
#%%
prompts = [
    "This sentence : \"%s\" means in one word : \"",
    "In this sentence %s, it means in one word :",
    "This sentence : \"%s\" means something",
    "This sentence : \"%s\" can be summarized as",
    "After thinking step by step, this sentence : \"%s\" means in one word : \"",
    "The essence of a sentence is often captured by its main subjects and actions, while descriptive terms provide additional but less central details. With this in mind , this sentence : \"%s\" means in one word : \""
]

rep_sent1 = X.get_data_represents(sent1_list, 50, None, "Prompt_Last", prompts[5], True)
rep_sent2 = X.get_data_represents(sent2_list, 50, None, "Prompt_Last", prompts[5], True)
print(rep_sent2.shape)

#%%
pearsons = []
for layer in range(rep_sent1.shape[1]):
    print("------Layer: %d-----" % layer)
    rep_sim = get_cos_similairty(rep_sent1[:,layer], rep_sent2[:,layer],ani_removal=(not rep_sent1[:,layer].equal(rep_sent2[:,layer])))
    print(rep_sim.shape)
    print("Max: %f\tMin: %f\tMean: %f" % (rep_sim.max().item(), rep_sim.min().item(), rep_sim.mean().item()))

    corr_p, _ = pearsonr(rep_sim, score)
    corr_s, _ = spearmanr(rep_sim, score)
    pearsons.append(corr_p)
    print("Pearson's r: %f" % corr_p)
    print("Spearsman: %f" % corr_s)

#%%
plt.figure()
plt.plot(pearsons, marker="o")
plt.title("Pearson's r on STS-B test set by P5")
plt.xlabel("Layer Index")
plt.ylabel("Pearson")
# %%
