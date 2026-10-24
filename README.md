# practice dl

My second practice project, this time deep learning. I go through 4 topics one step at a time and run everything on my mac (M-series, pytorch with mps).

- A - CNNs on images (CIFAR-10)
- B - text classification (NLP)
- C - deploying my BTC model as a streamlit app
- D - reinforcement learning (FrozenLake, Taxi, CartPole)

For every model i also run a simple baseline, so i know whether it's actually learning something.

## setup

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## A1 - CNN from scratch on CIFAR-10

`python a1_cnn_cifar.py`

Test accuracy on 10k photos the models never saw during training:

| model | test acc | weights |
|---|---|---|
| just guessing | 10.0% | - |
| linear (logistic regression on pixels) | 40.1% | 30,730 |
| MLP | 51.7% | 1,707,274 |
| CNN | 88.0% | 289,194 |

What i noticed:
- The CNN has about 6x fewer weights than the MLP and still beats it by more than 35 points. Sliding the same small filters over the whole image is a lot better than giving every pixel its own weight.
- Train and val accuracy stayed close the whole time (89.5% vs 88.6%), so with augmentation it barely overfit.
- Cat is the hardest class (74%) and it mostly gets confused with dog. Car/truck and plane/ship get mixed up too, which makes sense since they look alike at 32x32.

## A2 - transfer learning with pretrained ResNet18

`python a2_transfer.py` (needs a1_cnn.pt from A1)

Same test set, same 10 classes. Baseline is my A1 cnn.

| model | trained on | test acc |
|---|---|---|
| just guessing | - | 10.0% |
| A1 cnn from scratch | 45k imgs | 88.0% |
| frozen resnet + new last layer | 45k imgs | 85.9% |
| finetuned resnet (3 epochs) | 45k imgs | **95.2%** |
| cnn from scratch | 2k imgs | 67.0% |
| finetuned resnet | 2k imgs | 87.8% |

What i noticed:
- Finetuning the whole resnet for only 3 epochs (~2.5 min) beat my cnn by 7 points. Starting from imagenet weights is a huge head start.
- Frozen resnet (only the last layer trained, 5k weights) got 85.9%, which is almost as good as my whole cnn. The features it learned on imagenet already work for cifar without any changes.
- The small data test is the big one. With only 2,000 photos my cnn drops to 67% and overfits a lot (train 88% vs val 68%). The pretrained resnet with the same 2,000 photos gets 87.8%, about the same as my cnn trained on all 45k. So pretraining is basically worth 20x more labelled data here.
- Finetuned resnet on 2k got 100% train accuracy but only 88% val, so it memorised the training set. It still generalised way better than training from scratch though.

## B1 - bag of words on IMDB reviews

`python b1_bow_imdb.py`

Positive or negative movie review? 25k reviews for training, 25k for testing. Word order is ignored, so each review is just word counts.

| model | test acc |
|---|---|
| always say positive | 50.0% |
| my good/bad word list rule | 70.5% |
| naive bayes | 81.5% |
| logreg (tf-idf, single words) | 88.7% |
| logreg (tf-idf, words + pairs) | **90.6%** |

What i noticed:
- My hand-written word list got 70.5%. Just letting logistic regression learn the weights for 45k words gets 88.7%, so the model finds way more useful words than i could write down.
- The words it learned make sense: worst, awful, boring, waste on the negative side and great, excellent, perfect, superb on the positive side. "Unfortunately" and "nothing" are strongly negative too, which i wouldn't have guessed.
- Adding word pairs helped (+1.9 points). "not funny" gets -3.15 even though "funny" is +0.86.
- But it still fails on "this movie was not bad at all" (P(positive) = 0.01). "not bad" is only +1.48 while "bad" is -13.03, and since order doesn't matter it can't tell that "not" flips the meaning. That's the motivation for B2/B3, models that actually read the words in order.
- The reviews it gets wrong are mostly long ones that describe the plot, or say bad things about other movies before saying this one is good.

## B2 - word embeddings + BiLSTM

`python b2_embeddings.py`

Every word gets a vector of 100 numbers that the network learns (30k words, so 3M weights just for that). Same IMDB data, with 2.5k of the training reviews kept for validation.

| model | test acc |
|---|---|
| B1 logreg words + pairs (baseline) | **90.6%** |
| avg embeddings (order ignored) | 87.4% |
| bilstm (reads in order) | 87.1% |

What i noticed:
- Neither neural model beat my simple logistic regression from B1. With only 22.5k reviews, learning 3M embedding numbers from scratch isn't enough data. Bag of words + pairs is a really strong baseline for sentiment.
- The avg model overfits fast: train went up to 96% while val stopped at ~90% after epoch 4.
- The embeddings did learn meaning though. Closest to "awful" are waste, worst, disappointment, boring, horrible, and closest to "excellent" are wonderful, fantastic, gem, and also "7" (people write things like "7/10"). But they only learned sentiment, because that's the only thing the model was trained for. "actor" and "funny" got random neighbours.
- Reading in order didn't really fix "not bad at all" (bilstm 0.09). It did move "this movie was not good" and "great actors, terrible script" closer to the middle compared to the avg model, which said 0.00 for most of them. Short sentences are also very different from the 230-word reviews it was trained on.
- The quick run with 2.7k reviews gave nearly random neighbours and 56% for the lstm, so these models really need data.

## B3 - pretrained transformer (DistilBERT)

`python b3_transformer.py`

DistilBERT (67M weights) is already pretrained on a lot of English text. All three are tested on the same 5,000 test reviews, and reviews get cut at 256 tokens.

| model | imdb reviews it trained on | test acc |
|---|---|---|
| B1 logreg words + pairs (baseline) | 25,000 | **90.6%** |
| distilbert finetuned on SST-2 by someone else | 0 | 86.8% |
| my distilbert finetune (2 epochs, ~3.5 min) | 4,000 | 88.2% |

What i noticed:
- The transformer didn't beat my logreg in this first try. I think the main reasons are: (1) it only reads the first 256 tokens, and imdb reviews are often longer than that, with the actual verdict at the end, (2) i only used 4k of the 25k reviews, and (3) val was best after epoch 1 (88.4%) and went down in epoch 2.
- Still, with 4k reviews it beat both B2 models that used all 22.5k (~87%). Pretraining is worth a lot of data, same as resnet in A2.
- The SST-2 model got 86.8% without ever seeing an imdb review. It was trained on short sentences, which shows when you look at my test sentences: it's the only model that says "not bad at all" is positive (1.00), but it also says "great actors, terrible script" is positive (0.99).
- My finetune got "i expected it to be boring but it was great" right (0.85), but still not "not bad at all" (0.17).
- The quick run with only 300 reviews gave 67% and every sentence ~0.48, so 300 isn't enough to even move the new classifier layer much.
**Second try:** `python b3_transformer.py --train-size 8000 --max-len 384 --head-tail` (~10 min of training)

Here it reads 384 tokens instead of 256, and for long reviews it keeps the first 128 tokens + the end instead of cutting the end off. It also trains on 8k reviews instead of 4k.

| model | imdb reviews it trained on | test acc |
|---|---|---|
| B1 logreg words + pairs | 25,000 | 90.6% |
| sst-2 distilbert (no training, 384 head+tail) | 0 | 88.5% |
| my distilbert finetune | 8,000 | **91.8%** |

- Now the transformer beats logreg, and with only a third of the reviews. Val was 94.0% after epoch 2.
- The SST-2 model also went from 86.8% to 88.5% just from reading the end of the reviews, without any training. So cutting off the end really was losing information.
- My finetune now gets "great actors, terrible script" right (0.01) and "expected boring but great" (0.98). "not bad at all" is still wrong (0.08) for everything except SST-2. In imdb reviews "not bad" is probably used more for meh movies than for good ones.
- Lesson: how you cut the text mattered almost as much as the model.

- The "MISSING / UNEXPECTED" load report is normal. The pretraining layers get dropped, and the new classifier layer starts random, which is the part i finetune.

## C - BTC direction model as a live streamlit app

`streamlit run c_btc_app.py` and then open http://localhost:8501

Every few seconds it gets the BTC price from coinbase, checks the last predictions once the price moves, updates the online logreg, and predicts the next move. The logreg first learns from my old prices.csv (1,386 recorded moves).

First live run (~30 min, a price every 8 s, 119 moves checked):

| model | live accuracy |
|---|---|
| coin flip | 50% |
| majority (baseline) | 49% |
| bounce rule | **42%** |
| online logreg | 42% |

What i noticed:
- The bounce rule got 66% on my recorded data in the realtime project, but live it's at 42%, worse than a coin flip. If bounce is right 42% of the time, then "same as the last move" (momentum) would be right 58%. So right now the moves are continuing, not bouncing back.
- My guess is it's because of the sampling interval. I recorded prices.csv every 2 s, where the price jumps between bid and ask. At 8 s that bounce mostly disappears and you see the actual trend. 119 moves is still small though (42% is about 1.7 standard errors below 50%), so i need to let it run longer to be sure.
- The logreg learned "bounce" from the old data and just copied it (also 42%). It learned too slowly to adapt, so i raised its learning rate and added a momentum rule to the app so i can watch both.
- Big lesson: a pattern that worked on old data can stop working when the setup changes. That's concept drift, same thing step6 in the realtime project was about. An app has to keep checking itself live, which is why this one shows rolling accuracy.
**Second live run** (~15 min, a price every **3 s**, 119 moves, with the momentum rule added):

| model | live accuracy |
|---|---|
| coin flip | 50% |
| majority | 52% |
| bounce rule | **61%** |
| momentum rule | 39% |
| online logreg (faster lr) | 52% |

- At 3 s the bounce came back: 61% (and momentum 39%, the mirror image). At 8 s it was 42%. So my guess was right, the bounce pattern only shows up when you sample fast enough to catch the price flipping between bid and ask. The "model" is really a model of how coinbase quotes the price, not of where btc is going.
- The price chart now shows the real moves. Most of the jumps are a few dollars up and down around the same level, which is exactly the bounce.
- The logreg only got 52%. With the higher learning rate it reacts to every move and keeps changing its mind, so the simple rule wins. There's a tradeoff between adapting fast and being stable.

- Two display bugs i fixed: the price chart started the y axis at 0 so btc looked flat (switched to altair with zero=False), and "ticks with no move" was counting the flat ticks from prices.csv too (3,695).

## D1 - Q-learning (FrozenLake + Taxi)

`python d1_qlearning.py`

No dataset here. The agent plays the game, gets rewards, and fills a table Q[state, action] with how much reward it expects. It explores randomly at first and less over time.

| game | random agent | q-learning |
|---|---|---|
| FrozenLake 4x4 slippery (reach the goal) | 0.9% | **74.2%** |
| Taxi (avg reward per ride) | -767.7 | **+7.9** |

What it learned on FrozenLake:
```
← ↑ ↑ ↑
← H → H
↑ ↓ ← H
H → ↓ G
```

What i noticed:
- On the start square it goes ← (into the wall) instead of ↓ or →. That looked wrong at first, but the ice is slippery and you only go where you wanted 1/3 of the time, otherwise sideways. Pushing into the wall means the slip can only send you down or keep you in place, never into a hole. It found a safe strategy that i wouldn't have thought of.
- 74% is about as good as it gets on slippery 4x4. Some falls just can't be avoided.
- Random taxi basically never finishes (196 of max 200 steps) and loses a lot from wrong pickups/dropoffs. Q-learning does a ride in 13 steps: drive to the passenger, pick up, drive to the destination, drop off.
- Training was super fast (2-3 s) because the tables are tiny: 16x4 for FrozenLake and 500x6 for Taxi.

## D2 - DQN on CartPole

`python d2_dqn_cartpole.py`

Balance a pole on a cart by pushing left or right, +1 for every step it stays up, max 500. The state is 4 continuous numbers, so a plain table doesn't work. DQN uses a small neural network (4 -> 128 -> 128 -> 2) instead, plus a replay buffer and a slowly updated target network.

| agent | avg steps over 100 test episodes |
|---|---|
| random pushes | 21.9 |
| my hand rule (push towards where the pole falls) | 500.0 |
| q-table with bins (3,600 states, 3000 episodes) | 487.8 |
| dqn (400 episodes, 20 s on cpu) | **500.0** |

What i noticed:
- DQN got the perfect 500 without knowing any physics. It learned it all from the +1 rewards.
- My 1-line rule also got 500, so honestly CartPole is easy if you know the physics. The cool part is that DQN figured it out without being told.
- DQN training is really unstable. My greedy test hit 370 at episode 80, then dropped to 43 at episode 200, then 500 again at 340, 96 at 380, and 500 at 400. The network can "forget" what it learned when it trains on new experience. That's why the script tests every 20 episodes and keeps the best version.
- The first settings (explore for 10k steps, slower target network) got stuck around 300 steps. Exploring for a shorter time and updating the target net faster got to 500.
- The q-table with bins also did well (487.8) but needs me to pick the bins by hand. With more state numbers the table would explode (bins^dimensions), and that's the real reason for using a network.

## my main takeaways

1. **Pretrained models are worth a ton of data.** ResNet on 2k images matched my CNN on 45k (A2). DistilBERT on 8k reviews beat logreg on 25k (B3).
2. **Simple baselines are hard to beat.** Bag of words logreg beat both neural nets in B2. A 1-line physics rule balanced CartPole perfectly (D2).
3. **The data setup matters as much as the model.** Reading the end of reviews took DistilBERT from 88.2% to 91.8%. Changing the price interval from 3 s to 8 s turned the bounce rule from 61% into 42%.
4. **Patterns from old data can stop working live** (C), so a deployed model has to keep checking its own accuracy.
5. **Deep learning training is noisy.** DQN went 370 -> 43 -> 500, and val accuracy jumped around in every track. Keeping the best checkpoint on validation saved me every time.

## problems i ran into (and how i fixed them)

- **pip not found in my venv.** `pip install torchvision` gave `zsh: command not found: pip`, and `python -m pip` said `No module named pip`. My new DeepLearning folder's .venv had been made without pip. I fixed it with `python -m ensurepip --upgrade`, and since then i always use `python -m pip install ...` so it goes into the venv that's active.
- **VS Code "Extension activation failed" popup.** It didn't stop anything from running in the terminal, so i ignored it.
- **IMDB download failed in B1.** The script downloaded from Stanford's server (ai.stanford.edu) and got `URLError: [Errno 61] Connection refused`, so their server was down or blocking. It's the same dataset on Hugging Face, so i switched to downloading the parquet files from there (needs `pandas` + `pyarrow`) and kept Stanford as a backup. Lesson: don't depend on one download link for a dataset.
- **GPU out of memory in B2.** The bilstm trained fine, but testing it on all 25k reviews crashed with `RuntimeError: MPS backend out of memory`. I was predicting 500 reviews x 300 words at a time and the mac gpu memory had filled up during training. First i tried predicting in batches of 100 and calling `torch.mps.empty_cache()`, but it still crashed (and once i was actually running the old file because i hadn't saved the new one). In the end i made predict fall back to the cpu when the gpu runs out of memory, and it printed `(gpu out of memory, predicting on cpu instead)` and finished. The "other allocations: 28.90 GiB" in the error makes me think the mps lstm keeps memory from training around.
- **Streamlit app display bugs (C).** The price chart started at 0, so btc looked like a flat line. I switched to altair with `zero=False`. The "ticks with no move" counter also included the flat ticks from my old prices.csv (3,695), so i reset it after the warm start. When i restarted the app it opened on port 8502, because the old one was still running on 8501. Ctrl+C in the old terminal fixed that.
- **DQN stuck (D2).** My first settings (explore for 10k steps, tau 0.005) got stuck around 100-300 steps. Exploring for 5k steps and tau 0.01 got to 500.
