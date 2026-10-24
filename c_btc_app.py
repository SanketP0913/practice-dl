# C - my BTC direction model as a live web app (streamlit)
#
# In the realtime project (step5) the best way to guess the next BTC move was the
# "bounce" rule: say the opposite of the last move. It got 66% on my recorded data
# because of the bid/ask bounce, and even an LSTM couldn't beat it.
# This time i'm not trying to beat it, i want to put the model in an actual app
# that anyone can open in a browser and watch predict live.
#
# Every few seconds the app:
#   1. gets the BTC price from coinbase (free, no key)
#   2. if the price moved, checks the last predictions against what actually happened
#   3. updates the online logistic regression with that move
#   4. predicts the next move with every model
# Models: majority (baseline), bounce rule, momentum rule (the opposite of bounce),
# online logreg on the signs of the last 3 moves.
# If prices.csv from the realtime project is in the folder, the logreg learns from it first.
#
# pip install streamlit
# streamlit run c_btc_app.py
# then it opens in the browser at http://localhost:8501

import csv
import math
import os
import random
import time
from collections import deque

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.linear_model import SGDClassifier

URL = "https://api.coinbase.com/v2/prices/BTC-USD/spot"
N_MOVES = 3          # how many past moves the logreg looks at
ACC_WINDOW = 100     # rolling accuracy over the last 100 moves
MODELS = ["majority", "bounce", "momentum", "logreg"]


def fetch_price():
    try:
        return float(requests.get(URL, timeout=5).json()["data"]["amount"])
    except (requests.RequestException, KeyError, ValueError):   # no internet or a weird reply
        return None


def fake_price():
    # random walk + bid/ask bounce, so i can test the app without internet
    s = st.session_state
    if random.random() < 0.3:
        s.true_price *= math.exp(random.gauss(0, 0.0003))
    return round(s.true_price + random.choice([-1, 1]) * 2, 2)


class OnlineLogReg:
    def __init__(self):
        # eta0 was 0.01 but then it just stayed a copy of bounce from the old data.
        # bigger learning rate so it can change its mind faster when the market changes
        self.m = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=0.05, random_state=0)
        self.fitted = False

    def prob_up(self, x):
        if not self.fitted:
            return 0.5
        return float(self.m.predict_proba(np.array([x]))[0, 1])

    def learn(self, x, y):
        self.m.partial_fit(np.array([x]), [y], classes=[0, 1])
        self.fitted = True


def init_state():
    s = st.session_state
    s.prices = []                        # (time, price) for the chart
    s.prev = None
    s.moves = deque(maxlen=N_MOVES)      # signs of the last moves, +1 or -1
    s.pending = None                     # (features, predictions) waiting for the next move
    s.counts = {0: 0, 1: 0}              # downs and ups seen so far
    s.logreg = OnlineLogReg()
    s.hits = {m: [] for m in MODELS}     # 1 if the model was right, for each move
    s.log = []                           # table of recent predictions
    s.flat = 0
    s.warm = 0
    s.true_price = 60000.0
    if os.path.exists("prices.csv"):
        warm_start("prices.csv")


def predict_all(x):
    s = st.session_state
    p = s.logreg.prob_up(x)
    return {
        "majority": 1 if s.counts[1] >= s.counts[0] else 0,
        "bounce": 0 if x[-1] > 0 else 1,
        "momentum": 1 if x[-1] > 0 else 0,
        "logreg": 1 if p >= 0.5 else 0,
        "p_up": p,
    }


def update(price, live=True):
    # one new price -> score old predictions, learn, predict again.
    # same logic as step5 in the realtime project
    s = st.session_state
    if s.prev is None:
        s.prev = price
        return
    r = math.log(price / s.prev)
    s.prev = price
    if r == 0:
        s.flat += 1          # price didn't move, nothing to check
        return
    y = 1 if r > 0 else 0
    if s.pending is not None:
        x_prev, preds = s.pending
        if live:
            for m in MODELS:
                s.hits[m].append(int(preds[m] == y))
            s.log.append({"time": time.strftime("%H:%M:%S"), "price": price,
                          "actual": "UP" if y else "DOWN",
                          **{m: ("UP" if preds[m] else "DOWN") + (" ✓" if preds[m] == y else " ✗")
                             for m in MODELS}})
        s.logreg.learn(x_prev, y)
    s.counts[y] += 1
    s.moves.append(1 if y else -1)
    if len(s.moves) == N_MOVES:
        x = list(s.moves)
        s.pending = (x, predict_all(x))


def warm_start(path):
    # let the logreg learn from my old recorded prices before going live
    s = st.session_state
    with open(path) as f:
        prices = [float(row["price"]) for row in csv.DictReader(f)]
    for p in prices:
        update(p, live=False)
    s.warm = sum(s.counts.values())
    s.counts = {0: 0, 1: 0}    # majority should only count live moves
    s.prev = None              # don't compare the old last price with the live one
    s.moves.clear()
    s.pending = None
    s.flat = 0                 # only count flat ticks from the live part


def main():
    st.set_page_config(page_title="BTC next move", page_icon="📈", layout="wide")
    st.title("BTC: will the next move be up or down?")
    st.caption("Live price from Coinbase. Predictions get checked the next time the price moves.")

    if "prices" not in st.session_state:
        init_state()

    with st.sidebar:
        interval = st.slider("seconds between prices", 2, 15, 3)
        simulate = st.checkbox("use fake prices (offline test)", value=False)
        save = st.checkbox("also save prices to prices.csv", value=False)
        if st.button("reset"):
            init_state()
        s = st.session_state
        if s.warm:
            st.write(f"logreg learned from {s.warm:,} recorded moves in prices.csv before going live")
        st.write("**models**")
        st.write("- majority: whatever happened more often (baseline)\n"
                 "- bounce: opposite of the last move\n"
                 "- momentum: same as the last move\n"
                 f"- logreg: online logistic regression on the last {N_MOVES} move signs")

    @st.fragment(run_every=interval)
    def live():
        s = st.session_state
        price = fake_price() if simulate else fetch_price()
        if price is None:
            st.warning("couldn't get the price from coinbase, trying again...")
        else:
            s.prices.append((pd.Timestamp.now(), price))
            s.prices = s.prices[-500:]
            if save and not simulate:
                new = not os.path.exists("prices.csv")
                with open("prices.csv", "a", newline="") as f:
                    w = csv.writer(f)
                    if new:
                        w.writerow(["time", "price"])
                    w.writerow([time.time(), price])
            update(price)

        if not s.prices:
            st.info("waiting for the first price...")
            return

        n = len(s.hits["bounce"])
        cols = st.columns(4)
        last = s.prices[-1][1]
        change = last - s.prices[-2][1] if len(s.prices) > 1 else 0.0
        cols[0].metric("BTC price", f"${last:,.2f}", f"{change:+,.2f}")
        if s.pending:
            preds = s.pending[1]
            cols[1].metric("bounce says next move", "UP" if preds["bounce"] else "DOWN")
            cols[2].metric("logreg says next move", "UP" if preds["logreg"] else "DOWN",
                           f"P(up) = {preds['p_up']:.2f}", delta_color="off")
        else:
            cols[1].metric("bounce says next move", "...")
            cols[2].metric("logreg says next move", "...")
        cols[3].metric("moves checked", n, f"{s.flat} ticks with no move", delta_color="off")

        left, right = st.columns(2)
        with left:
            st.subheader("price")
            # st.line_chart always starts the y axis at 0, so btc looked like a flat line.
            # altair lets me zoom the axis to the actual prices
            df = pd.DataFrame(s.prices, columns=["time", "price"])
            chart = alt.Chart(df).mark_line().encode(
                x="time:T", y=alt.Y("price:Q", scale=alt.Scale(zero=False)))
            st.altair_chart(chart, width="stretch")
        with right:
            st.subheader(f"accuracy (last {ACC_WINDOW} moves)")
            if n:
                acc = {m: f"{np.mean(s.hits[m][-ACC_WINDOW:]):.0%}" for m in MODELS}
                st.write("  |  ".join(f"**{m}** {a}" for m, a in acc.items()) + "  |  coin flip 50%")
                roll = pd.DataFrame({m: pd.Series(s.hits[m]).rolling(ACC_WINDOW, min_periods=1).mean()
                                     for m in MODELS})
                roll["coin flip"] = 0.5
                st.line_chart(roll)
            else:
                st.write("no moves checked yet, the price needs to move a few times first")

        if s.log:
            st.subheader("last predictions")
            st.dataframe(pd.DataFrame(s.log[-15:][::-1]), hide_index=True, width="stretch")

    live()


main()
