import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, pd, plt


@app.cell
def _(pd):
    df = pd.read_csv("src/data/test_no_23.csv")
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    cols = [col for col in df.columns if col != "Time"]
    return (cols,)


@app.cell
def _(cols, df, plt):
    fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(7,10))
    for idx, col in enumerate(cols):
        df.plot(x="Time", y=col, ax=axes[idx])
    plt.show()
    return axes, col, fig, idx


@app.cell
def _(np):
    def genVal(low, high, n):
        return np.random.uniform(low=low, high=high, size=(n,))

    n = 50
    m_val_1 = genVal(100, 200, n)
    m_val_2 = genVal(1000, 1500, n)
    m_val_3 = genVal(1, 3, n)
    return genVal, m_val_1, m_val_2, m_val_3, n


@app.cell
def _(genVal, n, np):
    def genTimeSeries(params):
        tau = params.get("tau")
        period = params.get("period")
        shift = params.get("shift")
        amp = params.get("amp")
        base = params.get("base")
        x = np.arange(tau)
        y = amp * np.sin(2 * np.pi * x / period + shift) + base
        return x, y


    periods = genVal(20, 40, n)
    shifts = genVal(0, 100, n)
    amps = genVal(0, 3, n)
    taus = genVal(200, 300, n)
    bases = genVal(0, 10, n)

    tss = []
    for period, shift, amp, tau, base in zip(periods, shifts, amps, taus, bases):
        params = dict(period=period, shift=shift, amp=amp, tau=tau, base=base)
        ts = genTimeSeries(params)
        tss.append(ts)
    return (
        amp,
        amps,
        base,
        bases,
        genTimeSeries,
        params,
        period,
        periods,
        shift,
        shifts,
        tau,
        taus,
        ts,
        tss,
    )


@app.cell
def _(plt, tss):
    for i in range(10):
        x = tss[i][0]
        y = tss[i][1]
        plt.plot(x,y)
    plt.show()
    return i, x, y


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
