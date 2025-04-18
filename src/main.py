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
    df
    return (df,)


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
    taus = np.floor(genVal(200, 300, n))
    bases = genVal(0, 10, n)

    tsArr = []

    for period, shift, amp, tau, base in zip(periods, shifts, amps, taus, bases):
        params = dict(period=period, shift=shift, amp=amp, tau=tau, base=base)
        ts = genTimeSeries(params)
        tsArr.append(ts)
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
        tsArr,
    )


@app.cell
def _(plt, tsArr):
    for i in range(10):
        x = tsArr[i][0]
        y = tsArr[i][1]
        plt.plot(x,y)
    plt.show()
    return i, x, y


@app.cell
def _(
    amps,
    bases,
    m_val_1,
    m_val_2,
    m_val_3,
    pd,
    periods,
    shifts,
    taus,
    tsArr,
):
    data = {
        "m1": m_val_1,
        "m2": m_val_2,
        "m3": m_val_3,
        "ts": tsArr,
        "_period": periods,
        "_shift": shifts,
        "_amp": amps,
        "_tau": taus,
        "_base": bases,
    }

    dfData = pd.DataFrame(data=data)
    dfData
    return data, dfData


@app.cell
def _(dfData):
    dfDataStats = dfData.describe()
    dfDataStats
    return (dfDataStats,)


@app.cell
def _(dfData, dfDataStats, pd):
    def cal_y(row, stats):
        m1_mean = stats.loc["mean", "m1"]
        m2_mean = stats.loc["mean", "m2"]
        m3_mean = stats.loc["mean", "m3"]
        period_mean = stats.loc["mean", "_period"]
        y1 = (row["m1"]/m1_mean + row["_period"]/period_mean * 2) * 20
        y2 = (row["m2"]/m2_mean)  / (row["_period"]/period_mean)
        y3 = ((row["m3"]/m3_mean) - (row["_period"]/period_mean * 2)) * -100
        return pd.Series(data=[y1, y2, y3], index=["y1", "y2", "y3"])
        pass

    dfData[["y1", "y2", "y3"]] = dfData.apply(lambda row: cal_y(row, dfDataStats), axis=1)
    dfData
    return (cal_y,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
