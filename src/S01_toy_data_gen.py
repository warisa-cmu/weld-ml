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
def _(df, mo, plt):
    cols = [col for col in df.columns if col != "Time"]
    fig, axes = plt.subplots(nrows=4, ncols=1,figsize=(7,10))
    for idx, col in enumerate(cols):
        df.plot(x="Time", y=col, ax=axes[idx])
    mo.mpl.interactive(fig)
    return axes, col, cols, fig, idx


@app.cell
def _(np):
    def genVal(low, high, n):
        return np.random.uniform(low=low, high=high, size=(n,))

    n = 50
    m1Arr = genVal(100, 200, n)
    m2Arr = genVal(1000, 1500, n)
    m3Arr = genVal(1, 3, n)
    return genVal, m1Arr, m2Arr, m3Arr, n


@app.cell
def _(genVal, n, np):
    def genTimeSeries(params):
        tau = params.get("tau")
        period = params.get("period")
        shift = params.get("shift")
        amp = params.get("amp")
        base = params.get("base")
        t = np.arange(tau)
        s = amp * np.sin(2 * np.pi * t / period + shift) + base
        return t, s


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
def _(mo, plt, tsArr):
    for i in range(10):
        t = tsArr[i][0]
        s = tsArr[i][1]
        plt.plot(t,s)
    mo.mpl.interactive(plt.gcf())
    return i, s, t


@app.cell
def _(
    amps,
    bases,
    m1Arr,
    m2Arr,
    m3Arr,
    n,
    np,
    pd,
    periods,
    shifts,
    taus,
    tsArr,
):
    data = {
        "m1": m1Arr,
        "m2": m2Arr,
        "m3": m3Arr,
        "ts": tsArr,
        "_period": periods,
        "_shift": shifts,
        "_amp": amps,
        "_tau": taus,
        "_base": bases,
    }

    idxes = [f"E{str(i+1).zfill(3)}" for i in np.arange(n)]
    dfData = pd.DataFrame(data=data, index=idxes)
    dfData
    return data, dfData, idxes


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
def _(dfData, pd):
    def genTsfreshData(row):
        ts = row["ts"]
        t = ts[0]
        s = ts[1]
        data = {"time": t, "s1": s}
        df = pd.DataFrame(data)
        df["id"] = row.name
        return df

    temp = dfData.apply(genTsfreshData, axis=1)
    dfTsf = pd.concat(temp.values, axis=0)
    dfTsf
    return dfTsf, genTsfreshData, temp


@app.cell
def _(dfTsf):
    from tsfresh import extract_features

    dfEx = extract_features(dfTsf, column_id="id", column_sort="time")
    return dfEx, extract_features


@app.cell
def _(dfEx):
    dfEx
    return


@app.cell
def _(dfEx, pd):
    # Some features have NaN.
    from sklearn.impute import SimpleImputer

    # Drop columns which contains all NaN.
    dfExDrop = dfEx.dropna(axis=1, how="any")

    imputer = SimpleImputer(strategy="mean")
    exIm = imputer.fit_transform(dfExDrop)
    dfExIm = pd.DataFrame(exIm, columns=dfExDrop.columns, index=dfExDrop.index)
    return SimpleImputer, dfExDrop, dfExIm, exIm, imputer


@app.cell
def _(dfData, dfExIm):
    from tsfresh.feature_selection.relevance import calculate_relevance_table

    def calRelTable(dfEx, target):
        rt = calculate_relevance_table(dfExIm, dfData["y1"])
        rt = rt[rt.relevant].reset_index(drop=True)
        return rt

    r1 = calRelTable(dfExIm, dfData["y1"])
    r2 = calRelTable(dfExIm, dfData["y2"])
    r3 = calRelTable(dfExIm, dfData["y3"])
    return calRelTable, calculate_relevance_table, r1, r2, r3


@app.cell
def _(pd, r1, r2, r3):
    _dfRt = pd.concat([r1, r2, r3])
    dfRt = _dfRt[~_dfRt["feature"].duplicated()]
    dfRt
    return (dfRt,)


@app.cell
def _(dfEx, dfRt):
    dfExRel = dfEx[dfRt["feature"]]
    dfExRel
    return (dfExRel,)


@app.cell
def _(dfData, dfExRel, pd):
    cols1 = ["m1", "m2", "m3"]
    cols2 = ["y1", "y2", "y3"]
    dfTrain = pd.concat([dfData[cols1], dfExRel, dfData[cols2]], axis=1)
    dfTrain
    return cols1, cols2, dfTrain


@app.cell
def _(dfTrain):
    dfTrain.to_excel("src/output/data_train.xlsx", index=True)
    return


if __name__ == "__main__":
    app.run()
