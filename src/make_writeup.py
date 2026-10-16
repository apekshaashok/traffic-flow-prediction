"""Generate the 2-page PDF write-up from outputs/results.json (run after src.train).

Usage:
    python -m src.make_writeup --author "Your Name" --srn "PES..."
The Conclusions are auto-drafted from the numbers; read and edit them before submitting.
"""
import argparse
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--author", default="Your Name")
    ap.add_argument("--srn", default="")
    ap.add_argument("--out", default=str(OUT_DIR / "writeup.pdf"))
    args = ap.parse_args()

    r = json.loads((OUT_DIR / "results.json").read_text())
    res = r["results"]
    synthetic = r["source"] == "synthetic"

    ss = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.3, leading=11.8, spaceAfter=3)
    h = ParagraphStyle("h", parent=ss["Heading2"], fontSize=11, spaceBefore=5, spaceAfter=2)
    title = ParagraphStyle("t", parent=ss["Title"], fontSize=15, spaceAfter=2)
    sub = ParagraphStyle("s", parent=ss["Normal"], fontSize=9, alignment=1, textColor=colors.grey)

    best_lag = min((x for x in res if x["features"] == "Calendar + weather + lags"), key=lambda x: x["RMSE"])
    best_nolag = min((x for x in res if x["features"] == "Calendar + weather"), key=lambda x: x["RMSE"])
    persist = next(x for x in res if x["model"].startswith("Persistence"))
    hist = next(x for x in res if x["model"].startswith("Historical"))

    story = [Paragraph("Machine Learning for Traffic Flow Prediction in Urban Areas", title)]
    byline = args.author + (f" ({args.srn})" if args.srn else "")
    story.append(Paragraph(byline, sub))
    if synthetic:
        story.append(Paragraph("<font color='red'><b>WARNING: results below come from SYNTHETIC test data, not the real dataset.</b></font>", body))

    story += [
        Paragraph("1. Problem Statement", h),
        Paragraph(
            "Urban congestion costs time, fuel and safety. Accurate short-term traffic volume forecasts let "
            "signal control, routing and incident response act before a jam forms. We predict the hourly "
            "vehicle count on an urban interstate segment from calendar, weather and recent-traffic "
            "information. The project has the same purpose as the Stanford CS229 (Spring 2021) report "
            "<i>Machine Learning for Traffic Flow Prediction in Urban Areas</i> (Huang and Zhou), which "
            "predicted a five-level congestion label for 35 Manhattan roads from road, day and hour using a "
            "one-hidden-layer neural network (76.8% validation accuracy). We differ in both data and method: "
            "we use the public UCI Metro Interstate dataset, and we compare tree-based and linear models "
            "against forecasting baselines and a neural net. We solve two tasks: (1) <b>congestion-level "
            "classification</b> (the paper's task) and (2) <b>traffic volume regression</b> (a finer-grained view).", body),
        Paragraph("2. Dataset", h),
        Paragraph(
            f"We use the UCI <i>Metro Interstate Traffic Volume</i> dataset: hourly westbound I-94 traffic "
            f"(Minneapolis-St Paul) joined with weather and US holiday data. After removing "
            f"duplicate timestamps, rebuilding a regular hourly grid and dropping rows without a target or "
            f"lag history, we have <b>{r['n_rows']:,}</b> hourly samples from {r['date_start'][:10]} to "
            f"{r['date_end'][:10]}. Target: <i>traffic_volume</i> (mean {r['target_mean']:.0f}, max "
            f"{r['target_max']:.0f} vehicles/hour). Raw features: temperature, rain, snow, cloud cover, "
            f"weather category, holiday flag and timestamp. Known sensor errors (0 K temperatures and "
            f"implausible rain spikes) were treated as missing. The data is split chronologically: first "
            f"{r['n_train']:,} samples for training, last {r['n_test']:,} for testing, so models never see "
            f"the future.", body),
        Paragraph("3. Approach", h),
        Paragraph(
            "<b>Features.</b> (a) Calendar: hour (raw and sin/cos), day of week, month, weekend and holiday "
            "flags. (b) Weather: temperature, rain, snow, cloud cover, one-hot weather category. "
            "(c) Lags: traffic 1, 2, 3, 24 and 168 hours earlier (recent trend, same hour yesterday and last "
            f"week). This gives {r['n_features_no_lags']} features without lags and {r['n_features_with_lags']} with.<br/>"
            "<b>Models.</b> Two baselines: historical mean per (day-of-week, hour) and persistence (previous "
            "hour). Learned models: Ridge regression, Random Forest and Gradient Boosting, each trained on "
            "both feature sets to separate what calendar/weather explains from what recent traffic adds.<br/>"
            "<b>Congestion levels.</b> As in the paper, congestion is a label from 0 (free flow) to 4 (severe). "
            "We define it as hourly volume relative to the maximum training volume, in 20% bands "
            f"(level edges {', '.join(f'{e:.0f}' for e in r['level_edges'])} vehicles/hour). Classifiers: Logistic "
            "Regression, Random Forest, Gradient Boosting and an MLP, plus a paper-style MLP (4 hidden neurons, "
            "day and hour only) as a reference.<br/>"
            "<b>Metrics.</b> Regression: MAE, RMSE (vehicles/hour), R<super>2</super>. Classification: accuracy, "
            "within-one-level accuracy (off by at most one level) and macro-F1. All on the held-out final period.", body),
        Paragraph("4. Implementation", h),
        Paragraph(
            "Python 3 with pandas, scikit-learn and matplotlib. <font face='Courier'>src/data.py</font> "
            "downloads and cleans the data and builds features; <font face='Courier'>src/train.py</font> "
            "runs the chronological split, fits all models with fixed seeds, and writes metrics and the plot; "
            "<font face='Courier'>src/make_writeup.py</font> builds this report from the saved results. "
            "Random Forest uses 200 trees (min leaf 3); Gradient Boosting uses histogram-based trees "
            "(300 iterations, learning rate 0.08); Ridge uses standardized inputs, alpha 1. "
            "The classifiers use matching settings (Random Forest 200 trees, Gradient Boosting 200 iterations, "
            "Logistic Regression up to 2000 iterations). The MLP has one hidden layer of 32 units (4 units for the "
            "paper-style reference), standardized inputs, early stopping and at most 500 iterations. "
            "The whole pipeline reproduces with two commands (see the README).", body),
        Paragraph("5. Results", h),
    ]

    rows = [["Feature set", "Model", "MAE", "RMSE", "R²"]]
    for x in res:
        rows.append([x["features"], x["model"], f"{x['MAE']:.0f}", f"{x['RMSE']:.0f}", f"{x['R2']:.3f}"])
    t = Table(rows, colWidths=[4.2 * cm, 6.6 * cm, 1.8 * cm, 1.8 * cm, 1.8 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    cls = r["classification"]
    best_cls = max((x for x in cls if x["features"] == "Calendar + weather + lags"), key=lambda x: x["Accuracy"])
    paper_mlp = next(x for x in cls if x["model"].startswith("Paper-style"))
    cls_hist = next(x for x in cls if x["model"].startswith("Historical"))
    crow = [["Feature set", "Model", "Accuracy", "Within-1", "Macro-F1"]]
    for x in cls:
        crow.append([x["features"], x["model"], f"{100 * x['Accuracy']:.1f}%", f"{100 * x['Within-1']:.1f}%", f"{x['Macro-F1']:.3f}"])
    ct = Table(crow, colWidths=[4.2 * cm, 6.6 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm], repeatRows=1)
    ct.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6e6e6")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("ALIGN", (2, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
    ]))
    story += [
        Paragraph("<b>Congestion-level classification</b> (5 classes, the paper's task):", body), ct, Spacer(1, 5),
        Paragraph("<b>Traffic volume regression</b> (vehicles/hour):", body), t, Spacer(1, 4),
        Image(str(OUT_DIR / "pred_vs_actual.png"), width=16 * cm, height=5.0 * cm),
    ]
    cls_sentence = (
        f"For five-level congestion classification, the best model ({best_cls['model']} with lags) reaches "
        f"{100 * best_cls['Accuracy']:.1f}% accuracy ({100 * best_cls['Within-1']:.1f}% within one level), versus "
        f"{100 * cls_hist['Accuracy']:.1f}% for the historical-mean baseline and {100 * paper_mlp['Accuracy']:.1f}% for a "
        f"paper-style MLP on day and hour only; the paper reports 76.8% on a different dataset, so the numbers are "
        f"indicative rather than directly comparable. ")

    gain_vs_persist = 100 * (1 - best_lag["RMSE"] / persist["RMSE"])
    gain_lags = 100 * (1 - best_lag["RMSE"] / best_nolag["RMSE"])
    story += [
        Paragraph("6. Conclusions", h),
        Paragraph(
            f"The best model was <b>{best_lag['model']} with lag features</b> (RMSE {best_lag['RMSE']:.0f}, "
            f"R² {best_lag['R2']:.3f}), which cuts RMSE by {gain_vs_persist:.0f}% against the persistence "
            f"baseline ({persist['RMSE']:.0f}). Using only calendar and weather information, the best model "
            f"({best_nolag['model']}) reaches RMSE {best_nolag['RMSE']:.0f}; adding lag features lowers it by "
            f"a further {gain_lags:.0f}%, showing that recent traffic carries information that time of day and "
            f"weather cannot. The historical-mean baseline (RMSE {hist['RMSE']:.0f}) captures the daily and "
            f"weekly rhythm but misses day-specific deviations. {cls_sentence}Limitations: a single sensor, one corridor, "
            f"and no incident or event data. Future work: multi-step horizons, spatial neighbours from several "
            f"sensors, and sequence models (LSTM / graph networks).", body),
    ]

    doc = SimpleDocTemplate(args.out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.4 * cm, title="Traffic Flow Prediction", author=args.author)
    doc.build(story)
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
