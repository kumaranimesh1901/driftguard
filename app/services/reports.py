"""Risk-committee PDF report (ReportLab) built from a stored analysis report.

Sections: monitoring summary, baseline information, production batch
information, record counts, feature-level PSI/KS/JSD, DIS, highest-risk
features, model performance, risk assessment, recommended actions, and the
monitoring timestamp. Charts are drawn with ReportLab graphics from the
report's own values.
"""

from datetime import UTC, datetime
from io import BytesIO
from typing import Any

from reportlab.graphics.charts.barcharts import HorizontalBarChart, VerticalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#0B2545")
BLUE = colors.HexColor("#1D5FA8")
LIGHT_BLUE = colors.HexColor("#8FB8E8")
GREY = colors.HexColor("#5B6474")
LIGHT = colors.HexColor("#F2F5F9")
RISK_COLOURS = {
    "LOW RISK": colors.HexColor("#1E7B4F"),
    "MEDIUM RISK": colors.HexColor("#B7791F"),
    "HIGH RISK": colors.HexColor("#C2410C"),
    "CRITICAL RISK": colors.HexColor("#B91C1C"),
}


def _fmt(value: Any, spec: str = ".4f", none: str = "—") -> str:
    if value is None:
        return none
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def _pct(value: Any) -> str:
    return _fmt(value, ".1%")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "t",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
        ),
        "subtitle": ParagraphStyle(
            "st", parent=base["Normal"], fontSize=9.5, leading=13, textColor=GREY
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=NAVY,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle("b", parent=base["Normal"], fontSize=9, leading=12.5),
        "small": ParagraphStyle(
            "s", parent=base["Normal"], fontSize=7.5, leading=10, textColor=GREY
        ),
        "cell": ParagraphStyle("c", parent=base["Normal"], fontSize=7.8, leading=10),
        "cellb": ParagraphStyle(
            "cb",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7.8,
            leading=10,
        ),
        "banner": ParagraphStyle(
            "bn",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.white,
        ),
    }


def _table(rows: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    t = Table(rows, colWidths=widths, repeatRows=1 if header else 0)
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D5DCE6")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if header:
        style += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ]
    else:
        style += [
            ("BACKGROUND", (0, 0), (0, -1), LIGHT),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ]
    t.setStyle(TableStyle(style))
    return t


def _dis_chart(features: list[dict[str, Any]], width: float) -> Drawing:
    ranked = sorted(features, key=lambda f: f.get("dis", 0))
    names = [f["label"] for f in ranked]
    d = Drawing(width, 18 * len(ranked) + 50)
    chart = HorizontalBarChart()
    chart.x, chart.y = 120, 30
    chart.width, chart.height = width - 150, 18 * len(ranked)
    chart.data = [[f.get("d", 0) for f in ranked], [f.get("dis", 0) for f in ranked]]
    chart.categoryAxis.categoryNames = names
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.valueMin, chart.valueAxis.valueMax, chart.valueAxis.valueStep = (
        0,
        1,
        0.2,
    )
    chart.valueAxis.labels.fontSize = 7
    chart.bars[0].fillColor = LIGHT_BLUE
    chart.bars[1].fillColor = NAVY
    chart.barSpacing = 1
    d.add(chart)
    legend = Legend()
    legend.x, legend.y = 120, 12
    legend.fontSize = 7
    legend.alignment = "right"
    legend.columnMaximum = 1
    legend.colorNamePairs = [
        (LIGHT_BLUE, "Normalised drift D"),
        (NAVY, "DIS = D × FI × S"),
    ]
    d.add(legend)
    return d


def _histogram_chart(dist: dict[str, Any], label: str, width: float) -> Drawing | None:
    if not dist or not dist.get("series"):
        return None
    series = dist["series"]
    d = Drawing(width, 170)
    chart = VerticalBarChart()
    chart.x, chart.y, chart.width, chart.height = 40, 35, width - 60, 110
    chart.data = [series.get("baseline", []), series.get("production", [])]
    if dist["kind"] == "numerical":
        edges = dist["edges"]
        names = [
            f"{edges[i]:,.0f}" if abs(edges[i]) >= 100 else f"{edges[i]:.2f}"
            for i in range(len(edges) - 1)
        ]
        chart.categoryAxis.categoryNames = [
            n if i % 5 == 0 else "" for i, n in enumerate(names)
        ]
    else:
        chart.categoryAxis.categoryNames = dist["categories"]
    chart.categoryAxis.labels.fontSize = 6
    chart.valueAxis.labels.fontSize = 6
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = LIGHT_BLUE
    chart.bars[1].fillColor = NAVY
    chart.barSpacing = 0
    chart.groupSpacing = 1
    d.add(chart)
    d.add(
        String(
            40,
            155,
            f"{label}: baseline vs production (share of customers)",
            fontSize=8,
            fillColor=NAVY,
        )
    )
    legend = Legend()
    legend.x, legend.y, legend.fontSize = 40, 12, 7
    legend.alignment = "right"
    legend.columnMaximum = 1
    legend.colorNamePairs = [(LIGHT_BLUE, "Baseline 2020"), (NAVY, "Production")]
    d.add(legend)
    return d


def generate_committee_pdf(report: dict[str, Any]) -> bytes:
    """Render the committee PDF and return its bytes."""
    s = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title=f"DriftGuard Risk Committee Report — {report.get('batch_name', '')}",
        author="DriftGuard",
    )
    width = doc.width
    story: list[Any] = []
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    risk = report.get("risk_category", "LOW RISK")
    features = report.get("feature_details", [])
    health = report.get("model_health", {})

    story.append(
        Paragraph("DriftGuard — Credit Risk Model Monitoring Report", s["title"])
    )
    story.append(
        Paragraph(
            f"Prepared for the Credit Risk Committee · Analysis job #{report.get('job_id', '—')} · "
            f"Generated {generated}",
            s["subtitle"],
        )
    )
    story.append(
        HRFlowable(width="100%", thickness=1.2, color=NAVY, spaceBefore=6, spaceAfter=8)
    )

    if report.get("is_synthetic"):
        story.append(
            Paragraph(
                "<b>SYNTHETIC DATA NOTICE:</b> this analysis includes synthetically generated "
                "records produced for demonstration. They are not real banking customers.",
                s["body"],
            )
        )
        story.append(Spacer(1, 6))

    banner = Table(
        [
            [
                Paragraph(f"Risk assessment: {risk}", s["banner"]),
                Paragraph(
                    f"Overall DIS {_fmt(report.get('overall_dis'), '.3f')}", s["banner"]
                ),
            ]
        ],
        colWidths=[width * 0.6, width * 0.4],
    )
    banner.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), RISK_COLOURS.get(risk, NAVY)),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    story.append(banner)

    # 1. Monitoring summary
    story.append(Paragraph("1. Monitoring summary", s["h2"]))
    perf = report.get("performance_state", "unknown")
    perf_text = {
        "unknown": "not yet confirmed — production outcome labels are pending",
        "stable": "labelled outcomes show no meaningful degradation",
        "degraded": "CONFIRMED degradation from labelled outcomes",
        "severe": "CONFIRMED severe degradation from labelled outcomes",
    }.get(perf, perf)
    story.append(
        Paragraph(
            f"The {report.get('analysis_type', 'individual')} analysis of <b>{report.get('batch_name')}</b> "
            f"compared {report.get('customer_count', 0):,} production customers with "
            f"{report.get('baseline_count', 0):,} baseline customers across "
            f"{report.get('features_monitored', 0)} financial attributes. "
            f"{report.get('drifting_features_count', 0)} attributes show moderate or significant drift "
            f"({report.get('significant_drift_count', 0)} significant). Batch drift level is "
            f"<b>{report.get('drift_level', 'low')}</b>; model performance: <b>{perf_text}</b>. "
            "Statistical drift alone does not demonstrate that model accuracy has fallen.",
            s["body"],
        )
    )

    # 2–4. Baseline, batch information and record counts
    story.append(
        Paragraph("2. Historical baseline and 3. production batch information", s["h2"])
    )
    info = [
        ["Item", "Value"],
        [
            "Baseline dataset",
            f"{report.get('baseline_name', '—')} (dataset #{report.get('baseline_dataset_id', '—')})",
        ],
        ["Baseline customers", f"{report.get('baseline_count', 0):,}"],
        [
            "Baseline model",
            f"{health.get('baseline_metrics', {}) and 'XGBoost champion' or 'none (drift-only DIS)'}",
        ],
        ["Analysis mode", report.get("analysis_type", "individual")],
        [
            "DIS mode",
            (
                "model-based FI × S"
                if report.get("dis_mode") == "model"
                else "drift only (no labelled baseline)"
            ),
        ],
    ]
    story.append(_table(info, [width * 0.3, width * 0.7], header=True))
    story.append(Spacer(1, 6))
    story.append(Paragraph("4. Customer record counts", s["h2"]))
    rows = [["Batch", "Batch date", "Customers", "Labelled", "Description"]]
    for b in report.get("batches", []):
        rows.append(
            [
                b.get("batch_id"),
                b.get("batch_date"),
                f"{b.get('row_count', 0):,}",
                f"{b.get('labelled_rows', 0):,}",
                Paragraph(str(b.get("description") or ""), s["cell"]),
            ]
        )
    story.append(
        _table(
            rows, [width * 0.2, width * 0.13, width * 0.12, width * 0.11, width * 0.44]
        )
    )

    # 5. Feature-level metrics
    story.append(Paragraph("5. Feature-level PSI, KS and JSD", s["h2"]))
    rows = [["Feature", "PSI", "KS stat", "KS p-value", "JSD", "D", "Status", "Conv."]]
    for f in features:
        rows.append(
            [
                f["label"],
                _fmt(f.get("psi")),
                _fmt(f.get("ks_stat")),
                _fmt(f.get("ks_p"), ".2e"),
                _fmt(f.get("jsd")),
                _fmt(f.get("d"), ".2f"),
                f.get("drift_status"),
                "yes" if f.get("conventional_alert") else "no",
            ]
        )
    story.append(
        _table(rows, [width * x for x in (0.22, 0.1, 0.1, 0.12, 0.1, 0.08, 0.18, 0.1)])
    )
    story.append(
        Paragraph(
            "D is the null-floor normalised drift (bootstrap 95th-percentile noise floor removed, capped), "
            "aggregated across detectors. 'Conv.' marks conventional alerts (PSI > 0.25, KS p < 0.01 or JSD > 0.1).",
            s["small"],
        )
    )

    # 6. DIS
    story.append(
        KeepTogether(
            [
                Paragraph("6. Drift Impact Score (proposed)", s["h2"]),
                Paragraph(
                    "DIS = D × FI × S per feature, where FI is SHAP importance and S is permutation sensitivity of "
                    "the baseline credit model on this batch, each scaled to [0, 1]. Overall DIS is the maximum "
                    "feature DIS. DIS is a proposed prioritisation score, not an industry standard.",
                    s["body"],
                ),
                _dis_chart(features, width),
            ]
        )
    )

    # 7. Highest-risk features
    story.append(Paragraph("7. Highest-risk features", s["h2"]))
    top = sorted(features, key=lambda f: f.get("dis", 0), reverse=True)[:5]
    rows = [
        ["Rank", "Feature", "DIS", "D", "FI", "S", "Baseline mean", "Production mean"]
    ]
    for i, f in enumerate(top, 1):
        rs, ps = f.get("reference_stats") or {}, f.get("production_stats") or {}
        rows.append(
            [
                i,
                f["label"],
                _fmt(f.get("dis"), ".3f"),
                _fmt(f.get("d"), ".2f"),
                _fmt(f.get("fi"), ".2f"),
                _fmt(f.get("s"), ".2f"),
                _fmt(rs.get("mean"), ",.2f"),
                _fmt(ps.get("mean"), ",.2f"),
            ]
        )
    story.append(
        _table(
            rows, [width * x for x in (0.07, 0.25, 0.1, 0.08, 0.08, 0.08, 0.17, 0.17)]
        )
    )
    if top:
        chart = _histogram_chart(
            report.get("distributions", {}).get(top[0]["feature"]),
            top[0]["label"],
            width,
        )
        if chart:
            story.append(KeepTogether([Spacer(1, 6), chart]))

    # 8. Model performance
    story.append(Paragraph("8. Model performance evaluation", s["h2"]))
    story.append(Paragraph(health.get("status_message", ""), s["body"]))
    base_m = health.get("baseline_metrics") or {}
    prod_m = health.get("production_metrics") or {}
    if base_m:
        rows = [["Metric", "Baseline (holdout)", "Production"]]
        for key, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-score"),
            ("roc_auc", "ROC-AUC"),
            ("actual_default_rate", "Actual default rate"),
            ("predicted_default_rate", "Predicted default rate"),
        ):
            prod_val = prod_m.get(key)
            if key == "predicted_default_rate" and prod_val is None:
                prod_val = health.get("predicted_default_rate")
            spec = ".3f" if key == "roc_auc" else ".1%"
            rows.append(
                [
                    label,
                    _fmt(base_m.get(key), spec),
                    _fmt(prod_val, spec, "pending labels"),
                ]
            )
        story.append(Spacer(1, 4))
        story.append(_table(rows, [width * 0.4, width * 0.3, width * 0.3]))
        if prod_m.get("auc_ci"):
            lo, hi = prod_m["auc_ci"]
            story.append(
                Paragraph(
                    f"Production AUC 95% bootstrap CI [{lo:.3f}, {hi:.3f}]; drop versus baseline "
                    f"{prod_m.get('auc_drop', 0):+.3f}. Performance state: <b>{perf}</b>.",
                    s["small"],
                )
            )

    # 9–10. Risk assessment and actions
    story.append(Paragraph("9. Risk assessment", s["h2"]))
    alert = report.get("alert_type") or "none"
    story.append(
        Paragraph(
            f"<b>{risk}</b> — decision engine alert: <b>{alert}</b> (level {report.get('alert_level', 'low')}). "
            f"Engine recommendation: {report.get('recommendation', '')}.",
            s["body"],
        )
    )
    for e in report.get("evidence", []):
        story.append(Paragraph(f"• {e}", s["small"]))
    story.append(Paragraph("10. Recommended actions", s["h2"]))
    for a in report.get("recommended_actions", []):
        story.append(Paragraph(f"• {a}", s["body"]))
    story.append(
        Paragraph(
            "DriftGuard does not retrain models or modify loan sanctioning thresholds automatically; all "
            "actions require committee approval.",
            s["small"],
        )
    )

    # 11. Timestamp
    story.append(Paragraph("11. Monitoring timestamp", s["h2"]))
    story.append(
        _table(
            [
                ["Analysis performed", str(report.get("analysed_at", "—"))],
                ["Analysed by", str(report.get("analysed_by", "—"))],
                ["Report generated", generated],
                [
                    "Thresholds",
                    ", ".join(
                        f"{k}={v}" for k, v in (report.get("thresholds") or {}).items()
                    ),
                ],
            ],
            [width * 0.3, width * 0.7],
            header=False,
        )
    )

    def _footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(
            16 * mm, 8 * mm, "DriftGuard · Confidential — for credit risk committee use"
        )
        canvas.drawRightString(A4[0] - 16 * mm, 8 * mm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return buffer.getvalue()
