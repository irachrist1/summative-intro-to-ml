"""Render report.md into an academic-styled PDF with embedded figures."""
import re
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                Table, TableStyle, KeepTogether)
from PIL import Image as PILImage

ROOT = Path(__file__).parent
MD = (ROOT / "report.md").read_text()

styles = {
    "title": ParagraphStyle("title", fontName="Times-Bold", fontSize=16,
                            leading=20, alignment=TA_CENTER, spaceAfter=4),
    "meta": ParagraphStyle("meta", fontName="Times-Roman", fontSize=11,
                           leading=14, alignment=TA_CENTER, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Times-Bold", fontSize=13, leading=16,
                         spaceBefore=14, spaceAfter=6),
    "body": ParagraphStyle("body", fontName="Times-Roman", fontSize=11,
                           leading=14.8, alignment=TA_JUSTIFY, spaceAfter=7),
    "ref": ParagraphStyle("ref", fontName="Times-Roman", fontSize=10,
                          leading=12.5, spaceAfter=4, leftIndent=18,
                          firstLineIndent=-18),
    "caption": ParagraphStyle("caption", fontName="Times-Italic", fontSize=9,
                              leading=11, alignment=TA_CENTER, spaceBefore=3,
                              spaceAfter=10),
    "tabcap": ParagraphStyle("tabcap", fontName="Times-Bold", fontSize=9.5,
                             leading=12, alignment=TA_CENTER, spaceBefore=4,
                             spaceAfter=4),
}

CAPTIONS = {
    "fig1": "Fig. 1. Distribution of final results after excluding students who withdrew before day 30.",
    "fig2": "Fig. 2. Correlation between the main numeric features. The behavioural block (top left) is internally correlated but nearly orthogonal to the demographic block.",
    "fig3": "Fig. 3. First-30-day engagement (log of total clicks) by final outcome.",
    "fig4": "Fig. 4. Learning curve of the unconstrained random forest (Experiment 2), showing the variance pathology.",
    "fig5": "Fig. 5. Confusion matrix and one-vs-rest ROC curves for the tuned random forest (Experiment 3).",
    "fig6": "Fig. 6. Training history of the unregularised Sequential network (Experiment 5).",
    "fig7": "Fig. 7. Training history with dropout and L2 regularisation (Experiment 6).",
    "fig8": "Fig. 8. Confusion matrix and ROC curve for the binary Distinction-versus-rest model (Experiment 8).",
}
# figure placed after the first body paragraph containing its anchor text
ANCHORS = {
    "as Fig. 1 shows": ["fig1"],
    "The correlation analysis in Fig. 2": ["fig2"],
    "Fig. 3 plots first-month engagement": ["fig3"],
    "Fig. 4 shows the learning curve": ["fig4", "fig5"],
    "As Fig. 6 shows": ["fig6", "fig7"],
    "ROC curve in Fig. 8": ["fig8"],
}
FILES = {
    "fig1": "fig1_class_distribution.png", "fig2": "fig2_correlation_heatmap.png",
    "fig3": "fig3_engagement_by_outcome.png", "fig4": "fig4_exp2_learning_curve.png",
    "fig5": "fig5_exp3_cm_roc.png", "fig6": "fig6_exp5_history.png",
    "fig7": "fig7_exp6_history.png", "fig8": "fig8_exp8_cm_roc.png",
}


def inline(text):
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    return text


def figure_flowable(key, max_w=6.0 * inch, max_h=3.6 * inch):
    path = ROOT / "figures" / FILES[key]
    w, h = PILImage.open(path).size
    scale = min(max_w / w, max_h / h)
    img = Image(str(path), width=w * scale, height=h * scale)
    return KeepTogether([img, Paragraph(CAPTIONS[key], styles["caption"])])


def results_table(rows):
    header = ["#", "Model", "Key settings", "Acc.", "Macro-F1", "Macro AUC",
              "Dist. recall"]
    cell = ParagraphStyle("cell", fontName="Times-Roman", fontSize=8.5, leading=10)
    cellb = ParagraphStyle("cellb", fontName="Times-Bold", fontSize=8.5, leading=10)
    data = [[Paragraph(h, cellb) for h in header]]
    for r in rows:
        data.append([Paragraph(c, cell) for c in r])
    t = Table(data, colWidths=[0.32 * inch, 1.55 * inch, 1.75 * inch, 0.62 * inch,
                               0.72 * inch, 0.75 * inch, 0.79 * inch])
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.Color(0.92, 0.92, 0.94)),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


story = []
lines = MD.splitlines()
i = 0
in_refs = False
table_rows, in_table = [], False
title_done = False
meta_buf = []

while i < len(lines):
    line = lines[i].rstrip()
    if not line:
        i += 1
        continue
    if line.startswith("# ") and not title_done:
        story.append(Paragraph(inline(line[2:]), styles["title"]))
        story.append(Spacer(1, 6))
        title_done = True
        i += 1
        # author/meta block until first ## heading
        while i < len(lines) and not lines[i].startswith("## "):
            l = lines[i].strip()
            if l:
                story.append(Paragraph(inline(l), styles["meta"]))
            i += 1
        story.append(Spacer(1, 10))
        continue
    if line.startswith("## "):
        head = line[3:]
        in_refs = head.strip() == "References"
        story.append(Paragraph(inline(head), styles["h1"]))
        i += 1
        continue
    if line.startswith("|"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set(cells[0]) <= set("-: ") and cells[0]:
            i += 1
            continue
        if cells[0] != "#":   # skip header row, we supply our own
            table_rows.append(cells)
        in_table = True
        i += 1
        if i >= len(lines) or not lines[i].startswith("|"):
            story.append(results_table(table_rows))
            story.append(Spacer(1, 8))
            in_table = False
        continue
    if line.startswith("**Table 1"):
        story.append(Paragraph(inline(line.strip("*")), styles["tabcap"]))
        i += 1
        continue
    # ordinary paragraph (may span source lines until blank)
    para = [line]
    i += 1
    while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "**Table")):
        para.append(lines[i].strip())
        i += 1
    text = " ".join(para)
    story.append(Paragraph(inline(text), styles["ref" if in_refs else "body"]))
    for anchor, figs in ANCHORS.items():
        if anchor in text:
            for f in figs:
                story.append(figure_flowable(f))

doc = SimpleDocTemplate(str(ROOT / "summative_report.pdf"), pagesize=A4,
                        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                        topMargin=0.9 * inch, bottomMargin=0.9 * inch,
                        title="Early Identification of High-Aptitude Students "
                              "from Online Engagement Behaviour",
                        author="Christian Tonny")
doc.build(story)
print("wrote", ROOT / "summative_report.pdf")
