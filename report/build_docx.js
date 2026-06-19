// Render report.md into an editable Word document with embedded figures.
const fs = require("fs");
const path = require("path");
const zlib = require("zlib");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        ImageRun, AlignmentType, BorderStyle, WidthType, ShadingType,
        Footer, PageNumber, HeadingLevel } = require("docx");

const ROOT = __dirname;
const md = fs.readFileSync(path.join(ROOT, "report.md"), "utf8");

// A4 with 0.9" margins -> content width 9314 DXA
const MARGIN = 1296;
const CONTENT_W = 11906 - 2 * MARGIN;

const CAPTIONS = {
  fig1: "Fig. 1. Distribution of final results after excluding students who withdrew before day 30.",
  fig2: "Fig. 2. Correlation between the main numeric features. The behavioural block (top left) is internally correlated but nearly orthogonal to the demographic block.",
  fig3: "Fig. 3. First-30-day engagement (log of total clicks) by final outcome.",
  fig4: "Fig. 4. Learning curve of the unconstrained random forest (Experiment 2), showing the variance pathology.",
  fig5: "Fig. 5. Confusion matrix and one-vs-rest ROC curves for the tuned random forest (Experiment 3).",
  fig6: "Fig. 6. Training history of the unregularised Sequential network (Experiment 5).",
  fig7: "Fig. 7. Training history with dropout and L2 regularisation (Experiment 6).",
  fig8: "Fig. 8. Confusion matrix and ROC curve for the binary Distinction-versus-rest model (Experiment 8).",
};
const ANCHORS = {
  "as Fig. 1 shows": ["fig1"],
  "The correlation analysis in Fig. 2": ["fig2"],
  "Fig. 3 plots first-month engagement": ["fig3"],
  "Fig. 4 shows the learning curve": ["fig4", "fig5"],
  "As Fig. 6 shows": ["fig6", "fig7"],
  "ROC curve in Fig. 8": ["fig8"],
};
const FILES = {
  fig1: "fig1_class_distribution.png", fig2: "fig2_correlation_heatmap.png",
  fig3: "fig3_engagement_by_outcome.png", fig4: "fig4_exp2_learning_curve.png",
  fig5: "fig5_exp3_cm_roc.png", fig6: "fig6_exp5_history.png",
  fig7: "fig7_exp6_history.png", fig8: "fig8_exp8_cm_roc.png",
};

function pngSize(buf) {
  return { w: buf.readUInt32BE(16), h: buf.readUInt32BE(20) };
}

function figureParas(key) {
  const buf = fs.readFileSync(path.join(ROOT, "figures", FILES[key]));
  const { w, h } = pngSize(buf);
  const maxW = 590, maxH = 330; // px at 96dpi, fits A4 content width
  const scale = Math.min(maxW / w, maxH / h);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 40 },
      children: [new ImageRun({
        type: "png", data: buf,
        transformation: { width: Math.round(w * scale), height: Math.round(h * scale) },
        altText: { title: key, description: CAPTIONS[key], name: key },
      })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 160 },
      children: [new TextRun({ text: CAPTIONS[key], italics: true, size: 18 })],
    }),
  ];
}

// minimal **bold** / *italic* inline parser
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: tok.slice(1, -1), italics: true, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "888888" };
const borders = { top: border, bottom: border, left: border, right: border };
const COLW = [420, 2100, 2400, 800, 950, 1000, 1644];

function resultsTable(rows) {
  const header = ["#", "Model", "Key settings", "Acc.", "Macro-F1", "Macro AUC", "Dist. recall"];
  const mkCell = (text, bold, w) => new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    shading: bold ? { fill: "EAEAEE", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({ children: [new TextRun({ text, bold, size: 17 })] })],
  });
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: COLW,
    rows: [
      new TableRow({ children: header.map((h, j) => mkCell(h, true, COLW[j])) }),
      ...rows.map(r => new TableRow({ children: r.map((c, j) => mkCell(c, false, COLW[j])) })),
    ],
  });
}

const children = [];
const lines = md.split("\n");
let i = 0, titleDone = false, inRefs = false;
const tableRows = [];

while (i < lines.length) {
  const line = lines[i].trimEnd();
  if (!line.trim()) { i++; continue; }

  if (line.startsWith("# ") && !titleDone) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 80 },
      children: [new TextRun({ text: line.slice(2), bold: true, size: 32 })],
    }));
    titleDone = true; i++;
    while (i < lines.length && !lines[i].startsWith("## ")) {
      const l = lines[i].trim();
      if (l) children.push(new Paragraph({
        alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: runs(l, { size: 22 }),
      }));
      i++;
    }
    children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    continue;
  }

  if (line.startsWith("## ")) {
    const head = line.slice(3);
    inRefs = head.trim() === "References";
    children.push(new Paragraph({
      heading: HeadingLevel.HEADING_1,
      children: [new TextRun({ text: head, bold: true, size: 26 })],
    }));
    i++; continue;
  }

  if (line.startsWith("|")) {
    const cells = line.replace(/^\||\|$/g, "").split("|").map(c => c.trim());
    const isRule = cells.every(c => /^:?-+:?$/.test(c) || c === "");
    if (!isRule && cells[0] !== "#") tableRows.push(cells);
    i++;
    if (i >= lines.length || !lines[i].startsWith("|")) {
      children.push(resultsTable(tableRows));
      children.push(new Paragraph({ spacing: { after: 160 }, children: [] }));
    }
    continue;
  }

  if (line.startsWith("**Table 1")) {
    children.push(new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { before: 80, after: 80 },
      children: [new TextRun({ text: line.replace(/\*\*/g, ""), bold: true, size: 19 })],
    }));
    i++; continue;
  }

  // ordinary paragraph, may span lines
  const para = [line];
  i++;
  while (i < lines.length && lines[i].trim() &&
         !lines[i].startsWith("#") && !lines[i].startsWith("|") &&
         !lines[i].startsWith("**Table")) {
    para.push(lines[i].trim());
    i++;
  }
  const text = para.join(" ");
  children.push(new Paragraph({
    alignment: inRefs ? AlignmentType.LEFT : AlignmentType.JUSTIFIED,
    spacing: { after: inRefs ? 80 : 140, line: 300 },
    indent: inRefs ? { left: 360, hanging: 360 } : undefined,
    children: runs(text, { size: inRefs ? 20 : 22 }),
  }));
  for (const [anchor, figs] of Object.entries(ANCHORS)) {
    if (text.includes(anchor)) figs.forEach(f => children.push(...figureParas(f)));
  }
}

const doc = new Document({
  title: "Early Identification of High-Aptitude Students from Online Engagement Behaviour",
  creator: "Christian Tonny",
  styles: {
    default: { document: { run: { font: "Times New Roman", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Times New Roman", color: "000000" },
        paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 0 } },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: MARGIN, right: MARGIN, bottom: MARGIN, left: MARGIN },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ children: [PageNumber.CURRENT], size: 18 })],
      })] }),
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(path.join(ROOT, "summative_report.docx"), buf);
  console.log("wrote summative_report.docx,", buf.length, "bytes");
});
