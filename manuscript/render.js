// Render manuscript JSON (from build_manuscript.py) to .docx
const fs = require('fs');
const { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, ImageRun, Table, TableRow, TableCell,
        WidthType, BorderStyle, ShadingType, PageNumber, Footer, LineRuleType } = require('docx');
const M = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const FONT = 'Times New Roman';
// mini markup: **bold**, *italic*, _{sub}, ^{sup}
function runs(text, base = {}) {
  const out = []; let i = 0; let buf = ''; let st = { bold: false, italics: false };
  const flush = (extra = {}) => { if (buf) { out.push(new TextRun({ text: buf, font: FONT, size: base.size || 24, bold: st.bold || base.bold, italics: st.italics || base.italics, ...extra })); buf = ''; } };
  while (i < text.length) {
    if (text.startsWith('**', i)) { flush(); st.bold = !st.bold; i += 2; continue; }
    if (text[i] === '*' ) { flush(); st.italics = !st.italics; i += 1; continue; }
    if ((text.startsWith('_{', i) || text.startsWith('^{', i))) {
      flush(); const sub = text[i] === '_'; const j = text.indexOf('}', i); buf = text.slice(i + 2, j);
      flush(sub ? { subScript: true } : { superScript: true }); i = j + 1; continue;
    }
    buf += text[i]; i++;
  }
  flush(); return out;
}
const para = (t, o = {}) => new Paragraph({ children: runs(t, o), spacing: { after: 120, line: 360, lineRule: LineRuleType.AUTO }, alignment: o.align || AlignmentType.JUSTIFIED, ...(o.p || {}) });
const children = [];
for (const b of M.blocks) {
  if (b.type === 'title') children.push(new Paragraph({ children: runs(b.text, { bold: true, size: 32 }), alignment: AlignmentType.CENTER, spacing: { after: 240 } }));
  else if (b.type === 'center') children.push(para(b.text, { align: AlignmentType.CENTER }));
  else if (b.type === 'h1') children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(b.text, { bold: true, size: 28 }), spacing: { before: 240, after: 120 } }));
  else if (b.type === 'h2') children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(b.text, { bold: true, italics: false, size: 24 }), spacing: { before: 200, after: 80 } }));
  else if (b.type === 'p') children.push(para(b.text));
  else if (b.type === 'ref') children.push(new Paragraph({ children: runs(b.text, { size: 22 }), spacing: { after: 60 }, indent: { left: 440, hanging: 440 } }));
  else if (b.type === 'fig') {
    const img = fs.readFileSync(b.path); const w = b.width || 620; const h = Math.round(w * b.aspect);
    children.push(new Paragraph({ children: [new ImageRun({ type: 'png', data: img, transformation: { width: w, height: h } })], alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 } }));
    children.push(para(b.caption, { size: 20 }));
  } else if (b.type === 'table') {
    children.push(para(b.caption, { size: 20, p: { keepNext: true } }));
    const ncol = b.header.length; const total = 9360; const cw = b.widths ? b.widths.map(x => Math.round(x * total)) : Array(ncol).fill(Math.floor(total / ncol));
    const fix = total - cw.reduce((a, c) => a + c, 0); cw[cw.length - 1] += fix;
    const border = { style: BorderStyle.SINGLE, size: 4, color: '808080' }; const none = { style: BorderStyle.NONE, size: 0, color: 'FFFFFF' };
    const cell = (t, j, hdr, last) => new TableCell({ width: { size: cw[j], type: WidthType.DXA },
      borders: { top: hdr ? border : none, bottom: (hdr || last) ? border : none, left: none, right: none },
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({ children: runs(String(t), { size: 18, bold: hdr }), alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })] });
    const rows = [new TableRow({ tableHeader: true, children: b.header.map((t, j) => cell(t, j, true, false)) })];
    b.rows.forEach((r, ri) => rows.push(new TableRow({ children: r.map((t, j) => cell(t, j, false, ri === b.rows.length - 1)) })));
    children.push(new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: cw, rows }));
    if (b.note) children.push(para(b.note, { size: 18 }));
    else children.push(new Paragraph({ children: [], spacing: { after: 120 } }));
  }
}
const doc = new Document({
  styles: { default: { document: { run: { font: FONT, size: 24 } } }, paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 28, bold: true, color: '000000' }, paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true, run: { font: FONT, size: 24, bold: true, color: '000000' }, paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 } }] },
  sections: [{ properties: { page: { size: { width: 11906, height: 16838 }, margin: { top: 1440, bottom: 1440, left: 1273, right: 1273 } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children }]
});
Packer.toBuffer(doc).then(buf => { fs.writeFileSync(process.argv[3], buf); console.log('wrote', process.argv[3]); });
