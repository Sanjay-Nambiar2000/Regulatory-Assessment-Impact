# report_pdf.py
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors

def _p(text, style):
    return Paragraph(text.replace("\n", "<br/>"), style)

def _mk_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10, leading=13))
    styles.add(ParagraphStyle(name="Mono", parent=styles["BodyText"], fontName="Courier", fontSize=9, leading=11))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, textColor=colors.grey))
    styles.add(ParagraphStyle(name="Warn", parent=styles["BodyText"], fontSize=10, textColor=colors.red))
    styles.add(ParagraphStyle(name="OK",   parent=styles["BodyText"], fontSize=10, textColor=colors.green))
    return styles

def _mk_key_val_table(items, col_widths=None):
    data = [[f"<b>{k}</b>", v] for k, v in items]
    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("VALIGN", (0,0), (-1,-1), "TOP"),
        ("TEXTCOLOR", (0,0), (0,-1), colors.HexColor("#333333")),
        ("LINEBELOW", (0,0), (-1,-1), 0.25, colors.HexColor("#e0e0e0")),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
    ]))
    return tbl

def build_pdf_report(question: str, result: dict) -> bytes:
    """
    result schema (as produced by your /ask structured mode):
      - compliance_status: str
      - rationale: str
      - citations: [{source, page, quote, ...}]
      - violations_or_risks: [str]
      - alternative_suggestions: [str]
      - summary_proposal: str
      - human_supervision_required: bool
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=36, rightMargin=36, topMargin=42, bottomMargin=42)
    S = _mk_styles()
    story = []

    # Header
    story.append(_p("Regulatory Impact Assessment — Draft Output", S["H1"]))
    story.append(_p(f"Question", S["H2"]))
    story.append(_p(question or "-", S["Body"]))
    story.append(Spacer(1, 6))

    # Summary box
    status = (result or {}).get("compliance_status", "Unclear")
    rationale = (result or {}).get("rationale", "")
    human = (result or {}).get("human_supervision_required", True)

    status_style = S["OK"] if status.lower().startswith("compliant") else S["Warn"]
    story.append(_p("Summary", S["H2"]))
    story.append(_p(f"Compliance status: <b>{status}</b>", status_style))
    if human:
        story.append(_p("Human supervision required.", S["Warn"]))
    story.append(Spacer(1, 4))
    if rationale:
        story.append(_p(f"Rationale: {rationale}", S["Body"]))
    story.append(Spacer(1, 8))

    # Citations table
    cits = (result or {}).get("citations", []) or []
    story.append(_p("Key Citations", S["H2"]))
    if not cits:
        story.append(_p("— No citations returned —", S["Small"]))
    else:
        rows = []
        for c in cits:
            src = c.get("source", "-")
            page = c.get("page", "-")
            quote = c.get("quote", "")
            rows.append([
                f"{src} (p.{page})",
                quote.replace("\n", " ")
            ])
        tbl = Table([["Source (Page)", "Quoted Clause"]] + rows, colWidths=[180, 330])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#f2f2f2")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#333333")),
            ("ALIGN", (0,0), (-1,0), "LEFT"),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,0), 10),
            ("VALIGN", (0,0), (-1,-1), "TOP"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#fafafa")]),
            ("GRID", (0,0), (-1,-1), 0.25, colors.HexColor("#dddddd")),
        ]))
        story.append(tbl)
    story.append(Spacer(1, 10))

    # Violations / Risks
    story.append(_p("Violations / Risks", S["H2"]))
    vr = (result or {}).get("violations_or_risks", []) or []
    if not vr:
        story.append(_p("— None identified in retrieved evidence —", S["Small"]))
    else:
        for item in vr:
            story.append(_p(f"• {item}", S["Body"]))
    story.append(Spacer(1, 8))

    # Alternatives
    story.append(_p("Alternative Suggestions", S["H2"]))
    alts = (result or {}).get("alternative_suggestions", []) or []
    if not alts:
        story.append(_p("— None proposed —", S["Small"]))
    else:
        for item in alts:
            story.append(_p(f"• {item}", S["Body"]))
    story.append(Spacer(1, 8))

    # Proposal
    story.append(_p("Summary Proposal", S["H2"]))
    story.append(_p((result or {}).get("summary_proposal", "-"), S["Body"]))

    # Footer
    story.append(Spacer(1, 14))
    story.append(_p("Note: This draft is evidence-grounded. A qualified human reviewer must verify compliance before implementation.",
                    S["Small"]))

    doc.build(story)
    return buf.getvalue()
