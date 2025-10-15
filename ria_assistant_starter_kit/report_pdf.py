from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.units import cm

def _p(txt, style="BodyText"):
    return Paragraph(txt.replace("\n","<br/>"), getSampleStyleSheet()[style])

def write_report_pdf(data: dict, question: str, out_path: str):
    doc = SimpleDocTemplate(out_path, pagesize=A4, title="RIA Assessment Report")
    S = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Regulatory Impact Assessment — Automated Draft", S["Title"]))
    story.append(Spacer(1, 0.3*cm))
    story.append(_p(f"<b>Question:</b> {question}"))
    story.append(_p(f"<b>Compliance Status:</b> {data.get('compliance_status','Unclear')}"))
    story.append(_p(f"<b>Human Supervision Required:</b> {data.get('human_supervision_required', True)}"))
    story.append(Spacer(1, 0.2*cm))
    story.append(_p("<b>Rationale</b>", "Heading2"))
    story.append(_p(data.get("rationale","")))
    story.append(Spacer(1, 0.2*cm))

    cits = data.get("citations", [])
    if cits:
        story.append(_p("<b>Grounded Evidence (Citations)</b>", "Heading2"))
        items = []
        for c in cits:
            src = c.get("source","")
            pg = c.get("page","")
            qt = c.get("quote","").replace('"','&quot;')
            items.append(ListItem(_p(f"<b>{src}</b> (page {pg}) — “{qt}”")))
        story.append(ListFlowable(items, bulletType="bullet"))
        story.append(Spacer(1, 0.2*cm))

    def add_list(title, key):
        vals = data.get(key, [])
        if vals:
            story.append(_p(f"<b>{title}</b>", "Heading2"))
            items = [ListItem(_p(v)) for v in vals]
            story.append(ListFlowable(items, bulletType="bullet"))
            story.append(Spacer(1, 0.2*cm))

    add_list("Violations or Risks", "violations_or_risks")
    add_list("Alternative Suggestions", "alternative_suggestions")

    story.append(_p("<b>Summary Proposal</b>", "Heading2"))
    story.append(_p(data.get("summary_proposal","")))

    doc.build(story)
