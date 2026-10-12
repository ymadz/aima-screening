"""Render private proposal panels with the bundled PDF runtime, no model imports."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader


def create(output: Path) -> dict:
    rows = list(csv.DictReader((output/'glare_coverage_private.csv').open()))
    sheets = output/'review_sheets'; sheets.mkdir(mode=0o700)
    groups = ['uncertain_high_priority', 'likely_glare_present', 'empty_proposal_pending_review']
    target = sheets/'glare_review_all_119.pdf'
    if target.exists(): raise FileExistsError('Never overwrite review materials')
    pdf = canvas.Canvas(str(target), pagesize=(1190, 842))
    pdf.setTitle('AIMA Experiment 4B - private glare proposal review')
    pages = 0
    for group in groups:
        subset = [r for r in rows if r['review_group'] == group]
        for start in range(0, len(subset), 3):
            pages += 1
            pdf.setFont('Helvetica-Bold', 16)
            pdf.drawString(28, 809, f'AIMA 4B | {group.replace("_", " ")}')
            pdf.setFont('Helvetica', 10)
            pdf.drawString(28, 790, 'Private tissue crops. Cyan = proposed glare; green = approved anatomy. No glare approval is implied.')
            for position, row in enumerate(subset[start:start+3]):
                image = output/'review_panels'/f"{row['review_id']}.png"
                pdf.drawImage(ImageReader(str(image)), 24, 542-position*250, width=1142, height=243, preserveAspectRatio=True)
            pdf.setFont('Helvetica', 10)
            pdf.drawString(28, 20, f'Page {pages} | Review every proposed mask, including empty masks. Record accept/no-glare/correct/uncertain.')
            pdf.showPage()
    pdf.save()
    if len(PdfReader(target).pages) != pages: raise ValueError('PDF page count mismatch')
    return {'images': len(rows), 'pages': pages, 'pdf': str(target)}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    print(create(parser.parse_args().output.resolve()))
