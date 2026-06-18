"""
Generate 4-Page Process-Focused Portfolio Summary
Highlights workflow, failures, iterations, and business decisions.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io

# ============================================================
# CONFIGURATION
# ============================================================

class SummaryConfig:
    OUTPUT_FILENAME = "credit_threshold_process_summary.pdf"
    PAPER_SIZE = LETTER
    MARGINS = (0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch)
    
    PRIMARY_COLOR = colors.HexColor('#1a365d')
    SECONDARY_COLOR = colors.HexColor('#2b6cb0')
    GRAY = colors.HexColor('#4a5568')
    LIGHT_GRAY = colors.HexColor('#edf2f7')

# ============================================================
# DATA LOADING
# ============================================================

class SummaryData:
    def __init__(self):
        self.model_metrics = None
        self.baseline_validation = None
        self.load_data()
    
    def load_data(self):
        try:
            self.model_metrics = pd.read_csv(
                "reports/model_comparison/model_comparison_metrics.csv"
            )
            self.baseline_validation = pd.read_csv(
                "reports/baseline_validation.csv",
                header=None, names=['Metric', 'Value']
            )
            self.baseline_validation = self.baseline_validation.dropna().reset_index(drop=True)
            print("✓ Summary data loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load data: {e}")
            self.create_sample_data()
    
    def create_sample_data(self):
        self.model_metrics = pd.DataFrame({
            'plan': ['baseline_minimal'], 'model_type': ['lr'],
            'auc': [0.6671], 'brier': [0.1622],
            'optimal_threshold': [0.620], 'optimal_profit': [123795944]
        })
        self.baseline_validation = pd.DataFrame({
            'Metric': ['approve_all', 'model_best'],
            'Value': [-331903078, -331622596]
        })
    
    def get_best_model(self):
        if self.model_metrics is not None:
            best_idx = self.model_metrics['optimal_profit'].idxmax()
            return self.model_metrics.iloc[best_idx]
        return None

# ============================================================
# STYLES
# ============================================================

def create_styles():
    styles = getSampleStyleSheet()
    
    def add_style(name, parent, **kwargs):
        try:
            styles.add(ParagraphStyle(name=name, parent=parent, **kwargs))
        except KeyError:
            pass
    
    add_style('ReportTitle', styles['Title'], fontName='Helvetica-Bold', fontSize=24, 
              textColor=SummaryConfig.PRIMARY_COLOR, alignment=TA_CENTER, spaceAfter=12)
    
    add_style('ReportSubtitle', styles['Normal'], fontName='Helvetica', fontSize=14, 
              textColor=SummaryConfig.GRAY, alignment=TA_CENTER, spaceAfter=24)
    
    add_style('SectionHeader', styles['Heading2'], fontName='Helvetica-Bold', fontSize=16, 
              textColor=SummaryConfig.PRIMARY_COLOR, spaceBefore=12, spaceAfter=12)
    
    add_style('SubsectionHeader', styles['Heading3'], fontName='Helvetica-Bold', fontSize=13, 
              textColor=SummaryConfig.SECONDARY_COLOR, spaceBefore=8, spaceAfter=6)
    
    add_style('BodyText', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=15, spaceAfter=6, alignment=TA_JUSTIFY)
    
    add_style('ListItem', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=15, spaceAfter=4, leftIndent=20)
    
    add_style('Caption', styles['Normal'], fontName='Helvetica', fontSize=9, 
              textColor=SummaryConfig.GRAY, alignment=TA_CENTER, spaceBefore=4)
    
    add_style('TableCell', styles['Normal'], fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
    
    return styles

# ============================================================
# PDF GENERATOR (4-Page Process Summary)
# ============================================================

class SummaryGenerator:
    def __init__(self, data: SummaryData, styles):
        self.data = data
        self.styles = styles
        self.elements = []
        
    def generate(self):
        self.add_title_page()
        self.add_problem_and_approach()
        self.add_debugging_and_experiments()
        self.add_results_and_takeaways()
        
    def add_title_page(self):
        styles = self.styles
        self.elements.append(Spacer(1, 2.5 * inch))
        self.elements.append(Paragraph("Profit-Driven Credit Approval Thresholding", styles['ReportTitle']))
        self.elements.append(Paragraph("From Predictions to Portfolio Outcomes", styles['ReportSubtitle']))
        self.elements.append(Spacer(1, 0.5 * inch))
        self.elements.append(Paragraph("Ken Ira Lacson Talingting", 
            ParagraphStyle(name='AuthorName', parent=styles['Normal'], fontSize=14, 
                           textColor=SummaryConfig.PRIMARY_COLOR, alignment=TA_CENTER)))
        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph("Data Science Project Report", 
            ParagraphStyle(name='AuthorInfo', parent=styles['Normal'], fontSize=12, 
                           textColor=SummaryConfig.GRAY, alignment=TA_CENTER)))
        self.elements.append(Spacer(1, 0.3 * inch))
        self.elements.append(Paragraph(datetime.now().strftime("%B %d, %Y"), 
            ParagraphStyle(name='DateStyle', parent=styles['Normal'], fontSize=12, 
                           textColor=SummaryConfig.GRAY, alignment=TA_CENTER)))
        self.elements.append(PageBreak())

    def add_problem_and_approach(self):
        styles = self.styles
        
        # Problem Framing (From README)
        self.elements.append(Paragraph("1. The Problem", styles['SectionHeader']))
        self.elements.append(Paragraph(
            "Credit models predict the probability a borrower will default. But a prediction alone does not make a decision. "
            "Someone must pick a cutoff: above this probability, reject; below it, approve. "
            "Most projects evaluate models by AUC or accuracy, but the business question is different: "
            "<b>which threshold makes the most money?</b>",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # Core Philosophy
        self.elements.append(Paragraph("Framing", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "A marginal loan at 22% default probability might generate interest income that outweighs the loss. "
            "A conservative cutoff at 15% might reject profitable borrowers. "
            "The optimal threshold depends on the actual dollars involved, not just the probabilities.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # Approach
        self.elements.append(Paragraph("2. Approach & Pipeline", styles['SectionHeader']))
        self.elements.append(Paragraph(
            "Using 1.34M LendingClub loans (2007–2018), I built a pipeline that:",
            styles['BodyText']
        ))
        bullets = [
            "Calculated realized profit (total payments - funded amount) for each loan.",
            "Implemented 4 progressively complex feature engineering plans (Baseline, Domain, Interaction, ML-Informed).",
            "Trained Logistic Regression (Platt scaling) and Random Forest (Isotonic calibration) on each plan.",
            "Swept approval thresholds from 0.05 to 0.95 and calculated portfolio profit at each step."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(PageBreak())

    def add_debugging_and_experiments(self):
        styles = self.styles
        
        # The Failures & Fixes (From TODO.md)
        self.elements.append(Paragraph("3. Debugging & Iteration", styles['SectionHeader']))
        
        self.elements.append(Paragraph("<b>Target Leakage:</b> Initial models trained with a suspicious 0.9999 AUC.", styles['BodyText']))
        self.elements.append(Paragraph(
            "I identified 25 payment/collection columns (e.g., 'total_pymnt', 'recoveries') that were leaking the target. "
            "After implementing a pattern-based removal function, the AUC dropped to a realistic 0.6671, confirming the leakage was resolved.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("<b>Feature Dependency Failures:</b> The feature engineering pipeline initially failed due to missing columns.", styles['BodyText']))
        self.elements.append(Paragraph(
            "The 'installment' and 'inq_last_6mths' columns were not available, causing the pipeline to crash. "
            "I removed these dependencies and recalculated the features to successfully generate all 4 plans.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("<b>Baseline Validation:</b> To measure value added, I compared the model against 8 heuristic strategies.", styles['BodyText']))
        self.elements.append(Paragraph(
            "Strategies included: Approve-All, Reject-All, Fixed Thresholds (0.30, 0.50, 0.70), Grade-Based, DTI-Based, and FICO-Based. "
            "This helped contextualize the model's financial performance against simple rules.",
            styles['BodyText']
        ))
        self.elements.append(PageBreak())

    def add_results_and_takeaways(self):
        styles = self.styles
        
        # Metrics Table
        self.elements.append(Paragraph("4. Results", styles['SectionHeader']))
        
        if self.data.model_metrics is not None:
            df = self.data.model_metrics
            table_data = [["Feature Plan", "Model", "AUC", "Brier", "Optimal\nThreshold", "Optimal\nProfit"]]
            for _, row in df.iterrows():
                plan = row['plan'].replace('_', ' ').title()
                if plan in ['Mi Informed', 'Ml Informed']: plan = 'ML Informed'
                model = "Logistic Regression" if row['model_type'] == 'lr' else "Random Forest"
                profit_str = f"${row['optimal_profit']:,.0f}"
                if row['optimal_profit'] == df['optimal_profit'].max():
                    profit_str = f"<b>{profit_str}</b>"
                table_data.append([plan, model, f"{row['auc']:.4f}", f"{row['brier']:.4f}", 
                                   f"{row['optimal_threshold']:.3f}", profit_str])
            
            t = Table(table_data, colWidths=[1.3*inch, 1.7*inch, 0.7*inch, 0.7*inch, 0.9*inch, 1.1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), SummaryConfig.PRIMARY_COLOR),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
                ('BACKGROUND', (0,1), (-1,-1), SummaryConfig.LIGHT_GRAY),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            # Highlight best row
            best_idx = df['optimal_profit'].idxmax()
            t.setStyle(TableStyle([('BACKGROUND', (0, best_idx+1), (-1, best_idx+1), colors.lightblue)]))
            
            self.elements.append(t)
            self.elements.append(Spacer(1, 0.1 * inch))
            self.elements.append(Paragraph(
                "<i>Table 1: Model performance across all plans. Best model highlighted.</i>",
                styles['Caption']
            ))
            self.elements.append(Spacer(1, 0.3 * inch))

        # Confusion Matrix
        self.elements.append(Paragraph("Business Decision Profile", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "At the optimal threshold of 0.620, the model approved 268,935 loans and declined 29. "
            "This illustrates that the threshold is highly conservative, only rejecting loans when default risk is severe.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.1 * inch))

        cm_path = Path("reports/visualizations/confusion_matrix.png")
        if cm_path.exists():
            img = Image(str(cm_path), width=5*inch, height=4*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 1: Confusion matrix at optimal threshold (0.620).</i>",
                styles['Caption']
            ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # Final Takeaways
        self.elements.append(Paragraph("5. Key Takeaways", styles['SectionHeader']))
        self.elements.append(Paragraph(
            "The threshold, not the model alone, drives profitability. A well-calibrated Logistic Regression (AUC 0.6671) "
            "outperformed a higher-AUC Random Forest (0.7001) in total profit, reinforcing that calibration matters more than raw discrimination "
            "when making financial decisions.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph(
            "<b>Limitations:</b> Retrospective analysis; operational costs not modeled.",
            styles['BodyText']
        ))

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("GENERATING 4-PAGE PROCESS SUMMARY")
    print("="*60)
    
    data = SummaryData()
    styles = create_styles()
    generator = SummaryGenerator(data, styles)
    generator.generate()
    
    doc = SimpleDocTemplate(
        SummaryConfig.OUTPUT_FILENAME,
        pagesize=SummaryConfig.PAPER_SIZE,
        leftMargin=SummaryConfig.MARGINS[0],
        rightMargin=SummaryConfig.MARGINS[1],
        topMargin=SummaryConfig.MARGINS[2],
        bottomMargin=SummaryConfig.MARGINS[3]
    )
    doc.build(generator.elements)
    
    print(f"\n✓ Process summary generated successfully!")
    print(f"  Output: {SummaryConfig.OUTPUT_FILENAME}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()