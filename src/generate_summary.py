"""
Generate 5-Page Portfolio Case Study PDF
Focuses on Engineering, Business Logic, and Iteration.
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

class Config:
    OUTPUT_FILENAME = "credit_threshold_portfolio_case_study.pdf"
    PAPER_SIZE = LETTER
    MARGINS = (0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch)
    
    PRIMARY_COLOR = colors.HexColor('#1a365d')
    SECONDARY_COLOR = colors.HexColor('#2b6cb0')
    GRAY = colors.HexColor('#4a5568')
    LIGHT_GRAY = colors.HexColor('#edf2f7')

# ============================================================
# DATA LOADING
# ============================================================

class ProjectData:
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
            print("✓ Data loaded")
        except Exception as e:
            print(f"⚠ Error: {e}")
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
              textColor=Config.PRIMARY_COLOR, alignment=TA_CENTER, spaceAfter=12)
    
    add_style('ReportSubtitle', styles['Normal'], fontName='Helvetica', fontSize=14, 
              textColor=Config.GRAY, alignment=TA_CENTER, spaceAfter=24)
    
    add_style('SectionHeader', styles['Heading2'], fontName='Helvetica-Bold', fontSize=16, 
              textColor=Config.PRIMARY_COLOR, spaceBefore=12, spaceAfter=12)
    
    add_style('SubsectionHeader', styles['Heading3'], fontName='Helvetica-Bold', fontSize=13, 
              textColor=Config.SECONDARY_COLOR, spaceBefore=8, spaceAfter=6)
    
    add_style('BodyText', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=16, spaceAfter=6, alignment=TA_JUSTIFY)
    
    add_style('ListItem', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=16, spaceAfter=4, leftIndent=20)
    
    add_style('Caption', styles['Normal'], fontName='Helvetica', fontSize=9, 
              textColor=Config.GRAY, alignment=TA_CENTER, spaceBefore=4)
    
    add_style('TableCell', styles['Normal'], fontName='Helvetica', fontSize=9, alignment=TA_CENTER)
    
    return styles

# ============================================================
# PDF GENERATOR
# ============================================================

class ReportGenerator:
    def __init__(self, data: ProjectData, styles):
        self.data = data
        self.styles = styles
        self.elements = []
        
    def generate(self):
        self.add_title_page()
        self.add_problem_and_pipeline()
        self.add_iteration_and_fixes()
        self.add_results_and_validation()
        self.add_takeaways()
        
    def add_title_page(self):
        styles = self.styles
        self.elements.append(Spacer(1, 2.5 * inch))
        self.elements.append(Paragraph("Profit-Driven Credit Approval Thresholding", styles['ReportTitle']))
        self.elements.append(Paragraph("From Predictions to Portfolio Outcomes", styles['ReportSubtitle']))
        self.elements.append(Spacer(1, 0.5 * inch))
        self.elements.append(Paragraph("Ken Ira Lacson Talingting", 
            ParagraphStyle(name='AuthorName', parent=styles['Normal'], fontSize=14, 
                           textColor=Config.PRIMARY_COLOR, alignment=TA_CENTER)))
        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph("Data Science Project Report", 
            ParagraphStyle(name='AuthorInfo', parent=styles['Normal'], fontSize=12, 
                           textColor=Config.GRAY, alignment=TA_CENTER)))
        self.elements.append(Spacer(1, 0.3 * inch))
        self.elements.append(Paragraph(datetime.now().strftime("%B %d, %Y"), 
            ParagraphStyle(name='DateStyle', parent=styles['Normal'], fontSize=12, 
                           textColor=Config.GRAY, alignment=TA_CENTER)))
        self.elements.append(PageBreak())

    def add_problem_and_pipeline(self):
        styles = self.styles
        
        # 1. Problem Framing
        self.elements.append(Paragraph("1. Problem: Profit over Accuracy", styles['SectionHeader']))
        self.elements.append(Paragraph(
            "Credit models predict the probability a borrower will default. But a prediction alone does not make a decision. "
            "Someone must pick a cutoff: above this probability, reject; below it, approve. "
            "Most projects evaluate models by AUC, but the business question is different: "
            "<b>which threshold makes the most money?</b>",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("1.1 Why Threshold Matters", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "A marginal loan at 22% default probability might generate interest income that outweighs the loss. "
            "A conservative cutoff at 15% might reject profitable borrowers. "
            "The optimal threshold depends on the actual dollars involved, not just the probabilities.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # 2. Pipeline & Design
        self.elements.append(Paragraph("2. Pipeline: Data to Decision", styles['SectionHeader']))
        self.elements.append(Paragraph(
            "Using 1.34M LendingClub loans (2007–2018), I designed a pipeline to bridge the gap between predictions and profits.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        self.elements.append(Paragraph("2.1 Data Preparation", styles['SubsectionHeader']))
        bullets = [
            "<b>Cleaning:</b> Filtered resolved loans; created binary default target.",
            "<b>Profit Calculation:</b> Calculated realized profit as 'total_received - funded_amount'.",
            "<b>Split:</b> 80/20 temporal split by issue date (Train: 2007-2015, Test: 2016-2018) to prevent look-ahead bias."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        self.elements.append(Paragraph("2.2 Feature Engineering & Modeling", styles['SubsectionHeader']))
        bullets = [
            "Built 4 feature plans (Baseline Minimal, Domain Enhanced, Interaction Heavy, ML-Informed).",
            "Trained Logistic Regression (Platt scaling) and Random Forest (Isotonic calibration) on each plan.",
            "Calibrated probabilities using 5-fold CV to ensure reliable thresholding."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.1 * inch))

        self.elements.append(Paragraph("2.3 Optimization", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "Swept approval thresholds from 0.05 to 0.95 in increments of 0.01. "
            "Approved loans if the predicted default probability was below the threshold. "
            "Selected the threshold that maximized total portfolio profit.",
            styles['BodyText']
        ))
        self.elements.append(PageBreak())

    def add_iteration_and_fixes(self):
        styles = self.styles
        
        self.elements.append(Paragraph("3. Debugging & Iterative Improvement", styles['SectionHeader']))
        
        # 3.1 Leakage
        self.elements.append(Paragraph("3.1 Target Leakage", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "<b>Detection:</b> Initial models achieved an AUC of 0.9999 — a clear sign of target leakage.",
            styles['BodyText']
        ))
        self.elements.append(Paragraph(
            "<b>Resolution:</b> I identified 25 payment/collection columns (e.g., 'total_pymnt', 'recoveries', 'last_pymnt_d'). "
            "I implemented a pattern-based removal function to systematically exclude these features. "
            "<b>Result:</b> AUC dropped from 0.9999 to a realistic 0.6671, confirming the leakage was fixed.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # 3.2 Engineering failures
        self.elements.append(Paragraph("3.2 Feature Dependency Failures", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "<b>Issue:</b> The feature engineering pipeline failed due to missing 'installment' and 'inq_last_6mths' columns.",
            styles['BodyText']
        ))
        self.elements.append(Paragraph(
            "<b>Resolution:</b> I removed these dependencies and restructured the pipeline to generate all 4 plans successfully.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        # 3.3 Baseline Validation
        self.elements.append(Paragraph("3.3 Baseline Comparison", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "To quantify value added, I compared the model against 8 heuristic strategies:",
            styles['BodyText']
        ))
        baselines = [
            "Approve-All (no filtering)",
            "Reject-All (profit = $0)",
            "Random (50%)",
            "Fixed Thresholds (0.30, 0.50, 0.70)",
            "Grade-Based (approve A-C)",
            "DTI-Based (≤30%)",
            "FICO-Based (≥660)"
        ]
        for b in baselines:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(Paragraph(
            "This provided a practical benchmark for the model's financial performance.",
            styles['BodyText']
        ))
        self.elements.append(PageBreak())

    def add_results_and_validation(self):
        styles = self.styles
        
        self.elements.append(Paragraph("4. Results & Validation", styles['SectionHeader']))
        
        # Table
        if self.data.model_metrics is not None:
            df = self.data.model_metrics
            table_data = [["Feature Plan", "Model", "AUC", "Brier", "Optimal\nThreshold", "Optimal\nProfit"]]
            for _, row in df.iterrows():
                plan = row['plan'].replace('_', ' ').title()
                if plan in ['Mi Informed', 'Ml Informed']: plan = 'ML Informed'
                model = "Logistic Regression" if row['model_type'] == 'lr' else "Random Forest"
                
                # FIX: Ensure we pass a Paragraph object, not a raw string, for bold text
                profit_str = f"${row['optimal_profit']:,.0f}"
                if row['optimal_profit'] == df['optimal_profit'].max():
                    profit_cell = Paragraph(f"<b>{profit_str}</b>", styles['TableCell'])
                else:
                    profit_cell = Paragraph(profit_str, styles['TableCell'])
                
                table_data.append([plan, model, f"{row['auc']:.4f}", f"{row['brier']:.4f}", 
                                   f"{row['optimal_threshold']:.3f}", profit_cell])
            
            t = Table(table_data, colWidths=[1.3*inch, 1.7*inch, 0.7*inch, 0.7*inch, 0.9*inch, 1.1*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), Config.PRIMARY_COLOR),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
                ('BACKGROUND', (0,1), (-1,-1), Config.LIGHT_GRAY),
                ('PADDING', (0,0), (-1,-1), 4),
            ]))
            best_idx = df['optimal_profit'].idxmax()
            t.setStyle(TableStyle([('BACKGROUND', (0, best_idx+1), (-1, best_idx+1), colors.lightblue)]))
            
            self.elements.append(t)
            self.elements.append(Spacer(1, 0.1 * inch))
            self.elements.append(Paragraph(
                "<i>Table 1: Model performance across all plans. Best model highlighted in blue.</i>",
                styles['Caption']
            ))
            self.elements.append(Spacer(1, 0.3 * inch))

        # Confusion Matrix & Calibration Context
        self.elements.append(Paragraph("4.1 Business Decision Profile", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "At the optimal threshold of 0.620, the model approved 268,935 loans and declined 29. "
            "The high approval rate reflects the model's conservative nature: it only rejects loans when default risk is severe. "
            "This is financially optimal because the interest income from the 210,396 performing loans outweighs the losses from the 58,539 defaults.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.1 * inch))

        cm_path = Path("reports/visualizations/confusion_matrix.png")
        if cm_path.exists():
            img = Image(str(cm_path), width=5.5*inch, height=4*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 1: Business decisions at the optimal threshold.</i>",
                styles['Caption']
            ))
        self.elements.append(PageBreak())

    def add_takeaways(self):
        styles = self.styles
        
        # 5. Key Takeaways (Grounded, Data-Driven)
        self.elements.append(Paragraph("5. Observations & Limitations", styles['SectionHeader']))
        
        self.elements.append(Paragraph("5.1 Calibration > AUC", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "The Logistic Regression model (AUC 0.6671) generated higher profit than the Random Forest model (AUC 0.7001). "
            "This implies that calibration quality is often more important than raw discriminative power when a model is used to make threshold-based financial decisions.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("5.2 Retrospective Simulation & Limitations", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "This project is a retrospective analysis using historical data. While the results show that the threshold framework can add financial value, "
            "operational costs (underwriting, collections) and capital constraints were not modeled. "
            "The optimal threshold is conditional on the historical data distribution and may require updating as market conditions change.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("5.3 Technical Contributions", styles['SubsectionHeader']))
        bullets = [
            "Designed a custom `SafeFeatureEngineer` class to enforce strict train/test separation.",
            "Implemented `remove_leaking_features()` to eliminate target leakage.",
            "Generated 4 feature engineering plans to test the impact of complexity on profit.",
            "Conducted comprehensive baseline validation against 8 heuristic strategies."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.2 * inch))

        self.elements.append(Paragraph("5.4 Final Remarks", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "This project served as an exercise in connecting machine learning outputs to business decisions. "
            "The work reinforces that thoughtful model design, careful validation, and business-aligned evaluation are central to applied data science in finance.",
            styles['BodyText']
        ))

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("GENERATING 5-PAGE PORTFOLIO CASE STUDY")
    print("="*60)
    
    data = ProjectData()
    styles = create_styles()
    generator = ReportGenerator(data, styles)
    generator.generate()
    
    doc = SimpleDocTemplate(
        Config.OUTPUT_FILENAME,
        pagesize=Config.PAPER_SIZE,
        leftMargin=Config.MARGINS[0],
        rightMargin=Config.MARGINS[1],
        topMargin=Config.MARGINS[2],
        bottomMargin=Config.MARGINS[3]
    )
    doc.build(generator.elements)
    
    print(f"\n✓ Case study generated successfully!")
    print(f"  Output: {Config.OUTPUT_FILENAME}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()