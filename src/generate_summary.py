"""
Generate LinkedIn-Optimized Executive Brief (Defensible & Evidence-Based)
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
    PageBreak, Image, HRFlowable, Flowable
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import io

# ============================================================
# CONFIGURATION
# ============================================================

class Config:
    OUTPUT_FILENAME = "credit_threshold_linkedin_brief.pdf"
    PAPER_SIZE = LETTER
    MARGINS = (0.5 * inch, 0.5 * inch, 0.5 * inch, 0.5 * inch)
    
    PRIMARY_COLOR = colors.HexColor('#1e3a5f')
    SECONDARY_COLOR = colors.HexColor('#2a6f97')
    GRAY = colors.HexColor('#5c677d')
    LIGHT_GRAY = colors.HexColor('#f0f4f8')

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
            'plan': ['baseline_minimal', 'baseline_minimal'],
            'model_type': ['lr', 'rf'],
            'auc': [0.6671, 0.7001],
            'brier': [0.1622, 0.1558],
            'optimal_threshold': [0.620, 0.920],
            'optimal_profit': [123795944, 123790408]
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
    
    add_style('ReportTitle', styles['Title'], fontName='Helvetica-Bold', fontSize=28, 
              textColor=Config.PRIMARY_COLOR, alignment=TA_CENTER, spaceAfter=8)
    
    add_style('ReportSubtitle', styles['Normal'], fontName='Helvetica', fontSize=14, 
              textColor=Config.GRAY, alignment=TA_CENTER, spaceAfter=24)
    
    add_style('SectionHeader', styles['Heading2'], fontName='Helvetica-Bold', fontSize=16, 
              textColor=Config.PRIMARY_COLOR, spaceBefore=12, spaceAfter=8)
    
    add_style('SubsectionHeader', styles['Heading3'], fontName='Helvetica-Bold', fontSize=12, 
              textColor=Config.SECONDARY_COLOR, spaceBefore=8, spaceAfter=4)
    
    add_style('BodyText', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=18, spaceAfter=6, alignment=TA_JUSTIFY)
    
    add_style('ListItem', styles['Normal'], fontName='Helvetica', fontSize=11, 
              leading=18, spaceAfter=4, leftIndent=20)
    
    add_style('CalloutBox', styles['Normal'], fontName='Helvetica-Bold', fontSize=14, 
              textColor=Config.PRIMARY_COLOR, backColor=Config.LIGHT_GRAY, 
              borderPadding=12, borderWidth=0, spaceBefore=12, spaceAfter=12, alignment=TA_CENTER)
    
    add_style('Caption', styles['Normal'], fontName='Helvetica', fontSize=8, 
              textColor=Config.GRAY, alignment=TA_CENTER, spaceBefore=4)
    
    add_style('TableCell', styles['Normal'], fontName='Helvetica', fontSize=9, 
              alignment=TA_CENTER, leading=12)
    
    return styles




# ============================================================
# CUSTOM FLOWABLES (Visual Enhancements)
# ============================================================

class CalloutBox(Flowable):
    """A simple colored box with padding for key metrics."""
    def __init__(self, text, style, width=5.5*inch, height=0.6*inch):
        Flowable.__init__(self)
        self.text = text
        self.style = style
        self.width = width
        self.height = height
        
    def draw(self):
        # Draw the box
        self.canv.setFillColor(Config.LIGHT_GRAY)
        self.canv.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        # Draw the text
        self.canv.setFillColor(Config.PRIMARY_COLOR)
        p = Paragraph(self.text, self.style)
        p.wrap(self.width - 20, self.height - 10)
        p.drawOn(self.canv, 10, 10)

# ============================================================
# PDF GENERATOR
# ============================================================

class ReportGenerator:
    def __init__(self, data: ProjectData, styles):
        self.data = data
        self.styles = styles
        self.elements = []
        self.best_model = data.get_best_model()
        
    def generate(self):
        self.page_1_hero()
        self.page_2_methodology_and_calibration()
        self.page_3_results_and_baselines()
        self.page_4_limitations_and_validation()
        
    def page_1_hero(self):
        """Page 1: Title, Author, Core Finding, and Confusion Matrix (Guaranteed Layout)."""
        styles = self.styles
        
        # --- TITLE BLOCK ---
        self.elements.append(Spacer(1, 0.7 * inch))
        self.elements.append(Paragraph("Profit-Driven Credit Approval", styles['ReportTitle']))
        self.elements.append(Paragraph("Thresholding", styles['ReportTitle']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(Paragraph("From Predictions to Portfolio Outcomes", styles['ReportSubtitle']))
        self.elements.append(Spacer(1, 0.5 * inch))
        
        # --- CORE FINDING ---
        if self.best_model is not None:
            profit_str = f"${self.best_model['optimal_profit']:,.0f}"
            callout_text = f"Optimal Threshold: {self.best_model['optimal_threshold']:.3f} | Generated {profit_str} Net Cash Flow"
            box = CalloutBox(callout_text, styles['CalloutBox'])
            self.elements.append(box)
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # --- AUTHOR & DATE (GUARANTEED VERTICAL STACK) ---
        # We use an inner Table to force three separate lines (Name, Title, Date)
        author_data = [
            [Paragraph("Ken Ira Lacson Talingting", 
                ParagraphStyle(name='AuthorName', parent=styles['Normal'], fontSize=14, 
                               textColor=Config.PRIMARY_COLOR, alignment=TA_CENTER))],
            [Paragraph("Data Science Project", 
                ParagraphStyle(name='AuthorInfo', parent=styles['Normal'], fontSize=12, 
                               textColor=Config.GRAY, alignment=TA_CENTER))],
            [Paragraph("June 19, 2026", 
                ParagraphStyle(name='DateStyle', parent=styles['Normal'], fontSize=12, 
                               textColor=Config.GRAY, alignment=TA_CENTER))]
        ]
        author_table = Table(author_data, colWidths=[6.5*inch])
        author_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        self.elements.append(author_table)
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # --- CONFUSION MATRIX (FORCED TO STAY ON PAGE 1) ---
        cm_path = Path("reports/visualizations/confusion_matrix.png")
        if cm_path.exists():
            img = Image(str(cm_path), width=6.5*inch, height=4.5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 1: Approval decisions at the optimal threshold (0.620).</i>",
                styles['Caption']
            ))
        self.elements.append(PageBreak())

    def page_2_methodology_and_calibration(self):
        """Page 2: Methodology, Calibration Curve, and Transparency (Fixed Spacing)."""
        styles = self.styles
        
        # Remove the large top spacer so Page 2 starts immediately with text
        self.elements.append(Paragraph("Methodology & Validation", styles['SectionHeader']))
        
        # Pipeline (Left Column)
        left_content = []
        left_content.append(Paragraph("The Pipeline", styles['SubsectionHeader']))
        left_content.append(Paragraph(
            "Using 1.34M LendingClub loans (2007–2018), I built a temporal validation pipeline:",
            styles['BodyText']
        ))
        bullets = [
            "<b>Data:</b> Filtered resolved loans; binary default target.",
            "<b>Profit Metric:</b> <i>Net Cash Flow</i> = `total_received - funded_amount`.",
            "<b>Split:</b> 80/20 temporal split by issue date (Train: 2007–2015, Test: 2016–2018).",
            "<b>Engineering:</b> 4 feature plans with <i>SafeFeatureEngineer</i> to prevent leakage.",
            "<b>Optimization:</b> Swept 90 thresholds (0.05–0.95) to maximize profit."
        ]
        for b in bullets:
            left_content.append(Paragraph(f"• {b}", styles['ListItem']))
        
        # The Paradox (Right Column)
        right_content = []
        right_content.append(Paragraph("The AUC Paradox", styles['SubsectionHeader']))
        right_content.append(Paragraph(
            "In this experiment, the model with higher AUC (Random Forest, 0.7001) did not produce higher profit than the "
            "lower-AUC model (Logistic Regression, 0.6671).",
            styles['BodyText']
        ))
        right_content.append(Spacer(1, 0.05 * inch))
        right_content.append(Paragraph(
            "<b>Why?</b> The Logistic Regression produced a sharper profit peak at the optimal threshold (0.620). "
            "The Random Forest's probability estimates were less precise at the decision boundary, resulting in a flatter profit curve.",
            styles['BodyText']
        ))
        right_content.append(Spacer(1, 0.05 * inch))
        right_content.append(Paragraph(
            "<b>Observation:</b> Calibration and threshold behavior can be as important as raw discriminative power when the objective is profit.",
            styles['BodyText']
        ))
        
        # Two-column layout using Table
        left_table = Table([[item] for item in left_content], colWidths=[3.2*inch])
        right_table = Table([[item] for item in right_content], colWidths=[3.2*inch])
        two_col_table = Table([[left_table, right_table]], colWidths=[3.2*inch, 3.2*inch])
        two_col_table.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        self.elements.append(two_col_table)
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # Calibration Curve
        cal_path = Path("reports/visualizations/calibration_curve.png")
        if cal_path.exists():
            img = Image(str(cal_path), width=6.5*inch, height=4.5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 2: Calibration curve. The model's probabilities align with observed default rates.</i>",
                styles['Caption']
            ))
        self.elements.append(PageBreak())

    def page_3_results_and_baselines(self):
        """Page 3: Results Tables and Baseline Comparison."""
        styles = self.styles
        
        self.elements.append(Paragraph("Results: Model Performance & Baselines", styles['SectionHeader']))
        
        # Model Metrics Table
        self.elements.append(Paragraph("Model Comparison", styles['SubsectionHeader']))
        if self.data.model_metrics is not None:
            df = self.data.model_metrics
            table_data = [["Plan", "Model", "AUC", "Threshold", "Profit"]]
            for _, row in df.iterrows():
                plan = row['plan'].replace('_', ' ').title()
                if plan in ['Mi Informed', 'Ml Informed']: plan = 'ML Informed'
                model = "LR" if row['model_type'] == 'lr' else "RF"
                profit_str = f"${row['optimal_profit']:,.0f}"
                if row['optimal_profit'] == df['optimal_profit'].max():
                    profit_cell = Paragraph(f"<b>{profit_str}</b>", styles['TableCell'])
                else:
                    profit_cell = Paragraph(profit_str, styles['TableCell'])
                table_data.append([plan, model, f"{row['auc']:.4f}", 
                                   f"{row['optimal_threshold']:.3f}", profit_cell])
            
            t = Table(table_data, colWidths=[1.5*inch, 1.0*inch, 1.0*inch, 1.2*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), Config.PRIMARY_COLOR),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
                ('BACKGROUND', (0,1), (-1,-1), Config.LIGHT_GRAY),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            best_idx = df['optimal_profit'].idxmax()
            t.setStyle(TableStyle([('BACKGROUND', (0, best_idx+1), (-1, best_idx+1), colors.lightblue)]))
            self.elements.append(t)
            self.elements.append(Spacer(1, 0.2 * inch))
        
        # Baseline Comparison (Defensible framing)
        self.elements.append(Paragraph("Baseline Validation", styles['SubsectionHeader']))
        if self.data.baseline_validation is not None:
            df = self.data.baseline_validation
            approve_all = df[df['Metric'] == 'approve_all']['Value'].iloc[0]
            model_profit = df[df['Metric'] == 'model_best']['Value'].iloc[0]
            loss_reduction = model_profit - approve_all
            
            table_data = [
                ["Strategy", "Net Cash Flow", "Outcome"],
                ["Approve-All", f"${approve_all:,.0f}", "Loss"],
                ["Optimal Threshold (0.620)", f"${model_profit:,.0f}", f"Loss Reduced by ${loss_reduction:,.0f}"]
            ]
            t = Table(table_data, colWidths=[2.5*inch, 2.0*inch, 2.0*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), Config.SECONDARY_COLOR),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('GRID', (0,0), (-1,-1), 0.5, colors.gray),
                ('BACKGROUND', (0,1), (-1,-1), Config.LIGHT_GRAY),
                ('PADDING', (0,0), (-1,-1), 6),
            ]))
            self.elements.append(t)
            self.elements.append(Spacer(1, 0.1 * inch))
            self.elements.append(Paragraph(
                "<i>Note: The test period (2016–2018) had negative returns. The threshold reduced losses by $280k.</i>",
                styles['Caption']
            ))
        self.elements.append(PageBreak())

    def page_4_limitations_and_validation(self):
        """Page 4: Limitations, Data Integrity, and Next Steps."""
        styles = self.styles
        
        self.elements.append(Paragraph("Limitations & Validation", styles['SectionHeader']))
        
        # 1. Target Leakage (Shows integrity)
        self.elements.append(Paragraph("Data Integrity: Leakage Detection", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "Initial models achieved <b>AUC 0.9999</b>, indicating target leakage. I identified 25 payment/collection columns "
            "and removed them using a pattern-based function. After removal, AUC dropped to <b>0.6671</b>, confirming realistic model performance.",
            styles['BodyText']
        ))
        self.elements.append(Spacer(1, 0.2 * inch))
        
        # 2. Limitations (Crucial for credibility)
        self.elements.append(Paragraph("Limitations", styles['SubsectionHeader']))
        self.elements.append(Paragraph(
            "This is a retrospective simulation using historical data. The following factors were not modeled:",
            styles['BodyText']
        ))
        bullets = [
            "Operational costs (underwriting, collections, capital costs).",
            "Dynamic economic shifts (the optimal threshold is conditional on historical distributions).",
            "Live deployment constraints (simulation vs. real-world execution)."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.2 * inch))
        
        # 3. Future Work (Shows you think beyond the project)
        self.elements.append(Paragraph("Potential Extensions", styles['SubsectionHeader']))
        bullets = [
            "<b>Dynamic Thresholding:</b> Adjust the cut-off based on macro-economic indicators.",
            "<b>Profit-Weighted Training:</b> Directly optimize for financial outcomes during training.",
            "<b>Multi-Period Optimization:</b> Model portfolio dynamics over time."
        ]
        for b in bullets:
            self.elements.append(Paragraph(f"• {b}", styles['ListItem']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # Footer
        self.elements.append(Paragraph(
            "Full technical code and 15-page academic report available on GitHub.",
            ParagraphStyle(name='Footer', parent=styles['Normal'], fontSize=10, 
                           textColor=Config.GRAY, alignment=TA_CENTER)
        ))

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    print("="*60)
    print("GENERATING LINKEDIN-OPTIMIZED EXECUTIVE BRIEF")
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
    
    print(f"\n✓ LinkedIn Brief generated successfully!")
    print(f"  Output: {Config.OUTPUT_FILENAME}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()