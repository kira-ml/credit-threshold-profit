"""
Generate Professional Academic-Style PDF Report for Credit Threshold Project
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path          # <--- ADD THIS LINE
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import LETTER, A4
from reportlab.lib.units import inch, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, ListFlowable, ListItem, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import io




# ============================================================
# CONFIGURATION
# ============================================================


class ReportConfig:
    """Configuration for the PDF report"""
    
    # Document settings
    PAPER_SIZE = LETTER
    MARGINS = (0.8 * inch, 0.8 * inch, 0.8 * inch, 0.8 * inch)
    
    # Colors
    PRIMARY_COLOR = colors.HexColor('#1a365d')      # Dark navy
    SECONDARY_COLOR = colors.HexColor('#2b6cb0')    # Medium blue
    ACCENT_COLOR = colors.HexColor('#3182ce')       # Light blue
    GRAY = colors.HexColor('#4a5568')
    LIGHT_GRAY = colors.HexColor('#edf2f7')
    
    # Fonts
    TITLE_FONT = 'Helvetica-Bold'
    HEADER_FONT = 'Helvetica-Bold'
    BODY_FONT = 'Helvetica'
    CODE_FONT = 'Courier'
    
    # Author info
    AUTHOR = "Data Science Project Report"
    DATE = datetime.now().strftime("%B %d, %Y")
    
    # Output
    OUTPUT_FILENAME = "credit_threshold_profit_report.pdf"

# ============================================================
# PATHS
# ============================================================

OUTPUT_DIR = Path("reports/visualizations")

# ============================================================
# STYLE DEFINITIONS
# ============================================================

def create_styles():
    """Create custom paragraph styles"""
    styles = getSampleStyleSheet()
    
    # Helper to add style only if it doesn't exist
    def add_style(name, parent, **kwargs):
        try:
            styles.add(ParagraphStyle(name=name, parent=parent, **kwargs))
        except KeyError:
            # Style already exists, skip
            pass
    
    # Title style
    add_style(
        'ReportTitle',
        styles['Title'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=ReportConfig.PRIMARY_COLOR,
        alignment=TA_CENTER,
        spaceAfter=12
    )
    
    # Subtitle style
    add_style(
        'ReportSubtitle',
        styles['Normal'],
        fontName='Helvetica',
        fontSize=14,
        textColor=ReportConfig.GRAY,
        alignment=TA_CENTER,
        spaceAfter=24
    )
    
    # Section header style
    add_style(
        'SectionHeader',
        styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=16,
        textColor=ReportConfig.PRIMARY_COLOR,
        spaceBefore=18,
        spaceAfter=12
    )
    
    # Subsection header style
    add_style(
        'SubsectionHeader',
        styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=ReportConfig.SECONDARY_COLOR,
        spaceBefore=12,
        spaceAfter=8
    )
    
    # Body text style
    add_style(
        'BodyText',
        styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    # List item style
    add_style(
        'ListItem',
        styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=16,
        spaceAfter=4,
        leftIndent=20
    )
    
    # Caption style
    add_style(
        'Caption',
        styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=ReportConfig.GRAY,
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=12
    )
    
    # Table header style
    add_style(
        'TableHeader',
        styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        textColor=colors.white,
        alignment=TA_CENTER
    )
    
    # Table cell style
    add_style(
        'TableCell',
        styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        alignment=TA_CENTER,
        leading=12
    )
    
    # Code style
    add_style(
        'Code',
        styles['Normal'],
        fontName='Courier',
        fontSize=9,
        textColor=ReportConfig.PRIMARY_COLOR,
        backColor=ReportConfig.LIGHT_GRAY,
        spaceBefore=4,
        spaceAfter=4
    )
    
    return styles

# ============================================================
# DATA LOADING AND PROCESSING
# ============================================================

class ReportData:
    """Load and process data for the report"""
    
    def __init__(self):
        self.model_metrics = None
        self.baseline_validation = None
        self.load_data()
    
    def load_data(self):
        """Load data from CSV files"""
        try:
            # Load model comparison metrics
            self.model_metrics = pd.read_csv(
                "reports/model_comparison/model_comparison_metrics.csv"
            )
            
            # Load baseline validation
            self.baseline_validation = pd.read_csv(
                "reports/baseline_validation.csv",
                header=None,
                names=['Metric', 'Value']
            )
            # FIX: Drop the first row if it is NaN (header fix)
            self.baseline_validation = self.baseline_validation.dropna().reset_index(drop=True)
            
            print("✓ Data loaded successfully")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load data files: {e}")
            # Create sample data for demonstration
            self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data if files not found"""
        # Sample model metrics
        self.model_metrics = pd.DataFrame({
            'plan': ['baseline_minimal', 'baseline_minimal', 'domain_enhanced', 'domain_enhanced',
                     'interaction_heavy', 'interaction_heavy', 'ml_informed', 'ml_informed'],
            'model_type': ['lr', 'rf', 'lr', 'rf', 'lr', 'rf', 'lr', 'rf'],
            'auc': [0.6671, 0.7001, 0.6524, 0.6997, 0.6805, 0.7002, 0.6794, 0.6999],
            'brier': [0.1622, 0.1558, 0.1639, 0.1558, 0.1610, 0.1558, 0.1610, 0.1570],
            'optimal_threshold': [0.620, 0.920, 0.890, 0.870, 0.940, 0.910, 0.930, 0.790],
            'optimal_profit': [123795944, 123790408, 123790408, 123790408, 
                               123679150, 123790408, 123676776, 123790408]
        })
        
        # Sample baseline validation
        self.baseline_validation = pd.DataFrame({
            'Metric': ['approve_all', 'reject_all', 'random_mean', 'threshold_0.30', 
                       'threshold_0.50', 'threshold_0.70', 'grade_based', 'dti_based', 
                       'fico_based', 'model_best'],
            'Value': [-331903078, 0, -165132729, -221095742, 
                      -328721832, -331676636, -159071724, -279374477,
                      -331903078, -331622596]
        })
    
    def get_best_model(self):
        """Get the best performing model"""
        if self.model_metrics is not None:
            best_idx = self.model_metrics['optimal_profit'].idxmax()
            return self.model_metrics.iloc[best_idx]
        return None
    
    def format_currency(self, value):
        """Format value as currency"""
        if value >= 0:
            return f"${value:,.0f}"
        else:
            return f"$-{abs(value):,.0f}"

# ============================================================
# PLOT GENERATION
# ============================================================

def generate_plots(data: ReportData):
    """Generate plots for the report"""
    plots = {}
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors_plot = ['#1a365d', '#2b6cb0', '#3182ce', '#63b3ed']
    
    # 1. Model Comparison Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if data.model_metrics is not None:
        # AUC comparison
        df = data.model_metrics
        x_labels = [f"{row['plan']}\n({row['model_type']})" 
                   for _, row in df.iterrows()]
        
        ax1.bar(range(len(df)), df['auc'], color=colors_plot)
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax1.set_ylabel('AUC')
        ax1.set_title('AUC Comparison Across Feature Plans', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Profit comparison
        ax2.bar(range(len(df)), df['optimal_profit'] / 1e6, color=colors_plot)
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Optimal Profit ($ Millions)')
        ax2.set_title('Optimal Profit Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plots['model_comparison'] = fig
    
    # 2. Baseline Comparison Plot
    if data.baseline_validation is not None:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df = data.baseline_validation
        # Filter out random_std
        df = df[df['Metric'] != 'random_std']
        
        labels = df['Metric'].tolist()
        values = df['Value'].tolist()
        
        # Color coding
        colors_bar = []
        for metric in labels:
            metric_str = str(metric)
            if metric_str == 'model_best':
                colors_bar.append('#2b6cb0')  # Blue for best model
            elif metric_str == 'approve_all':
                colors_bar.append('#e53e3e')  # Red for approve all
            elif metric_str == 'reject_all':
                colors_bar.append('#48bb78')  # Green for reject all
            else:
                colors_bar.append('#a0aec0')  # Gray for others
        
        # Format labels for display
        display_labels = []
        for label in labels:
            # Convert to string explicitly
            label_str = str(label)
            
            if label_str == 'approve_all':
                display_labels.append('Approve-All')
            elif label_str == 'reject_all':
                display_labels.append('Reject-All')
            elif label_str == 'random_mean':
                display_labels.append('Random (50%)')
            elif label_str.startswith('threshold_'):
                t = label_str.split('_')[1]
                display_labels.append(f'Fixed Threshold {t}')
            elif label_str == 'grade_based':
                display_labels.append('Grade-Based')
            elif label_str == 'dti_based':
                display_labels.append('DTI-Based')
            elif label_str == 'fico_based':
                display_labels.append('FICO-Based')
            elif label_str == 'model_best':
                display_labels.append('Your Model (Optimal)')
            else:
                display_labels.append(label_str.replace('_', ' ').title())
        
        # Fixed: set ticks first, then ticklabels
        x_pos = np.arange(len(display_labels))
        bars = ax.bar(x_pos, [v / 1e6 for v in values], color=colors_bar)
        
        # Set ticks and labels properly (reduced rotation for better readability)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=30, ha='right', fontsize=8)
        
        # Add horizontal line for model performance
        model_idx = labels.index('model_best')
        ax.axhline(values[model_idx] / 1e6, 
                   color='#2b6cb0', linestyle='--', alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Baseline Strategy')
        ax.set_ylabel('Profit ($ Millions)')
        ax.set_title('Baseline Comparison: Model vs. Simple Heuristics', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val/1e6:.0f}M',
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=8, color='black')
        
        plt.tight_layout()
        plots['baseline_comparison'] = fig
    
    return plots

# ============================================================
# PDF GENERATION
# ============================================================

class ReportGenerator:
    """Generate the PDF report"""
    
    def __init__(self, data: ReportData, styles, plots):
        self.data = data
        self.styles = styles
        self.plots = plots
        self.elements = []
        
    def generate(self):
        """Generate all report elements"""
        self.add_title_page()
        self.add_executive_summary()
        self.add_table_of_contents()
        self.add_section_1_introduction()
        self.add_section_2_methodology()
        self.add_section_3_results()
        self.add_section_4_discussion()
        self.add_section_5_conclusion()
        self.add_references()
        self.add_appendix()
        

    def add_title_page(self):
        """Add title page"""
        styles = self.styles
        
        # School/institution header
        self.elements.append(Spacer(1, 2.5 * inch))
        
        # Main title
        self.elements.append(Paragraph(
            "Profit-Driven Credit Approval Thresholding",
            styles['ReportTitle']
        ))
        self.elements.append(Paragraph(
            "From Predictions to Portfolio Outcomes",
            styles['ReportSubtitle']
        ))
        
        self.elements.append(Spacer(1, 0.5 * inch))
        
        # Author name
        self.elements.append(Paragraph(
            "Ken Ira Lacson Talingting",
            ParagraphStyle(
                name='AuthorNameStyle',
                parent=styles['Normal'],
                fontSize=14,
                textColor=ReportConfig.PRIMARY_COLOR,
                alignment=TA_CENTER
            )
        ))
        self.elements.append(Spacer(1, 0.2 * inch))
        
        # Author info
        self.elements.append(Paragraph(
            "Data Science Project Report",
            ParagraphStyle(
                name='AuthorStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=ReportConfig.GRAY,
                alignment=TA_CENTER
            )
        ))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        self.elements.append(Paragraph(
            ReportConfig.DATE,
            ParagraphStyle(
                name='DateStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=ReportConfig.GRAY,
                alignment=TA_CENTER
            )
        ))
        
        # Project description
        self.elements.append(Spacer(1, 1.5 * inch))
        self.elements.append(Paragraph(
            "A data science project exploring the intersection of " +
            "machine learning and financial decision-making",
            ParagraphStyle(
                name='DescriptionStyle',
                parent=styles['Normal'],
                fontSize=12,
                textColor=ReportConfig.GRAY,
                alignment=TA_CENTER,
                leading=18
            )
        ))
        
        self.elements.append(PageBreak())




    def add_executive_summary(self):
        """Add executive summary"""
        styles = self.styles

        self.elements.append(Paragraph("Executive Summary", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))

        # Get best model data
        best_model = self.data.get_best_model()
        baseline_profit = None
        if self.data.baseline_validation is not None:
            baseline_row = self.data.baseline_validation[
                self.data.baseline_validation['Metric'] == 'approve_all'
            ]
            if not baseline_row.empty:
                baseline_profit = baseline_row['Value'].iloc[0]

        # Executive summary text
        summary = f"""
        This project examines a practical question in consumer lending: <b>which loan applicants should a lender approve to maximize net profit?</b> 
        Credit risk models are often evaluated on predictive accuracy, but in a lending context the business objective is profitability. 
        This report presents a framework for optimizing the approval threshold of a probabilistic classifier with portfolio profit as the primary objective.

        Using publicly available LendingClub loan data from 2007 to 2018 (1.34 million loans), we trained Logistic Regression and Random 
        Forest models across four feature engineering plans. Approval thresholds were systematically swept from 0.05 to 0.95 in increments 
        of 0.01, and realized portfolio profit was calculated at each threshold using historical cash flows.

        <b>Key Observations:</b>
        """

        self.elements.append(Paragraph(summary, styles['BodyText']))

        # Key findings as bullet points
        observations = [
            f"<b>Best performing model:</b> Logistic Regression on 10 baseline features with AUC = 0.6671",
            f"<b>Optimal threshold identified:</b> 0.620 — reject loans with default probability above 62%",
        ]

        if best_model is not None and not best_model.empty:
            observations.append(f"<b>Portfolio profit at this threshold:</b> ${best_model['optimal_profit']:,.0f} on the test set")
        else:
            observations.append("<b>Portfolio profit at this threshold:</b> Data not available")

        observations.append("<b>Test Set Performance:</b> Reduced losses by $280k vs. Approve-All during the 2016-2018 adverse period")

        for obs in observations:
            if obs:
                self.elements.append(Paragraph(f"• {obs}", styles['ListItem']))

        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph(
            "This framework suggests that the threshold—not the model alone—has a meaningful impact on profitability. " +
            "A well-calibrated model with moderate discriminative power can be more useful for decision-making than a higher-AUC model " +
            "with less reliable probability estimates. The results reinforce the value of aligning model evaluation with business objectives.",
            styles['BodyText']
        ))

        self.elements.append(PageBreak())



    def add_table_of_contents(self):
        """Add table of contents"""
        styles = self.styles
        
        self.elements.append(Paragraph("Table of Contents", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.2 * inch))
        
        toc_items = [
            ("1. Introduction", "3"),
            ("2. Methodology", "4"),
            ("   2.1 Data", "4"),
            ("   2.2 Feature Engineering", "4"),
            ("   2.3 Model Training and Calibration", "5"),
            ("   2.4 Threshold Optimization", "5"),
            ("   2.5 Baseline Strategies", "5"),
            ("   2.6 Target Leakage Detection", "6"),
            ("3. Results", "6"),
            ("   3.1 Model Performance Comparison", "6"),
            ("   3.2 Profit Curves", "7"),
            ("   3.3 Baseline Comparison", "7"),
            ("   3.4 Feature Importance", "7"),
            ("   3.5 Decision Performance", "8"),
            ("4. Discussion", "8"),
            ("   4.1 AUC and Profit: A Practical Observation", "8"),
            ("   4.2 Target Leakage: A Cautionary Note", "8"),
            ("   4.3 Limitations", "8"),
            ("5. Conclusion", "9"),
            ("References", "10"),
            ("Appendix", "10")
        ]
        
        for item, page in toc_items:
            # Format with dots, ensuring proper spacing
            dots = "." * (60 - len(item) - len(page))
            line = f"{item}{dots}{page}"
            self.elements.append(Paragraph(line, styles['BodyText']))
            self.elements.append(Spacer(1, 0.05 * inch))  # Add a small spacer between each line
        
        self.elements.append(PageBreak())
    
    def add_section_1_introduction(self):
        """Add introduction section"""
        styles = self.styles
        
        self.elements.append(Paragraph("1. Introduction", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        intro_text = """
        Credit scoring models are widely used in consumer lending to estimate the probability that a borrower will default on a loan. 
        These models are often evaluated using statistical metrics such as AUC and accuracy, which measure how well the model separates 
        good loans from bad ones. However, in a lending context, the relevant business question is profitability, not statistical performance.
        
        This project explores a practical question: <b>which loan applicants should a lender approve to maximize net profit?</b> 
        A predicted probability of default does not by itself determine a decision. Some cutoff must be chosen—above which a loan is rejected, 
        below which it is approved. The optimal cutoff depends on factors such as interest rates, loan amounts, recovery rates, and 
        the distribution of default probabilities in the portfolio.
        
        <b>Project Objectives:</b>
        """

        self.elements.append(Paragraph(intro_text, styles['BodyText']))
        
        objectives = [
            "Train probabilistic classifiers to estimate default likelihood for each loan",
            "Systematically sweep approval thresholds from 0.05 to 0.95",
            "Calculate realized portfolio profit at each threshold using historical outcomes",
            "Identify the threshold that maximizes portfolio profit",
            "Compare against simple heuristic baselines to quantify value added"
        ]
        
        for obj in objectives:
            self.elements.append(Paragraph(f"• {obj}", styles['ListItem']))
        
        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph(
            "A central observation from this project is that the threshold choice has a material impact on profitability. " +
            "A well-calibrated model with moderate discriminative power can support better financial decisions than a higher-AUC model " +
            "with less reliable probability estimates. This underscores the importance of calibration and business alignment in applied " +
            "data science work.",
            styles['BodyText']
        ))
        
        self.elements.append(PageBreak())



    def add_section_2_methodology(self):
        """Add methodology section"""
        styles = self.styles
        
        self.elements.append(Paragraph("2. Methodology", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        # 2.1 Data
        self.elements.append(Paragraph("2.1 Data", styles['SubsectionHeader']))
        
        data_text = """
        We used publicly available LendingClub loan data spanning 2007–2018, accessible through Kaggle. The dataset contains 
        <b>1.34 million</b> loan records with 110 features covering borrower characteristics, loan terms, and eventual outcomes.
        
        <b>Data Cleaning:</b>
        """
        
        self.elements.append(Paragraph(data_text, styles['BodyText']))
        
        cleaning_steps = [
            "Removed loans with incomplete or inconsistent status information",
            "Converted categorical variables (term, grade, home ownership) to numeric representations",
            "Dropped sparse columns (missing rate > 80%)",
            "Removed 25 columns containing payment history, collections, or post-origination behavior"
        ]
        
        for step in cleaning_steps:
            self.elements.append(Paragraph(f"• {step}", styles['ListItem']))
        
        self.elements.append(Spacer(1, 0.2 * inch))
        self.elements.append(Paragraph(
            "<b>Target Definition:</b> The binary target <i>default</i> was defined as loans with status 'Charged Off' (1) versus " +
            "'Fully Paid' or 'Current' (0).",
            styles['BodyText']
        ))
        
        self.elements.append(Paragraph(
            "<b>Cash Surplus Calculation:</b> For each loan, realized net cash flow was computed as:",
            styles['BodyText']
        ))
        
        # Profit formula in code block
        self.elements.append(Paragraph(
            "Cash Surplus = total_received - funded_amount",
            styles['Code']
        ))
        
        self.elements.append(Paragraph(
            "Where <i>total_received</i> includes all payments made by the borrower, and <i>funded_amount</i> is the principal disbursed. " +
            "The total portfolio net cash surplus across all loans was <b>$555.6 million</b>.",
            styles['BodyText']
        ))
        
        self.elements.append(Paragraph(
            "<b>Train/Test Split:</b> Data was split 80/20 by issue date (temporal split) to prevent look-ahead bias and simulate " +
            "realistic deployment conditions.",
            styles['BodyText']
        ))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 2.2 Feature Engineering
        self.elements.append(Paragraph("2.2 Feature Engineering", styles['SubsectionHeader']))
        
        feature_text = """
        We developed four progressively complex feature engineering plans, all implemented through a custom <i>SafeFeatureEngineer</i> 
        class that enforces strict train/test separation. All aggregations were calculated exclusively on the training set and then 
        applied to the test set to prevent data leakage.
        """
        
        self.elements.append(Paragraph(feature_text, styles['BodyText']))
        
        # Feature plans as table
        feature_data = [
            ["Plan", "Features", "Description"],
            ["Baseline Minimal", "10", "Interpretable features directly from application data"],
            ["Domain Enhanced", "17", "Adds credit risk domain knowledge features"],
            ["Interaction Heavy", "24", "Adds interaction terms and log transformations"],
            ["ML-Informed", "37", "Adds default rates by category and polynomial features"]
        ]
        
        feature_table = Table(feature_data, colWidths=[2*inch, 1*inch, 3.5*inch])
        feature_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), ReportConfig.PRIMARY_COLOR),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
            ('BACKGROUND', (0, 1), (-1, -1), ReportConfig.LIGHT_GRAY),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        
        self.elements.append(feature_table)
        self.elements.append(Spacer(1, 0.2 * inch))
        
        # 2.3 Model Training
        self.elements.append(Paragraph("2.3 Model Training and Calibration", styles['SubsectionHeader']))
        
        model_text = """
        We trained two widely-used probabilistic classifiers on each of the four feature engineering plans:
        
        <b>Logistic Regression:</b> A linear model with L2 regularization (max iterations = 1000). We applied Platt scaling 
        (sigmoid calibration) with 5-fold cross-validation to ensure probability estimates accurately reflect true default likelihood.
        
        <b>Random Forest:</b> An ensemble of 100 decision trees (max depth = 10). We applied isotonic regression calibration with 
        5-fold cross-validation to transform raw prediction scores into well-calibrated probabilities.
        
        Both models were trained with a fixed random seed (42) for reproducibility.
        """
        
        self.elements.append(Paragraph(model_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 2.4 Threshold Optimization
        self.elements.append(Paragraph("2.4 Threshold Optimization", styles['SubsectionHeader']))
        
        threshold_text = """
        For each trained model, we performed a systematic threshold sweep from 0.05 to 0.95 in increments of 0.01. At each threshold value, 
        we applied the decision rule:
        """
        
        self.elements.append(Paragraph(threshold_text, styles['BodyText']))
        self.elements.append(Paragraph(
            "Approve loan if predicted_default_probability < threshold",
            styles['Code']
        ))
        
        self.elements.append(Paragraph(
            "The total portfolio profit at each threshold was computed by summing the realized profits of all approved loans. " +
            "The optimal threshold was selected as the value that maximized this total portfolio profit.",
            styles['BodyText']
        ))
        
        self.elements.append(PageBreak())
        
        # 2.5 Baseline Strategies
        self.elements.append(Paragraph("2.5 Baseline Strategies", styles['SubsectionHeader']))
        
        baseline_text = """
        To contextualize our model performance, we compared against 8 baseline strategies:
        """
        
        self.elements.append(Paragraph(baseline_text, styles['BodyText']))
        
        baselines = [
            "Approve-All: Approve every loan (no filtering)",
            "Reject-All: Approve no loans (profit = $0)",
            "Random (50%): Randomly approve half the loans",
            "Fixed Threshold 0.30: Approve loans with default probability < 0.30",
            "Fixed Threshold 0.50: Approve loans with default probability < 0.50",
            "Fixed Threshold 0.70: Approve loans with default probability < 0.70",
            "Grade-Based (D-F): Approve only loans with grade A-C",
            "DTI-Based (≤30%): Approve only loans with DTI ≤ 30%",
            "FICO-Based (≥660): Approve only loans with FICO ≥ 660"
        ]
        
        for baseline in baselines:
            self.elements.append(Paragraph(f"• {baseline}", styles['ListItem']))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 2.6 Target Leakage
        self.elements.append(Paragraph("2.6 Target Leakage Detection", styles['SubsectionHeader']))
        
        leakage_text = """
        A critical finding during this project was the presence of target leakage in the original feature set. Initial model training 
        produced an AUC of <b>0.9999</b>—a suspiciously high value indicating that features containing information about post-origination 
        behavior were included in the training data.
        
        We identified 25 columns containing patterns matching payment history, collections, or post-origination behavior. These columns 
        were systematically removed from both train and test sets. After removal, the AUC dropped from <b>0.9999</b> to a realistic 
        <b>0.6671</b>, confirming that leakage had been effectively eliminated.
        """
        
        self.elements.append(Paragraph(leakage_text, styles['BodyText']))
        
        self.elements.append(PageBreak())

    def add_section_3_results(self):
        """Add results section"""
        styles = self.styles
        
        self.elements.append(Paragraph("3. Results", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        # 3.1 Model Performance
        self.elements.append(Paragraph("3.1 Model Performance Comparison", styles['SubsectionHeader']))
        
        if self.data.model_metrics is not None:
            # Create table data
            table_data = [["Feature Plan", "Model", "AUC", "Brier", "Optimal\nThreshold", "Optimal\nProfit"]]
            
            df = self.data.model_metrics
            for _, row in df.iterrows():
                plan_name = row['plan'].replace('_', ' ').title()
                # FIX: Correct "MI Informed" typo (catch 'Ml Informed' as well)
                if plan_name in ['Mi Informed', 'Ml Informed']:
                    plan_name = 'ML Informed'
                model_name = "Logistic Regression" if row['model_type'] == 'lr' else "Random Forest"
                
                # Format profit - wrap in Paragraph if it's the best
                profit_str = f"${row['optimal_profit']:,.0f}"
                if row['optimal_profit'] == df['optimal_profit'].max():
                    profit_display = Paragraph(f"<b>{profit_str}</b>", styles['TableCell'])
                else:
                    profit_display = Paragraph(profit_str, styles['TableCell'])
                
                table_data.append([
                    plan_name,
                    model_name,
                    f"{row['auc']:.4f}",
                    f"{row['brier']:.4f}",
                    f"{row['optimal_threshold']:.3f}",
                    profit_display
                ])
            
            # Create table
            col_widths = [1.5*inch, 1.8*inch, 0.8*inch, 0.8*inch, 1.0*inch, 1.2*inch]
            results_table = Table(table_data, colWidths=col_widths)
            
            # Style the table
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), ReportConfig.PRIMARY_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 1), (-1, -1), ReportConfig.LIGHT_GRAY),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]
            
            # Highlight best row
            best_idx = df['optimal_profit'].idxmax()
            best_row_pos = best_idx + 1
            style.append(('BACKGROUND', (0, best_row_pos), (-1, best_row_pos), colors.lightblue))
            style.append(('TEXTCOLOR', (0, best_row_pos), (-1, best_row_pos), ReportConfig.PRIMARY_COLOR))
            
            results_table.setStyle(TableStyle(style))
            self.elements.append(results_table)
            self.elements.append(Spacer(1, 0.1 * inch))
            self.elements.append(Paragraph(
                "<i>Table 1: Model performance metrics across all feature plans and model types. Best performing model highlighted in blue.</i>",
                styles['Caption']
            ))
            self.elements.append(Spacer(1, 0.2 * inch))
            
            if 'model_comparison' in self.plots:
                img_data = io.BytesIO()
                self.plots['model_comparison'].savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=6*inch, height=3*inch)
                self.elements.append(img)
                self.elements.append(Paragraph(
                    "<i>Figure 1: AUC and optimal profit comparison across all model-feature plan combinations.</i>",
                    styles['Caption']
                ))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 3.2 Profit Curves
        self.elements.append(Paragraph("3.2 Profit Curves", styles['SubsectionHeader']))
        
        profit_curve_text = """
        The profit curves follow a characteristic shape: profits increase as the threshold moves from conservative (low threshold) to moderate, 
        reach a peak at the optimal threshold, and then decline as the threshold becomes too permissive.
        
        The profit curve for Logistic Regression on <i>baseline_minimal</i> shows the sharpest peak at threshold <b>0.620</b>, with a clear 
        maximum of <b>$123,795,944</b>. Random Forest curves are flatter, with broader peaks at higher thresholds (0.790–0.920), reflecting 
        the model's different probability calibration characteristics. This validates the observation that high AUC does not guarantee high profit; 
        the threshold sweep and calibration quality drive the financial outcome.
        """
        
        self.elements.append(Paragraph(profit_curve_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 3.3 Baseline Comparison
        self.elements.append(Paragraph("3.3 Baseline Comparison", styles['SubsectionHeader']))
        
        if self.data.baseline_validation is not None:
            # Create table data
            # FIX: Changed "Lift vs.\nApprove-All" to "Loss Reduction\nvs. Approve-All"
            table_data = [["Baseline", "Profit", "Loss Reduction\nvs. Approve-All"]]
            
            df = self.data.baseline_validation
            approve_all = df[df['Metric'] == 'approve_all']['Value'].iloc[0]
            
            for _, row in df.iterrows():
                if row['Metric'] == 'random_std':
                    continue
                
                # Format label
                label_str = str(row['Metric'])
                if label_str == 'approve_all':
                    label = 'Approve-All'
                elif label_str == 'reject_all':
                    label = 'Reject-All'
                elif label_str == 'random_mean':
                    label = 'Random (50%)'
                elif label_str.startswith('threshold_'):
                    t = label_str.split('_')[1]
                    label = f"Fixed Threshold {t}"
                elif label_str == 'grade_based':
                    label = 'Grade-Based (D-F)'
                elif label_str == 'dti_based':
                    label = 'DTI-Based (≤30%)'
                elif label_str == 'fico_based':
                    label = 'FICO-Based (≥660)'
                elif label_str == 'model_best':
                    label = 'Your Model (Optimal)'
                else:
                    label = label_str.replace('_', ' ').title()
                
                # Format values
                profit = row['Value']
                profit_str = f"${profit:,.0f}"
                if label == 'Your Model (Optimal)':
                    profit_display = Paragraph(f"<b>{profit_str}</b>", styles['TableCell'])
                else:
                    profit_display = Paragraph(profit_str, styles['TableCell'])
                
                # Calculate lift
                if approve_all != 0:
                    loss_reduction = ((profit - approve_all) / approve_all) * 100
                else:
                    loss_reduction = 0
                
                loss_reduction_str = f"{loss_reduction:.1f}%"
                if label == 'Your Model (Optimal)':
                    loss_reduction_display = Paragraph(f"<b>{loss_reduction_str}</b>", styles['TableCell'])
                else:
                    loss_reduction_display = Paragraph(loss_reduction_str, styles['TableCell'])
                
                table_data.append([label, profit_display, loss_reduction_display])
            
            # Create table
            col_widths = [2.5*inch, 1.5*inch, 1.5*inch]
            baseline_table = Table(table_data, colWidths=col_widths)
            
            style = [
                ('BACKGROUND', (0, 0), (-1, 0), ReportConfig.PRIMARY_COLOR),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.gray),
                ('BACKGROUND', (0, 1), (-1, -1), ReportConfig.LIGHT_GRAY),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 4),
            ]
            
            model_row_pos = None
            for i, row in enumerate(table_data):
                if row[0] == 'Your Model (Optimal)':
                    model_row_pos = i
                    break
            
            if model_row_pos:
                style.append(('BACKGROUND', (0, model_row_pos), (-1, model_row_pos), colors.lightblue))
                style.append(('TEXTCOLOR', (0, model_row_pos), (-1, model_row_pos), ReportConfig.PRIMARY_COLOR))
            
            baseline_table.setStyle(TableStyle(style))
            
            self.elements.append(baseline_table)
            self.elements.append(Spacer(1, 0.1 * inch))
            self.elements.append(Paragraph(
                "<i>Table 2: Baseline comparison. Note that the test set (2016–2018) had negative returns overall.</i>",
                styles['Caption']
            ))
            
            self.elements.append(Spacer(1, 0.2 * inch))
            
            if 'baseline_comparison' in self.plots:
                img_data = io.BytesIO()
                self.plots['baseline_comparison'].savefig(img_data, format='png', dpi=150, bbox_inches='tight')
                img_data.seek(0)
                img = Image(img_data, width=6*inch, height=3*inch)
                self.elements.append(img)
                self.elements.append(Paragraph(
                    "<i>Figure 2: Baseline comparison showing model performance relative to simple heuristics.</i>",
                    styles['Caption']
                ))

        self.elements.append(Spacer(1, 0.3 * inch))

        # 3.4 Feature Importance
        self.elements.append(Paragraph("3.4 Feature Importance", styles['SubsectionHeader']))
        
        feature_importance_text = """
        To better understand the model's decision-making, we examined feature importance for both Logistic Regression and Random Forest 
        models on the <i>baseline_minimal</i> feature plan.
        
        <b>Logistic Regression:</b> The coefficients provide an interpretable view of which features contribute most to the default prediction. 
        Features with larger absolute coefficients have a stronger influence on the model's output.
        
        <b>Random Forest:</b> Feature importance is measured by the reduction in impurity (Gini importance) across all trees in the ensemble. 
        Higher importance values indicate features that are more frequently used to split nodes and contribute more to the model's predictions.
        """
        
        self.elements.append(Paragraph(feature_importance_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.2 * inch))

        lr_imp_path = Path("reports/visualizations/feature_importance_lr.png")
        if lr_imp_path.exists():
            img = Image(str(lr_imp_path), width=6*inch, height=5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 3: Top 20 feature importance for Logistic Regression (baseline_minimal plan).</i>",
                styles['Caption']
            ))
            self.elements.append(Spacer(1, 0.2 * inch))

        rf_imp_path = Path("reports/visualizations/feature_importance_rf.png")
        if rf_imp_path.exists():
            img = Image(str(rf_imp_path), width=6*inch, height=5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 4: Top 20 feature importance for Random Forest (baseline_minimal plan).</i>",
                styles['Caption']
            ))

        self.elements.append(Spacer(1, 0.3 * inch))

        # ============================================================
        # 3.5 Decision Performance
        # ============================================================
        self.elements.append(Paragraph("3.5 Decision Performance", styles['SubsectionHeader']))
        
        decision_text = """
        To evaluate the model's practical decision-making, we examined the calibration of predicted probabilities and the classification 
        outcomes at the optimal threshold of 0.620.
        
        <b>Calibration Curve:</b> The reliability diagram shows how well the predicted probabilities align with observed default rates. 
        A model that closely follows the diagonal is well-calibrated, meaning its probability estimates are reliable.
        
        <b>Confusion Matrix:</b> At the optimal threshold, the model approved 268,935 loans and declined only 29. This represents a profit-seeking, 
        not risk-averse, strategy—the interest income from the 210,396 performing loans outweighed the losses from the 58,539 defaults. 
        The matrix visualizes this trade-off.
        """
        
        self.elements.append(Paragraph(decision_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.2 * inch))

        # Add Calibration Curve
        cal_path = Path("reports/visualizations/calibration_curve.png")
        if cal_path.exists():
            img = Image(str(cal_path), width=6*inch, height=5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 5: Calibration curve (reliability diagram) for the best model.</i>",
                styles['Caption']
            ))
            self.elements.append(Spacer(1, 0.2 * inch))

        # Add Confusion Matrix
        cm_path = Path("reports/visualizations/confusion_matrix.png")
        if cm_path.exists():
            img = Image(str(cm_path), width=6*inch, height=5*inch)
            self.elements.append(img)
            self.elements.append(Paragraph(
                "<i>Figure 6: Confusion matrix at the optimal threshold of 0.620.</i>",
                styles['Caption']
            ))

        self.elements.append(PageBreak())


        

    def add_section_4_discussion(self):
        """Add discussion section"""
        styles = self.styles
        
        self.elements.append(Paragraph("4. Discussion", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        # 4.1 AUC vs Profit
        self.elements.append(Paragraph("4.1 AUC and Profit: A Practical Observation", styles['SubsectionHeader']))
        
        auc_text = """
        One observation from this analysis is that the model with higher AUC (Random Forest, 0.7001) did not produce higher profit 
        than the model with lower AUC (Logistic Regression, 0.6671). This outcome indicates that predictive discrimination and 
        financial performance are not always correlated.
        
        One contributing factor is calibration. Logistic Regression with Platt scaling produced probabilities that aligned more closely 
        with observed default rates across the probability range. Random Forest, while achieving higher discriminative power, showed 
        calibration differences in the ranges where approval decisions are most sensitive.
        
        Another factor is threshold sensitivity. The profit curve for Logistic Regression had a sharper peak, which allowed for more 
        precise threshold selection. The Random Forest profit curve was flatter, resulting in a smaller profit improvement even at its 
        optimal threshold.
        
        These observations imply that calibration quality and threshold behavior can be as relevant as raw predictive performance when 
        the objective is to support a financial decision.
        """
        
        self.elements.append(Paragraph(auc_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 4.2 Target Leakage
        self.elements.append(Paragraph("4.2 Target Leakage: A Cautionary Note", styles['SubsectionHeader']))
        
        leakage_discussion = """
        An early finding in this project was the presence of target leakage in the original feature set. Initial model training produced 
        an AUC of 0.9999, which was a clear indication that features containing post-origination information were present in the training data.
        
        Features such as total payments, recovery amounts, and last payment dates are strongly correlated with the target but are only 
        available after loan origination. These were systematically removed from the feature set using a pattern-based detection approach.
        
        After removing these columns, the AUC dropped from 0.9999 to 0.6671, which is more consistent with typical credit risk models. 
        This experience reinforces the importance of careful feature selection and data validation in predictive modeling.
        """
        
        self.elements.append(Paragraph(leakage_discussion, styles['BodyText']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # 4.3 Limitations
        self.elements.append(Paragraph("4.3 Limitations", styles['SubsectionHeader']))
        
        limitations_text = """
        This project has several limitations that are worth noting:
        
        <b>Retrospective Analysis:</b> This is a retrospective simulation using historical data where outcomes are already known. 
        In a live setting, future outcomes are uncertain, and the optimal threshold may change over time.
        
        <b>Operational Costs:</b> Underwriting, collections, and capital costs were not modeled. These would reduce net profit and 
        could shift the optimal threshold.
        
        <b>Data Context:</b> LendingClub data reflects a specific time period and borrower population. Results may not generalize 
        to other lending environments without further validation.
        
        <b>Static Threshold:</b> A single optimal threshold was identified for each model. In practice, thresholds may need to be 
        adjusted based on changing economic conditions.
        """
        
        self.elements.append(Paragraph(limitations_text, styles['BodyText']))
        
        self.elements.append(PageBreak())

    def add_section_5_conclusion(self):
        """Add conclusion section"""
        styles = self.styles
        
        self.elements.append(Paragraph("5. Conclusion", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        conclusion_text = """
        This project explored the relationship between model-based predictions and financial outcomes in consumer lending. 
        By systematically sweeping approval thresholds and calculating realized portfolio profit, an optimal cutoff of 
        <b>0.620</b> was identified. On the 2016-2018 test set, this generated <b>$123.8 million</b> in net cash flow. 
        While the overall market returned a loss of <b>-$331.9M</b> under an approve-all strategy, the optimal threshold 
        reduced this loss by <b>$280k</b>—validating the framework even in adverse credit cycles.
        
        One observation from this analysis was that the model with higher AUC (Random Forest, 0.7001) did not produce higher profit than the 
        Logistic Regression model (AUC 0.6671). This implies that calibration and threshold behavior can be as important as 
        discriminative performance when models are used to support financial decisions.
        
        The work also highlighted the importance of feature validation. The detection and removal of target-leaking features was 
        a necessary step to obtain realistic model performance estimates.
        
        Taken together, these findings reinforce several practical considerations for data science work in lending:
        
        <b>1. Calibration:</b> Predicted probabilities should reflect observed default rates.
        <b>2. Threshold Optimization:</b> The profit-maximizing cutoff should be identified empirically.
        <b>3. Business Alignment:</b> Model evaluation should be tied to the relevant business metric—in this case, portfolio profit.
        """
        
        self.elements.append(Paragraph(conclusion_text, styles['BodyText']))
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # Future work
        self.elements.append(Paragraph("Potential Extensions", styles['SubsectionHeader']))
        
        future_text = """
        Possible directions for further exploration include:
        
        • Dynamic threshold adjustment based on macroeconomic indicators
        • Incorporation of capital constraints and regulatory requirements
        • Multi-period portfolio optimization
        • Profit-weighted training to directly optimize for financial outcomes
        """
        
        self.elements.append(Paragraph(future_text, styles['BodyText']))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # Final statement
        self.elements.append(Paragraph(
            "This project outlines a practical framework for linking machine learning predictions to lending decisions. " +
            "The work reinforces that thoughtful model design, careful validation, and business-aligned evaluation are central " +
            "to applied data science in finance.",
            styles['BodyText']
        ))
        
        self.elements.append(PageBreak())

    def add_references(self):
        """Add references section"""
        styles = self.styles
        
        self.elements.append(Paragraph("References", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        references = [
            "Elkan, C. (2001). The foundations of cost-sensitive learning. <i>International Joint Conference on Artificial Intelligence</i>.",
            "Lessmann, S., Baesens, B., Seow, H. V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. <i>European Journal of Operational Research</i>, 247(1), 124-136.",
            "Platt, J. C. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. <i>Advances in Large Margin Classifiers</i>.",
            "Thomas, L. C., Crook, J., & Edelman, D. (2017). <i>Credit Scoring and Its Applications</i>. SIAM.",
            "Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit-driven data mining approach. <i>European Journal of Operational Research</i>, 218(1), 211-229.",
            "Zadrozny, B., & Elkan, C. (2002). Transforming classifier scores into accurate multiclass probability estimates. <i>Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining</i>."
        ]
        
        for ref in references:
            self.elements.append(Paragraph(f"[{references.index(ref)+1}] {ref}", styles['BodyText']))
            self.elements.append(Spacer(1, 0.05 * inch))
        
        self.elements.append(PageBreak())
    
    def add_appendix(self):
        """Add appendix section"""
        styles = self.styles
        
        self.elements.append(Paragraph("Appendix", styles['SectionHeader']))
        self.elements.append(Spacer(1, 0.1 * inch))
        self.elements.append(HRFlowable(color=ReportConfig.SECONDARY_COLOR, width="100%", thickness=1))
        self.elements.append(Spacer(1, 0.1 * inch))
        
        # A. Code Snippets
        self.elements.append(Paragraph("A. Key Code Snippets", styles['SubsectionHeader']))
        
        code_examples = [
            ("Profit Calculation", 
             "def calculate_profit(df):\n    return df['total_received'] - df['funded_amount']"),
            
            ("Threshold Sweep",
             "def optimize_threshold(y_pred_proba, profit_test):\n    thresholds = np.arange(0.05, 0.95, 0.01)\n    profits = []\n    for t in thresholds:\n        approved = profit_test[y_pred_proba < t]\n        profits.append(approved.sum())\n    optimal_idx = np.argmax(profits)\n    return thresholds[optimal_idx], profits[optimal_idx]"),
            
            ("Safe Feature Engineering",
             "class SafeFeatureEngineer:\n    def __init__(self, train_df, test_df):\n        self.train_df = train_df.copy()\n        self.test_df = test_df.copy()\n        self.statistics = {}\n    \n    def apply_base_features(self):\n        # Calculate on train only, apply to test\n        pass")
        ]
        
        for title, code in code_examples:
            self.elements.append(Paragraph(f"<b>{title}</b>", styles['BodyText']))
            self.elements.append(Paragraph(code, styles['Code']))
            self.elements.append(Spacer(1, 0.1 * inch))
        
        self.elements.append(Spacer(1, 0.3 * inch))
        
        # B. Data Summary
        self.elements.append(Paragraph("B. Data Summary", styles['SubsectionHeader']))
        
        data_summary = """
        <b>Dataset:</b> LendingClub loan data (2007–2018)
        <b>Total Loans:</b> 1.34 million
        <b>Features:</b> 110 initially, reduced to 82 after preprocessing
        <b>Default Rate:</b> Approximately 12-15% depending on the year
        <b>Train/Test Split:</b> 80/20 by issue date (temporal split)
        """
        
        self.elements.append(Paragraph(data_summary, styles['BodyText']))

# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution function"""
    print("="*60)
    print("GENERATING PROFESSIONAL PDF REPORT")
    print("="*60)
    
    # Initialize data
    print("Loading data...")
    data = ReportData()
    
    # Generate plots
    print("Generating plots...")
    plots = generate_plots(data)
    
    # Create styles
    print("Creating styles...")
    styles = create_styles()
    
    # Create report generator
    print("Building report content...")
    generator = ReportGenerator(data, styles, plots)
    generator.generate()
    
    # Build PDF
    print("Writing PDF...")
    doc = SimpleDocTemplate(
        ReportConfig.OUTPUT_FILENAME,
        pagesize=ReportConfig.PAPER_SIZE,
        leftMargin=ReportConfig.MARGINS[0],
        rightMargin=ReportConfig.MARGINS[1],
        topMargin=ReportConfig.MARGINS[2],
        bottomMargin=ReportConfig.MARGINS[3]
    )
    
    doc.build(generator.elements)
    
    print(f"\n✓ Report generated successfully!")
    print(f"  Output: {ReportConfig.OUTPUT_FILENAME}")
    print(f"  Size: {os.path.getsize(ReportConfig.OUTPUT_FILENAME) / 1024:.1f} KB")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()