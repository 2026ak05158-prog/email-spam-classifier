import os
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
    Image as RLImage
)

def create_assignment_pdf(
    output_filename="ML_Assignment_2_Submission_2026AK05158.pdf",
    github_url="https://github.com/lokeshpanda/email-spam-classifier-bits",
    streamlit_url="https://email-spam-classifier-bits.streamlit.app",
    screenshot_path=None
):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Simple, academic typography (no extra decorative colors or border headers)
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=0, # Left-aligned clean
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.black,
        spaceAfter=4
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.black,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'Heading1_Clean',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.black,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Clean',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Clean',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.black,
        spaceAfter=6
    )

    body_bold = ParagraphStyle(
        'Body_Bold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )

    link_style = ParagraphStyle(
        'Link_Clean',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.navy
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.black
    )

    story = []

    # Title block
    story.append(Paragraph("Machine Learning Assignment - 2", title_style))
    story.append(Paragraph("Work Integrated Learning Programmes Division | BITS Pilani", subtitle_style))
    story.append(Paragraph("M.Tech (AIML) &bull; Course: Machine Learning (S1-26_AIMLZG565)", subtitle_style))
    story.append(Paragraph("<b>Student ID:</b> 2026AK05158 &nbsp;&nbsp;|&nbsp;&nbsp; <b>Student Name:</b> Lokesh Panda", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceBefore=2, spaceAfter=12))

    # Section 1: Mandatory Submission Links
    story.append(Paragraph("1. Mandatory Submission Links", h1_style))
    
    links_data = [
        [
            Paragraph("<b>1. GitHub Repository Link:</b>", body_style),
            Paragraph(f'<a href="{github_url}"><font color="#0000EE"><u>{github_url}</u></font></a><br/><font size="8" color="#333333">Contains source code (app.py, train.py), requirements.txt, test_data.csv, README.md, and model/ artifacts.</font>', link_style)
        ],
        [
            Paragraph("<b>2. Live Streamlit App Link:</b>", body_style),
            Paragraph(f'<a href="{streamlit_url}"><font color="#0000EE"><u>{streamlit_url}</u></font></a><br/><font size="8" color="#333333">Deployed on Streamlit Community Cloud (interactive evaluation on hold-out test data).</font>', link_style)
        ]
    ]

    t_links = Table(links_data, colWidths=[150, 354])
    t_links.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_links)
    story.append(Spacer(1, 10))

    # Section 2: Screenshot of Assignment Execution on BITS Virtual Lab
    story.append(Paragraph("2. Screenshot of Assignment Execution on BITS Virtual Lab", h1_style))
    story.append(Paragraph("As specified in the assignment guidelines, proof of execution on BITS Virtual Lab is provided below:", body_style))

    if screenshot_path and os.path.exists(screenshot_path):
        try:
            img = RLImage(screenshot_path, width=480, height=260)
            story.append(Spacer(1, 4))
            story.append(img)
            story.append(Spacer(1, 4))
            story.append(Paragraph("<i>Figure 1: Screenshot verifying execution of assignment environment on BITS Virtual Lab.</i>", subtitle_style))
        except Exception as e:
            story.append(Paragraph(f"Error loading screenshot: {e}", body_style))
    else:
        # Placeholder frame for screenshot
        placeholder_data = [[
            Paragraph(
                "<br/><br/><b>[ BITS Virtual Lab Execution Screenshot ]</b><br/><br/>"
                "<i>Replace this box with your actual BITS Virtual Lab desktop / terminal screenshot.<br/>"
                "Proof of execution on BITS Virtual Lab as mandated by Step 2 / Step 3 guidelines (1 mark).</i><br/><br/>",
                ParagraphStyle('Placeholder', parent=body_style, alignment=1)
            )
        ]]
        t_placeholder = Table(placeholder_data, colWidths=[504], rowHeights=[140])
        t_placeholder.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 1, colors.gray),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke)
        ]))
        story.append(t_placeholder)

    story.append(Spacer(1, 14))

    # Section 3: GitHub README Content
    story.append(Paragraph("3. GitHub README Content", h1_style))
    story.append(Paragraph("The complete documentation and experimental findings from the project repository README.md are reproduced below as required by Step 5.", body_style))

    # a. Problem Statement
    story.append(Paragraph("a. Problem Statement", h2_style))
    p_statement = (
        "Unsolicited electronic mail (spam) presents substantial security vulnerabilities (such as credential harvesting, "
        "ransomware transmission, and social engineering) alongside operational overhead for mail delivery networks. "
        "The objective of this assignment is to develop, evaluate, and deploy an end-to-end machine learning classification system "
        "capable of categorizing incoming messages into legitimate correspondence (ham, 0) and unsolicited spam (1). "
        "The evaluation balances overall discriminatory power (AUC, MCC) with operational precision, ensuring that legitimate emails "
        "are not erroneously discarded (False Positives) while maintaining robust spam filtration (Recall)."
    )
    story.append(Paragraph(p_statement, body_style))

    # b. Dataset Description
    story.append(Paragraph("b. Dataset Description", h2_style))
    p_data = (
        "<b>Dataset:</b> UCI Spambase Dataset (Hewlett-Packard Laboratories benchmark, available via UCI Repository & OpenML).<br/>"
        "&bull; <b>Instance Size:</b> 4,601 instances (2,788 legitimate messages [60.6%] and 1,813 spam messages [39.4%]). Exceeds the minimum requirement of 500 instances.<br/>"
        "&bull; <b>Feature Count:</b> 57 continuous numeric predictor attributes (exceeds the minimum requirement of 12 features):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;1. <i>Word Frequency Attributes (48 features):</i> Percentage of words in an email matching specific words (e.g., 'make', 'address', 'free', 'money', 'credit', 'hp', 'george').<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;2. <i>Character Frequency Attributes (6 features):</i> Percentage of characters matching specific punctuation symbols (';', '(', '[', '!', '$', '#').<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;3. <i>Capital Run-Length Attributes (3 features):</i> Measures average, maximum, and total lengths of consecutive capital letter sequences.<br/>"
        "&bull; <b>Target Variable:</b> 'class' (0 = legitimate / ham, 1 = spam).<br/>"
        "&bull; <b>Data Partitioning:</b> Stratified 80/20 train/test split (3,680 training records, 921 test records saved to <code>test_data.csv</code>)."
    )
    story.append(Paragraph(p_data, body_style))

    # c. GitHub Repository Link
    story.append(Paragraph("c. GitHub Repository Link", h2_style))
    p_repo = f"Repository URL: <a href='{github_url}'><font color='#0000EE'><u>{github_url}</u></font></a>"
    story.append(Paragraph(p_repo, body_style))

    # d. Models Used & Comparison Table
    story.append(Paragraph("d. Models Used & Evaluation Metrics", h2_style))
    story.append(Paragraph("All five mandated classification models were trained on the identical training partition and evaluated on the hold-out test partition (<code>test_data.csv</code>, N = 921). The calculated metrics are detailed in the comparison table below:", body_style))

    # Comparison Table
    table_raw = [
        [
            Paragraph("<b>ML Model Name</b>", table_header),
            Paragraph("<b>Accuracy</b>", table_header),
            Paragraph("<b>AUC</b>", table_header),
            Paragraph("<b>Precision</b>", table_header),
            Paragraph("<b>Recall</b>", table_header),
            Paragraph("<b>F1</b>", table_header),
            Paragraph("<b>MCC</b>", table_header),
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_bold),
            Paragraph("0.9294", table_cell),
            Paragraph("0.9702", table_cell),
            Paragraph("0.9209", table_cell),
            Paragraph("0.8981", table_cell),
            Paragraph("0.9093", table_cell),
            Paragraph("0.8518", table_cell),
        ],
        [
            Paragraph("<b>Decision Tree</b>", table_cell_bold),
            Paragraph("0.9034", table_cell),
            Paragraph("0.9266", table_cell),
            Paragraph("0.9308", table_cell),
            Paragraph("0.8154", table_cell),
            Paragraph("0.8693", table_cell),
            Paragraph("0.7976", table_cell),
        ],
        [
            Paragraph("<b>KNN</b>", table_cell_bold),
            Paragraph("0.9077", table_cell),
            Paragraph("0.9506", table_cell),
            Paragraph("0.8861", table_cell),
            Paragraph("0.8788", table_cell),
            Paragraph("0.8824", table_cell),
            Paragraph("0.8065", table_cell),
        ],
        [
            Paragraph("<b>Naive Bayes</b>", table_cell_bold),
            Paragraph("0.8339", table_cell),
            Paragraph("0.9449", table_cell),
            Paragraph("0.7178", table_cell),
            Paragraph("0.9532", table_cell),
            Paragraph("0.8189", table_cell),
            Paragraph("0.6941", table_cell),
        ],
        [
            Paragraph("<b>Support Vector Machine</b>", table_cell_bold),
            Paragraph("0.9273", table_cell),
            Paragraph("0.9667", table_cell),
            Paragraph("0.9277", table_cell),
            Paragraph("0.8843", table_cell),
            Paragraph("0.9055", table_cell),
            Paragraph("0.8471", table_cell),
        ],
    ]

    t_metrics = Table(table_raw, colWidths=[130, 62, 62, 62, 62, 62, 62])
    t_metrics.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_metrics)
    story.append(Spacer(1, 10))

    # Observations Table
    story.append(Paragraph("e. Observations on Model Performance", h2_style))
    story.append(Paragraph("Observations regarding the empirical performance of each model on the Spambase dataset are summarized in the table below:", body_style))

    obs_table_data = [
        [
            Paragraph("<b>ML Model Name</b>", table_header),
            Paragraph("<b>Observation about model performance</b>", table_header)
        ],
        [
            Paragraph("<b>Logistic Regression</b>", table_cell_bold),
            Paragraph("Delivered the highest Accuracy (0.9294), highest AUC (0.9702), highest F1-Score (0.9093), and highest MCC (0.8518). Because the Spambase dataset has 57 additive numeric predictors with distinct log-odds directional weights (positive for currency tokens, negative for institutional contact words), standardizing the inputs and fitting a linear decision boundary with L2 regularization generalized exceptionally well without overfitting.", table_cell)
        ],
        [
            Paragraph("<b>Decision Tree</b>", table_cell_bold),
            Paragraph("Achieved the highest Precision (0.9308) among all models, misclassifying only 22 legitimate emails as spam. By regularizing the tree to max_depth=6, it identified clean orthogonal splitting thresholds on high-leverage indicators ('char_freq_dollar' and 'capital_run_length_longest'). However, Recall was lower (0.8154, 67 missed spam messages) because axis-parallel boundaries fail to capture subtle multi-feature correlations.", table_cell)
        ],
        [
            Paragraph("<b>KNN</b>", table_cell_bold),
            Paragraph("Exhibited solid and balanced performance (Accuracy 0.9077, AUC 0.9506, Precision 0.8861, Recall 0.8788, MCC 0.8065). With standardized Euclidean distances, query instances mapped accurately into homogeneous neighborhood clusters. The primary trade-off is inference latency and memory storage during deployment compared to parametric models.", table_cell)
        ],
        [
            Paragraph("<b>Naive Bayes</b>", table_cell_bold),
            Paragraph("Produced the highest Recall of all models (0.9532), capturing 346 of 363 spam messages (only 17 false negatives). However, it suffered the lowest Precision (0.7178) and lowest Accuracy (0.8339), generating 136 false alarms. Its conditional independence assumption overstates likelihoods when correlated spam terms co-occur in marketing emails.", table_cell)
        ],
        [
            Paragraph("<b>Support Vector Machine</b>", table_cell_bold),
            Paragraph("Delivered the second-best overall performance (Accuracy 0.9273, AUC 0.9667, Precision 0.9277, Recall 0.8843, F1 0.9055, MCC 0.8471). The non-linear RBF kernel mapped the 57 continuous dimensions into a high-dimensional feature space, establishing a smooth, robust maximum-margin boundary that handles noisy inputs effectively.", table_cell)
        ],
        [
            Paragraph("<b>Overall, Winner for your dataset?</b>", table_cell_bold),
            Paragraph("<b>Logistic Regression is the overall winner for this dataset.</b> It ranks first across Accuracy (92.94%), ROC-AUC (0.9702), F1-Score (0.9093), and Matthews Correlation Coefficient (0.8518), while restricting false positives to just 28 out of 558 legitimate emails. Coupled with near-instantaneous inference speed and minimal memory footprint (1.3 KB serialized model size), it is the most robust and practical model for production deployment.", table_cell)
        ]
    ]

    t_obs = Table(obs_table_data, colWidths=[130, 374])
    t_obs.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_obs)

    doc.build(story)
    print(f"PDF generated successfully: {output_filename}")

if __name__ == '__main__':
    create_assignment_pdf()
