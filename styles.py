"""
UI styling and HTML template generation for SI Data Generator.

This module provides CSS styles and HTML template functions for consistent
UI rendering throughout the application, including step progress indicators,
cards, and information boxes.
"""

import streamlit as st
from typing import List, Dict


def get_main_css() -> str:
    """
    Return main application CSS styles.
    
    Provides comprehensive styling for the Snowflake Intelligence Data
    Generator application including responsive design, animations, and
    brand-consistent colors.
    
    Returns:
        CSS style string to be injected via st.markdown
    """
    return """
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
        overflow-x: hidden;
    }
    
    /* Step progress indicator */
    .step-progress {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 24px;
        background: linear-gradient(135deg, #056fb7 0%, #29B5E8 100%);
        color: white;
        border-radius: 12px;
        margin-bottom: 32px;
        gap: 0;
    }
    
    .step-item {
        display: flex;
        flex-direction: column;
        align-items: center;
        min-width: 120px;
    }
    
    .step-number {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(255,255,255,0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 8px;
        color: rgba(255,255,255,0.6);
        transition: all 0.3s ease;
    }
    
    .step-number.active {
        background: white;
        color: #056fb7;
        width: 50px;
        height: 50px;
        font-size: 1.5rem;
        box-shadow: 0 0 20px rgba(255,255,255,0.8), 
                    0 0 40px rgba(255,255,255,0.4);
        transform: scale(1.1);
    }
    
    .step-number.completed {
        background: rgba(14,165,233,0.8);
        color: white;
    }
    
    .step-label {
        font-size: 0.9rem;
        text-align: center;
        color: rgba(255,255,255,0.7);
        transition: all 0.3s ease;
    }
    
    .step-item:has(.step-number.active) .step-label {
        font-weight: bold;
        font-size: 1rem;
        color: white;
    }
    
    .step-item:has(.step-number.completed) .step-label {
        color: white;
    }
    
    .step-connector {
        width: 100px;
        height: 2px;
        background: rgba(255,255,255,0.3);
        margin: 0 8px 28px 8px;
        flex-shrink: 0;
    }
    
    /* Card styling */
    .demo-card {
        border: 2px solid #d1d5db;
        border-radius: 12px;
        padding: 24px;
        background: white;
        transition: all 0.3s ease;
        min-height: 480px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        display: flex;
        flex-direction: column;
    }
    
    .demo-card:hover {
        border-color: #29B5E8;
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.3);
        transform: translateY(-2px);
    }
    
    .demo-card-selected {
        border: 3px solid #29B5E8;
        background: #EBF8FF;
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.3);
    }
    
    /* Progress indicators */
    .progress-item {
        padding: 16px;
        margin: 8px 0;
        border-left: 4px solid #29B5E8;
        background: #f8f9fa;
        border-radius: 4px;
        transition: all 0.3s ease;
    }
    
    .progress-item.success {
        border-left-color: #0EA5E9;
        background: #d4edda;
    }
    
    .progress-item.error {
        border-left-color: #DC3545;
        background: #f8d7da;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #056fb7 0%, #29B5E8 100%);
        color: white;
        padding: 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #056fb7 0%, #29B5E8 100%);
        color: white;
        padding: 24px;
        border-radius: 10px;
        margin: 16px 0;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .step-progress {
            flex-direction: column;
        }
        .step-connector {
            display: none;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 8px 32px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* Page header styling */
    .page-header {
        text-align: center;
        padding: 0rem 0;
    }
    
    .page-title {
        color: #29B5E8;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .page-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Selection box styling */
    .selection-box {
        background: #EBF8FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #29B5E8;
    }
    
    /* About page styling */
    .about-hero {
        background: linear-gradient(135deg, #056fb7 0%, #29B5E8 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .about-hero h3 {
        color: white;
        margin-top: 0;
    }
    
    .about-hero p {
        color: white;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    /* Footer styling */
    .page-footer {
        text-align: center;
        color: #666;
        padding: 2rem 0;
        border-top: 1px solid #e0e0e0;
    }
    
    /* Loading spinner */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,0.3);
        border-top: 3px solid white;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    /* Results table styling */
    .results-table-item {
        font-size: 0.875rem;
        color: #6b7280;
    }
    
    .results-list {
        margin: 0;
        padding-left: 1.5rem;
        line-height: 1.8;
    }
    
    /* Query card sections */
    .query-section {
        margin-bottom: 2rem;
    }
    
    .query-section-header {
        margin-bottom: 0.5rem;
        font-size: 1.25rem;
    }
    
    .query-section-desc {
        font-size: 0.875rem;
        color: #6b7280;
        font-style: italic;
        margin-bottom: 1rem;
    }
    
    .query-section.analyst .query-section-header {
        color: #0ea5e9;
    }
    
    .query-section.search .query-section-header {
        color: #8b5cf6;
    }
    
    .query-section.intelligence .query-section-header {
        color: #06b6d4;
    }
    
    /* Difficulty badges */
    .difficulty-badge {
        font-weight: bold;
    }
    
    .difficulty-badge.basic {
        color: #10b981;
    }
    
    .difficulty-badge.intermediate {
        color: #f59e0b;
    }
    
    .difficulty-badge.advanced {
        color: #ef4444;
    }
    
    /* Value card with custom height */
    .value-card-full {
        max-height: none !important;
    }
    
    .value-card-content-full {
        max-height: none !important;
        overflow-y: visible !important;
    }
    
    /* Override default value-card max-height for query results */
    .value-card.query-results-card {
        max-height: none !important;
    }
    
    .value-card.query-results-card .value-card-content {
        max-height: none !important;
        overflow-y: visible !important;
    }
    
    /* Coverage section styling */
    .coverage-intro {
        margin-bottom: 1rem;
        color: #6b7280;
        font-size: 0.95rem;
    }
    
    .coverage-question-box {
        margin: 1.5rem 0;
        padding: 1rem;
        background: #f9fafb;
        border-left: 4px solid #29B5E8;
        border-radius: 4px;
    }
    
    .coverage-question-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    .coverage-feedback {
        margin-top: 0.75rem;
        font-size: 0.9rem;
    }
    
    .coverage-feedback-table {
        color: #056fb7;
        font-weight: bold;
    }
    
    .coverage-feedback-item {
        margin-left: 1rem;
        color: #374151;
    }
    
    .coverage-note {
        margin-top: 1rem;
        padding: 1rem;
        background: #ecfdf5;
        border-radius: 8px;
        border: 1px solid #a7f3d0;
    }
    
    .coverage-note-title {
        color: #065f46;
        font-weight: 500;
    }
    
    .coverage-note-text {
        color: #047857;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    
    /* Infrastructure cards */
    .infra-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        min-height: 350px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
    }
    
    .infra-card:hover {
        border-color: #29B5E8;
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.2);
        transform: translateY(-2px);
    }
    
    .infra-card h3 {
        color: #056fb7;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .infra-stat {
        display: flex;
        align-items: center;
        padding: 0.75rem 0;
        border-bottom: 1px solid #f3f4f6;
    }
    
    .infra-stat:last-child {
        border-bottom: none;
    }
    
    .infra-stat-icon {
        font-size: 1.5rem;
        margin-right: 1rem;
        min-width: 40px;
        text-align: center;
    }
    
    .infra-stat-content {
        flex: 1;
    }
    
    .infra-stat-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    
    .infra-stat-desc {
        font-size: 0.875rem;
        color: #6b7280;
    }
    
    /* Value cards */
    .value-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        height: 100%;
        min-height: 350px;
        max-height: 500px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .value-card:hover {
        border-color: #29B5E8;
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.2);
    }
    
    .value-card h2 {
        color: #056fb7;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #29B5E8;
        padding-bottom: 0.5rem;
        flex-shrink: 0;
    }
    
    .value-card-content {
        flex: 1;
        overflow-y: auto;
        padding-right: 0.5rem;
    }
    
    .value-card-content::-webkit-scrollbar {
        width: 6px;
    }
    
    .value-card-content::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 3px;
    }
    
    .value-card-content::-webkit-scrollbar-thumb {
        background: #29B5E8;
        border-radius: 3px;
    }
    
    .value-card-content::-webkit-scrollbar-thumb:hover {
        background: #056fb7;
    }
    
    /* Demo flow cards */
    .demo-flow-card {
        background: linear-gradient(135deg, #056fb7 0%, #29B5E8 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        display: flex;
        flex-direction: column;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        height: 100%;
        min-height: 100%;
    }
    
    .demo-flow-card h3 {
        color: white;
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .demo-flow-card p {
        color: white;
        font-size: 0.95rem;
        line-height: 1.6;
        margin-bottom: 0;
    }
    
    /* Equal height card containers */
    [data-testid="stHorizontalBlock"] {
        display: flex !important;
        align-items: stretch !important;
        gap: 1rem;
    }
    
    [data-testid="stHorizontalBlock"] > div {
        display: flex !important;
        flex: 1 !important;
    }
    
    [data-testid="stHorizontalBlock"] > div > div {
        width: 100% !important;
    }
    
    /* Demo flow card additional styles */
    .demo-flow-card .ask-text {
        color: #ffd700;
        font-weight: bold;
        margin-bottom: 1rem;
        font-style: italic;
        flex-shrink: 0;
        word-wrap: break-word;
    }
    
    .demo-flow-card .check-item {
        color: white;
        margin-bottom: 0.5rem;
        padding-left: 0;
        word-wrap: break-word;
    }
    
    .demo-flow-card .check-item::before {
        content: '✅ ';
    }
    
    /* Ensure equal height columns */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    
    [data-testid="column"] > div {
        height: 100%;
    }
    
    .value-card-content p,
    .value-card-content ol,
    .value-card-content ul {
        margin-bottom: 0.5rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .infra-card, .value-card {
            margin-bottom: 1rem;
        }
        .demo-flow-card {
            min-height: auto;
            margin-bottom: 1rem;
        }
        [data-testid="stHorizontalBlock"] {
            flex-direction: column;
        }
    }
    
    /* History page styles */
    .history-container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1rem;
    }
    
    .history-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .history-card:hover {
        border-color: #29B5E8;
        box-shadow: 0 4px 12px rgba(41, 181, 232, 0.2);
        transform: translateY(-2px);
    }
    
    .history-header {
        display: flex;
        justify-content: space-between;
        align-items: start;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f3f4f6;
    }
    
    .history-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #056fb7;
        margin-bottom: 0.5rem;
    }
    
    .history-timestamp {
        font-size: 0.875rem;
        color: #6b7280;
        font-style: italic;
    }
    
    .history-company {
        font-size: 1rem;
        color: #1f2937;
        font-weight: 500;
        margin-bottom: 0.25rem;
    }
    
    .history-details {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .history-detail-item {
        display: flex;
        flex-direction: column;
    }
    
    .history-detail-label {
        font-size: 0.75rem;
        color: #6b7280;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    
    .history-detail-value {
        font-size: 0.95rem;
        color: #1f2937;
        font-weight: 500;
    }
    
    .history-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin: 1rem 0;
    }
    
    .history-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
    }
    
    .history-badge.enabled {
        background: #d1fae5;
        color: #065f46;
    }
    
    .history-badge.disabled {
        background: #f3f4f6;
        color: #6b7280;
    }
    
    .history-badge.advanced {
        background: #fef3c7;
        color: #92400e;
    }
    
    .history-actions {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
    }
    
    .history-section {
        margin: 1rem 0;
        padding: 1rem;
        background: #f9fafb;
        border-radius: 8px;
    }
    
    .history-section-title {
        font-size: 0.875rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    
    .history-list {
        list-style: none;
        padding-left: 0;
        margin: 0;
    }
    
    .history-list li {
        padding: 0.25rem 0;
        color: #4b5563;
        font-size: 0.9rem;
    }
    
    .history-list li::before {
        content: '▸ ';
        color: #29B5E8;
        font-weight: bold;
    }
    
    .history-empty {
        text-align: center;
        padding: 3rem 1rem;
        color: #6b7280;
    }
    
    .history-empty-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
        opacity: 0.5;
    }
    
    .history-pagination {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        margin: 2rem 0;
    }
    
    .export-button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .export-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    
    .reuse-button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        border: none;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .reuse-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* History page - darker text for disabled inputs */
    input:disabled, textarea:disabled {
        color: #1f2937 !important;
        opacity: 1 !important;
        -webkit-text-fill-color: #1f2937 !important;
    }
    
    /* Fix card overflow issues */
    .infra-card {
        overflow: auto;
        word-wrap: break-word;
    }
    
    .infra-stat-title, .infra-stat-desc {
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
    }
    
    .results-list {
        padding-left: 1.5rem;
        margin: 0;
    }
    
    .results-list li {
        word-wrap: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
        line-height: 1.8;
    }
    
    .results-list li strong {
        display: inline-block;
        max-width: 100%;
        word-break: break-all;
    }
</style>
"""


def show_step_progress(current_step: int):
    """
    Display step progress indicator with visual feedback.
    
    Shows a multi-step progress indicator with completed, active, and
    pending states. Steps are connected with visual connectors.
    
    Args:
        current_step: Current step number (1-4)
    """
    steps = [
        {"num": 1, "label": "Customer Info"},
        {"num": 2, "label": "Select Demo"},
        {"num": 3, "label": "Configure"},
        {"num": 4, "label": "Generate"}
    ]
    
    html = '<div class="step-progress">'
    for i, step in enumerate(steps):
        status_class = ""
        if step["num"] < current_step:
            status_class = "completed"
        elif step["num"] == current_step:
            status_class = "active"
        
        # Add step item
        html += (
            f'<div class="step-item">'
            f'<div class="step-number {status_class}">{step["num"]}</div>'
            f'<div class="step-label">{step["label"]}</div>'
            f'</div>'
        )
        
        # Add connector AFTER step item (not inside it)
        if i < len(steps) - 1:
            html += '<div class="step-connector"></div>'
    
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_info_box(content: str):
    """
    Render an informational box with Snowflake styling.
    
    Args:
        content: HTML or text content to display in the box
    """
    html = f'<div class="info-box">{content}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_success_box(content: str):
    """
    Render a success/completion box with Snowflake styling.
    
    Args:
        content: HTML or text content to display in the box
    """
    html = f'<div class="success-box">{content}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_header(title: str, subtitle: str = ""):
    """
    Render application header with title and optional subtitle.
    
    Args:
        title: Main title text
        subtitle: Optional subtitle text
    """
    html = f"""
    <div class='page-header'>
        <h1 class='page-title'>
            ❄️ {title}
        </h1>
    """
    
    if subtitle:
        html += f"""
        <p class='page-subtitle'>
            {subtitle}
        </p>
        """
    
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_step_container(step_text: str):
    """
    Render a step container header.
    
    Args:
        step_text: Text to display in the step container
    """
    html = f"<div class='step-container'>{step_text}</div>"
    st.markdown(html, unsafe_allow_html=True)


def apply_main_styles():
    """
    Apply main application CSS styles.
    
    Should be called once at the beginning of the application to inject
    all necessary styles into the Streamlit app.
    """
    st.markdown(get_main_css(), unsafe_allow_html=True)


def render_selection_box(content: str):
    """
    Render a selection/info box with blue styling.
    
    Args:
        content: HTML or text content to display in the box
    """
    html = f'<div class="selection-box">{content}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_about_hero(title: str, content: str):
    """
    Render the about page hero section.
    
    Args:
        title: Hero section title
        content: Hero section content text
    """
    html = f"""
    <div class="about-hero">
        <h3>{title}</h3>
        <p>{content}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_page_footer(text: str):
    """
    Render page footer.
    
    Args:
        text: Footer text content
    """
    html = f'<div class="page-footer"><p>{text}</p></div>'
    st.markdown(html, unsafe_allow_html=True)


def render_loading_info(text: str):
    """
    Render info box with loading spinner.
    
    Args:
        text: Text to display next to spinner
    """
    html = f"""
    <div class='info-box' style='display: flex; align-items: center; gap: 12px;'>
        <div class='loading-spinner'></div>
        <span>{text}</span>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_demo_header(company_name: str, demo_title: str):
    """
    Render demo generation header.
    
    Args:
        company_name: Name of the company
        demo_title: Title of the demo
    """
    html = f"<h2 style='margin: 0; color: white;'>🎯 {company_name} Demo: {demo_title}</h2>"
    st.markdown(html, unsafe_allow_html=True)


def render_results_table_list(tables: List[Dict]) -> str:
    """
    Render list of created tables with descriptions.
    
    Args:
        tables: List of table dictionaries with 'table', 'type', and 'description' keys
        
    Returns:
        HTML string for the table list
    """
    items = []
    for table in tables:
        if table.get('type') == 'structured':
            desc = table.get('description', '')
            items.append(f"<li><strong>{table['table']}</strong><br/><span class='results-table-item'>{desc}</span></li>")
        elif table.get('type') == 'unstructured':
            items.append(f"<li><strong>{table['table']}</strong><br/><span class='results-table-item'>Searchable text content for knowledge retrieval</span></li>")
    
    html = f"<ul class='results-list'>{''.join(items)}</ul>"
    return html


def render_query_results(analytics_questions: List[str], search_questions: List[str], intelligence_questions: Dict[str, List[str]]):
    """
    Render the unified query results card with all three sections.
    
    Args:
        analytics_questions: List of Cortex Analyst questions
        search_questions: List of Cortex Search questions
        intelligence_questions: Dict with 'basic', 'intermediate', 'advanced' question lists
    """
    html = "<div class='value-card query-results-card'><div class='value-card-content'>"
    
    # Cortex Analyst Section
    html += "<div class='query-section analyst'>"
    html += "<h3 class='query-section-header'>🔍 Cortex Analyst</h3>"
    html += "<div class='query-section-desc'>Natural language to SQL queries</div>"
    if analytics_questions:
        html += "<ol class='results-list'>"
        for q_text in analytics_questions[:3]:
            html += f'<li>"{q_text}"</li>'
        html += "</ol>"
    html += "</div>"
    
    # Cortex Search Section
    html += "<div class='query-section search'>"
    html += "<h3 class='query-section-header'>🔎 Cortex Search</h3>"
    html += "<div class='query-section-desc'>Semantic search for unstructured data</div>"
    if search_questions:
        html += "<ol class='results-list'>"
        for q_text in search_questions[:3]:
            html += f'<li>"{q_text}"</li>'
        html += "</ol>"
    html += "</div>"
    
    # Snowflake Intelligence Section
    html += "<div class='query-section intelligence'>"
    html += "<h3 class='query-section-header'>🤖 Snowflake Intelligence</h3>"
    html += "<div class='query-section-desc'>Multi-tool AI agent orchestration</div>"
    
    for difficulty, label, emoji in [('basic', 'Basic', '🟢'), ('intermediate', 'Intermediate', '🟡'), ('advanced', 'Advanced', '🔴')]:
        questions = intelligence_questions.get(difficulty, [])
        if questions:
            html += f"<div style='margin-bottom: 1rem;'><strong class='difficulty-badge {difficulty}'>{emoji} {label}:</strong>"
            html += "<ol style='margin: 0.5rem 0 0 1.5rem; padding-left: 0; line-height: 1.8;'>"
            for q_text in questions[:2]:
                html += f'<li>"{q_text}"</li>'
            html += "</ol></div>"
    
    html += "</div></div></div>"
    st.markdown(html, unsafe_allow_html=True)

