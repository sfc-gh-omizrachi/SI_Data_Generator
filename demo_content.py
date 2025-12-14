"""
Demo content generation for SI Data Generator.

This module consolidates all functionality related to generating demo content:
- Demo templates (fallback scenarios)
- Question generation (natural language questions for demos)
- Data generation (structured and unstructured data)
- Data validation (ensuring data can answer target questions)

This consolidates what was previously spread across demo_templates.py and
multiple sections of SI_Generator.py.
"""

import streamlit as st
import pandas as pd
import json
import re
import random
import time
import traceback
from metrics import timeit
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

from utils import (
    LLM_MODEL,
    call_cortex_with_retry,
    MAX_FOLLOW_UP_QUESTIONS,
    DEFAULT_QUESTION_COUNT,
    MAX_RETRY_ATTEMPTS,
    DEFAULT_NUMERIC_RANGE_MAX,
    DATE_LOOKBACK_DAYS,
    SAMPLE_DATA_LIMIT
)
from errors import (
    ErrorCode,
    ErrorSeverity,
    safe_json_parse
)
from prompts import (
    get_company_analysis_prompt,
    get_question_generation_prompt,
    get_follow_up_questions_prompt,
    get_target_question_analysis_prompt,
    get_schema_generation_prompt,
    get_collective_validation_prompt,
    get_single_table_validation_prompt,
    get_unstructured_data_generation_prompt
)
from utils import (
    validate_language_content,
    enhance_prompt_with_language,
    add_language_metadata_to_chunks,
    get_language_display_name,
    safe_parallel_execute
)


# ============================================================================
# DEMO TEMPLATES (from demo_templates.py)
# ============================================================================

def get_fallback_demo_ideas(
    company_name: str,
    team_members: str,
    use_cases: str,
    target_questions: Optional[List[str]] = None,
    advanced_mode: bool = False
) -> List[Dict]:
    """
    Return fallback demo ideas when Cortex LLM is unavailable.
    
    Provides three high-quality demo templates covering E-commerce,
    Financial Services, and Healthcare industries. These templates
    are designed to showcase Cortex Analyst and Cortex Search capabilities.
    
    Note: Fallback templates are in standard mode (3 structured + 1 unstructured
    table). For advanced mode with 3-5 tables, Cortex LLM should be used.
    
    Args:
        company_name: Company name to personalize templates
        team_members: Target audience description
        use_cases: Specific use cases to address
        target_questions: Optional list of questions (not used in templates)
        advanced_mode: Whether advanced mode was requested
        
    Returns:
        List of demo configuration dictionaries
    """
    # Note about mode limitations
    if advanced_mode:
        st.info(
            "ℹ️ Advanced mode (3-5 tables) not available in fallback. "
            "Using standard mode templates (3 tables). For advanced mode, "
            "ensure Cortex LLM is available."
        )
    
    # Note: Fallback templates are standard mode (3 tables: 1 fact + 2 dimensions + 1 unstructured)
    # Advanced mode (3-5 tables) requires Cortex LLM
    
    demos = [
        {
            "title": "E-commerce Analytics & Customer Intelligence",
            "description": (
                f"Comprehensive e-commerce analytics solution for "
                f"{company_name} combining transactional sales data with "
                f"customer intelligence to drive revenue growth and improve "
                f"customer lifetime value"
            ),
            "industry_focus": "E-commerce/Retail",
            "business_value": (
                "Optimize sales performance, predict customer churn, identify "
                "upsell opportunities, and personalize customer experiences "
                "through unified analytics across transactions, customer "
                "profiles, and product reviews. Enable data-driven decision "
                "making for marketing, product, and sales teams."
            ),
            "tables": {
                "structured_1": {
                    "name": "SALES_TRANSACTIONS",
                    "description": (
                        "Transaction-level sales data including order IDs, "
                        "customer IDs, product SKUs, purchase amounts, "
                        "discounts applied, timestamps, payment methods, "
                        "shipping details, geographic location, order status, "
                        "and revenue metrics across all sales channels"
                    ),
                    "purpose": (
                        "Enable Cortex Analyst to answer questions about "
                        "sales trends, revenue patterns, product performance, "
                        "regional analysis, discount effectiveness, and "
                        "customer purchasing behavior. Supports time-series "
                        "analysis, cohort analysis, and predictive forecasting."
                    ),
                    "table_type": "fact"
                },
                "structured_2": {
                    "name": "CUSTOMER_PROFILES",
                    "description": (
                        "Customer demographic and behavioral data including "
                        "customer IDs, acquisition dates, lifetime value, "
                        "purchase frequency, average order value, preferred "
                        "product categories, engagement scores, churn risk "
                        "indicators, subscription status, and customer segment "
                        "classifications"
                    ),
                    "purpose": (
                        "Support customer analytics including segmentation, "
                        "churn prediction, lifetime value analysis, and "
                        "personalization strategies. Enables Cortex Analyst "
                        "to join with transaction data for comprehensive "
                        "customer journey analysis and targeted marketing "
                        "insights."
                    ),
                    "table_type": "dimension"
                },
                "structured_3": {
                    "name": "PRODUCT_CATALOG",
                    "description": (
                        "Product dimension containing product IDs, SKUs, "
                        "product names, categories, subcategories, brands, "
                        "pricing tiers, inventory status, supplier information, "
                        "product attributes, and merchandising classifications"
                    ),
                    "purpose": (
                        "Provide product context and hierarchy for sales "
                        "analysis. Enables product performance tracking, "
                        "category analytics, and merchandising insights when "
                        "joined with transaction data."
                    ),
                    "table_type": "dimension"
                },
                "unstructured": {
                    "name": "PRODUCT_REVIEWS_CHUNKS",
                    "description": (
                        "Chunked customer product reviews and feedback "
                        "including review text, star ratings, product mentions, "
                        "sentiment indicators, helpfulness votes, and "
                        "user-generated content from website, mobile app, and "
                        "third-party platforms"
                    ),
                    "purpose": (
                        "Enable Cortex Search for semantic search across "
                        "customer feedback to identify product issues, feature "
                        "requests, competitive comparisons, and sentiment "
                        "trends. Supports voice-of-customer analysis and "
                        "product improvement prioritization."
                    )
                }
            }
        },
        {
            "title": "Financial Services Risk & Compliance",
            "description": (
                f"Risk management and compliance monitoring system for "
                f"{company_name} that combines real-time transaction "
                f"monitoring with regulatory compliance tracking and policy "
                f"documentation for comprehensive risk intelligence"
            ),
            "industry_focus": "Financial Services",
            "business_value": (
                "Enhance fraud detection, reduce regulatory risk, automate "
                "compliance reporting, and accelerate investigation workflows "
                "through AI-powered analytics and intelligent document search. "
                "Reduce false positives, improve audit preparedness, and "
                "ensure regulatory adherence across all business units."
            ),
            "tables": {
                "structured_1": {
                    "name": "TRANSACTION_MONITORING",
                    "description": (
                        "Financial transaction records including transaction "
                        "IDs, account numbers, amounts, currencies, "
                        "transaction types (wire, ACH, card), counterparty "
                        "information, geographic locations, timestamps, risk "
                        "scores, anomaly flags, AML alerts, pattern indicators, "
                        "and investigator notes"
                    ),
                    "purpose": (
                        "Enable Cortex Analyst to perform transaction pattern "
                        "analysis, identify suspicious activity, track risk "
                        "score trends, analyze false positive rates, and "
                        "measure investigation efficiency. Supports predictive "
                        "risk modeling and anomaly detection queries."
                    ),
                    "table_type": "fact"
                },
                "structured_2": {
                    "name": "COMPLIANCE_EVENTS",
                    "description": (
                        "Regulatory compliance events including event IDs, "
                        "event types (KYC, AML, sanctions screening), account "
                        "information, violation indicators, severity levels, "
                        "investigation status, remediation actions, regulatory "
                        "deadlines, audit findings, and resolution timestamps"
                    ),
                    "purpose": (
                        "Support compliance reporting, trend analysis of "
                        "regulatory events, remediation tracking, and audit "
                        "trail generation through Cortex Analyst. Enables "
                        "cross-referencing with transaction data for "
                        "comprehensive risk assessment and regulatory reporting."
                    ),
                    "table_type": "dimension"
                },
                "structured_3": {
                    "name": "ACCOUNT_PROFILES",
                    "description": (
                        "Account dimension containing account IDs, account types, "
                        "customer demographics, risk classifications, account "
                        "opening dates, KYC status, sanctions screening results, "
                        "geographic regions, relationship managers, and account "
                        "status indicators"
                    ),
                    "purpose": (
                        "Provide account context for transaction and compliance "
                        "analysis. Enables account segmentation, risk profiling, "
                        "and demographic analysis when joined with transaction "
                        "monitoring data."
                    ),
                    "table_type": "dimension"
                },
                "unstructured": {
                    "name": "REGULATORY_DOCS_CHUNKS",
                    "description": (
                        "Chunked regulatory documents, compliance policies, "
                        "procedure manuals, regulatory guidance, industry "
                        "standards, internal control documentation, audit "
                        "reports, and regulatory correspondence for "
                        "comprehensive policy knowledge base"
                    ),
                    "purpose": (
                        "Enable Cortex Search for rapid policy lookup, "
                        "regulatory guidance retrieval, compliance procedure "
                        "verification, and audit preparation. Supports "
                        "investigators and compliance officers in finding "
                        "relevant regulations and internal controls quickly "
                        "during case review."
                    )
                }
            }
        },
        {
            "title": "Healthcare Patient Analytics & Research",
            "description": (
                f"Patient outcomes and research data platform for "
                f"{company_name} integrating clinical outcomes with treatment "
                f"protocols and medical research for evidence-based care "
                f"delivery and continuous quality improvement"
            ),
            "industry_focus": "Healthcare",
            "business_value": (
                "Improve patient outcomes, reduce readmission rates, optimize "
                "treatment protocols, accelerate clinical research, and "
                "support evidence-based medicine through comprehensive "
                "analytics and intelligent research discovery. Enable "
                "clinicians to make data-informed decisions backed by "
                "real-world evidence and published research."
            ),
            "tables": {
                "structured_1": {
                    "name": "PATIENT_OUTCOMES",
                    "description": (
                        "Patient treatment outcomes including patient IDs "
                        "(anonymized), admission dates, discharge dates, "
                        "diagnoses (ICD-10 codes), procedures (CPT codes), "
                        "outcome measures, quality indicators, readmission "
                        "flags, length of stay, complication rates, patient "
                        "satisfaction scores, and recovery metrics"
                    ),
                    "purpose": (
                        "Enable Cortex Analyst to perform clinical performance "
                        "analysis, identify outcome patterns, benchmark against "
                        "quality metrics, predict readmission risk, and analyze "
                        "treatment effectiveness. Supports population health "
                        "analysis and quality improvement initiatives."
                    ),
                    "table_type": "fact"
                },
                "structured_2": {
                    "name": "TREATMENT_PROTOCOLS",
                    "description": (
                        "Treatment protocols and care pathways including "
                        "protocol IDs, condition types, medication regimens, "
                        "dosage information, treatment sequences, clinical "
                        "guidelines, contraindications, success rates, cost "
                        "data, and evidence levels for each treatment approach"
                    ),
                    "purpose": (
                        "Support treatment effectiveness comparison, protocol "
                        "optimization, cost-benefit analysis, and clinical "
                        "decision support through Cortex Analyst. Enables "
                        "joining with outcomes data to measure real-world "
                        "protocol effectiveness and identify best practices."
                    ),
                    "table_type": "dimension"
                },
                "structured_3": {
                    "name": "PROVIDER_DIRECTORY",
                    "description": (
                        "Healthcare provider dimension containing provider IDs, "
                        "provider types (physician, nurse, specialist), "
                        "specialties, certifications, facility affiliations, "
                        "patient ratings, case volumes, years of experience, "
                        "and performance metrics"
                    ),
                    "purpose": (
                        "Provide provider context for outcomes analysis. Enables "
                        "provider performance tracking, specialty comparisons, "
                        "and resource allocation when joined with patient "
                        "outcomes data."
                    ),
                    "table_type": "dimension"
                },
                "unstructured": {
                    "name": "RESEARCH_PAPERS_CHUNKS",
                    "description": (
                        "Chunked medical research papers, clinical studies, "
                        "meta-analyses, systematic reviews, case reports, "
                        "clinical trial results, and evidence summaries from "
                        "peer-reviewed journals covering relevant medical "
                        "specialties and treatment modalities"
                    ),
                    "purpose": (
                        "Enable Cortex Search for evidence-based research "
                        "discovery, clinical guideline verification, treatment "
                        "option exploration, and staying current with medical "
                        "literature. Supports clinicians and researchers in "
                        "finding relevant studies and evidence during care "
                        "planning and protocol development."
                    )
                }
            }
        }
    ]
    
    # Add target audience and customization to each demo
    for demo in demos:
        demo['target_audience'] = (
            f"Designed for presentation to: {team_members}"
        )
        if use_cases:
            demo['customization'] = f"Tailored for: {use_cases}"
    
    return demos


def get_template_by_industry(industry: str) -> Optional[Dict]:
    """
    Get a specific demo template by industry.
    
    Args:
        industry: Industry name (case-insensitive)
        
    Returns:
        Demo template dictionary or None if not found
    """
    # Get all templates with placeholder values
    all_templates = get_fallback_demo_ideas(
        company_name="COMPANY",
        team_members="Team",
        use_cases="General Analytics"
    )
    
    # Match by industry
    industry_lower = industry.lower()
    for template in all_templates:
        if industry_lower in template['industry_focus'].lower():
            return template
    
    return None


def get_available_template_industries() -> List[str]:
    """
    Get list of industries with available templates.
    
    Returns:
        List of industry names
    """
    return [
        "E-commerce/Retail",
        "Financial Services",
        "Healthcare"
    ]


# ============================================================================
# QUESTION GENERATION
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_company_url(
    company_url: str,
    _session,
    _error_handler
) -> Dict[str, Any]:
    """
    Analyze company URL to extract industry and business context using LLM.
    
    CACHED FUNCTION: Results cached for 1 hour to avoid redundant LLM calls.
    
    Args:
        company_url: Company website URL
        _session: Snowflake session (prefixed with _ to skip hashing)
        _error_handler: ErrorHandler instance (prefixed with _ to skip hashing)
        
    Returns:
        Dictionary with industry, domain, context, and keywords
    """
    # Restore original parameter names for use in function body
    session = _session
    error_handler = _error_handler
    result = {
        'industry': 'Technology',  # Default fallback
        'domain': '',
        'context': '',
        'keywords': []
    }
    
    if not company_url or company_url == "":
        return result
    
    try:
        # Extract domain from URL
        domain_match = re.search(
            r'(?:https?://)?(?:www\.)?([^/]+)', company_url
        )
        if domain_match:
            result['domain'] = domain_match.group(1)
        
        # Use LLM to analyze the company URL and infer industry
        prompt = get_company_analysis_prompt(company_url)
        
        response_text = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?)",
            params=[LLM_MODEL, prompt]
        ).collect()[0][0]
        
        # Parse JSON response
        # Extract JSON from response
        json_match = re.search(
            r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL
        )
        if json_match:
            analysis = json.loads(json_match.group())
            result['industry'] = analysis.get('industry', result['industry'])
            result['keywords'] = analysis.get('keywords', [])
            result['context'] = analysis.get('context', '')
    
    except Exception:
        # Fallback: Use simple heuristics based on domain keywords
        domain_lower = result['domain'].lower() if result['domain'] else ''
        if any(word in domain_lower for word in [
            'bank', 'finance', 'capital', 'invest', 'credit'
        ]):
            result['industry'] = 'Finance'
            result['keywords'] = [
                'financial', 'risk', 'portfolio', 'transactions'
            ]
            result['context'] = (
                'Financial services with focus on risk management and '
                'portfolio analysis'
            )
        elif any(word in domain_lower for word in [
            'health', 'medical', 'pharma', 'clinic', 'hospital'
        ]):
            result['industry'] = 'Healthcare'
            result['keywords'] = ['patient', 'clinical', 'medical', 'treatment']
            result['context'] = (
                'Healthcare provider with patient care and clinical data '
                'analysis needs'
            )
        elif any(word in domain_lower for word in [
            'retail', 'shop', 'store', 'commerce', 'buy'
        ]):
            result['industry'] = 'Retail'
            result['keywords'] = ['sales', 'customer', 'inventory', 'orders']
            result['context'] = (
                'Retail business with customer analytics and inventory '
                'management focus'
            )
        elif any(word in domain_lower for word in [
            'tech', 'software', 'data', 'cloud', 'saas'
        ]):
            result['industry'] = 'Technology'
            result['keywords'] = ['user', 'product', 'platform', 'engagement']
            result['context'] = (
                'Technology company with product analytics and user '
                'engagement focus'
            )
        elif any(word in domain_lower for word in [
            'manu', 'factory', 'industrial', 'supply'
        ]):
            result['industry'] = 'Manufacturing'
            result['keywords'] = [
                'production', 'quality', 'supply chain', 'operations'
            ]
            result['context'] = (
                'Manufacturing with supply chain and operational efficiency '
                'needs'
            )
    
    return result


def generate_contextual_questions(
    demo_data: Dict,
    semantic_model_info: Optional[Dict],
    company_name: str,
    session,
    error_handler,
    num_questions: int = 12,
    company_url: Optional[str] = None,
    table_schemas: Optional[Dict] = None,
    rich_table_contexts: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Generate contextual questions using AI with optional URL analysis and
    table schemas.
    
    Args:
        demo_data: Demo configuration dictionary
        semantic_model_info: Optional semantic model information
        company_name: Company name
        session: Snowflake session
        error_handler: ErrorHandler instance
        num_questions: Number of questions to generate
        company_url: Optional company URL for context
        table_schemas: Optional table schemas for better questions (deprecated, use rich_table_contexts)
        rich_table_contexts: Optional rich context with full schema, data analysis, and cardinality
        
    Returns:
        List of question dictionaries
    """
    industry = demo_data.get(
        'industry_focus',
        demo_data.get('industry', 'Business Intelligence')
    )
    
    # Analyze company URL if provided to enhance context
    url_context = None
    if company_url:
        try:
            url_context = analyze_company_url(
                company_url, session, error_handler
            )
            # Override industry if URL analysis provides better context
            if url_context.get('industry'):
                industry = url_context['industry']
        except Exception:
            pass  # Continue with default industry if URL analysis fails
    
    questions = generate_questions_with_llm(
        demo_data,
        semantic_model_info,
        company_name,
        industry,
        session,
        error_handler,
        num_questions,
        url_context,
        table_schemas,
        rich_table_contexts
    )
    
    # Validate questions against schema if rich contexts available
    if rich_table_contexts and questions:
        validated_questions = validate_questions_against_schema(
            questions,
            rich_table_contexts
        )
        
        # If we lost more than half the questions, regenerate with stricter guidance
        if len(validated_questions) < len(questions) / 2:
            import streamlit as st
            st.warning(
                f"⚠️ Question quality low ({len(validated_questions)}/{len(questions)} passed validation). "
                "Regenerating with stricter rules..."
            )
            # Retry with same parameters - the enhanced prompt should do better
            questions = generate_questions_with_llm(
                demo_data,
                semantic_model_info,
                company_name,
                industry,
                session,
                error_handler,
                num_questions * 2,  # Ask for more since we'll filter
                url_context,
                table_schemas,
                rich_table_contexts
            )
            validated_questions = validate_questions_against_schema(
                questions,
                rich_table_contexts
            )
        
        return validated_questions[:num_questions]
    
    return questions[:num_questions]


def generate_questions_with_llm(
    demo_data: Dict,
    semantic_model_info: Optional[Dict],
    company_name: str,
    industry: str,
    session,
    error_handler,
    num_questions: int,
    url_context: Optional[Dict] = None,
    table_schemas: Optional[Dict] = None,
    rich_table_contexts: Optional[List[Dict]] = None
) -> List[Dict[str, Any]]:
    """
    Generate questions using Cortex LLM with optional URL context and actual
    table schemas.
    
    Args:
        demo_data: Demo configuration dictionary
        semantic_model_info: Optional semantic model information
        company_name: Company name
        industry: Industry name
        session: Snowflake session
        error_handler: ErrorHandler instance
        num_questions: Number of questions to generate
        url_context: Optional URL analysis context
        table_schemas: Optional actual table schemas (deprecated, use rich_table_contexts)
        rich_table_contexts: Optional rich context with full schema, data analysis, and cardinality
        
    Returns:
        List of question dictionaries
    """
    table_info = ""
    if 'structured_table_1' in demo_data:
        table_info += (
            f"- {demo_data['structured_table_1']['name']}: "
            f"{demo_data['structured_table_1']['description']}\n"
        )
    if 'structured_table_2' in demo_data:
        table_info += (
            f"- {demo_data['structured_table_2']['name']}: "
            f"{demo_data['structured_table_2']['description']}\n"
        )
    
    # Add actual column information if available
    columns_info = ""
    if table_schemas:
        columns_info = "\n\nActual Columns Available:\n"
        if 'table1' in table_schemas:
            columns_info += (
                f"- {table_schemas['table1']['name']}: "
                f"{', '.join(table_schemas['table1']['columns'])}\n"
            )
        if 'table2' in table_schemas:
            columns_info += (
                f"- {table_schemas['table2']['name']}: "
                f"{', '.join(table_schemas['table2']['columns'])}\n"
            )
    
    # Use prompt from prompts module
    prompt = get_question_generation_prompt(
        num_questions=num_questions,
        industry=industry,
        company_name=company_name,
        demo_data=demo_data,
        url_context=url_context,
        table_info=table_info,
        columns_info=columns_info,
        rich_table_contexts=rich_table_contexts
    )
    
    try:
        result = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[LLM_MODEL, prompt]
        ).collect()
        
        response = result[0]['RESPONSE']
        
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            questions_data = json.loads(json_match.group(0))
            for q in questions_data:
                q['source'] = 'ai'
            return questions_data
        else:
            return []
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def validate_questions_against_schema(
    questions: List[Dict],
    rich_table_contexts: List[Dict]
) -> List[Dict]:
    """
    Validate that questions are answerable with actual schema.
    
    CACHED FUNCTION: Results cached to avoid redundant validation for same schema.
    
    Filters out questions that:
    - Reference columns that don't exist
    - Ask for "top N" where N exceeds unique value count
    - Make assumptions about data not present
    
    Args:
        questions: List of question dictionaries
        rich_table_contexts: Rich context with schema and cardinality info
    
    Returns:
        List of validated question dictionaries
    """
    import re
    
    valid_questions = []
    all_columns = {}  # {col_name_lower: {unique_count, type}}
    
    # Collect all column info from contexts
    for ctx in rich_table_contexts:
        for col in ctx['columns']:
            col_lower = col['name'].lower()
            all_columns[col_lower] = {
                'unique_count': col.get('unique_count'),
                'type': col.get('type'),
                'name': col['name']
            }
    
    for q in questions:
        question_text = q['text'].lower()
        is_valid = True
        
        # Check 1: "top N" validation
        top_n_pattern = r'top\s+(\d+)'
        top_n_matches = re.findall(top_n_pattern, question_text)
        
        if top_n_matches:
            for n_str in top_n_matches:
                n = int(n_str)
                # Check if any column has unique_count and if so, ensure N is safe
                has_cardinality_check = False
                for col_info in all_columns.values():
                    if col_info['unique_count'] and col_info['unique_count'] < n:
                        is_valid = False
                        has_cardinality_check = True
                        break
                
                # If we found a cardinality issue, skip this question
                if not is_valid:
                    break
        
        # Check 2: Common field names that might not exist
        # Look for common keywords that might not be in actual columns
        suspicious_keywords = [
            'promotional', 'promotion', 'campaign',
            'weather', 'event', 'holiday',
            'waste', 'defect', 'quality_issue',
            'restaurant', 'store', 'location'
        ]
        
        for keyword in suspicious_keywords:
            if keyword in question_text:
                # Check if there's actually a column with this keyword
                has_matching_column = False
                for col_name in all_columns.keys():
                    if keyword in col_name:
                        has_matching_column = True
                        break
                
                # If keyword is mentioned but no matching column, it's suspicious
                # However, don't mark as invalid yet - just note it
                # (The LLM should have followed the rules, so if it used the keyword,
                #  it should have seen it in the data)
        
        # Check 3: Ensure question doesn't ask for things beyond 100 records
        # (e.g., "year-over-year" when we only have 100 records)
        if 'year-over-year' in question_text or 'yoy' in question_text:
            # This is risky with only 100 records - skip it
            is_valid = False
        
        if is_valid:
            valid_questions.append(q)
    
    return valid_questions


def create_question_chains(
    questions: List[Dict[str, Any]],
    session,
    error_handler,
    max_chains: int = 3
) -> Dict[str, List[str]]:
    """
    Create question chains with follow-ups IN PARALLEL.
    
    This function parallelizes the generation of follow-up questions to significantly
    reduce total execution time (from ~24s sequential to ~6s parallel for 12 questions).
    
    Args:
        questions: List of question dictionaries
        session: Snowflake session
        error_handler: ErrorHandler instance
        max_chains: Maximum number of chains to create
        
    Returns:
        Dictionary mapping primary questions to follow-up lists
    """
    chains = {}
    
    primary_questions = [
        q for q in questions
        if q.get('difficulty') in ['basic', 'intermediate']
    ][:max_chains]
    
    # Build argument list for parallel execution
    args_list = [
        [primary_q['text'], session, error_handler]
        for primary_q in primary_questions
    ]
    
    # Execute all follow-up generation calls in parallel (max 4 workers)
    follow_ups_list = safe_parallel_execute(
        generate_follow_up_questions,
        args_list,
        max_workers=min(4, len(primary_questions)),
        fallback_value=[]
    )
    
    # Build chains dictionary
    for q_dict, follow_ups in zip(primary_questions, follow_ups_list):
        if follow_ups:
            chains[q_dict['text']] = follow_ups
    
    return chains


def generate_follow_up_questions(
    primary_question: str,
    session,
    error_handler
) -> List[str]:
    """
    Generate follow-up questions for a primary question.
    
    Args:
        primary_question: The primary question text
        session: Snowflake session
        error_handler: ErrorHandler instance
        
    Returns:
        List of follow-up question strings
    """
    prompt = get_follow_up_questions_prompt(primary_question)
    
    try:
        result = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[LLM_MODEL, prompt]
        ).collect()
        
        response = result[0]['RESPONSE']
        
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            follow_ups = json.loads(json_match.group(0))
            return follow_ups[:MAX_FOLLOW_UP_QUESTIONS]
        else:
            return []
    except Exception:
        return []


def analyze_target_questions(
    questions: List[str],
    session,
    error_handler
) -> Dict[str, Any]:
    """
    Analyze target questions to extract required data dimensions, metrics,
    and characteristics.
    
    Args:
        questions: List of target questions that the demo should answer
        session: Snowflake session
        error_handler: ErrorHandler instance
    
    Returns:
        dict: Analysis results with required_dimensions, metrics_needed,
              and data_characteristics
    """
    if not questions or len(questions) == 0:
        return {
            'required_dimensions': [],
            'metrics_needed': [],
            'data_characteristics': {},
            'has_target_questions': False
        }
    
    prompt = get_target_question_analysis_prompt(questions)
    
    try:
        response = call_cortex_with_retry(prompt, session, error_handler)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed:
                # Add metadata
                parsed['has_target_questions'] = True
                parsed['original_questions'] = questions
                return parsed
    except Exception as e:
        st.warning(f"⚠️ Could not analyze target questions: {str(e)}")
    
    # Fallback: return basic structure
    return {
        'required_dimensions': [],
        'metrics_needed': [],
        'data_characteristics': {},
        'has_target_questions': True,
        'original_questions': questions
    }


def format_questions_for_display(
    questions: List[Dict[str, Any]]
) -> str:
    """
    Format questions for display in UI.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Formatted markdown string
    """
    output = "## Generated Questions\n\n"
    
    basic_questions = [
        q for q in questions if q.get('difficulty') == 'basic'
    ]
    intermediate_questions = [
        q for q in questions if q.get('difficulty') == 'intermediate'
    ]
    advanced_questions = [
        q for q in questions if q.get('difficulty') == 'advanced'
    ]
    
    if basic_questions:
        output += "### Basic Questions\n\n"
        for idx, q in enumerate(basic_questions, 1):
            output += f"{idx}. {q['text']}\n"
        output += "\n"
    
    if intermediate_questions:
        output += "### Intermediate Questions\n\n"
        for idx, q in enumerate(intermediate_questions, 1):
            output += f"{idx}. {q['text']}\n"
        output += "\n"
    
    if advanced_questions:
        output += "### Advanced Questions\n\n"
        for idx, q in enumerate(advanced_questions, 1):
            output += f"{idx}. {q['text']}\n"
        output += "\n"
    
    return output


# ============================================================================
# DATA GENERATION
# ============================================================================

def normalize_field_name_to_sql(field_name: str) -> str:
    """
    Normalize natural language field names to SQL conventions.
    
    Converts "store ids" -> "STORE_ID"
    Converts "employee counts" -> "EMPLOYEE_COUNT"
    Converts "geographic coordinates" -> "GEOGRAPHIC_COORDINATE"
    
    Args:
        field_name: Natural language field name (may have spaces, plural)
        
    Returns:
        SQL-compliant field name (underscores, singular, uppercase)
    """
    import re
    
    # Convert to lowercase for processing
    normalized = field_name.lower().strip()
    
    # Replace spaces with underscores
    normalized = re.sub(r'\s+', '_', normalized)
    
    # Remove common plurals (be careful to preserve words that end in 's' naturally)
    # Handle common plural patterns
    plural_patterns = [
        (r'ies$', 'y'),      # quantities -> quantity, categories -> category
        (r'ses$', 's'),      # addresses -> address, classes -> class  
        (r'ches$', 'ch'),    # batches -> batch
        (r'xes$', 'x'),      # indexes -> index
        (r'([^s])s$', r'\1') # ids -> id, counts -> count (but not 'class' -> 'clas')
    ]
    
    for pattern, replacement in plural_patterns:
        if re.search(pattern, normalized):
            normalized = re.sub(pattern, replacement, normalized)
            break  # Only apply one pattern
    
    # Convert to uppercase
    normalized = normalized.upper()
    
    return normalized


def extract_required_fields_from_description(
    table_description: str
) -> List[Dict[str, Any]]:
    """
    Parse table description to extract explicitly mentioned field names.
    
    Uses regex patterns to identify:
    - Fields mentioned with parentheses: "movement_type (receipt, usage, waste)"
    - Fields in "including" lists: "including restaurant_id, supplier_id"
    - Common patterns: "with delivery_date", "containing quality_score"
    
    Args:
        table_description: The table description text to parse
        
    Returns:
        List of dicts with extracted field information:
        [
            {
                'field_name': 'MOVEMENT_TYPE',
                'suggested_type': 'STRING',
                'sample_values': ['receipt', 'usage', 'waste', 'transfer'],
                'description': 'Type of inventory movement'
            },
            ...
        ]
    """
    import re
    
    required_fields = []
    seen_fields = set()
    
    # Pattern 1: field_name (value1, value2, value3)
    # Matches: "movement_type (receipt, usage, waste, transfer)"
    pattern1 = r'\b([a-z_][a-z0-9_]*)\s*\(([^)]+)\)'
    matches1 = re.findall(pattern1, table_description.lower())
    
    for field_name, values_str in matches1:
        field_normalized = normalize_field_name_to_sql(field_name)
        if field_normalized not in seen_fields:
            # Extract values from parentheses
            values = [v.strip() for v in values_str.split(',')]
            
            # Infer type based on values
            suggested_type = 'STRING'
            if all(v.replace('.', '').replace('-', '').isdigit() for v in values if v):
                suggested_type = 'NUMBER'
            
            required_fields.append({
                'field_name': field_normalized,
                'suggested_type': suggested_type,
                'sample_values': values[:10],  # Limit to first 10
                'description': f'{field_name.replace("_", " ").title()}'
            })
            seen_fields.add(field_normalized)
    
    # Pattern 2: "including field1, field2, field3"
    pattern2 = r'including\s+([a-z0-9_,\s]+?)(?:\s+with|\s+and\s+|[,.]|\s*$)'
    matches2 = re.findall(pattern2, table_description.lower())
    
    for match in matches2:
        # Split by commas and "and"
        fields = re.split(r',|\s+and\s+', match)
        for field_name in fields:
            field_name = field_name.strip()
            if field_name and len(field_name) > 2:
                field_normalized = normalize_field_name_to_sql(field_name)
                if field_normalized not in seen_fields:
                    # Infer type from field name (use normalized version for checking)
                    suggested_type = 'STRING'
                    field_lower = field_normalized.lower()
                    if any(keyword in field_lower for keyword in ['_id', '_key', '_number', '_count', '_quantity']):
                        suggested_type = 'NUMBER'
                    elif any(keyword in field_lower for keyword in ['_date', '_time', '_timestamp']):
                        suggested_type = 'TIMESTAMP' if 'time' in field_lower or 'timestamp' in field_lower else 'DATE'
                    elif any(keyword in field_lower for keyword in ['_amount', '_cost', '_price', '_value', '_rate', '_score']):
                        suggested_type = 'FLOAT'
                    
                    required_fields.append({
                        'field_name': field_normalized,
                        'suggested_type': suggested_type,
                        'sample_values': [],
                        'description': f'{field_name.replace("_", " ").title()}'
                    })
                    seen_fields.add(field_normalized)
    
    # Pattern 3: "with field_name" or "containing field_name"
    pattern3 = r'(?:with|containing)\s+([a-z_][a-z0-9_\s]*)'
    matches3 = re.findall(pattern3, table_description.lower())
    
    for field_name in matches3:
        field_name = field_name.strip()
        if len(field_name) > 3:
            field_normalized = normalize_field_name_to_sql(field_name)
            if field_normalized not in seen_fields:
                # Infer type from field name (use normalized version)
                suggested_type = 'STRING'
                field_lower = field_normalized.lower()
                if any(keyword in field_lower for keyword in ['_id', '_key', '_number', '_count', '_quantity']):
                    suggested_type = 'NUMBER'
                elif any(keyword in field_lower for keyword in ['_date', '_time', '_timestamp']):
                    suggested_type = 'TIMESTAMP' if 'time' in field_lower or 'timestamp' in field_lower else 'DATE'
                elif any(keyword in field_lower for keyword in ['_amount', '_cost', '_price', '_value', '_rate', '_score']):
                    suggested_type = 'FLOAT'
                
                required_fields.append({
                    'field_name': field_normalized,
                    'suggested_type': suggested_type,
                    'sample_values': [],
                    'description': f'{field_name.replace("_", " ").title()}'
                })
                seen_fields.add(field_normalized)
    
    # Pattern 4: Common data field keywords
    keywords_to_extract = [
        'promotional_period', 'promotional_periods', 'promotion_type',
        'weather_condition', 'weather_conditions', 'weather_pattern',
        'local_event', 'local_events', 'event_type',
        'holiday', 'holidays', 'holiday_type',
        'season', 'seasons', 'seasonal',
        'waste_amount', 'waste_quantity', 'waste_type',
        'quality_score', 'quality_rating', 'quality_metric',
        'performance_metric', 'performance_score', 'performance_rating'
    ]
    
    for keyword in keywords_to_extract:
        if keyword in table_description.lower():
            field_normalized = normalize_field_name_to_sql(keyword)
            if field_normalized not in seen_fields:
                # Infer type
                suggested_type = 'STRING'
                if 'amount' in keyword or 'quantity' in keyword or 'score' in keyword or 'rating' in keyword or 'metric' in keyword:
                    suggested_type = 'FLOAT'
                
                required_fields.append({
                    'field_name': field_normalized,
                    'suggested_type': suggested_type,
                    'sample_values': [],
                    'description': f'{keyword.replace("_", " ").title()}'
                })
                seen_fields.add(field_normalized)
    
    return required_fields


@timeit
def generate_schema_for_table(
    table_name: str,
    table_description: str,
    company_name: str,
    session,
    error_handler,
    max_attempts: int = 3,
    target_questions: Optional[List[str]] = None,
    question_analysis: Optional[Dict] = None,
    required_fields: Optional[List[Dict]] = None
) -> Optional[List[Dict]]:
    """
    Use Cortex to generate a realistic schema for a table with validation
    and retry logic.
    
    Args:
        table_name: Name of the table
        table_description: Description of the table's purpose
        company_name: Company context for realistic data
        session: Snowflake session
        error_handler: ErrorHandler instance
        max_attempts: Maximum number of generation attempts if validation fails
        target_questions: Optional list of target questions
        question_analysis: Optional analysis dict from analyze_target_questions()
        required_fields: Optional list of mandatory fields extracted from description
    
    Returns:
        list: List of column definitions or None if all attempts fail
    """
    # Use prompt from prompts module with required fields enforcement
    prompt = get_schema_generation_prompt(
        table_name=table_name,
        table_description=table_description,
        company_name=company_name,
        target_questions=target_questions,
        question_analysis=question_analysis,
        required_fields=required_fields
    )

    for attempt in range(max_attempts):
        try:
            response = call_cortex_with_retry(prompt, session, error_handler)
            
            if response:
                parsed = safe_json_parse(response)
                if parsed and "columns" in parsed:
                    columns = parsed.get("columns", [])
                    # Validate that we got at least ENTITY_ID and one other column
                    if len(columns) >= 2 and any(
                        col.get('name') == 'ENTITY_ID' for col in columns
                    ):
                        # Validate that all required fields are present
                        # Since we've normalized field names to SQL conventions, we can do simpler matching
                        if required_fields:
                            generated_field_names = {col['name'].upper() for col in columns}
                            missing_fields = []
                            
                            for req_field in required_fields:
                                required_name = req_field['field_name'].upper()
                                found_match = False
                                
                                # Check for exact match
                                if required_name in generated_field_names:
                                    found_match = True
                                else:
                                    # Try with an 'S' appended or removed (final safety net)
                                    if required_name.endswith('S'):
                                        singular = required_name[:-1]
                                        if singular in generated_field_names:
                                            found_match = True
                                    else:
                                        plural = required_name + 'S'
                                        if plural in generated_field_names:
                                            found_match = True
                                
                                if not found_match:
                                    missing_fields.append(req_field['field_name'])
                            
                            if missing_fields:
                                st.warning(
                                    f"⚠️ Schema for {table_name} missing required fields: "
                                    f"{', '.join(missing_fields)}. Retrying... (attempt {attempt + 1}/{max_attempts})"
                                )
                                continue  # Retry
                        
                        # All validations passed
                        return columns
                    else:
                        st.warning(
                            f"⚠️ Schema validation failed for {table_name} "
                            f"(attempt {attempt + 1}/{max_attempts}): "
                            f"Invalid column structure"
                        )
                else:
                    st.warning(
                        f"⚠️ Failed to parse schema JSON for {table_name} "
                        f"(attempt {attempt + 1}/{max_attempts})"
                    )
            else:
                st.warning(
                    f"⚠️ No response from LLM for {table_name} "
                    f"(attempt {attempt + 1}/{max_attempts})"
                )
            
            # If not the last attempt, wait a bit before retrying
            if attempt < max_attempts - 1:
                time.sleep(1)
                
        except Exception as e:
            error_handler.log_error(
                error_code=ErrorCode.DATA_GENERATION_FAILED,
                error_type=type(e).__name__,
                severity=ErrorSeverity.WARNING,
                message=(
                    f"Schema generation failed for {table_name} "
                    f"(attempt {attempt + 1}/{max_attempts}): {str(e)}"
                ),
                stack_trace=traceback.format_exc()
            )
            if attempt < max_attempts - 1:
                time.sleep(1)
    
    # All attempts failed
    error_handler.log_error(
        error_code=ErrorCode.DATA_GENERATION_FAILED,
        error_type="SchemaGenerationError",
        severity=ErrorSeverity.ERROR,
        message=(
            f"Failed to generate valid schema for {table_name} after "
            f"{max_attempts} attempts"
        ),
        function_name="generate_schema_for_table"
    )
    return None


@timeit
def generate_data_from_schema(
    schema_data: List[Dict],
    num_records: int,
    table_info: Dict,
    company_name: str,
    join_key_values: Optional[List[int]] = None
) -> Dict[str, List]:
    """
    Generate realistic data based on LLM-provided schema with sample_values.
    
    Returns data in column-oriented format {col_name: [values]} instead of 
    row-oriented [{col: val}] for better pandas DataFrame creation performance.
    This approach is more efficient as pandas DataFrames are column-major
    internally.
    
    Args:
        schema_data: List of column definitions with name, type, and sample_values
        num_records: Number of records to generate
        table_info: Metadata about the table
        company_name: Company name for context-aware generation
        join_key_values: Optional list of ENTITY_ID values for joinable tables
    
    Returns:
        dict: Column-oriented data structure {column_name: [list_of_values]}
    """
    data = {}
    
    if join_key_values:
        data['ENTITY_ID'] = join_key_values
        num_records = len(join_key_values)
    else:
        data['ENTITY_ID'] = list(range(1, num_records + 1))
    
    for column in schema_data:
        if column['name'] == 'ENTITY_ID':
            continue
            
        col_type = column['type'].upper()
        col_name = column['name']
        sample_values = column.get('sample_values', [])
        
        # PREFER LLM-provided sample_values (like native_app does)
        if col_type in ['STRING', 'VARCHAR', 'TEXT']:
            if sample_values:
                data[col_name] = [
                    random.choice(sample_values) for _ in range(num_records)
                ]
            else:
                st.warning(
                    f"⚠️ Column {col_name} missing sample_values from LLM. "
                    f"Data quality may be degraded."
                )
                data[col_name] = [
                    f"{col_name}_{i+1}" for i in range(num_records)
                ]
                
        elif col_type in ['NUMBER', 'INTEGER', 'INT']:
            if 'ID' in col_name.upper() and col_name != 'ENTITY_ID':
                data[col_name] = [i + 1 for i in range(num_records)]
            elif sample_values:
                numeric_samples = [
                    int(x) for x in sample_values
                    if str(x).replace('-', '').isdigit()
                ]
                if numeric_samples:
                    data[col_name] = [
                        random.choice(numeric_samples)
                        for _ in range(num_records)
                    ]
                else:
                    data[col_name] = [
                        random.randint(1, 1000) for _ in range(num_records)
                    ]
            else:
                data[col_name] = [
                    random.randint(1, 1000) for _ in range(num_records)
                ]
                
        elif col_type in ['FLOAT', 'DECIMAL', 'DOUBLE']:
            if sample_values:
                try:
                    float_samples = [float(x) for x in sample_values]
                    data[col_name] = [
                        random.choice(float_samples)
                        for _ in range(num_records)
                    ]
                except:
                    data[col_name] = [
                        round(random.uniform(0, 1000), 2)
                        for _ in range(num_records)
                    ]
            else:
                data[col_name] = [
                    round(random.uniform(0, 1000), 2)
                    for _ in range(num_records)
                ]
                
        elif col_type == 'DATE':
            start_date = datetime.now() - timedelta(days=365)
            data[col_name] = [
                (start_date + timedelta(days=random.randint(0, 365)))
                .strftime('%Y-%m-%d')
                for _ in range(num_records)
            ]
            
        elif 'TIMESTAMP' in col_type or col_type == 'DATETIME':
            # Generate recent timestamps (past 7 days) for realistic queries
            start_date = datetime.now() - timedelta(days=7)
            data[col_name] = [
                (start_date + timedelta(
                    days=random.randint(0, 7),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59),
                    seconds=random.randint(0, 59)
                ))
                for _ in range(num_records)
            ]
            
        elif col_type == 'BOOLEAN':
            data[col_name] = [
                random.choice([True, False]) for _ in range(num_records)
            ]
            
        else:
            # Default fallback
            if sample_values:
                data[col_name] = [
                    random.choice(sample_values) for _ in range(num_records)
                ]
            else:
                data[col_name] = [
                    f"Value_{i+1}" for i in range(num_records)
                ]
    
    return data


def build_rich_table_context(
    table_key: str,
    table_info: Dict,
    table_schema: List[Dict],
    table_data: Dict[str, List],
    num_sample_rows: int = 5
) -> Dict:
    """
    Build comprehensive context about a table for question generation.
    
    Analyzes both the LLM-generated schema and actual generated data to provide
    rich context including data types, descriptions, sample values, cardinality,
    numeric ranges, and sample rows.
    
    Args:
        table_key: Internal key for the table (e.g., 'structured_1')
        table_info: Table metadata including name and description
        table_schema: LLM-generated schema with columns, types, descriptions, sample_values
        table_data: Actual generated data in column-oriented format {col_name: [values]}
        num_sample_rows: Number of sample rows to include
    
    Returns:
        dict: Comprehensive table context with structure:
            {
                'name': str,
                'description': str,
                'columns': [
                    {
                        'name': str,
                        'type': str,
                        'description': str,
                        'llm_sample_values': List,  # From schema generation
                        'unique_count': int,  # For categorical columns
                        'sample_actual_values': List,  # From generated data
                        'numeric_range': {min, max, avg},  # For numeric columns
                        'has_nulls': bool
                    }
                ],
                'sample_rows': List[Dict],  # First N rows
                'row_count': int
            }
    """
    # Extract basic table info
    context = {
        'name': table_info['name'],
        'description': table_info.get('description', ''),
        'row_count': len(table_data.get('ENTITY_ID', [])),
        'columns': [],
        'sample_rows': []
    }
    
    # Analyze each column
    for col_def in table_schema:
        col_name = col_def['name']
        col_type = col_def['type'].upper()
        col_description = col_def.get('description', '')
        llm_sample_values = col_def.get('sample_values', [])
        
        col_context = {
            'name': col_name,
            'type': col_type,
            'description': col_description,
            'llm_sample_values': llm_sample_values,
            'has_nulls': False
        }
        
        # Get actual data for this column
        if col_name in table_data:
            col_values = table_data[col_name]
            
            # Check for nulls/None values
            col_context['has_nulls'] = any(v is None for v in col_values)
            
            # For categorical columns (STRING, VARCHAR, TEXT), analyze cardinality
            if col_type in ['STRING', 'VARCHAR', 'TEXT']:
                unique_values = list(set(col_values))
                col_context['unique_count'] = len(unique_values)
                # Sample up to 10 actual values
                col_context['sample_actual_values'] = unique_values[:10]
            
            # For numeric columns, calculate range and statistics
            elif col_type in ['NUMBER', 'INTEGER', 'INT', 'FLOAT', 'DECIMAL', 'DOUBLE']:
                numeric_values = [v for v in col_values if v is not None and isinstance(v, (int, float))]
                if numeric_values:
                    col_context['numeric_range'] = {
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': sum(numeric_values) / len(numeric_values)
                    }
                    col_context['unique_count'] = len(set(numeric_values))
            
            # For date/timestamp columns, get range
            elif 'DATE' in col_type or 'TIMESTAMP' in col_type:
                # For datetime objects
                if col_values and hasattr(col_values[0], 'strftime'):
                    date_values = [v for v in col_values if v is not None]
                    if date_values:
                        col_context['date_range'] = {
                            'min': str(min(date_values)),
                            'max': str(max(date_values))
                        }
                # For string dates
                elif col_values and isinstance(col_values[0], str):
                    date_values = [v for v in col_values if v is not None]
                    if date_values:
                        col_context['date_range'] = {
                            'min': min(date_values),
                            'max': max(date_values)
                        }
        
        context['columns'].append(col_context)
    
    # Build sample rows (convert column-oriented to row-oriented)
    if context['row_count'] > 0:
        num_rows_to_sample = min(num_sample_rows, context['row_count'])
        for i in range(num_rows_to_sample):
            row = {}
            for col_def in table_schema:
                col_name = col_def['name']
                if col_name in table_data and i < len(table_data[col_name]):
                    row[col_name] = table_data[col_name][i]
            context['sample_rows'].append(row)
    
    return context


@timeit
def generate_unstructured_data(
    table_name: str,
    table_description: str,
    num_chunks: int,
    company_name: str,
    session,
    error_handler,
    language_code: str = "en"
) -> List[Dict]:
    """
    Generate unstructured text data for Cortex Search with language support.
    
    Args:
        table_name: Name of the unstructured table
        table_description: Description of the content
        num_chunks: Number of text chunks to generate
        company_name: Company name for context
        session: Snowflake session
        error_handler: ErrorHandler instance
        language_code: Language code for content generation
        
    Returns:
        List of chunk dictionaries with metadata
    """
    base_prompt = f"""Generate {min(num_chunks, 5)} realistic text chunks for this unstructured data:

Type: {table_name}
Description: {table_description}
Company: {company_name}

Each chunk should be 2-3 paragraphs of realistic business content.
Format as JSON array:
[
  {{"chunk_text": "Content here...", "document_type": "type", "source_system": "system"}}
]"""

    prompt = enhance_prompt_with_language(base_prompt, language_code)
    
    response = call_cortex_with_retry(prompt, session, error_handler)
    
    chunks_data = []
    
    if response:
        parsed = safe_json_parse(response)
        if parsed and isinstance(parsed, list):
            base_chunks = parsed
            
            while len(chunks_data) < num_chunks:
                for chunk in base_chunks:
                    if len(chunks_data) >= num_chunks:
                        break
                    
                    chunk_text = chunk.get(
                        'chunk_text',
                        f"Sample content {len(chunks_data)}"
                    )
                    
                    is_valid, error_msg = validate_language_content(
                        chunk_text, language_code
                    )
                    if not is_valid and language_code != "en":
                        st.warning(f"Language validation warning: {error_msg}")
                    
                    chunks_data.append({
                        'CHUNK_ID': len(chunks_data) + 1,
                        'DOCUMENT_ID': f"DOC_{(len(chunks_data) // 5) + 1}",
                        'CHUNK_TEXT': chunk_text,
                        'DOCUMENT_TYPE': chunk.get('document_type', 'general'),
                        'SOURCE_SYSTEM': chunk.get('source_system', company_name)
                    })
    
    if not chunks_data:
        for i in range(num_chunks):
            chunks_data.append({
                'CHUNK_ID': i + 1,
                'DOCUMENT_ID': f"DOC_{(i // 5) + 1}",
                'CHUNK_TEXT': (
                    f"This is sample content chunk {i+1} for "
                    f"{table_description}. " * 10
                ),
                'DOCUMENT_TYPE': table_name.lower(),
                'SOURCE_SYSTEM': company_name
            })
    
    chunks_data = add_language_metadata_to_chunks(chunks_data, language_code)
    
    return chunks_data


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_tables_collectively(
    all_tables_info: List[Dict],
    target_questions: List[str],
    session,
    error_handler
) -> Dict[str, Any]:
    """
    Validate that ALL tables TOGETHER can answer the target questions.
    This addresses cross-table questions that require joins.
    
    Args:
        all_tables_info: List of dicts with {name, schema, data, role}
        target_questions: List of target questions
        session: Snowflake session
        error_handler: ErrorHandler instance
    
    Returns:
        dict: Validation results with per-question assessment
    """
    if not target_questions or len(target_questions) == 0:
        return {'overall_valid': True, 'questions': []}
    
    # Build comprehensive tables summary
    tables_summary = []
    for table_info in all_tables_info:
        table_name = table_info['name']
        schema = table_info['schema']
        data = table_info['data']
        
        # Get enhanced sample data (10-15 rows instead of 3-5)
        sample_size = min(15, len(data.get('ENTITY_ID', [])))
        sample_rows = []
        for i in range(sample_size):
            row = {}
            for col_name, values in data.items():
                if i < len(values):
                    row[col_name] = values[i]
            sample_rows.append(row)
        
        # Calculate distribution statistics
        stats = {}
        for col in schema:
            col_name = col['name']
            if col_name in data and len(data[col_name]) > 0:
                col_data = data[col_name]
                if col['type'].upper() in [
                    'NUMBER', 'INTEGER', 'INT', 'FLOAT', 'DECIMAL'
                ]:
                    try:
                        numeric_vals = [
                            float(v) for v in col_data if v is not None
                        ]
                        if numeric_vals:
                            stats[col_name] = {
                                'min': min(numeric_vals),
                                'max': max(numeric_vals),
                                'unique_count': len(set(numeric_vals))
                            }
                    except:
                        pass
                else:
                    unique_vals = set(str(v) for v in col_data if v is not None)
                    stats[col_name] = {
                        'unique_count': len(unique_vals),
                        'sample_values': list(unique_vals)[:10]
                    }
        
        column_list = ", ".join([
            f"{col['name']} ({col['type']})" for col in schema
        ])
        tables_summary.append(
            f"**{table_name}**:\n  Columns: {column_list}\n  "
            f"Row count: {len(data.get('ENTITY_ID', []))}\n  "
            f"Statistics: {json.dumps(stats, default=str)[:200]}"
        )
    
    tables_text = "\n\n".join(tables_summary)
    questions_text = "\n".join([
        f"{i+1}. {q}" for i, q in enumerate(target_questions)
    ])
    
    prompt = f"""Validate if these tables TOGETHER can answer the target questions.

Available Tables:
{tables_text}

Join Key: All tables can join on ENTITY_ID (with ~70% overlap expected)

Target Questions:
{questions_text}

For EACH question, determine:
1. Which table(s) are needed? (single table or join required)
2. What columns/operations are needed?
3. Is the answer CALCULABLE from available data and joins?
4. Rate confidence: HIGH (definitely answerable), MEDIUM (probably answerable), LOW (missing data)

Return ONLY a JSON object:
{{
  "overall_assessment": "Summary of whether all questions are answerable",
  "questions": [
    {{
      "question": "question text",
      "tables_needed": ["TABLE1", "TABLE2"],
      "columns_needed": ["column1", "column2"],
      "answerable": true/false,
      "confidence": "HIGH/MEDIUM/LOW",
      "requires_join": true/false,
      "notes": "Specific explanation of how to answer or what's missing"
    }}
  ]
}}

IMPORTANT: A question requiring data from multiple tables should NOT mark individual tables as invalid. Instead, mark requires_join=true and assess if the JOIN will produce the answer."""
    
    try:
        response = call_cortex_with_retry(prompt, session, error_handler)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed:
                return parsed
    except Exception as e:
        st.warning(f"⚠️ Collective validation error: {str(e)}")
    
    # Fallback
    return {
        'overall_assessment': 'Validation completed (limited confidence)',
        'questions': []
    }


def validate_data_against_questions(
    table_data: Dict,
    target_questions: List[str],
    table_schema: List[Dict],
    table_name: str,
    session,
    error_handler
) -> Tuple[bool, str]:
    """
    Validate that generated data can plausibly answer the target questions.
    NOTE: This is now a helper for single-table validation. Use
    validate_tables_collectively() for comprehensive validation.
    
    Args:
        table_data: Dict of column-oriented data
        target_questions: List of target questions
        table_schema: List of column definitions
        table_name: Name of the table
        session: Snowflake session
        error_handler: ErrorHandler instance
    
    Returns:
        tuple: (is_valid, feedback_message)
    """
    if not target_questions or len(target_questions) == 0:
        return (True, "No target questions to validate")
    
    # Enhanced sample size: 10-15 rows instead of 3-5
    sample_size = min(15, len(table_data.get('ENTITY_ID', [])))
    sample_data_rows = []
    if sample_size > 0:
        for i in range(sample_size):
            row = {}
            for col_name, values in table_data.items():
                if i < len(values):
                    row[col_name] = values[i]
            sample_data_rows.append(row)
    
    # Calculate distribution statistics for better validation
    column_summary = []
    for col in table_schema:
        col_name = col.get('name', 'Unknown')
        col_type = col.get('type', 'Unknown')
        sample_vals = col.get('sample_values', [])[:8]  # Show more samples
        
        # Add distribution info if available
        dist_info = ""
        if col_name in table_data and len(table_data[col_name]) > 0:
            col_data = table_data[col_name]
            if col_type.upper() in [
                'NUMBER', 'INTEGER', 'INT', 'FLOAT', 'DECIMAL'
            ]:
                try:
                    numeric_vals = [
                        float(v) for v in col_data if v is not None
                    ]
                    if numeric_vals:
                        dist_info = (
                            f" [range: {min(numeric_vals):.1f}-"
                            f"{max(numeric_vals):.1f}, "
                            f"unique: {len(set(numeric_vals))}]"
                        )
                except:
                    pass
            else:
                unique_vals = set(str(v) for v in col_data if v is not None)
                dist_info = f" [unique values: {len(unique_vals)}]"
        
        column_summary.append(
            f"- {col_name} ({col_type}): {sample_vals}{dist_info}"
        )
    
    columns_text = "\n".join(column_summary)
    questions_text = "\n".join([
        f"{i+1}. {q}" for i, q in enumerate(target_questions)
    ])
    sample_data_text = json.dumps(
        sample_data_rows[:10], indent=2, default=str
    )
    
    prompt = f"""Validate if this SINGLE table can contribute to answering the target questions.

IMPORTANT: Some questions may require joining with other tables. For those questions, assess if THIS table has the necessary columns and data to participate in the join.

Table: {table_name}

Columns (with distribution stats):
{columns_text}

Sample Data (first 10 rows):
{sample_data_text}

Target Questions:
{questions_text}

For each question, determine:
1. Can this table ALONE answer it? Or does it need to join with other tables?
2. If it needs a join, does this table have the necessary columns to participate?
3. Is the data quality sufficient (good distribution, realistic values)?

Return ONLY a JSON object:
{{
  "table_role": "primary/supporting/not_needed",
  "feedback": "Brief summary focusing on what THIS table provides",
  "questions_coverage": [
    {{"question": "question text", "role_for_question": "answers_alone/needs_join/not_relevant", "notes": "explanation"}}
  ]
}}"""
    
    try:
        response = call_cortex_with_retry(prompt, session, error_handler)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed:
                feedback = parsed.get('feedback', 'Validation completed')
                questions_coverage = parsed.get('questions_coverage', [])
                
                # Build detailed feedback with better messaging
                detailed_feedback = f"{feedback}\n\n"
                for qc in questions_coverage:
                    question = qc.get('question', 'Unknown')
                    role = qc.get('role_for_question', 'unknown')
                    notes = qc.get('notes', '')
                    
                    if role == 'answers_alone':
                        status = "✅"
                        msg = "Can answer independently"
                    elif role == 'needs_join':
                        status = "🔗"
                        msg = "Supports via join"
                    else:
                        status = "➖"
                        msg = "Not relevant to this table"
                    
                    detailed_feedback += (
                        f"{status} {question}: {msg} - {notes}\n"
                    )
                
                # Don't mark as invalid if table just needs to join
                is_valid = True  # Individual tables are valid if they contribute
                return (is_valid, detailed_feedback)
    except Exception as e:
        return (True, f"Validation skipped due to error: {str(e)}")
    
    # Fallback: assume valid if we can't validate
    return (True, "Validation completed (no specific issues found)")


@timeit
def save_structured_table_to_snowflake(
    schema_name: str,
    table_name: str,
    table_schema: List[Dict],
    table_data: Dict,
    table_info: Dict,
    num_records: int,
    status_container,
    session,
    overlap_info: Optional[str] = None
) -> Dict:
    """
    Helper function to save a structured table to Snowflake and display results.
    
    Args:
        schema_name: Schema name
        table_name: Table name
        table_schema: List of column definitions
        table_data: Column-oriented data dict
        table_info: Table metadata
        num_records: Number of records
        status_container: Streamlit container for status messages
        session: Snowflake session
        overlap_info: Optional string describing join overlap
    
    Returns:
        dict: Result metadata for the created table
    """
    dataframe = pd.DataFrame(table_data)
    snowpark_dataframe = session.create_dataframe(dataframe)
    snowpark_dataframe.write.mode("overwrite").save_as_table(
        f"{schema_name}.{table_name}"
    )
    session.sql(
        f"ALTER TABLE {schema_name}.{table_name} ADD PRIMARY KEY (ENTITY_ID)"
    ).collect()
    
    # Create status message
    status_msg = f"✅ {table_name} created ({num_records} records"
    if overlap_info:
        status_msg += f", {overlap_info}"
    status_msg += ")"
    
    # Add sample data preview in expander
    with status_container:
        with st.expander(status_msg, expanded=False):
            st.caption(
                f"**Columns:** {', '.join([col['name'] for col in table_schema])}"
            )
            st.dataframe(dataframe.head(3), use_container_width=True)
    
    return {
        'table': table_name,
        'records': num_records,
        'description': table_info['description'],
        'columns': [col['name'] for col in table_schema],
        'type': 'structured',
        'table_type': table_info.get('table_type', 'dimension'),
        'sample_data': dataframe.head(3).to_dict('records')  # MEMORY OPTIMIZATION: Convert DataFrame to dict
    }

