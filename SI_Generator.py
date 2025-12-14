"""
SI Data Generator - Enhanced Dashboard
Production-ready with comprehensive error handling, agent automation,
question generation, and multi-language support
"""

import streamlit as st
import pandas as pd
from snowflake.snowpark import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.types import StructType, StructField, StringType, IntegerType, FloatType, DateType, TimestampType
import random
from datetime import datetime, timedelta
import json
import re
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import time
import traceback
from typing import Optional, Dict, Any, Callable, List, Tuple
from enum import Enum

# Import from new modular components
from utils import (
    LLM_MODEL,
    SNOWFLAKE_COLORS,
    MAX_FACTS_PER_TABLE,
    MAX_TOTAL_FACTS,
    TABLE_JOIN_OVERLAP_PERCENTAGE,
    MAX_DIMENSIONS_PER_TABLE,
    MAX_TOTAL_DIMENSIONS,
    create_local_session,
    call_cortex_with_retry,
    get_company_info_from_url,
    generate_demo_ideas_with_cortex
)
from errors import (
    ErrorSeverity,
    ErrorCode,
    RecoverableError,
    FatalError,
    CortexServiceError,
    DataGenerationError,
    ErrorHandler,
    retry_with_exponential_backoff,
    check_cortex_availability,
    safe_json_parse
)
from styles import (
    apply_main_styles,
    show_step_progress,
    render_header,
    render_selection_box,
    render_about_hero,
    render_page_footer,
    render_loading_info,
    render_demo_header,
    render_results_table_list,
    render_query_results
)
from utils import (
    SUPPORTED_LANGUAGES,
    get_language_config,
    get_language_display_name,
    validate_language_content,
    enhance_prompt_with_language,
    add_language_metadata_to_chunks
)
from demo_content import (
    get_fallback_demo_ideas,
    analyze_company_url,
    generate_contextual_questions,
    analyze_target_questions,
    format_questions_for_display,
    generate_schema_for_table,
    extract_required_fields_from_description,
    generate_data_from_schema,
    build_rich_table_context,
    generate_unstructured_data,
    validate_tables_collectively,
    validate_data_against_questions,
    save_structured_table_to_snowflake
)
from infrastructure import (
    verify_snowflake_intelligence_setup,
    create_agent_automatically,
    generate_agent_documentation,
    test_agent_with_questions,
    extract_questions_from_semantic_model,
    create_semantic_view,
    create_cortex_search_service,
    analyze_table_relationships
)
from prompts import (
    get_company_analysis_prompt,
    get_question_generation_prompt,
    get_follow_up_questions_prompt,
    get_target_question_analysis_prompt,
    get_agent_system_prompt,
    get_agent_persona_prompt,
    get_demo_generation_prompt,
    get_schema_generation_prompt,
    get_collective_validation_prompt,
    get_single_table_validation_prompt,
    get_unstructured_data_generation_prompt,
    get_table_relationships_analysis_prompt
)
from utils import execute_parallel_llm_calls
from metrics import timeit, display_performance_summary

st.set_page_config(
    page_title="SI Data Generator",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply main CSS styles from styles module
apply_main_styles()


# ============================================================================
# SESSION AND CONFIG (now imported from utils module)
# ============================================================================

# Initialize session with spinner feedback
with st.spinner("🔄 Connecting to Snowflake..."):
    session = create_local_session()
    warehouse_name = session.get_current_warehouse()

error_handler = ErrorHandler(session)

# All question generation and data generation functions have been moved to demo_content.py

def save_to_history(session, company_name, company_url, demo_data, schema_name, num_records, 
                    language_code, team_members, use_cases, enable_semantic_view, 
                    enable_search_service, enable_agent, advanced_mode, results, 
                    target_questions, generated_questions):
    """
    Save demo generation details to history table.
    
    Args:
        session: Snowflake session
        company_name: Company name
        company_url: Company URL
        demo_data: Full demo data structure
        schema_name: Generated schema name
        num_records: Number of records per table
        language_code: Content language code
        team_members: Team members/audience
        use_cases: Specific use cases
        enable_semantic_view: Whether semantic view was created
        enable_search_service: Whether search service was created
        enable_agent: Whether agent was created
        advanced_mode: Whether advanced mode was used
        results: List of created tables/services
        target_questions: User-provided target questions
        generated_questions: AI-generated questions
    """
    try:
        import uuid
        
        # Generate unique history ID
        history_id = str(uuid.uuid4())
        
        # Extract table names from results
        table_names = [r['table'] for r in results if r.get('table')]
        
        # Prepare question data
        target_questions_array = target_questions if target_questions else []
        generated_questions_array = []
        if generated_questions:
            for q in generated_questions:
                if isinstance(q, dict):
                    generated_questions_array.append(q.get('text', str(q)))
                else:
                    generated_questions_array.append(str(q))
        
        # Prepare data for insertion
        demo_title = demo_data.get('title', 'Untitled Demo')
        demo_description = demo_data.get('description', '')
        
        # Convert arrays and objects to JSON strings and escape for SQL
        # Must escape backslashes first, then newlines, then single quotes
        def escape_json_for_sql(obj):
            json_str = json.dumps(obj)
            json_str = json_str.replace('\\', '\\\\')  # Escape backslashes first
            json_str = json_str.replace('\n', '\\n')   # Escape newlines
            json_str = json_str.replace('\r', '\\r')   # Escape carriage returns
            json_str = json_str.replace("'", "''")     # Escape single quotes
            return json_str
        
        table_names_json = escape_json_for_sql(table_names)
        target_questions_json = escape_json_for_sql(target_questions_array)
        generated_questions_json = escape_json_for_sql(generated_questions_array)
        demo_data_json = escape_json_for_sql(demo_data)
        
        # Escape single quotes in string fields
        company_name_escaped = company_name.replace("'", "''") if company_name else ''
        company_url_escaped = company_url.replace("'", "''") if company_url else ''
        demo_title_escaped = demo_title.replace("'", "''")
        demo_description_escaped = demo_description.replace("'", "''")
        schema_name_escaped = schema_name.replace("'", "''")
        team_members_escaped = team_members.replace("'", "''") if team_members else ''
        use_cases_escaped = use_cases.replace("'", "''") if use_cases else ''
        language_code_escaped = language_code.replace("'", "''")
        
        # Create insert statement using SELECT with PARSE_JSON (workaround for VALUES clause limitation)
        insert_sql = f"""
        INSERT INTO SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY (
            HISTORY_ID, CREATED_AT, COMPANY_NAME, COMPANY_URL, DEMO_TITLE, 
            DEMO_DESCRIPTION, SCHEMA_NAME, NUM_RECORDS, LANGUAGE_CODE, 
            TEAM_MEMBERS, USE_CASES, ENABLE_SEMANTIC_VIEW, ENABLE_SEARCH_SERVICE, 
            ENABLE_AGENT, ADVANCED_MODE, TABLE_NAMES, TARGET_QUESTIONS, 
            GENERATED_QUESTIONS, DEMO_DATA_JSON
        )
        SELECT 
            '{history_id}', 
            CURRENT_TIMESTAMP, 
            '{company_name_escaped}', 
            '{company_url_escaped}', 
            '{demo_title_escaped}', 
            '{demo_description_escaped}', 
            '{schema_name_escaped}', 
            {num_records}, 
            '{language_code_escaped}', 
            '{team_members_escaped}', 
            '{use_cases_escaped}', 
            {enable_semantic_view}, 
            {enable_search_service}, 
            {enable_agent}, 
            {advanced_mode}, 
            PARSE_JSON('{table_names_json}'), 
            PARSE_JSON('{target_questions_json}'), 
            PARSE_JSON('{generated_questions_json}'), 
            PARSE_JSON('{demo_data_json}')
        """
        
        # Execute insert
        session.sql(insert_sql).collect()
        
        return history_id
        
    except Exception as e:
        # Log error but don't fail the demo generation
        st.warning(f"⚠️ Could not save to history: {str(e)}")
        return None


def get_history_records(session, limit=50, offset=0):
    """
    Fetch history records from the database.
    
    Args:
        session: Snowflake session
        limit: Maximum number of records to fetch
        offset: Offset for pagination
        
    Returns:
        List of history records as dictionaries
    """
    try:
        query = f"""
        SELECT 
            HISTORY_ID,
            CREATED_AT,
            COMPANY_NAME,
            COMPANY_URL,
            DEMO_TITLE,
            DEMO_DESCRIPTION,
            SCHEMA_NAME,
            NUM_RECORDS,
            LANGUAGE_CODE,
            TEAM_MEMBERS,
            USE_CASES,
            ENABLE_SEMANTIC_VIEW,
            ENABLE_SEARCH_SERVICE,
            ENABLE_AGENT,
            ADVANCED_MODE,
            TABLE_NAMES,
            TARGET_QUESTIONS,
            GENERATED_QUESTIONS,
            DEMO_DATA_JSON
        FROM SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY
        ORDER BY CREATED_AT DESC
        LIMIT {limit} OFFSET {offset}
        """
        
        result = session.sql(query).collect()
        
        history_records = []
        for row in result:
            record = {
                'history_id': row['HISTORY_ID'],
                'created_at': row['CREATED_AT'],
                'company_name': row['COMPANY_NAME'],
                'company_url': row['COMPANY_URL'],
                'demo_title': row['DEMO_TITLE'],
                'demo_description': row['DEMO_DESCRIPTION'],
                'schema_name': row['SCHEMA_NAME'],
                'num_records': row['NUM_RECORDS'],
                'language_code': row['LANGUAGE_CODE'],
                'team_members': row['TEAM_MEMBERS'],
                'use_cases': row['USE_CASES'],
                'enable_semantic_view': row['ENABLE_SEMANTIC_VIEW'],
                'enable_search_service': row['ENABLE_SEARCH_SERVICE'],
                'enable_agent': row['ENABLE_AGENT'],
                'advanced_mode': row['ADVANCED_MODE'],
                'table_names': json.loads(row['TABLE_NAMES']) if row['TABLE_NAMES'] else [],
                'target_questions': json.loads(row['TARGET_QUESTIONS']) if row['TARGET_QUESTIONS'] else [],
                'generated_questions': json.loads(row['GENERATED_QUESTIONS']) if row['GENERATED_QUESTIONS'] else [],
                'demo_data_json': json.loads(row['DEMO_DATA_JSON']) if row['DEMO_DATA_JSON'] else {}
            }
            history_records.append(record)
        
        return history_records
        
    except Exception as e:
        st.error(f"Error fetching history: {str(e)}")
        return []


def get_history_by_id(session, history_id):
    """
    Fetch a specific history record by ID.
    
    Args:
        session: Snowflake session
        history_id: History ID to fetch
        
    Returns:
        History record as dictionary or None
    """
    try:
        query = """
        SELECT 
            HISTORY_ID,
            CREATED_AT,
            COMPANY_NAME,
            COMPANY_URL,
            DEMO_TITLE,
            DEMO_DESCRIPTION,
            SCHEMA_NAME,
            NUM_RECORDS,
            LANGUAGE_CODE,
            TEAM_MEMBERS,
            USE_CASES,
            ENABLE_SEMANTIC_VIEW,
            ENABLE_SEARCH_SERVICE,
            ENABLE_AGENT,
            ADVANCED_MODE,
            TABLE_NAMES,
            TARGET_QUESTIONS,
            GENERATED_QUESTIONS,
            DEMO_DATA_JSON
        FROM SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY
        WHERE HISTORY_ID = ?
        """
        
        result = session.sql(query).bind(history_id).collect()
        
        if result:
            row = result[0]
            return {
                'history_id': row['HISTORY_ID'],
                'created_at': row['CREATED_AT'],
                'company_name': row['COMPANY_NAME'],
                'company_url': row['COMPANY_URL'],
                'demo_title': row['DEMO_TITLE'],
                'demo_description': row['DEMO_DESCRIPTION'],
                'schema_name': row['SCHEMA_NAME'],
                'num_records': row['NUM_RECORDS'],
                'language_code': row['LANGUAGE_CODE'],
                'team_members': row['TEAM_MEMBERS'],
                'use_cases': row['USE_CASES'],
                'enable_semantic_view': row['ENABLE_SEMANTIC_VIEW'],
                'enable_search_service': row['ENABLE_SEARCH_SERVICE'],
                'enable_agent': row['ENABLE_AGENT'],
                'advanced_mode': row['ADVANCED_MODE'],
                'table_names': json.loads(row['TABLE_NAMES']) if row['TABLE_NAMES'] else [],
                'target_questions': json.loads(row['TARGET_QUESTIONS']) if row['TARGET_QUESTIONS'] else [],
                'generated_questions': json.loads(row['GENERATED_QUESTIONS']) if row['GENERATED_QUESTIONS'] else [],
                'demo_data_json': json.loads(row['DEMO_DATA_JSON']) if row['DEMO_DATA_JSON'] else {}
            }
        
        return None
        
    except Exception as e:
        st.error(f"Error fetching history record: {str(e)}")
        return None


def export_history_to_json(history_records):
    """
    Export history records to JSON format.
    
    Args:
        history_records: List of history record dictionaries
        
    Returns:
        JSON string of all history records
    """
    try:
        # Convert datetime objects to strings for JSON serialization
        export_data = []
        for record in history_records:
            record_copy = record.copy()
            if record_copy.get('created_at'):
                record_copy['created_at'] = str(record_copy['created_at'])
            export_data.append(record_copy)
        
        return json.dumps(export_data, indent=2)
        
    except Exception as e:
        st.error(f"Error exporting history: {str(e)}")
        return None


def load_configuration_from_history(history_record):
    """
    Load configuration from a history record into session state.
    
    Args:
        history_record: History record dictionary
    """
    try:
        # Clear any existing widget states that might conflict
        keys_to_clear = ['company_url', 'team_members', 'use_cases', 'num_records', 
                        'content_language', 'advanced_mode', 'enable_semantic', 
                        'enable_search', 'enable_agent']
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        # Now populate session state with history data
        st.session_state['company_url'] = history_record.get('company_url', '')
        st.session_state['team_members'] = history_record.get('team_members', '')
        st.session_state['use_cases'] = history_record.get('use_cases', '')
        st.session_state['num_records'] = history_record.get('num_records', 100)
        st.session_state['content_language'] = history_record.get('language_code', 'en')
        st.session_state['advanced_mode'] = history_record.get('advanced_mode', False)
        st.session_state['enable_semantic'] = history_record.get('enable_semantic_view', True)
        st.session_state['enable_search'] = history_record.get('enable_search_service', True)
        st.session_state['enable_agent'] = history_record.get('enable_agent', True)
        
        # Load target questions
        target_questions = history_record.get('target_questions', [])
        st.session_state['target_questions'] = target_questions if target_questions else []
        
        # Load company info and demo data
        st.session_state['company_name'] = history_record.get('company_name', '')
        demo_data = history_record.get('demo_data_json', {})
        if demo_data:
            st.session_state['selected_demo'] = demo_data
            st.session_state['demo_ideas'] = [demo_data]
            st.session_state['selected_demo_idx'] = 0
        
        # Clear any infrastructure flags
        if 'infrastructure_started' in st.session_state:
            del st.session_state['infrastructure_started']
        if 'infrastructure_complete' in st.session_state:
            del st.session_state['infrastructure_complete']
        
        # Set flag to show info message
        st.session_state['config_loaded_from_history'] = True
        st.session_state['loaded_history_id'] = history_record.get('history_id')
        
    except Exception as e:
        st.error(f"Error loading configuration: {str(e)}")


@timeit
def create_tables_in_snowflake(schema_name, demo_data, num_records, company_name, 
                               enable_search_service, enable_semantic_view, enable_agent,
                               language_code, progress_placeholder, status_container, company_url=None, target_questions=None):
    """Create tables in Snowflake with progress updates and enhancements"""
    
    results = []
    has_target_questions = target_questions and len(target_questions) > 0
    
    # Dynamically detect all structured tables to calculate accurate progress steps
    num_structured_tables = sum(1 for key in demo_data['tables'].keys() if key.startswith('structured_'))
    num_unstructured_tables = sum(1 for key in demo_data['tables'].keys() if key.startswith('unstructured'))
    
    # Progress steps breakdown:
    # 1. Create schema
    # 2. Generate all schemas (1 step)
    # 3. Generate data for each table (N steps for N tables)
    # 4. Save each table to Snowflake (N steps for N tables)
    # 5. Create unstructured table(s) (1-2 steps)
    # 6. Optional: Question analysis (if target_questions)
    # 7. Optional: Semantic view (if enabled)
    # 8. Optional: Search service (if enabled)
    # 9. Optional: Agent (if enabled)
    # 10. Generate sample questions
    # 11. Completion
    
    base_steps = 1 + 1 + num_structured_tables + num_structured_tables + num_unstructured_tables + 2  # schema + schema gen + data gen + saves + unstructured + questions + completion
    total_steps = base_steps
    if has_target_questions:
        total_steps += 1  # Question analysis
    if enable_semantic_view:
        total_steps += 1
    if enable_search_service:
        total_steps += 1
    if enable_agent:
        total_steps += 1
    
    current_step = 0
    
    start_time = time.time()
    
    # Analyze target questions if provided
    question_analysis = None
    if has_target_questions:
        current_step += 1
        progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Analyzing target questions...")
        question_analysis = analyze_target_questions(target_questions, session, error_handler)
        
        if question_analysis.get('has_target_questions'):
            with status_container:
                st.info(f"🎯 Question analysis complete: {len(question_analysis.get('required_dimensions', []))} dimensions identified")
    
    # Store validation results for later display
    validation_results = {}
    
    try:
        current_step += 1
        progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Creating schema...")
        session.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}").collect()
        
        # Dynamically detect all structured tables (supports 2-5 tables)
        structured_tables = []
        for key in sorted(demo_data['tables'].keys()):
            if key.startswith('structured_'):
                structured_tables.append((key, demo_data['tables'][key]))
        
        num_structured_tables = len(structured_tables)
        
        current_step += 1
        progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Generating schemas for {num_structured_tables} tables with AI in parallel...")
        
        # Generate schemas for all structured tables IN PARALLEL
        # This is a major performance optimization - reduces from N*3-5s to max(3-5s)
        schema_tasks = {}
        for key, table_info in structured_tables:
            # Extract required fields from table description
            required_fields = extract_required_fields_from_description(
                table_info['description']
            )
            
            # Build task for parallel execution
            schema_tasks[key] = (
                generate_schema_for_table,
                [
                    table_info['name'], 
                    table_info['description'], 
                    company_name,
                    session,
                    error_handler,
                    3,  # max_attempts
                    target_questions,
                    question_analysis,
                    required_fields
                ]
            )
        
        # Execute all schema generation calls in parallel (max 3 workers to avoid overwhelming Cortex)
        table_schemas_raw = execute_parallel_llm_calls(schema_tasks, max_workers=min(3, num_structured_tables))
        
        # Validate results and handle any failures
        table_schemas = {}
        for key, schema in table_schemas_raw.items():
            if isinstance(schema, Exception):
                # Schema generation failed
                user_msg = error_handler.get_user_friendly_message(ErrorCode.DATA_GENERATION_FAILED)
                st.error(f"{user_msg['title']}: Failed to generate schema for {key}: {str(schema)[:100]}")
                return None
            elif not schema:
                user_msg = error_handler.get_user_friendly_message(ErrorCode.DATA_GENERATION_FAILED)
                st.error(f"{user_msg['title']}: Failed to generate schema for {key}")
                return None
            table_schemas[key] = schema
        
        # Generate data for all structured tables
        # First table gets fresh IDs, subsequent tables get join keys with overlap
        table_data_dict = {}
        tables_for_validation = []
        base_entity_ids = None
        
        for idx, (key, table_info) in enumerate(structured_tables):
            current_step += 1
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Generating data for {table_info['name']}...")
            
            if idx == 0:
                # First table: generate fresh data with sequential ENTITY_IDs
                table_data = generate_data_from_schema(table_schemas[key], num_records, table_info, company_name, join_key_values=None)
                base_entity_ids = table_data['ENTITY_ID'].copy()
            else:
                # Subsequent tables: create join keys with controlled overlap to first table
                overlap_count = int(num_records * TABLE_JOIN_OVERLAP_PERCENTAGE)
                overlap_keys = random.sample(base_entity_ids, overlap_count)
                max_base_id = max(base_entity_ids)
                remaining_keys = [max_base_id + i + 1 for i in range(num_records - overlap_count)]
                join_keys = overlap_keys + remaining_keys
                random.shuffle(join_keys)
                
                table_data = generate_data_from_schema(table_schemas[key], num_records, table_info, company_name, join_key_values=join_keys)
            
            table_data_dict[key] = table_data
            
            # Store for validation
            if has_target_questions:
                tables_for_validation.append({
                    'name': table_info['name'],
                    'schema': table_schemas[key],
                    'data': table_data,
                    'role': 'structured'
                })
        
        # Run COLLECTIVE validation on all tables together
        if has_target_questions and tables_for_validation:
            with status_container:
                st.info("🔍 Validating that tables can answer target questions collectively...")
            
            collective_validation = validate_tables_collectively(tables_for_validation, target_questions, session, error_handler)
            
            if collective_validation and collective_validation.get('questions'):
                validation_results['collective'] = collective_validation
                
                answerable_count = sum(1 for q in collective_validation['questions'] if q.get('answerable', False))
                total_questions = len(collective_validation['questions'])
                
                with status_container:
                    if answerable_count == total_questions:
                        st.success(f"✅ All {total_questions} target question(s) are answerable with the generated data!")
                    elif answerable_count > 0:
                        st.info(f"✓ {answerable_count}/{total_questions} target question(s) are answerable. See details in results.")
                    else:
                        st.warning(f"⚠️ Generated data may need adjustment to fully answer target questions. See details below.")
        
        # Save all structured tables to Snowflake
        for idx, (key, table_info) in enumerate(structured_tables):
            current_step += 1
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Saving {table_info['name']} to Snowflake...")
            
            overlap_info = None
            if idx > 0:  # Tables after the first have join overlap
                overlap_pct = int(TABLE_JOIN_OVERLAP_PERCENTAGE * 100)
                overlap_info = f"{overlap_pct}% join overlap"
            
            table_result = save_structured_table_to_snowflake(
                schema_name, table_info['name'], table_schemas[key], table_data_dict[key],
                table_info, num_records, status_container, session, overlap_info=overlap_info
            )
            results.append(table_result)
        
        # Create unstructured table(s) - support for 1-2 tables
        unstructured_tables = []
        if 'unstructured' in demo_data['tables']:
            unstructured_tables.append(('unstructured', demo_data['tables']['unstructured']))
        if 'unstructured_2' in demo_data['tables']:
            unstructured_tables.append(('unstructured_2', demo_data['tables']['unstructured_2']))
        
        lang_display = get_language_display_name(language_code)
        
        for unstructured_key, unstructured in unstructured_tables:
            current_step += 1
            # Avoid duplicate _CHUNKS suffix if already present
            base_name = unstructured['name']
            if base_name.endswith('_CHUNKS'):
                table_name = base_name
            else:
                table_name = f"{base_name}_CHUNKS"
            
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Creating {table_name} in {lang_display}...")
            
            chunks_data = generate_unstructured_data(
                unstructured['name'],
                unstructured['description'],
                num_records,
                company_name,
                session,
                error_handler,
                language_code
            )
            
            chunks_df = pd.DataFrame(chunks_data)
            snowpark_chunks_df = session.create_dataframe(chunks_df)
            snowpark_chunks_df.write.mode("overwrite").save_as_table(f"{schema_name}.{table_name}")
            
            # Add sample data preview in expander
            with status_container:
                with st.expander(f"✅ {table_name} created ({len(chunks_data)} chunks, {lang_display})", expanded=False):
                    st.caption(f"**Columns:** CHUNK_ID, DOCUMENT_ID, CHUNK_TEXT, DOCUMENT_TYPE, SOURCE_SYSTEM, LANGUAGE")
                    st.dataframe(chunks_df.head(3), use_container_width=True)
            
            results.append({
                'table': table_name,
                'records': len(chunks_data),
                'description': unstructured['description'],
                'columns': ['CHUNK_ID', 'DOCUMENT_ID', 'CHUNK_TEXT', 'DOCUMENT_TYPE', 'SOURCE_SYSTEM', 'LANGUAGE'],
                'type': 'unstructured',
                'sample_data': chunks_df.head(3).to_dict('records')  # MEMORY OPTIMIZATION: Convert DataFrame to dict
            })
        
        semantic_view_info = None
        if enable_semantic_view:
            current_step += 1
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Creating semantic view...")
            
            # Create semantic view with all structured tables
            if len(structured_tables) >= 2:
                semantic_view_info = create_semantic_view(
                    schema_name, structured_tables, demo_data, company_name, session, error_handler
                )
            else:
                semantic_view_info = None
            
            if semantic_view_info:
                with status_container:
                    with st.expander(f"✅ Semantic view {semantic_view_info['view_name']} created", expanded=False):
                        # Show all tables in the view
                        table_names_str = ", ".join(semantic_view_info.get('table_names', []))
                        st.caption(f"Combining {semantic_view_info.get('num_tables', 2)} tables: {table_names_str}")
                        st.caption(f"Join key: ENTITY_ID across all tables")
                        
                        if semantic_view_info.get('example_queries'):
                            st.caption(f"**Example queries:** {len(semantic_view_info['example_queries'])} pre-configured")
                        
                        # Display the CREATE SQL
                        if semantic_view_info.get('create_sql'):
                            st.caption("\n**SQL Command Used:**")
                            st.code(semantic_view_info['create_sql'], language='sql')
                        
                        # Always display permissions section
                        st.caption("\n**Permissions:**")
                        grant_msgs = semantic_view_info.get('grant_messages', [])
                        if grant_msgs:
                            for msg in grant_msgs:
                                st.caption(msg)
                        else:
                            st.caption("⚠️ No permission grants recorded - this may cause access issues")
                
                results.append({
                    'table': semantic_view_info['view_name'],
                    'records': 'View',
                    'description': f"Semantic view combining {semantic_view_info.get('num_tables', 2)} tables: {table_names_str}",
                    'columns': ['Joined view with all columns from all tables'],
                    'type': 'semantic_view',
                    'example_queries': semantic_view_info['example_queries'],
                    'join_key': semantic_view_info['join_key']
                })
        
        search_service_name = None
        if enable_search_service:
            current_step += 1
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Creating Cortex Search service...")
            
            result = create_cortex_search_service(schema_name, table_name, session, warehouse_name, error_handler, language_code)
            search_service_name = result[0] if isinstance(result, tuple) else result
            grant_msgs = result[1] if isinstance(result, tuple) and len(result) > 1 else []
            
            if search_service_name:
                with status_container:
                    with st.expander(f"✅ Cortex Search service {search_service_name} created", expanded=False):
                        st.caption(f"Semantic search enabled on {table_name}")
                        if grant_msgs:
                            st.caption("\n**Permissions:**")
                            for msg in grant_msgs:
                                st.caption(msg)
                
                results.append({
                    'table': search_service_name,
                    'records': 'Service',
                    'description': f"Cortex Search service for {table_name}",
                    'columns': ['Search service for semantic text search'],
                    'type': 'search_service'
                })
        
        current_step += 1
        progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Generating intelligent questions...")
        
        # Build rich context with full schema and data analysis for better question generation
        rich_table_contexts = []
        for idx, (key, table_info) in enumerate(structured_tables):
            rich_context = build_rich_table_context(
                table_key=key,
                table_info=table_info,
                table_schema=table_schemas[key],
                table_data=table_data_dict[key],
                num_sample_rows=5
            )
            rich_table_contexts.append(rich_context)
        
        questions = generate_contextual_questions(
            demo_data,
            semantic_view_info,
            company_name,
            session,
            error_handler,
            num_questions=12,
            company_url=company_url,
            rich_table_contexts=rich_table_contexts
        )
        
        if questions:
            st.session_state['generated_questions'] = questions
        
        agent_config = None
        # Create AI Agent if requested and at least one tool is available
        if enable_agent and (semantic_view_info or search_service_name):
            current_step += 1
            progress_placeholder.progress(current_step / total_steps, text=f"Step {current_step}/{total_steps}: Creating AI agent in Snowflake Intelligence...")
            
            # Prepare sample questions for agent
            agent_sample_questions = questions if questions else []
            
            # Strip _SEMANTIC_MODEL suffix as agent_automation will add it back
            semantic_view_name_for_agent = None
            # Use full_schema_name (DATABASE.SCHEMA) for agent if available from semantic view
            agent_schema_name = schema_name
            if semantic_view_info:
                semantic_view_name_for_agent = semantic_view_info['view_name'].replace('_SEMANTIC_MODEL', '')
                # Use full schema name if available to ensure database is included
                if 'full_schema_name' in semantic_view_info:
                    agent_schema_name = semantic_view_info['full_schema_name']
            
            agent_config = create_agent_automatically(
                session=session,
                schema_name=agent_schema_name,
                demo_data=demo_data,
                semantic_view_name=semantic_view_name_for_agent,
                search_service_name=search_service_name,
                company_name=company_name,
                warehouse_name=warehouse_name,
                call_cortex_func=call_cortex_with_retry,
                sample_questions=agent_sample_questions
            )
            
            if agent_config:
                # Check if there was an error
                if agent_config.get('error', False):
                    error_msg = agent_config.get('error_message', 'Unknown error')
                    
                    if agent_config.get('setup_required', False):
                        # Infrastructure setup required
                        with status_container:
                            st.error("⚠️ Snowflake Intelligence setup required for agent creation")
                            with st.expander("📋 Setup Instructions", expanded=True):
                                st.markdown("**Run this SQL to set up the infrastructure:**")
                                st.code(error_msg.split('\n', 1)[1] if '\n' in error_msg else error_msg, language='sql')
                                st.info("💡 After running the SQL, regenerate the demo to create the agent")
                    else:
                        # Other error
                        with status_container:
                            st.warning(f"⚠️ Agent creation skipped: {error_msg}")
                else:
                    # Success - agent created
                    with status_container:
                        with st.expander(f"✅ AI Agent {agent_config['name']} created", expanded=False):
                            st.caption(f"**Location:** {agent_config.get('schema', 'SNOWFLAKE_INTELLIGENCE.AGENTS')}")
                            st.caption(f"**Display Name:** {agent_config.get('display_name', agent_config['name'])}")
                            st.caption(f"**Tools:** {', '.join(agent_config.get('tools', []))}")
                            st.caption(f"**Access:** Snowsight → {agent_config.get('ui_path', 'AI & ML » Agents')}")
                            
                            # Display resource configuration
                            if 'grant_results' in agent_config and agent_config['grant_results']:
                                st.caption("\n**Resource Configuration:**")
                                for result in agent_config['grant_results']:
                                    st.caption(result)
                    
                    results.append({
                        'table': agent_config['name'],
                        'records': 'Agent',
                        'description': f"AI agent for {demo_data.get('title', 'demo')} - Access in Snowsight UI",
                        'columns': ['Snowflake Intelligence Agent'],
                        'type': 'agent',
                        'agent_config': agent_config
                    })
                    
                    st.session_state['agent_config'] = agent_config
        
        progress_placeholder.progress(1.0, text="✅ Infrastructure creation complete!")
        
        elapsed_time = time.time() - start_time
        with status_container:
            st.success(f"⏱️ Total generation time: {elapsed_time:.1f} seconds")
            st.info("🎉 Demo Infrastructure Created Successfully! Your demo environment is ready to use.")
        
        # Store validation results in session state for display
        if has_target_questions and validation_results:
            st.session_state['validation_results'] = validation_results
            st.session_state['target_questions_for_display'] = target_questions
        
        # Debug expander for table relationships (if debug mode enabled)
        if st.session_state.get('debug_mode_infrastructure', False):
            with status_container:
                with st.expander("🔍 Debug: Table Relationships & Data Model", expanded=False):
                    # Basic Info Section
                    st.subheader("Basic Information")
                    
                    # Show all tables created
                    st.write("**Tables Created:**")
                    for result in results:
                        if result.get('type') == 'structured':
                            table_type_label = result.get('table_type', 'structured').title()
                            st.write(f"- {result['table']} ({result['records']} records, {table_type_label})")
                    
                    # Show overlap information
                    st.write("\n**Join Relationships:**")
                    st.write("- Primary Key: ENTITY_ID (present in all structured tables)")
                    st.write(f"- Join Overlap: {int(TABLE_JOIN_OVERLAP_PERCENTAGE * 100)}% between tables")
                    
                    # LLM Analysis Section
                    structured_count = len([r for r in results if r.get('type') == 'structured'])
                    if structured_count >= 2 and structured_tables:
                        st.subheader("Data Model Analysis")
                        
                        with st.spinner("Analyzing table relationships..."):
                            relationship_analysis = analyze_table_relationships(
                                structured_tables,
                                demo_data,
                                session,
                                error_handler,
                                target_questions=target_questions if has_target_questions else None
                            )
                        
                        if relationship_analysis:
                            st.write(f"**Model Type:** {relationship_analysis.get('model_type', 'Unknown')}")
                            
                            st.write("\n**Relationships:**")
                            for rel in relationship_analysis.get('relationships', []):
                                st.write(f"- {rel}")
                            
                            if relationship_analysis.get('join_paths'):
                                st.write("\n**Example Join Queries:**")
                                for join_query in relationship_analysis['join_paths']:
                                    st.code(join_query, language='sql')
                            
                            if relationship_analysis.get('question_mapping'):
                                st.write("\n**Question to Join Mapping:**")
                                for mapping in relationship_analysis['question_mapping']:
                                    st.write(f"- {mapping}")
                            
                            # Store in session state for later reference
                            st.session_state['last_relationship_analysis'] = relationship_analysis
                        else:
                            st.warning("Could not generate relationship analysis")
                    
                    # Input Configuration Section
                    st.markdown("---")
                    st.subheader("📝 Input Configuration")
                    
                    input_col1, input_col2 = st.columns(2)
                    with input_col1:
                        st.write("**Company Information:**")
                        st.write(f"- Company Name: {company_name}")
                        if company_url:
                            st.write(f"- Company URL: {company_url}")
                        if st.session_state.get('team_members'):
                            st.write(f"- Team Members: {st.session_state.get('team_members')}")
                        if st.session_state.get('use_cases'):
                            st.write(f"- Use Cases: {st.session_state.get('use_cases')}")
                    
                    with input_col2:
                        st.write("**Generation Settings:**")
                        st.write(f"- Records per Table: {num_records}")
                        st.write(f"- Language: {get_language_display_name(language_code)}")
                        st.write(f"- Advanced Mode: {'Yes' if st.session_state.get('advanced_mode', False) else 'No'}")
                        st.write(f"- Schema Name: {schema_name}")
                    
                    st.write("**Infrastructure Options:**")
                    infra_options = []
                    if enable_semantic_view:
                        infra_options.append("✓ Semantic View (Cortex Analyst)")
                    if enable_search_service:
                        infra_options.append("✓ Search Service (Cortex Search)")
                    if enable_agent:
                        infra_options.append("✓ AI Agent")
                    st.write(", ".join(infra_options) if infra_options else "None")
                    
                    if has_target_questions and target_questions:
                        st.write(f"\n**Target Questions ({len(target_questions)}):**")
                        for i, q in enumerate(target_questions, 1):
                            st.write(f"{i}. {q}")
                    
                    # Demo Selection Section
                    st.markdown("---")
                    st.subheader("🎯 Demo Scenario Selected")
                    st.write(f"**Title:** {demo_data.get('title', 'Unknown')}")
                    st.write(f"**Description:** {demo_data.get('description', 'N/A')}")
                    if demo_data.get('industry_focus'):
                        st.write(f"**Industry Focus:** {demo_data.get('industry_focus')}")
                    if demo_data.get('business_value'):
                        st.write(f"**Business Value:** {demo_data.get('business_value')}")
                    
                    # Performance Metrics Section
                    st.markdown("---")
                    display_performance_summary()
        
        return results
        
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.DATA_GENERATION_FAILED,
            error_type=type(e).__name__,
            severity=ErrorSeverity.ERROR,
            message=f"Error creating tables: {str(e)}",
            stack_trace=traceback.format_exc()
        )
        st.error(f"Error creating tables: {str(e)}")
        return None

# Show main page header only if not viewing history or about page
if not st.session_state.get('show_history', False) and not st.session_state.get('show_about', False):
    st.markdown("""
        <div class='page-header'>
            <h1 class='page-title'>
                ❄️ Snowflake Intelligence Data Generator
            </h1>
            <p class='page-subtitle'>
                Generate tailored demo data infrastructure for Cortex Analyst, Cortex Search, and Snowflake Intelligence
            </p>
        </div>
    """, unsafe_allow_html=True)

with st.sidebar:

    st.markdown("### 📚 Resources")
    st.markdown("- [Cortex Analyst Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)")
    st.markdown("- [Cortex Search Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview)")
    st.markdown("- [Snowflake Intelligence Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/snowflake-intelligence)")
    st.markdown("---")
    # st.markdown("### ⚙️ Configuration")
    
    # st.markdown(f"""
    # <div class='info-box' style='font-size: 0.9rem;'>
    #     <strong>Account:</strong> {config['account']}<br>
    #     <strong>User:</strong> {config['user']}<br>
    #     <strong>Role:</strong> {config['runtime']['role']}<br>
    #     <strong>Warehouse:</strong> {warehouse_name}<br>
    #     <strong>Database:</strong> {config['runtime']['database']}
    # </div>
    # """, unsafe_allow_html=True)
    
    if st.button("📖 About", use_container_width=True):
        st.session_state['show_about'] = not st.session_state.get('show_about', False)
    
    if st.button("📜 View History", use_container_width=True):
        st.session_state['show_history'] = not st.session_state.get('show_history', False)

    
    
 

    


# Show About page if requested
if st.session_state.get('show_about', False):
    # Back button at the top
    if st.button("← Back to Main Page", type="secondary"):
        st.session_state['show_about'] = False
        st.rerun()
    
    st.markdown("## 📖 About Snowflake Intelligence Data Generator")
    st.markdown("---")
    
    # Overview section
    render_about_hero(
        "🎯 Purpose",
        "The Snowflake Intelligence Data Generator is an enterprise-grade tool that automatically creates "
        "production-ready demo environments showcasing Snowflake's AI and analytics capabilities. Perfect "
        "for Solutions Engineers, Account Executives, and data professionals who need to quickly demonstrate "
        "the power of Snowflake Cortex Analyst, Cortex Search, and Snowflake Intelligence to customers."
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### ✨ Key Features
        
        #### 🤖 AI-Powered Generation
        - **Context-Aware Demo Scenarios** - Analyzes company URLs to generate industry-specific demos
        - **Claude 4 Sonnet Integration** - Uses latest LLM for high-quality, realistic content
        - **Intelligent Question Generation** - Creates 12+ questions across basic, intermediate, and advanced levels
        - **Multi-Language Support** - Generate content in English, Spanish, French, German, Japanese, and Chinese
        
        #### 🏗️ Infrastructure Automation
        - **Semantic Views** - Automatically creates Cortex Analyst-ready semantic views with facts and dimensions
        - **Cortex Search Services** - Sets up search services for unstructured content with full-text indexing
        - **Snowflake Intelligence Agents** - Automated agent creation with tool orchestration
        - **Smart Join Keys** - 70% overlap between tables for realistic data relationships
        - **Unique Schema Naming** - Timestamp-based naming prevents conflicts across multiple demos
        
        #### 📊 Data Quality
        - **LLM-Generated Schemas** - Context-aware column definitions with realistic sample values
        - **Business-Realistic Data** - No generic "Type_A" values - all data is contextually appropriate
        - **Time-Series Ready** - Recent timestamps (past 7 days) for "last 24 hours" type queries
        - **Join-Ready Tables** - PRIMARY KEY constraints and proper foreign key relationships
        
        #### 🎨 User Experience
        - **4-Step Guided Flow** - Intuitive wizard: Customer Info → Select Demo → Configure → Generate
        - **Real-Time Progress Tracking** - Visual progress indicators with detailed status updates
        - **Flexible Configuration** - Choose which components to create (semantic views, search services, agents)
        - **Direct Agent Access** - One-click button to launch Snowflake Intelligence UI
        - **Comprehensive Documentation** - Built-in resources and links to Snowflake docs
        
        #### 🛡️ Enterprise Features
        - **Robust Error Handling** - Comprehensive retry logic with exponential backoff
        - **Session Management** - Smart handling of token expiration and reconnection
        - **Permission Automation** - Automatically grants necessary permissions to roles and agents
        - **Deployment Flexibility** - Runs locally, in Streamlit in Snowflake, or as a Native App
        
        ### 🚀 How It Works
        
        **Step 1: Customer Information**
        - Enter company URL, audience, and specific use cases
        - System analyzes URL to determine industry and business context
        
        **Step 2: Select Demo Scenario**
        - AI generates 3 tailored demo scenarios using Claude 4 Sonnet
        - Each demo includes detailed descriptions, business value, and table structures
        - Fallback to high-quality templates if LLM is unavailable
        
        **Step 3: Configure Infrastructure**
        - Choose number of records (20 - 10,000)
        - Select content language (6 languages supported)
        - Enable/disable semantic views, search services, and AI agents
        - Customize schema name with unique timestamp
        
        **Step 4: Generate**
        - Automated infrastructure creation with progress tracking
        - Real-time status updates for each component
        - Generated questions organized by difficulty and system
        - Direct access to Snowflake Intelligence UI
        
        ### 💼 Use Cases
        
        - **Sales Demonstrations** - Quickly create industry-specific demos for prospect meetings
        - **POCs and Evaluations** - Generate production-like environments for customer testing
        - **Training and Enablement** - Create learning environments for team training
        - **Testing and Development** - Generate test data for Cortex feature development
        - **Customer Workshops** - Build hands-on demo environments for workshop sessions
        """)
    
    with col2:
        st.markdown("### 📦 What Gets Created")
        st.markdown("""
        **Data Tables:**
        - 2 structured tables (100-10K records)
        - 1 unstructured table (text chunks)
        - PRIMARY KEY constraints
        - 70% join overlap
        
        **AI Services:**
        - Semantic View (optional)
        - Cortex Search Service (optional)
        - Snowflake Intelligence Agent (optional)
        
        **Questions & Insights:**
        - 12+ AI-generated questions
        - Basic, Intermediate, Advanced levels
        - Cortex Analyst queries
        - Cortex Search queries
        - Intelligence Agent prompts
        
        **Organization:**
        - Isolated schemas (timestamp-based)
        - Proper permissions configured
        - Full documentation included
        - Ready for immediate demo
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 🔧 Technical Stack")
        st.markdown("""
        **AI/ML:**
        - Claude 4 Sonnet (LLM)
        - Cortex Analyst
        - Cortex Search
        - Snowflake Intelligence
        
        **Framework:**
        - Streamlit (UI)
        - Snowpark (Data)
        
        **Deployment:**
        - Local development
        - Streamlit in Snowflake
        - Native App ready
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("### 📚 Resources")
        st.markdown("""
        - [Cortex Analyst Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst)
        - [Cortex Search Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search)
        - [Snowflake Intelligence Docs](https://docs.snowflake.com/en/user-guide/snowflake-cortex/snowflake-intelligence)
        """)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_page_footer("Powered by ❄️ Snowflake")
    
    st.stop()

# Show History page if requested
if st.session_state.get('show_history', False):
    # Back button at the top
    if st.button("← Back to Main Page", type="secondary"):
        st.session_state['show_history'] = False
        st.rerun()
    
    # History page header (replaces main header)
    st.markdown("""
        <div class='page-header'>
            <h1 class='page-title'>
                📜 Demo Generation History
            </h1>
            <p class='page-subtitle'>
                View and re-use your previously generated demo configurations
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch all history records for selector
    with st.spinner("Loading history..."):
        all_history = get_history_records(session, limit=1000, offset=0)
    
    if not all_history:
        st.markdown("""
        <div class='history-empty'>
            <div class='history-empty-icon'>📭</div>
            <h3>No History Yet</h3>
            <p>Generate your first demo to start tracking your history!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # History selector at the top
        # Create options for selectbox
        history_options = {}
        for record in all_history:
            created_at_str = record['created_at'].strftime("%b %d, %Y %I:%M %p") if record['created_at'] else "Unknown"
            label = f"{record['company_name']} - {record['demo_title'][:40]} ({created_at_str})"
            history_options[label] = record
        
        # Initialize selected history if not set
        if 'selected_history_idx' not in st.session_state:
            st.session_state['selected_history_idx'] = 0
        
        selected_label = st.selectbox(
            "Select a demo generation to view:",
            options=list(history_options.keys()),
            index=st.session_state.get('selected_history_idx', 0),
            key="history_selector"
        )
        
        selected_record = history_options[selected_label]
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Display selected history record in main page format
        created_at_str = selected_record['created_at'].strftime("%B %d, %Y at %I:%M %p") if selected_record['created_at'] else "Unknown"
        
        # Header with timestamp
        st.markdown(f"""
        <div class='success-box'>
            <h2 style='margin: 0; color: white;'>🎯 {selected_record['company_name']} Demo: {selected_record['demo_title']}</h2>
            <p style='margin-top: 0.5rem; color: rgba(255,255,255,0.9);'>Generated: {created_at_str}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 1: Customer Information
        st.markdown("<div class='step-container'>Step 1: Customer Information</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Company URL", value=selected_record.get('company_url', ''), disabled=True, key="hist_url")
            st.text_input("Team Members / Audience", value=selected_record.get('team_members', ''), disabled=True, key="hist_team")
            st.number_input("Records per Table", value=selected_record.get('num_records', 100), disabled=True, key="hist_records")
        
        with col2:
            st.text_area("Specific Use Cases", value=selected_record.get('use_cases', ''), disabled=True, height=120, key="hist_cases")
            lang_display = get_language_display_name(selected_record.get('language_code', 'en'))
            st.text_input("Content Language", value=lang_display, disabled=True, key="hist_lang")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Advanced mode is now always on, so we don't need to show this badge anymore
        # if selected_record.get('advanced_mode'):
        #     st.info("⚡ Advanced Mode was enabled (3-5 tables with richer relationships)")
        
        # Show target questions if any
        if selected_record.get('target_questions'):
            with st.expander("🎯 Target Questions", expanded=False):
                for i, q in enumerate(selected_record['target_questions'], 1):
                    st.markdown(f"**{i}.** {q}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 2: Demo Scenario
        st.markdown("<div class='step-container'>Step 2: Demo Scenario Selected</div>", unsafe_allow_html=True)
        
        demo_data = selected_record.get('demo_data_json', {})
        if demo_data:
            with st.container(border=True):
                st.subheader(demo_data.get('title', 'Demo'))
                st.write(demo_data.get('description', ''))
                
                if demo_data.get('industry_focus'):
                    st.info(f"🏭 **Industry Focus:** {demo_data['industry_focus']}")
                
                if demo_data.get('business_value'):
                    st.info(f"💼 **Business Value:** {demo_data['business_value']}")
                
                if demo_data.get('target_audience'):
                    st.info(f"👥 {demo_data['target_audience']}")
                
                if demo_data.get('customization'):
                    st.info(f"🎯 {demo_data['customization']}")
                
                st.write("**📊 Data Tables:**")
                
                # First row: structured_1, structured_2, unstructured
                if demo_data.get('tables'):
                    tables = demo_data['tables']
                    
                    # Check if standard mode (3 structured + 1 unstructured) for 2-column layout
                    has_structured_3 = 'structured_3' in tables
                    has_structured_4 = 'structured_4' in tables
                    has_structured_5 = 'structured_5' in tables
                    has_unstructured_2 = 'unstructured_2' in tables
                    
                    # Standard mode: exactly 3 structured tables, use 2 columns x 2 rows
                    is_standard_mode = has_structured_3 and not has_structured_4 and not has_structured_5 and not has_unstructured_2
                    
                    if is_standard_mode:
                        # Row 1: structured_1, structured_2
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'structured_1' in tables:
                                st.write("**Structured Table 1**")
                                st.write(f"🏷️ **{tables['structured_1']['name']}**")
                                st.caption(tables['structured_1']['description'])
                                if 'purpose' in tables['structured_1']:
                                    st.caption(f"💡 {tables['structured_1']['purpose']}")
                                if 'table_type' in tables['structured_1']:
                                    st.caption(f"📁 Type: {tables['structured_1']['table_type'].title()}")
                        
                        with col2:
                            if 'structured_2' in tables:
                                st.write("**Structured Table 2**")
                                st.write(f"🏷️ **{tables['structured_2']['name']}**")
                                st.caption(tables['structured_2']['description'])
                                if 'purpose' in tables['structured_2']:
                                    st.caption(f"💡 {tables['structured_2']['purpose']}")
                                if 'table_type' in tables['structured_2']:
                                    st.caption(f"📁 Type: {tables['structured_2']['table_type'].title()}")
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Row 2: structured_3, unstructured
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            if 'structured_3' in tables:
                                st.write("**Structured Table 3**")
                                st.write(f"🏷️ **{tables['structured_3']['name']}**")
                                st.caption(tables['structured_3']['description'])
                                if 'purpose' in tables['structured_3']:
                                    st.caption(f"💡 {tables['structured_3']['purpose']}")
                                if 'table_type' in tables['structured_3']:
                                    st.caption(f"📁 Type: {tables['structured_3']['table_type'].title()}")
                        
                        with col4:
                            if 'unstructured' in tables:
                                st.write("**Unstructured Table**")
                                st.write(f"🏷️ **{tables['unstructured']['name']}**")
                                st.caption(tables['unstructured']['description'])
                                if 'purpose' in tables['unstructured']:
                                    st.caption(f"💡 {tables['unstructured']['purpose']}")
                    
                    else:
                        # Advanced mode: 3 columns for first row
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if 'structured_1' in tables:
                                st.write("**Structured Table 1**")
                                st.write(f"🏷️ **{tables['structured_1']['name']}**")
                                st.caption(tables['structured_1']['description'])
                                if 'purpose' in tables['structured_1']:
                                    st.caption(f"💡 {tables['structured_1']['purpose']}")
                                if 'table_type' in tables['structured_1']:
                                    st.caption(f"📁 Type: {tables['structured_1']['table_type'].title()}")
                        
                        with col2:
                            if 'structured_2' in tables:
                                st.write("**Structured Table 2**")
                                st.write(f"🏷️ **{tables['structured_2']['name']}**")
                                st.caption(tables['structured_2']['description'])
                                if 'purpose' in tables['structured_2']:
                                    st.caption(f"💡 {tables['structured_2']['purpose']}")
                                if 'table_type' in tables['structured_2']:
                                    st.caption(f"📁 Type: {tables['structured_2']['table_type'].title()}")
                        
                        with col3:
                            if 'unstructured' in tables:
                                st.write("**Unstructured Table**")
                                st.write(f"🏷️ **{tables['unstructured']['name']}**")
                                st.caption(tables['unstructured']['description'])
                                if 'purpose' in tables['unstructured']:
                                    st.caption(f"💡 {tables['unstructured']['purpose']}")
                        
                        # Check if there are additional tables (advanced mode)
                        additional_tables = []
                        for i in range(3, 6):  # Check for structured_3, structured_4, structured_5
                            table_key = f'structured_{i}'
                            if table_key in tables:
                                additional_tables.append((f"Structured Table {i}", table_key, tables[table_key], True))
                        
                        # Check for second unstructured table
                        if 'unstructured_2' in tables:
                            additional_tables.append(("Unstructured Table 2", "unstructured_2", tables['unstructured_2'], False))
                        
                        # Display additional tables in a second row if they exist
                        if additional_tables:
                            st.markdown("<br>", unsafe_allow_html=True)
                            
                            # Create columns based on number of additional tables
                            num_additional = len(additional_tables)
                            if num_additional == 1:
                                col_additional = st.columns([1, 2, 1])
                                cols = [col_additional[1]]
                            elif num_additional == 2:
                                cols = st.columns(2)
                            else:
                                cols = st.columns(3)
                            
                            for col_idx, (table_label, table_key, table_info, is_structured) in enumerate(additional_tables):
                                if col_idx < len(cols):  # Ensure we don't exceed column count
                                    with cols[col_idx]:
                                        st.write(f"**{table_label}**")
                                        st.write(f"🏷️ **{table_info['name']}**")
                                        st.caption(table_info['description'])
                                        if 'purpose' in table_info:
                                            st.caption(f"💡 {table_info['purpose']}")
                                        if is_structured and 'table_type' in table_info:
                                            st.caption(f"📁 Type: {table_info['table_type'].title()}")
                
                st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 3: Configuration
        st.markdown("<div class='step-container'>Step 3: Infrastructure Configuration</div>", unsafe_allow_html=True)
        
        st.text_input("Schema Name", value=selected_record.get('schema_name', ''), disabled=True, key="hist_schema")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.checkbox("📊 Semantic View", value=selected_record.get('enable_semantic_view', False), disabled=True, key="hist_sem")
        with col2:
            st.checkbox("🔍 Cortex Search Service", value=selected_record.get('enable_search_service', False), disabled=True, key="hist_search")
        with col3:
            st.checkbox("🤖 AI Agent", value=selected_record.get('enable_agent', False), disabled=True, key="hist_agent")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 4: Results Summary
        st.markdown("<div class='step-container'>Step 4: Generated Infrastructure</div>", unsafe_allow_html=True)
        st.markdown("## 📊 Infrastructure Created")
        
        # Infrastructure summary
        table_names = selected_record.get('table_names', [])
        structured_count = sum(1 for t in table_names if not any(x in t.upper() for x in ['CHUNKS', 'SEARCH', 'AGENT', 'SEMANTIC_MODEL']))
        unstructured_count = sum(1 for t in table_names if 'CHUNKS' in t.upper())
        has_semantic = selected_record.get('enable_semantic_view') and any('SEMANTIC_MODEL' in t.upper() for t in table_names)
        has_search = selected_record.get('enable_search_service') and any('SEARCH' in t.upper() for t in table_names)
        has_agent = selected_record.get('enable_agent') and any('AGENT' in t.upper() for t in table_names)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Build semantic view stat HTML
            semantic_stat = f"""
            <div class='infra-stat'>
                <div class='infra-stat-icon'>🔗</div>
                <div class='infra-stat-content'>
                    <div class='infra-stat-title'>1 Semantic View</div>
                    <div class='infra-stat-desc'>AI-ready data relationships</div>
                </div>
            </div>
            """ if has_semantic else ""
            
            # Build search service stat HTML
            search_stat = f"""
            <div class='infra-stat'>
                <div class='infra-stat-icon'>🔍</div>
                <div class='infra-stat-content'>
                    <div class='infra-stat-title'>1 Cortex Search Service</div>
                    <div class='infra-stat-desc'>Intelligent document retrieval</div>
                </div>
            </div>
            """ if has_search else ""
            
            # Build agent stat HTML
            agent_stat = f"""
            <div class='infra-stat'>
                <div class='infra-stat-icon'>🤖</div>
                <div class='infra-stat-content'>
                    <div class='infra-stat-title'>1 AI Agent</div>
                    <div class='infra-stat-desc'>Automated tools and capabilities</div>
                </div>
            </div>
            """ if has_agent else ""
            
            st.markdown(f"""
            <div class='infra-card'>
                <h3>🏗️ Infrastructure Summary</h3>
                <div class='infra-stat'>
                    <div class='infra-stat-icon'>📊</div>
                    <div class='infra-stat-content'>
                        <div class='infra-stat-title'>{structured_count} Structured Tables</div>
                        <div class='infra-stat-desc'>{selected_record.get('num_records', 0):,} records with ENTITY_ID PRIMARY KEY and 70% join overlap</div>
                    </div>
                </div>
                <div class='infra-stat'>
                    <div class='infra-stat-icon'>📄</div>
                    <div class='infra-stat-content'>
                        <div class='infra-stat-title'>{unstructured_count} Unstructured Table</div>
                        <div class='infra-stat-desc'>Text chunks for semantic search</div>
                    </div>
                </div>
                {semantic_stat}
                {search_stat}
                {agent_stat}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Create mock results structure for render_results_table_list
            tables_for_render = []
            for table_name in table_names:
                if not any(x in table_name.upper() for x in ['SEARCH_SERVICE', 'AGENT', 'SEMANTIC_MODEL']):
                    table_type = 'unstructured' if 'CHUNKS' in table_name.upper() else 'structured'
                    tables_for_render.append({
                        'table': table_name,
                        'type': table_type,
                        'description': ''
                    })
            
            tables_html = render_results_table_list(tables_for_render)
            st.markdown(f"""
            <div class='infra-card'>
                <h3>📋 Tables Created</h3>
                {tables_html}
            </div>
            """, unsafe_allow_html=True)
        
        # Generated Questions
        if selected_record.get('generated_questions'):
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📋 Generated Questions")
            
            with st.expander("View All Generated Questions", expanded=False):
                for i, q in enumerate(selected_record['generated_questions'], 1):
                    st.markdown(f"{i}. {q}")
        
        # Action buttons at the bottom
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Center the button
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            if st.button("🔄 Re-use This Configuration", use_container_width=True, type="primary"):
                load_configuration_from_history(selected_record)
                st.session_state['show_history'] = False
                st.success("✅ Configuration loaded! Redirecting to main page...")
                time.sleep(1)
                st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    render_page_footer("Powered by ❄️ Snowflake")
    
    st.stop()

# Main content area
current_step = 1
if 'demo_ideas' in st.session_state and st.session_state['demo_ideas']:
    current_step = 2
if 'selected_demo' in st.session_state:
    current_step = 3
    # If infrastructure creation has started, move to step 4
    if 'infrastructure_started' in st.session_state and st.session_state['infrastructure_started']:
        current_step = 4

show_step_progress(current_step)

st.markdown("<div class='step-container'>Step 1: Customer Information</div>", unsafe_allow_html=True)

# Show notification if configuration was loaded from history
if st.session_state.get('config_loaded_from_history', False):
    st.info(f"ℹ️ Configuration loaded from history (ID: {st.session_state.get('loaded_history_id', 'unknown')[:8]}...)")
    # Clear the flag after showing the message once
    st.session_state['config_loaded_from_history'] = False

col1, col2 = st.columns(2)

with col1:
    company_url = st.text_input(
        "Company URL",
        placeholder="https://example.com",
        help="Customer's website URL",
        key="company_url"
    )
    team_members = st.text_input(
        "Team Members / Audience",
        placeholder="e.g., CTO, Data Team, Sales Director",
        help="Who will be attending the demo",
        key="team_members"
    )
    num_records = st.number_input(
        "Records per Table",
        min_value=20,
        max_value=10000,
        value=1000,
        step=10,
        help="Number of records to generate per table",
        key="num_records"
    )

with col2:
    use_cases = st.text_area(
        "Specific Use Cases (Optional)",
        placeholder="e.g., Customer 360, Risk Analytics, etc.",
        help="Specific requirements or use cases for the demo",
        height=120,
        key="use_cases"
    )
    
    language_code = st.selectbox(
        "Content Language",
        options=list(SUPPORTED_LANGUAGES.keys()),
        format_func=lambda x: get_language_display_name(x),
        help="Select the language for generated text content",
        key="content_language"
    )

# Advanced Mode - Now default (hidden checkbox)
st.markdown("<br>", unsafe_allow_html=True)
# Advanced mode is now always enabled by default
# Keeping the checkbox code commented for future reference if needed
# advanced_mode = st.checkbox(
#     "🚀 Advanced Mode (Generate 3-5 tables with richer relationships)",
#     value=True,
#     help="Standard mode: 3 structured tables with star schema. Advanced mode: 3-5 structured tables with star/snowflake schema for more complex analytics.",
#     key="advanced_mode"
# )
advanced_mode = True  # Always use advanced mode
st.session_state['advanced_mode'] = True  # Store in session state for consistency

# if advanced_mode:
#     st.info("💡 Advanced Mode will generate 3-5 structured tables plus 1-2 unstructured tables with fact and dimension tables forming a star/snowflake schema for more complex analytics.")

# Target Questions Section (Optional - Advanced)
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("🎯 Target Questions (Optional - Advanced)", expanded=False):
    st.markdown("""
    **Question-Driven Data Generation**: Provide specific questions you want the demo to answer. 
    The AI will design demo scenarios and generate data structures to support these questions.
    
    Examples:
    - "What percentage of users are in the 25-34 age range?"
    - "Which products had the highest sales growth in Q4?"
    - "What are the top 5 customer segments by revenue?"
    """)
    
    # Initialize target_questions in session state if not exists
    if 'target_questions' not in st.session_state:
        st.session_state['target_questions'] = []
    
    # Input for new question
    new_question = st.text_area(
        "Enter a question you want the demo to answer:",
        placeholder="e.g., What percentage of customers are in the high-value segment?",
        height=80,
        key="new_target_question_input"
    )
    
    col_add, col_spacer = st.columns([1, 3])
    with col_add:
        if st.button("➕ Add Question", disabled=not new_question or not new_question.strip()):
            if new_question and new_question.strip():
                st.session_state['target_questions'].append(new_question.strip())
                # Clear the input by using a rerun
                st.rerun()
    
    # Display existing questions
    if st.session_state['target_questions']:
        st.markdown("---")
        st.markdown("**Added Questions:**")
        for idx, question in enumerate(st.session_state['target_questions']):
            col_q, col_btn = st.columns([5, 1])
            with col_q:
                st.markdown(f"**{idx + 1}.** {question}")
            with col_btn:
                if st.button("🗑️", key=f"remove_question_{idx}", help="Remove this question"):
                    st.session_state['target_questions'].pop(idx)
                    st.rerun()
        
        st.info(f"💡 {len(st.session_state['target_questions'])} question(s) will guide the data generation")

# Left-aligned button
generate_btn = st.button(
    "🤖 Generate Demo Ideas",
    type="primary",
    disabled=not (company_url and team_members)
)

if generate_btn:
    # Clear cache for demo ideas to generate fresh ideas each time button is clicked
    generate_demo_ideas_with_cortex.clear()
    
    if not company_url or not team_members:
        st.error("⚠️ Please provide company URL and team members")
    else:
        # Show loading indicator in info box (CSS animation keeps spinning during long operations)
        progress_placeholder = st.empty()
        progress_placeholder.markdown("""
        <div class='info-box' style='display: flex; align-items: center; gap: 12px;'>
            <div style='display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-top: 3px solid white; border-radius: 50%; animation: spin 1s linear infinite;'></div>
            <span>🤖 AI is analyzing customer context and generating tailored demo ideas...</span>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """, unsafe_allow_html=True)
        
        company_info = get_company_info_from_url(company_url)
        company_name = company_info['name']
        
        st.session_state['company_name'] = company_name
        st.session_state['company_info'] = company_info
        
        is_cortex_available = check_cortex_availability(session)
        
        # Get target questions and advanced mode from session state
        target_questions = st.session_state.get('target_questions', [])
        advanced_mode = st.session_state.get('advanced_mode', False)
        
        if is_cortex_available:
            demo_ideas = generate_demo_ideas_with_cortex(
                company_name, team_members, use_cases, session, error_handler,
                num_ideas=3, target_questions=target_questions,
                advanced_mode=advanced_mode
            )
            if demo_ideas and len(demo_ideas) > 0:
                st.session_state['used_fallback_demos'] = False
        else:
            user_msg = error_handler.get_user_friendly_message(ErrorCode.CORTEX_UNAVAILABLE)
            demo_ideas = None
        
        if not demo_ideas or len(demo_ideas) == 0:
            st.info("Using template demo scenarios")
            st.session_state['used_fallback_demos'] = True
            demo_ideas = get_fallback_demo_ideas(company_name, team_members, use_cases, target_questions=target_questions, advanced_mode=advanced_mode)
        
        st.session_state['demo_ideas'] = demo_ideas
        progress_placeholder.empty()
        st.rerun()

if 'demo_ideas' in st.session_state and st.session_state['demo_ideas']:
    template_label = " (template generated)" if st.session_state.get('used_fallback_demos', False) else ""
    st.markdown(f"<br><div class='step-container'>Step 2: Select Your Demo Scenario{template_label}</div>", unsafe_allow_html=True)
    
    demo_ideas = st.session_state['demo_ideas']
    
    # Determine if we should show target indicator
    has_target_questions = bool(st.session_state.get('target_questions', []))
    using_cortex_demos = not st.session_state.get('used_fallback_demos', False)
    show_target_indicator = has_target_questions and using_cortex_demos
    
    # Create tab labels
    tab_labels = []
    for i, demo in enumerate(demo_ideas):
        if i == 0 and show_target_indicator:
            tab_labels.append(f"🎯 Target : Demo {i+1}: {demo['title'].split(':')[0]}")
        else:
            tab_labels.append(f"Demo {i+1}: {demo['title'].split(':')[0]}")
    
    # Create tabs for each demo idea
    tabs = st.tabs(tab_labels)
    
    for idx, (tab, demo) in enumerate(zip(tabs, demo_ideas)):
        with tab:
            # Wrap entire demo content in a bordered container (card)
            with st.container(border=True):
                st.subheader(demo['title'])
                st.write(demo['description'])
                
                # Industry focus
                if 'industry_focus' in demo:
                    st.info(f"🏭 **Industry Focus:** {demo['industry_focus']}")
                
                # Business value
                if 'business_value' in demo:
                    st.info(f"💼 **Business Value:** {demo['business_value']}")
                
                # Target audience
                if 'target_audience' in demo:
                    st.info(f"👥 {demo['target_audience']}")
                
                # Customization
                if 'customization' in demo:
                    st.info(f"🎯 {demo['customization']}")
                
                st.write("**📊 Data Tables:**")
                
                # Check if standard mode (3 structured + 1 unstructured) for 2-column layout
                has_structured_3 = 'structured_3' in demo['tables']
                has_structured_4 = 'structured_4' in demo['tables']
                has_structured_5 = 'structured_5' in demo['tables']
                has_unstructured_2 = 'unstructured_2' in demo['tables']
                
                # Standard mode: exactly 3 structured tables, use 2 columns x 2 rows
                is_standard_mode = has_structured_3 and not has_structured_4 and not has_structured_5 and not has_unstructured_2
                
                if is_standard_mode:
                    # Row 1: structured_1, structured_2
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Structured Table 1**")
                        st.write(f"🏷️ **{demo['tables']['structured_1']['name']}**")
                        st.caption(demo['tables']['structured_1']['description'])
                        if 'purpose' in demo['tables']['structured_1']:
                            st.caption(f"💡 {demo['tables']['structured_1']['purpose']}")
                        if 'table_type' in demo['tables']['structured_1']:
                            st.caption(f"📁 Type: {demo['tables']['structured_1']['table_type'].title()}")
                    
                    with col2:
                        st.write("**Structured Table 2**")
                        st.write(f"🏷️ **{demo['tables']['structured_2']['name']}**")
                        st.caption(demo['tables']['structured_2']['description'])
                        if 'purpose' in demo['tables']['structured_2']:
                            st.caption(f"💡 {demo['tables']['structured_2']['purpose']}")
                        if 'table_type' in demo['tables']['structured_2']:
                            st.caption(f"📁 Type: {demo['tables']['structured_2']['table_type'].title()}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Row 2: structured_3, unstructured
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        st.write("**Structured Table 3**")
                        st.write(f"🏷️ **{demo['tables']['structured_3']['name']}**")
                        st.caption(demo['tables']['structured_3']['description'])
                        if 'purpose' in demo['tables']['structured_3']:
                            st.caption(f"💡 {demo['tables']['structured_3']['purpose']}")
                        if 'table_type' in demo['tables']['structured_3']:
                            st.caption(f"📁 Type: {demo['tables']['structured_3']['table_type'].title()}")
                    
                    with col4:
                        st.write("**Unstructured Table**")
                        st.write(f"🏷️ **{demo['tables']['unstructured']['name']}**")
                        st.caption(demo['tables']['unstructured']['description'])
                        if 'purpose' in demo['tables']['unstructured']:
                            st.caption(f"💡 {demo['tables']['unstructured']['purpose']}")
                
                else:
                    # Advanced mode: 3 columns for first row
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write("**Structured Table 1**")
                        st.write(f"🏷️ **{demo['tables']['structured_1']['name']}**")
                        st.caption(demo['tables']['structured_1']['description'])
                        if 'purpose' in demo['tables']['structured_1']:
                            st.caption(f"💡 {demo['tables']['structured_1']['purpose']}")
                        if 'table_type' in demo['tables']['structured_1']:
                            st.caption(f"📁 Type: {demo['tables']['structured_1']['table_type'].title()}")
                    
                    with col2:
                        st.write("**Structured Table 2**")
                        st.write(f"🏷️ **{demo['tables']['structured_2']['name']}**")
                        st.caption(demo['tables']['structured_2']['description'])
                        if 'purpose' in demo['tables']['structured_2']:
                            st.caption(f"💡 {demo['tables']['structured_2']['purpose']}")
                        if 'table_type' in demo['tables']['structured_2']:
                            st.caption(f"📁 Type: {demo['tables']['structured_2']['table_type'].title()}")
                    
                    with col3:
                        st.write("**Unstructured Table**")
                        st.write(f"🏷️ **{demo['tables']['unstructured']['name']}**")
                        st.caption(demo['tables']['unstructured']['description'])
                        if 'purpose' in demo['tables']['unstructured']:
                            st.caption(f"💡 {demo['tables']['unstructured']['purpose']}")
                    
                    # Check if there are additional tables (advanced mode)
                    additional_tables = []
                    for i in range(3, 6):  # Check for structured_3, structured_4, structured_5
                        table_key = f'structured_{i}'
                        if table_key in demo['tables']:
                            additional_tables.append((f"Structured Table {i}", table_key, demo['tables'][table_key], True))
                    
                    # Check for second unstructured table
                    if 'unstructured_2' in demo['tables']:
                        additional_tables.append(("Unstructured Table 2", "unstructured_2", demo['tables']['unstructured_2'], False))
                    
                    # Display additional tables in a second row if they exist
                    if additional_tables:
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # Create columns based on number of additional tables
                        num_additional = len(additional_tables)
                        if num_additional == 1:
                            col_additional = st.columns([1, 2, 1])
                            cols = [col_additional[1]]
                        elif num_additional == 2:
                            cols = st.columns(2)
                        else:
                            cols = st.columns(3)
                        
                        for col_idx, (table_label, table_key, table_info, is_structured) in enumerate(additional_tables):
                            if col_idx < len(cols):  # Ensure we don't exceed column count
                                with cols[col_idx]:
                                    st.write(f"**{table_label}**")
                                    st.write(f"🏷️ **{table_info['name']}**")
                                    st.caption(table_info['description'])
                                    if 'purpose' in table_info:
                                        st.caption(f"💡 {table_info['purpose']}")
                                    if is_structured and 'table_type' in table_info:
                                        st.caption(f"📁 Type: {table_info['table_type'].title()}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
            # Select button for this demo
            if st.button(f"🚀 Select Demo {idx+1}", key=f"select_demo_{idx}", type="primary"):
                st.session_state['selected_demo_idx'] = idx
                st.session_state['selected_demo'] = demo
                # Reset infrastructure flags when selecting a new demo
                st.session_state['infrastructure_started'] = False
                st.session_state['infrastructure_complete'] = False
                st.success(f"✅ Selected: {demo['title']}")
                st.rerun()

if 'selected_demo' in st.session_state:
    st.markdown("<br><div class='step-container'>Step 3: Create Demo Infrastructure</div>", unsafe_allow_html=True)
    
    selected_demo = st.session_state['selected_demo']
    company_name = st.session_state.get('company_name', 'DEMO')
    
    render_selection_box(
        f"<strong>Selected:</strong> {selected_demo['title']} ({selected_demo.get('industry_focus', selected_demo.get('industry', 'Business Intelligence'))})"
    )
    
    # Create unique schema name with date and time to avoid conflicts
    unique_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    schema_name = st.text_input(
        "Schema Name",
        value=f"{company_name}_DEMO_{unique_timestamp}",
        help="Name for the database schema (includes timestamp for uniqueness)",
        key="schema_name"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_semantic_view = st.checkbox(
            "📊 Create Semantic View",
            value=True,
            help="Create a semantic view with Cortex Analyst extension",
            key="enable_semantic"
        )
        if enable_semantic_view:
            st.caption("✨ Enables advanced join queries and relationships for Cortex Analyst")
    
    with col2:
        enable_search_service = st.checkbox(
            "🔍 Create Cortex Search Service",
            value=True,
            help="Create Cortex Search service for unstructured data",
            key="enable_search"
        )
        if enable_search_service:
            st.caption("✨ Enables semantic search on text content with Cortex Search")
    
    with col3:
        # Agent requires at least one tool (semantic view or search service)
        can_create_agent = enable_semantic_view or enable_search_service
        enable_agent = st.checkbox(
            "🤖 Create AI Agent",
            value=can_create_agent,
            disabled=not can_create_agent,
            help="Create an AI agent that can interact with your data using Cortex Analyst and/or Cortex Search",
            key="enable_agent"
        )
        if enable_agent and can_create_agent:
            st.caption("✨ Enables conversational AI interface with your demo data")
        elif not can_create_agent:
            st.caption("⚠️ Requires Semantic View or Search Service")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Debug mode checkbox
    debug_mode = st.checkbox(
        "🔍 Enable Debug Mode (Show detailed table relationships & performance metrics)",
        value=False,
        help="Generate detailed analysis of table relationships, joins, question mapping, and display performance statistics",
        key="debug_mode_infrastructure"
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    create_btn = st.button(
        "🚀 Create Demo Infrastructure",
        type="primary",
        key="create_infra"
    )
    
    if create_btn:
        # Set flag to advance to step 4
        st.session_state['infrastructure_started'] = True
        # Force rerun to update step progress indicator to step 4
        st.rerun()
    
    # Check if infrastructure creation should proceed
    if st.session_state.get('infrastructure_started', False) and not st.session_state.get('infrastructure_complete', False):
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📈 Creating Infrastructure")
        progress_placeholder = st.empty()
        progress_placeholder.progress(0, text="Starting infrastructure creation...")
        
        status_container = st.container()
        
        results = create_tables_in_snowflake(
            schema_name,
            selected_demo,
            num_records,
            company_name,
            enable_search_service,
            enable_semantic_view,
            enable_agent,
            language_code,
            progress_placeholder,
            status_container,
            company_url=company_url,
            target_questions=st.session_state.get('target_questions', [])
        )
        
        if results:
            # Mark infrastructure as complete
            st.session_state['infrastructure_complete'] = True
            
            # Save to history
            history_id = save_to_history(
                session=session,
                company_name=company_name,
                company_url=company_url if company_url else '',
                demo_data=selected_demo,
                schema_name=schema_name,
                num_records=num_records,
                language_code=language_code,
                team_members=st.session_state.get('team_members', ''),
                use_cases=st.session_state.get('use_cases', ''),
                enable_semantic_view=enable_semantic_view,
                enable_search_service=enable_search_service,
                enable_agent=enable_agent,
                advanced_mode=st.session_state.get('advanced_mode', False),
                results=results,
                target_questions=st.session_state.get('target_questions', []),
                generated_questions=st.session_state.get('generated_questions', [])
            )
            
            if history_id:
                st.session_state['last_history_id'] = history_id
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Separate different types of objects for display
            regular_tables = [t for t in results if t.get('type') not in ['semantic_view', 'search_service', 'agent']]
            semantic_views = [t for t in results if t.get('type') == 'semantic_view']
            search_services = [t for t in results if t.get('type') == 'search_service']
            agents = [t for t in results if t.get('type') == 'agent']
            
            structured_tables = [t for t in regular_tables if t.get('type') == 'structured']
            unstructured_tables = [t for t in regular_tables if t.get('type') == 'unstructured']
            
            total_records = sum(t['records'] for t in regular_tables if isinstance(t['records'], int))
            industry_focus = selected_demo.get('industry_focus', 'Business Intelligence')
            
            # Display demo header in blue success box
            st.markdown(f"""
            <div class='success-box'>
                <h2 style='margin: 0; color: white;'>🎯 {company_name} Demo: {selected_demo['title']}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            
            # Infrastructure Created - Card Layout
            st.markdown("## 📊 Infrastructure Created")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Build semantic view stat HTML
                semantic_stat = f"""
                <div class='infra-stat'>
                    <div class='infra-stat-icon'>🔗</div>
                    <div class='infra-stat-content'>
                        <div class='infra-stat-title'>{len(semantic_views)} Semantic View</div>
                        <div class='infra-stat-desc'>AI-ready data relationships</div>
                    </div>
                </div>
                """ if semantic_views else ""
                
                # Build search service stat HTML
                search_stat = f"""
                <div class='infra-stat'>
                    <div class='infra-stat-icon'>🔍</div>
                    <div class='infra-stat-content'>
                        <div class='infra-stat-title'>{len(search_services)} Cortex Search Service</div>
                        <div class='infra-stat-desc'>Intelligent document retrieval</div>
                    </div>
                </div>
                """ if search_services else ""
                
                # Build agent stat HTML
                agent_stat = f"""
                <div class='infra-stat'>
                    <div class='infra-stat-icon'>🤖</div>
                    <div class='infra-stat-content'>
                        <div class='infra-stat-title'>{len(agents)} AI Agent</div>
                        <div class='infra-stat-desc'>Automated tools and capabilities</div>
                    </div>
                </div>
                """ if agents else ""
                
                st.markdown(f"""
                <div class='infra-card'>
                    <h3>🏗️ Infrastructure Summary</h3>
                    <div class='infra-stat'>
                        <div class='infra-stat-icon'>📊</div>
                        <div class='infra-stat-content'>
                            <div class='infra-stat-title'>{len(structured_tables)} Structured Tables</div>
                            <div class='infra-stat-desc'>{total_records:,} records with ENTITY_ID PRIMARY KEY and 70% join overlap</div>
                        </div>
                    </div>
                    <div class='infra-stat'>
                        <div class='infra-stat-icon'>📄</div>
                        <div class='infra-stat-content'>
                            <div class='infra-stat-title'>{len(unstructured_tables)} Unstructured Table</div>
                            <div class='infra-stat-desc'>Text chunks for semantic search</div>
                        </div>
                    </div>
                    {semantic_stat}
                    {search_stat}
                    {agent_stat}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                tables_list = ""
                # Prepare tables for rendering (with truncation)
                tables_for_render = []
                for table in structured_tables:
                    table_copy = table.copy()
                    desc = table['description']
                    if len(desc) > 150:
                        table_copy['description'] = desc[:150] + "..."
                    tables_for_render.append(table_copy)
                tables_for_render.extend(unstructured_tables)
                
                tables_html = render_results_table_list(tables_for_render)
                st.markdown(f"""
                <div class='infra-card'>
                    <h3>📋 Tables Created</h3>
                    {tables_html}
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Single unified card for all example queries
            st.markdown("## 📋 Example Queries")
            
            # Get all generated questions
            all_questions = st.session_state.get('generated_questions', [])
            
            # Organize questions by category
            analytics_questions = []
            search_questions = []
            intelligence_questions = {'basic': [], 'intermediate': [], 'advanced': []}
            
            for q in all_questions:
                category = q.get('category', '')
                difficulty = q.get('difficulty', 'basic')
                
                if category == 'analytics':
                    analytics_questions.append(q['text'])
                    # Also use for intelligence section
                    intelligence_questions[difficulty].append(q['text'])
                elif category == 'search':
                    search_questions.append(q['text'])
                    # Also use for intelligence section
                    intelligence_questions[difficulty].append(q['text'])
            
            # Use the helper function to render query results
            render_query_results(analytics_questions, search_questions, intelligence_questions)
            
            # Target Questions Coverage Section
            if 'validation_results' in st.session_state and 'target_questions_for_display' in st.session_state:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("## 🎯 Target Questions Coverage")
                
                target_questions = st.session_state['target_questions_for_display']
                validation_results = st.session_state['validation_results']
                
                # Build coverage display
                coverage_html = "<div class='value-card'><div class='value-card-content'>"
                coverage_html += "<div class='coverage-intro'>"
                coverage_html += f"The generated data has been validated against your {len(target_questions)} target question(s):"
                coverage_html += "</div>"
                
                for idx, question in enumerate(target_questions, 1):
                    coverage_html += "<div class='coverage-question-box'>"
                    coverage_html += f"<div class='coverage-question-title'>Question {idx}: {question}</div>"
                    
                    # Show validation feedback for each table
                    for table_name, validation_info in validation_results.items():
                        feedback = validation_info.get('feedback', '')
                        if feedback and feedback != "No target questions to validate":
                            coverage_html += "<div class='coverage-feedback'>"
                            coverage_html += f"<strong class='coverage-feedback-table'>{table_name}:</strong><br>"
                            # Convert feedback to HTML-safe format
                            feedback_lines = feedback.split('\n')
                            for line in feedback_lines:
                                if line.strip():
                                    coverage_html += f"<div class='coverage-feedback-item'>{line}</div>"
                            coverage_html += "</div>"
                    
                    coverage_html += "</div>"
                
                coverage_html += "<div class='coverage-note'>"
                coverage_html += "<div class='coverage-note-title'>💡 Note:</div>"
                coverage_html += "<div class='coverage-note-text'>"
                coverage_html += "The data has been generated with your target questions in mind. You can now test these questions using Cortex Analyst or Snowflake Intelligence."
                coverage_html += "</div></div>"
                
                coverage_html += "</div></div>"
                
                st.markdown(coverage_html, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Demo flow with equal-height cards
            st.markdown("## 🚀 Step-by-Step Demo Flow")
            
            
            # Get generated questions for demo flow
            questions = st.session_state.get('generated_questions', [])
            analytics_questions = [q for q in questions if q.get('category') == 'analytics']
            search_questions = [q for q in questions if q.get('category') == 'search']
            
            # Get specific questions for each step with fallbacks
            step1_question = analytics_questions[0]['text'] if len(analytics_questions) > 0 else "What are the top 5 performing entities and their key metrics?"
            step2_question = analytics_questions[1]['text'] if len(analytics_questions) > 1 else "What could be the reasons for these performance differences?"
            step3_question = search_questions[0]['text'] if len(search_questions) > 0 else "Find relevant best practices or recommendations"
            
            flow_col1, flow_col2, flow_col3 = st.columns(3)
            
            with flow_col1:
                st.markdown(f"""
<div class="demo-flow-card">
    <h3>Step 1: Structured Data Analysis</h3>
    <div class="ask-text">Ask: "{step1_question}"</div>
    <div class="check-item">Cortex Analyst queries structured tables</div>
    <div class="check-item">Joins data using ENTITY_ID</div>
    <div class="check-item">Returns analytical insights with charts</div>
</div>
""", unsafe_allow_html=True)
            
            with flow_col2:
                st.markdown(f"""
<div class="demo-flow-card">
    <h3>Step 2: AI Reasoning Follow-up</h3>
    <div class="ask-text">Ask: "{step2_question}"</div>
    <div class="check-item">Agent uses AI reasoning (not querying data)</div>
    <div class="check-item">Provides business insights and hypotheses</div>
    <div class="check-item">Suggests potential factors and correlations</div>
</div>
""", unsafe_allow_html=True)
            
            with flow_col3:
                st.markdown(f"""
<div class="demo-flow-card">
    <h3>Step 3: Unstructured Knowledge Retrieval</h3>
    <div class="ask-text">Ask: "{step3_question}"</div>
    <div class="check-item">Cortex Search queries unstructured content</div>
    <div class="check-item">Returns contextual information from text data</div>
    <div class="check-item">Combines with previous analysis for insights</div>
</div>
""", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🎯 Next Steps")
            
            # Get first structured table name for example query
            first_table = next((r['table'] for r in results if r.get('type') == 'structured'), 'TABLE')
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                **Query Your Data:**
                ```sql
                USE SCHEMA {schema_name};
                SELECT * FROM {first_table} LIMIT 10;
                ```
                """)
            
            with col2:
                st.markdown("""
                **Try Cortex Analyst:**
                - Ask natural language questions
                - Analyze trends and patterns
                - Generate insights automatically
                """)
            
            with col3:
                # Get account info for Snowflake Intelligence URL
                try:
                    account_result = session.sql("SELECT CURRENT_ACCOUNT_NAME() as account_locator, CURRENT_ACCOUNT() as account_name").collect()
                    if account_result:
                        account_locator = account_result[0]['ACCOUNT_LOCATOR']
                        account_name = account_result[0]['ACCOUNT_NAME']
                        intelligence_url = f"https://ai.snowflake.com/{account_locator}/{account_name}/"
                    else:
                        intelligence_url = "https://ai.snowflake.com"
                except:
                    intelligence_url = "https://ai.snowflake.com"
                
                st.markdown("**Snowflake Intelligence:**")
                st.link_button(
                    "🤖 Use Your Snowflake Intelligence Agent",
                    intelligence_url,
                    use_container_width=True
                )
            
            st.session_state['last_created_schema'] = schema_name
            st.session_state['last_results'] = results

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
render_page_footer("Powered by ❄️ Snowflake")

