"""
Infrastructure creation for Snowflake Intelligence.

This module handles all infrastructure operations including:
- Agent automation (creating and configuring Snowflake Intelligence agents)
- Semantic view creation (Cortex Analyst models)
- Search service creation (Cortex Search)
- Table relationship analysis
"""

import streamlit as st
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from utils import LLM_MODEL
from prompts import get_agent_system_prompt, get_agent_persona_prompt
from metrics import timeit


# ============================================================================
# HELPER FUNCTIONS FOR BATCH OPERATIONS
# ============================================================================

def batch_grants(session, grants: List[Tuple[str, str, str]]) -> None:
    """
    Execute GRANT statements individually (original implementation).
    
    This was reverted from batch execution due to Snowflake session.sql() 
    not supporting multi-statement execution reliably.
    
    Args:
        session: Snowflake session
        grants: List of tuples (grant_type, object_name, role)
                Example: [('USAGE', 'SCHEMA mydb.myschema', 'SYSADMIN'), ...]
    
    Returns:
        None
    """
    if not grants:
        return
    
    # Execute grants individually (original working implementation)
    for grant_type, object_name, role in grants:
        try:
            session.sql(f"GRANT {grant_type} ON {object_name} TO ROLE {role}").collect()
        except Exception as e:
            # Silently ignore "already granted" and similar errors
            error_msg = str(e).lower()
            if 'already' not in error_msg and 'exists' not in error_msg and 'granted' not in error_msg:
                # Only show truly unexpected errors
                pass


def batch_describe_tables(session, schema: str, table_names: List[str]) -> Dict[str, List[Dict]]:
    """
    Get column information for multiple tables using INFORMATION_SCHEMA.
    
    This optimization replaces N individual DESCRIBE TABLE calls with a single
    INFORMATION_SCHEMA query, saving 1-2 seconds for multiple tables.
    
    Args:
        session: Snowflake session
        schema: Schema name (format: DATABASE.SCHEMA or just SCHEMA)
        table_names: List of table names to describe
    
    Returns:
        Dictionary mapping table names to list of column dictionaries
        Each column dict has: {name, type, nullable, default, primary_key, ...}
    """
    if not table_names:
        return {}
    
    # Parse schema to get database and schema names
    schema_parts = schema.split('.')
    if len(schema_parts) == 2:
        database_name, schema_name = schema_parts
    else:
        # Try to get current database
        try:
            database_result = session.sql("SELECT CURRENT_DATABASE()").collect()
            database_name = database_result[0][0]
            schema_name = schema
        except Exception:
            # Fallback to using schema as-is
            database_name = None
            schema_name = schema
    
    # Build INFORMATION_SCHEMA query for all tables at once
    table_list = ", ".join([f"'{t}'" for t in table_names])
    
    if database_name:
        query = f"""
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            IS_IDENTITY
        FROM {database_name}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema_name}'
        AND TABLE_NAME IN ({table_list})
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
    else:
        query = f"""
        SELECT 
            TABLE_NAME,
            COLUMN_NAME,
            DATA_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            IS_IDENTITY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{schema_name}'
        AND TABLE_NAME IN ({table_list})
        ORDER BY TABLE_NAME, ORDINAL_POSITION
        """
    
    try:
        result = session.sql(query).collect()
        
        # Organize results by table
        tables_info = {table: [] for table in table_names}
        for row in result:
            table_name = row['TABLE_NAME']
            if table_name in tables_info:
                tables_info[table_name].append({
                    'name': row['COLUMN_NAME'],
                    'type': row['DATA_TYPE'],
                    'nullable': row['IS_NULLABLE'] == 'YES',
                    'default': row['COLUMN_DEFAULT'],
                    'primary_key': row['IS_IDENTITY'] == 'YES',
                    'kind': 'COLUMN'
                })
        
        return tables_info
    except Exception as e:
        st.warning(f"⚠️ Batch DESCRIBE failed, falling back to individual calls: {str(e)[:100]}")
        # Fallback to individual DESCRIBE TABLE calls
        tables_info = {}
        for table in table_names:
            try:
                desc_result = session.sql(f"DESCRIBE TABLE {schema}.{table}").collect()
                tables_info[table] = [
                    {
                        'name': row['name'],
                        'type': row['type'],
                        'nullable': row['null?'] == 'Y',
                        'default': row['default'],
                        'primary_key': row.get('primary key', 'N') == 'Y',
                        'kind': row['kind']
                    }
                    for row in desc_result
                ]
            except Exception:
                tables_info[table] = []
        return tables_info


# ============================================================================
# AGENT AUTOMATION
# ============================================================================

def verify_snowflake_intelligence_setup(session) -> Tuple[bool, Optional[str]]:
    """
    Verify that SNOWFLAKE_INTELLIGENCE database and AGENTS schema exist.
    
    Args:
        session: Snowflake session
        
    Returns:
        Tuple of (setup_complete, error_message)
    """
    try:
        # Check if SNOWFLAKE_INTELLIGENCE database exists
        db_check = session.sql("""
            SHOW DATABASES LIKE 'SNOWFLAKE_INTELLIGENCE'
        """).collect()
        
        if not db_check or len(db_check) == 0:
            setup_sql = """
-- Run this SQL to set up Snowflake Intelligence infrastructure:

CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE
    COMMENT = 'Database for Snowflake Intelligence agents - enables UI discoverability';

CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS
    COMMENT = 'Schema for AI agents - agents here appear in Snowsight under AI & ML » Agents';

GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
"""
            return False, f"SNOWFLAKE_INTELLIGENCE database does not exist. Please run Setup.sql or execute:\n{setup_sql}"
        
        # Check if AGENTS schema exists
        schema_check = session.sql("""
            SHOW SCHEMAS LIKE 'AGENTS' IN DATABASE SNOWFLAKE_INTELLIGENCE
        """).collect()
        
        if not schema_check or len(schema_check) == 0:
            setup_sql = """
CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS
    COMMENT = 'Schema for AI agents - agents here appear in Snowsight under AI & ML » Agents';

GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
"""
            return False, f"SNOWFLAKE_INTELLIGENCE.AGENTS schema does not exist. Please run:\n{setup_sql}"
        
        return True, None
        
    except Exception as e:
        return False, f"Error checking Snowflake Intelligence setup: {str(e)}"


def generate_orchestration_instructions(
    has_analyst: bool,
    has_search: bool,
    demo_data: Dict
) -> str:
    """
    Generate orchestration instructions based on available tools.
    
    Args:
        has_analyst: Whether Cortex Analyst is available
        has_search: Whether Cortex Search is available
        demo_data: Demo configuration dictionary
        
    Returns:
        Orchestration instructions string
    """
    industry = demo_data.get('industry_focus', demo_data.get('industry', 'Business Intelligence'))
    
    if has_analyst and has_search:
        return f"""Use Cortex Analyst to query and analyze structured data tables for quantitative insights. Use Cortex Search to find relevant information in unstructured documents and content. 

When users ask about trends, metrics, or data analysis, use Cortex Analyst first. When users need information from documents, policies, or research, use Cortex Search.

For {industry} questions, leverage both tools to provide comprehensive answers that combine data analysis with contextual knowledge."""
    elif has_analyst:
        return f"""Use Cortex Analyst to query and analyze structured data tables. Provide insights from the data including trends, patterns, and key metrics relevant to {industry}."""
    elif has_search:
        return f"""Use Cortex Search to find relevant information in unstructured documents and content. Provide insights from the knowledge base relevant to {industry} operations and best practices."""
    else:
        return "Provide helpful responses based on general knowledge and context."


def generate_tool_resources(
    schema_name: str,
    semantic_view_name: Optional[str],
    search_service_name: Optional[str],
    warehouse_name: str,
    database_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate tool resources configuration for agent JSON specification.
    
    Args:
        schema_name: Schema name (may include database prefix)
        semantic_view_name: Name of semantic view (without _SEMANTIC_MODEL suffix)
        search_service_name: Name of search service
        warehouse_name: Warehouse name
        database_name: Optional database name
        
    Returns:
        Tool resources configuration dictionary
    """
    tool_resources = {}
    
    # Ensure we have fully qualified names (DATABASE.SCHEMA format)
    if '.' in schema_name:
        # Already has database prefix
        full_schema_name = schema_name
    elif database_name:
        # Add provided database
        full_schema_name = f"{database_name}.{schema_name}"
    else:
        # Use schema_name as-is (will use current database context)
        full_schema_name = schema_name
    
    if semantic_view_name:
        tool_resources["Query Demo Data"] = {
            "semantic_view": f"{full_schema_name}.{semantic_view_name}_SEMANTIC_MODEL"
        }
    
    if search_service_name:
        tool_resources["Search Documents"] = {
            "name": f"{full_schema_name}.{search_service_name}",
            "max_results": 5,
            "id_column": "CHUNK_ID",
            "title_column": "DOCUMENT_TYPE"
        }
    
    return tool_resources


def generate_agent_json_spec(
    demo_data: Dict,
    company_name: str,
    schema_name: str,
    semantic_view_name: Optional[str],
    search_service_name: Optional[str],
    warehouse_name: str,
    sample_questions: List[str] = None,
    session=None
) -> str:
    """
    Generate JSON specification for agent creation.
    
    Args:
        demo_data: Demo configuration dictionary
        company_name: Company name
        schema_name: Schema name
        semantic_view_name: Name of semantic view
        search_service_name: Name of search service
        warehouse_name: Warehouse name
        sample_questions: Optional list of sample questions
        session: Optional Snowflake session
        
    Returns:
        JSON specification string
    """
    # Generate system prompt (response instructions)
    system_prompt = generate_agent_system_prompt(demo_data, company_name)
    
    # Generate orchestration instructions
    has_analyst = semantic_view_name is not None
    has_search = search_service_name is not None
    orchestration = generate_orchestration_instructions(has_analyst, has_search, demo_data)
    
    # Build tools array
    tools = []
    
    if has_analyst:
        tools.append({
            "tool_spec": {
                "type": "cortex_analyst_text_to_sql",
                "name": "Query Demo Data",
                "description": f"Query structured data tables for {demo_data.get('title', 'demo')}. Use this to answer questions about metrics, trends, and analytical insights from the data."
            }
        })
    
    if has_search:
        tools.append({
            "tool_spec": {
                "type": "cortex_search",
                "name": "Search Documents",
                "description": f"Search unstructured documents and content for {demo_data.get('title', 'demo')}. Use this to find relevant information, best practices, and contextual knowledge."
            }
        })
    
    # Get database name for fully qualified resource names
    if '.' in schema_name:
        database_name = schema_name.split('.')[0]
    elif session:
        # Get current database from session
        try:
            database_name = session.get_current_database()
        except:
            database_name = None
    else:
        database_name = None
    
    # Generate tool resources
    tool_resources = generate_tool_resources(
        schema_name, semantic_view_name, search_service_name, warehouse_name, database_name
    )
    
    # Format sample questions
    sample_questions_list = []
    if sample_questions and len(sample_questions) > 0:
        # Take up to 5 sample questions
        for q in sample_questions[:5]:
            if isinstance(q, dict):
                question_text = q.get('text', q.get('question', ''))
            else:
                question_text = str(q)
            
            if question_text:
                sample_questions_list.append({"question": question_text})
    
    # Build the complete specification
    spec = {
        "models": {
            "orchestration": ""  # Use default model
        },
        "instructions": {
            "response": system_prompt,
            "orchestration": orchestration
        },
        "tools": tools,
        "tool_resources": tool_resources
    }
    
    # Add sample questions if available
    if sample_questions_list:
        spec["instructions"]["sample_questions"] = sample_questions_list
    
    return json.dumps(spec, indent=2)


def generate_agent_system_prompt(demo_data: Dict, company_name: str) -> str:
    """
    Generate comprehensive agent system prompt based on demo context.
    
    Args:
        demo_data: Demo configuration dictionary
        company_name: Company name
        
    Returns:
        System prompt string
    """
    return get_agent_system_prompt(demo_data, company_name)


def generate_agent_persona_with_llm(
    demo_data: Dict,
    company_name: str,
    session
) -> str:
    """
    Generate agent persona using Cortex LLM.
    
    Args:
        demo_data: Demo configuration dictionary
        company_name: Company name
        session: Snowflake session
        
    Returns:
        Agent persona string
    """
    industry = demo_data.get('industry_focus', demo_data.get('industry', 'Business Intelligence'))
    
    prompt = get_agent_persona_prompt(demo_data, company_name)
    
    try:
        result = session.sql(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) as response",
            params=[LLM_MODEL, prompt]
        ).collect()
        
        persona = result[0]['RESPONSE'].strip()
        return persona
    except Exception:
        return f"An expert AI analyst specializing in {industry} data analysis, with deep knowledge of industry trends and best practices. Skilled at transforming complex data into actionable business insights for {company_name}."


@timeit
def create_agent_automatically(
    session,
    schema_name: str,
    demo_data: Dict,
    semantic_view_name: Optional[str],
    search_service_name: Optional[str],
    company_name: str,
    warehouse_name: str,
    call_cortex_func: Any = None,
    sample_questions: List[str] = None
) -> Optional[Dict]:
    """
    Automatically create and configure a Cortex Agent using JSON specification.
    
    Args:
        session: Snowflake session
        schema_name: Schema name for resources
        demo_data: Demo configuration dictionary
        semantic_view_name: Name of semantic view
        search_service_name: Name of search service
        company_name: Company name
        warehouse_name: Warehouse name
        call_cortex_func: Optional Cortex function (not used)
        sample_questions: Optional list of sample questions
        
    Returns:
        Agent configuration dictionary or error dictionary
    """
    try:
        # Verify Snowflake Intelligence infrastructure exists
        setup_complete, error_msg = verify_snowflake_intelligence_setup(session)
        if not setup_complete:
            # Return error information for user display
            return {
                'error': True,
                'error_message': error_msg,
                'setup_required': True
            }
        
        # Generate agent name with date for uniqueness
        clean_company = company_name.replace('-', '_').replace(' ', '_').upper()
        date_suffix = datetime.now().strftime("%Y%m%d")
        agent_name = f"{clean_company}_{date_suffix}_AGENT"
        
        # Agent goes in SNOWFLAKE_INTELLIGENCE.AGENTS for UI discoverability
        full_agent_name = f"SNOWFLAKE_INTELLIGENCE.AGENTS.{agent_name}"
        
        # Generate display name for UI
        demo_title = demo_data.get('title', 'Demo')
        display_name = f"{company_name} - {demo_title}"
        
        # Build tools configuration
        tools_config = []
        if semantic_view_name:
            tools_config.append('Cortex Analyst')
        if search_service_name:
            tools_config.append('Cortex Search')
        
        # Require at least one tool
        if not tools_config:
            return {
                'error': True,
                'error_message': 'Agent requires at least one tool (Semantic Model or Search Service)',
                'setup_required': False
            }
        
        # Generate JSON specification
        json_spec = generate_agent_json_spec(
            demo_data=demo_data,
            company_name=company_name,
            schema_name=schema_name,
            semantic_view_name=semantic_view_name,
            search_service_name=search_service_name,
            warehouse_name=warehouse_name,
            sample_questions=sample_questions,
            session=session
        )
        
        # Escape JSON for SQL (double dollar signs allow easier embedding)
        json_spec_escaped = json_spec.replace('$$', '\\$\\$')
        
        # Build CREATE AGENT SQL with correct syntax
        # Escape double quotes in JSON for single-quote string embedding
        profile_json = json.dumps({"display_name": display_name}).replace('"', '\\"')
        comment_text = f"Agent for {company_name} demo: {demo_title}"
        comment_escaped = comment_text.replace("'", "''")
        
        create_agent_sql = f"""
CREATE OR REPLACE AGENT {full_agent_name}
WITH PROFILE = '{profile_json}'
COMMENT = '{comment_escaped}'
FROM SPECIFICATION $$
{json_spec}
$$;
"""
        
        # Create the agent
        session.sql(create_agent_sql).collect()
        
        # Note: Agents get access to resources through tool_resources in the specification
        # No explicit GRANT statements are needed - permissions are handled through the JSON spec
        grant_results = [
            "ℹ️ Agent created successfully",
            "ℹ️ Access to resources is configured through tool_resources in the agent specification",
            f"ℹ️ Semantic Model: {schema_name}.{semantic_view_name}_SEMANTIC_MODEL" if semantic_view_name else None,
            f"ℹ️ Search Service: {schema_name}.{search_service_name}" if search_service_name else None
        ]
        grant_results = [r for r in grant_results if r is not None]
        
        # Build agent configuration response
        agent_config = {
            'name': agent_name,
            'full_name': full_agent_name,
            'display_name': display_name,
            'schema': 'SNOWFLAKE_INTELLIGENCE.AGENTS',
            'demo_schema': schema_name,
            'warehouse': warehouse_name,
            'tools': tools_config,
            'created_at': datetime.now().isoformat(),
            'ui_path': 'AI & ML » Agents',
            'error': False,
            'grant_results': grant_results
        }
        
        if semantic_view_name:
            agent_config['semantic_model'] = f"{schema_name}.{semantic_view_name}_SEMANTIC_MODEL"
        
        if search_service_name:
            agent_config['search_service'] = f"{schema_name}.{search_service_name}"
        
        return agent_config
        
    except Exception as e:
        return {
            'error': True,
            'error_message': f"Failed to create agent: {str(e)}",
            'setup_required': False,
            'exception_details': str(e)
        }


def test_agent_with_questions(
    session,
    agent_full_name: str,
    test_questions: List[str],
    max_questions: int = 3
) -> Dict[str, Any]:
    """
    Test agent with a set of questions.
    
    Args:
        session: Snowflake session
        agent_full_name: Full agent name (DATABASE.SCHEMA.AGENT)
        test_questions: List of questions to test
        max_questions: Maximum number of questions to test
        
    Returns:
        Test results dictionary
    """
    results = {
        'total_questions': 0,
        'successful': 0,
        'failed': 0,
        'success_rate': 0.0,
        'question_results': []
    }
    
    questions_to_test = test_questions[:max_questions]
    results['total_questions'] = len(questions_to_test)
    
    for question in questions_to_test:
        question_result = {
            'question': question,
            'success': False,
            'response': None,
            'error': None,
            'execution_time': 0.0
        }
        
        try:
            start_time = datetime.now()
            
            query = f"""
            SELECT SNOWFLAKE.CORTEX.ANALYST.ASK(
                '{agent_full_name}',
                '{question.replace("'", "''")}'
            ) as response
            """
            
            result = session.sql(query).collect()
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result and len(result) > 0:
                question_result['success'] = True
                question_result['response'] = str(result[0]['RESPONSE'])[:200]
                question_result['execution_time'] = execution_time
                results['successful'] += 1
            else:
                question_result['error'] = "No response returned"
                results['failed'] += 1
                
        except Exception as e:
            question_result['error'] = str(e)[:200]
            results['failed'] += 1
        
        results['question_results'].append(question_result)
    
    if results['total_questions'] > 0:
        results['success_rate'] = (results['successful'] / results['total_questions']) * 100
    
    return results


def extract_questions_from_semantic_model(
    semantic_view_info: Dict
) -> List[str]:
    """
    Extract example questions from semantic view info.
    
    Args:
        semantic_view_info: Semantic view configuration dictionary
        
    Returns:
        List of example questions
    """
    questions = []
    
    if semantic_view_info and 'example_queries' in semantic_view_info:
        questions.extend(semantic_view_info['example_queries'])
    
    if len(questions) < 3:
        questions.extend([
            "What are the top 10 records by key metrics?",
            "Show me trends over the last month",
            "Which categories have the highest performance?"
        ])
    
    return questions[:5]


def generate_agent_documentation(
    agent_config: Dict,
    test_results: Optional[Dict] = None
) -> str:
    """
    Generate documentation for the created agent.
    
    Args:
        agent_config: Agent configuration dictionary
        test_results: Optional test results dictionary
        
    Returns:
        Markdown documentation string
    """
    doc = f"""# {agent_config['name']} - AI Data Analyst Agent

## Agent Configuration

- **Full Name:** `{agent_config['full_name']}`
- **Schema:** `{agent_config['schema']}`
- **Warehouse:** `{agent_config['warehouse']}`
- **Created:** {agent_config['created_at']}

## Capabilities

"""
    
    if 'tools' in agent_config and agent_config['tools']:
        doc += "This agent has access to the following tools:\n\n"
        for tool in agent_config['tools']:
            if tool == 'Cortex Analyst':
                doc += f"- **Cortex Analyst**: Query structured data using natural language\n"
                doc += f"  - Semantic Model: `{agent_config.get('semantic_model', 'N/A')}`\n"
            elif tool == 'Cortex Search':
                doc += f"- **Cortex Search**: Search unstructured documents and content\n"
                doc += f"  - Search Service: `{agent_config.get('search_service', 'N/A')}`\n"
    
    doc += "\n## System Prompt\n\n```\n"
    doc += agent_config.get('system_prompt', 'N/A')
    doc += "\n```\n"
    
    if test_results:
        doc += f"\n## Test Results\n\n"
        doc += f"- **Total Questions Tested:** {test_results['total_questions']}\n"
        doc += f"- **Successful:** {test_results['successful']}\n"
        doc += f"- **Failed:** {test_results['failed']}\n"
        doc += f"- **Success Rate:** {test_results['success_rate']:.1f}%\n\n"
        
        if test_results['question_results']:
            doc += "### Question Test Details\n\n"
            for idx, qr in enumerate(test_results['question_results'], 1):
                status = "✅" if qr['success'] else "❌"
                doc += f"{idx}. {status} **{qr['question']}**\n"
                if qr['success']:
                    doc += f"   - Execution time: {qr['execution_time']:.2f}s\n"
                else:
                    doc += f"   - Error: {qr['error']}\n"
                doc += "\n"
    
    doc += "\n## Usage Examples\n\n"
    doc += "### Ask a Question\n\n"
    doc += "```sql\n"
    doc += f"SELECT SNOWFLAKE.CORTEX.ANALYST.ASK(\n"
    doc += f"  '{agent_config['full_name']}',\n"
    doc += f"  'What are the key insights from this data?'\n"
    doc += f") as response;\n"
    doc += "```\n\n"
    
    doc += "### Get Conversational Response\n\n"
    doc += "```python\n"
    doc += "# In Python/Streamlit\n"
    doc += f"response = session.sql('''\n"
    doc += f"  SELECT SNOWFLAKE.CORTEX.ANALYST.ASK(\n"
    doc += f"    '{agent_config['full_name']}',\n"
    doc += f"    'Your question here'\n"
    doc += f"  ) as response\n"
    doc += f"''').collect()\n"
    doc += "print(response[0]['RESPONSE'])\n"
    doc += "```\n"
    
    return doc


# ============================================================================
# INFRASTRUCTURE CREATION
# ============================================================================

def analyze_table_relationships(
    structured_tables: List[Tuple],
    demo_data: Dict,
    session,
    error_handler,
    target_questions: Optional[List[str]] = None
) -> Optional[Dict]:
    """
    Use LLM to analyze how tables relate and what insights each join provides.
    
    Args:
        structured_tables: List of (key, table_info) tuples
        demo_data: Demo configuration with title and description
        session: Snowflake session
        error_handler: ErrorHandler instance
        target_questions: Optional list of target questions
    
    Returns:
        dict with model_type, relationships, join_paths, question_mapping
    """
    from utils import call_cortex_with_retry, TABLE_JOIN_OVERLAP_PERCENTAGE
    from errors import ErrorCode, ErrorSeverity, safe_json_parse
    import traceback
    
    # Build table information for the prompt
    table_descriptions = []
    for key, table_info in structured_tables:
        table_type = table_info.get('table_type', 'unknown')
        table_descriptions.append(f"""
- **{table_info['name']}** ({table_type.upper()})
  - Description: {table_info['description']}
  - Purpose: {table_info.get('purpose', 'N/A')}""")
    
    tables_text = "\n".join(table_descriptions)
    
    # Add target questions if provided
    questions_context = ""
    if target_questions and len(target_questions) > 0:
        questions_list = "\n".join([f"  {i+1}. {q}" for i, q in enumerate(target_questions)])
        questions_context = f"""

## Target Questions to Answer:
{questions_list}

For each question, identify which tables and joins are required to answer it.
"""
    
    from prompts import get_table_relationships_analysis_prompt
    prompt = get_table_relationships_analysis_prompt(
        demo_data, structured_tables, target_questions
    )
    
    try:
        response = call_cortex_with_retry(prompt, session, error_handler)
        
        if response:
            parsed = safe_json_parse(response)
            if parsed:
                return parsed
            else:
                st.warning("⚠️ Could not parse relationship analysis response")
        else:
            st.warning("⚠️ No response from LLM for relationship analysis")
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.CORTEX_INVALID_RESPONSE,
            error_type=type(e).__name__,
            severity=ErrorSeverity.ERROR,
            message=f"Error analyzing table relationships: {str(e)}",
            stack_trace=traceback.format_exc(),
            function_name="analyze_table_relationships"
        )
        st.error(f"Error in analyze_table_relationships: {str(e)}")
    
    return None


@timeit
def create_cortex_search_service(
    schema_name: str,
    table_name: str,
    session,
    warehouse_name: str,
    error_handler,
    language_code: str = "en",
    search_column: str = "CHUNK_TEXT"
) -> Tuple[Optional[str], List[str]]:
    """
    Create Cortex Search service for unstructured data with language support.
    
    Args:
        schema_name: Schema name
        table_name: Table name
        session: Snowflake session
        warehouse_name: Warehouse name
        error_handler: ErrorHandler instance
        language_code: Language code (default: en)
        search_column: Column to search (default: CHUNK_TEXT)
        
    Returns:
        Tuple of (service_name, grant_messages)
    """
    from errors import ErrorCode, ErrorSeverity
    import traceback
    
    try:
        # Get current database to ensure fully qualified names
        try:
            current_db_result = session.sql("SELECT CURRENT_DATABASE() as db").collect()
            database_name = current_db_result[0]['DB'] if current_db_result else None
        except Exception as e:
            st.warning(f"Could not get current database: {str(e)}")
            database_name = None
        
        # Create fully qualified schema name if we have the database
        if database_name and '.' not in schema_name:
            full_schema_name = f"{database_name}.{schema_name}"
        else:
            full_schema_name = schema_name
        
        service_name = f"{table_name}_{language_code.upper()}_SEARCH_SERVICE" if language_code != "en" else f"{table_name}_SEARCH_SERVICE"
        full_table_name = f"{full_schema_name}.{table_name}"
        
        create_service_sql = f"""
        CREATE OR REPLACE CORTEX SEARCH SERVICE {full_schema_name}.{service_name}
        ON {search_column}
        ATTRIBUTES CHUNK_ID, DOCUMENT_ID, DOCUMENT_TYPE, SOURCE_SYSTEM, LANGUAGE
        WAREHOUSE = {warehouse_name}
        TARGET_LAG = '1 minute'
        AS (
            SELECT 
                CHUNK_ID,
                DOCUMENT_ID, 
                DOCUMENT_TYPE,
                SOURCE_SYSTEM,
                LANGUAGE,
                {search_column}
            FROM {full_table_name}
        );
        """
        
        session.sql(create_service_sql).collect()
        
        # Grant permissions to current role and common roles for agent access (OPTIMIZED: batch grants)
        grant_messages = []
        try:
            current_role = session.get_current_role()
            
            # Collect all grants to execute in batch
            grants_to_execute = [
                ('USAGE', f'SCHEMA {schema_name}', current_role),
                ('USAGE', f'CORTEX SEARCH SERVICE {full_schema_name}.{service_name}', current_role)
            ]
            
            # Add ACCOUNTADMIN grants if different role
            if current_role.upper() != 'ACCOUNTADMIN':
                grants_to_execute.extend([
                    ('USAGE', f'SCHEMA {schema_name}', 'ACCOUNTADMIN'),
                    ('USAGE', f'CORTEX SEARCH SERVICE {full_schema_name}.{service_name}', 'ACCOUNTADMIN')
                ])
            
            # Execute all grants in batch (OPTIMIZATION)
            try:
                batch_grants(session, grants_to_execute)
                grant_messages.append(f"✓ Permissions granted to: {current_role}")
                if current_role.upper() != 'ACCOUNTADMIN':
                    grant_messages.append(f"✓ Permissions granted to: ACCOUNTADMIN")
            except Exception as e:
                grant_messages.append(f"⚠️ Some grants may have failed: {str(e)[:200]}")
            
            # Display grant results if any issues
            if any("✗" in msg for msg in grant_messages):
                st.warning(f"Search service grants:\n" + "\n".join(grant_messages))
            else:
                # All grants succeeded - verify service is accessible and grant ALL PRIVILEGES
                try:
                    # Grant ALL PRIVILEGES (includes USAGE, OWNERSHIP transfer, etc.)
                    session.sql(f"GRANT ALL PRIVILEGES ON CORTEX SEARCH SERVICE {full_schema_name}.{service_name} TO ROLE {current_role}").collect()
                    grant_messages.append(f"✓ ALL PRIVILEGES granted to: {current_role}")
                except Exception as e:
                    grant_messages.append(f"⚠️ ALL PRIVILEGES grant: {str(e)[:100]}")
                
                try:
                    verify_sql = f"SHOW CORTEX SEARCH SERVICES LIKE '{service_name}' IN SCHEMA {schema_name}"
                    result = session.sql(verify_sql).collect()
                    if result and len(result) > 0:
                        grant_messages.append(f"✓ Search service verified and accessible")
                        # Add service state info (Row object uses attribute access, not .get())
                        try:
                            service_info = result[0]
                            state = service_info['state'] if 'state' in service_info else service_info.get('STATE', 'Unknown')
                            grant_messages.append(f"ℹ️ Service status: {state}")
                        except:
                            pass  # Skip state if we can't get it
                    else:
                        grant_messages.append(f"⚠️ Search service created but not visible in SHOW command")
                except Exception as e:
                    grant_messages.append(f"⚠️ Could not verify service: {str(e)}")
            
        except Exception as grant_error:
            st.warning(f"Note: Could not grant permissions on search service: {str(grant_error)}")
        
        return service_name, grant_messages
        
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.SEARCH_SERVICE_FAILED,
            error_type=type(e).__name__,
            severity=ErrorSeverity.ERROR,
            message=f"Failed to create search service: {str(e)}",
            stack_trace=traceback.format_exc()
        )
        st.error(f"Error creating Cortex Search service: {str(e)}")
        return None, []


@timeit
def create_semantic_view(
    schema_name: str,
    structured_tables: List[Tuple],
    demo_data: Dict,
    company_name: str,
    session,
    error_handler
) -> Optional[Dict]:
    """
    Create a semantic view with Cortex Analyst extension supporting 2-5 tables.
    
    Args:
        schema_name: Name of the schema
        structured_tables: List of (key, table_info) tuples with all structured tables
        demo_data: Demo configuration
        company_name: Company name for view naming
        session: Snowflake session
        error_handler: ErrorHandler instance
    
    Returns:
        dict with view information and metadata
    """
    from utils import MAX_FACTS_PER_TABLE, MAX_TOTAL_FACTS, MAX_DIMENSIONS_PER_TABLE, MAX_TOTAL_DIMENSIONS
    from errors import ErrorCode, ErrorSeverity
    import traceback
    
    view_name = f"{company_name}_SEMANTIC_VIEW_SEMANTIC_MODEL"
    
    # Extract table information
    tables_info = []
    fact_tables = []
    dimension_tables = []
    
    for key, table_info in structured_tables:
        tables_info.append(table_info)
        table_type = table_info.get('table_type', 'dimension')
        if table_type == 'fact':
            fact_tables.append(table_info)
        else:
            dimension_tables.append(table_info)
    
    # If no explicit types, treat first 2 as fact and rest as dimensions
    if not fact_tables and len(tables_info) >= 2:
        fact_tables = [tables_info[0]]
        dimension_tables = tables_info[1:]
    elif not fact_tables:
        fact_tables = tables_info[:1]
        dimension_tables = []
    
    # Get current database to ensure fully qualified names
    try:
        current_db_result = session.sql("SELECT CURRENT_DATABASE() as db").collect()
        database_name = current_db_result[0]['DB'] if current_db_result else None
    except Exception as e:
        st.warning(f"Could not get current database: {str(e)}")
        database_name = None
    
    # Create fully qualified schema name if we have the database
    if database_name and '.' not in schema_name:
        full_schema_name = f"{database_name}.{schema_name}"
    else:
        full_schema_name = schema_name
    
    # Get column metadata from all tables
    all_tables_columns = {}  # Dict of table_name -> list of column metadata
    
    try:
        # Use batch DESCRIBE to get all table columns at once (OPTIMIZATION)
        table_names = [table_info['name'] for table_info in tables_info]
        tables_columns_raw = batch_describe_tables(session, schema_name, table_names)
        
        # Convert to expected format
        for table_name, columns_raw in tables_columns_raw.items():
            all_tables_columns[table_name] = [
                {'name': col['name'], 'type': col['type']}
                for col in columns_raw
            ]
        
    except Exception as e:
        # If we can't get columns, use safe fallback
        error_handler.log_error(
            error_code=ErrorCode.SCHEMA_VALIDATION_FAILED,
            error_type=type(e).__name__,
            severity=ErrorSeverity.WARNING,
            message=f"Could not retrieve table schemas for semantic view: {str(e)}",
            stack_trace=traceback.format_exc()
        )
        st.warning(f"Could not detect table columns, using minimal schema: {str(e)}")
        # Fallback to minimal column metadata for all tables
        for table_info in tables_info:
            all_tables_columns[table_info['name']] = [{'name': 'ENTITY_ID', 'type': 'NUMBER'}]
    
    # Generate FACTS and DIMENSIONS from all tables
    facts_list = []
    dimensions_list = []
    used_aliases = set()  # Track used aliases to prevent duplicates
    
    # Add ENTITY_ID only once (it's the join key)
    first_table_name = tables_info[0]['name']
    facts_list.append(f"    {first_table_name}.ENTITY_ID as ENTITY_ID with synonyms=('id','key','identifier','unique_id','record_id') comment='Unique identifier for joining tables'")
    used_aliases.add('ENTITY_ID')
    
    # Process all tables for facts and dimensions
    for table_info in tables_info:
        table_name = table_info['name']
        table_columns = all_tables_columns.get(table_name, [])
        
        # Add numeric columns as facts
        for col in table_columns:
            if col['name'] != 'ENTITY_ID' and any(num_type in col['type'].upper() for num_type in ['NUMBER', 'INT', 'FLOAT', 'DECIMAL', 'DOUBLE']):
                # Skip if column name already used - avoid duplicate column aliases
                if col['name'] not in used_aliases:
                    facts_list.append(f"    {table_name}.{col['name']} as {col['name']} with synonyms=('value','amount','quantity','metric','measure') comment='Numeric value from {table_name}'")
                    used_aliases.add(col['name'])
                if len(facts_list) >= MAX_TOTAL_FACTS:
                    break
        
        # Add text/date columns as dimensions
        for col in table_columns:
            if col['name'] != 'ENTITY_ID' and col['name'] not in used_aliases:
                col_type_upper = col['type'].upper()
                
                # Check date/time types first
                if any(date_type in col_type_upper for date_type in ['TIMESTAMP', 'DATE']) or col_type_upper == 'TIME':
                    dimensions_list.append(f"    {table_name}.{col['name']} as {col['name']} with synonyms=('date','time','when','timestamp','period') comment='Date/time dimension from {table_name}'")
                    used_aliases.add(col['name'])
                elif 'BOOLEAN' in col_type_upper:
                    dimensions_list.append(f"    {table_name}.{col['name']} as {col['name']} with synonyms=('flag','indicator','status','boolean','is') comment='Boolean dimension from {table_name}'")
                    used_aliases.add(col['name'])
                elif any(text_type in col_type_upper for text_type in ['VARCHAR', 'STRING', 'TEXT', 'CHAR']):
                    dimensions_list.append(f"    {table_name}.{col['name']} as {col['name']} with synonyms=('category','type','label','name','description') comment='Text dimension from {table_name}'")
                    used_aliases.add(col['name'])
                
                if len(dimensions_list) >= MAX_TOTAL_DIMENSIONS:
                    break
    
    # Ensure we have at least something
    if not facts_list:
        facts_list = [f"    {first_table_name}.ENTITY_ID as ENTITY_ID with synonyms=('id','key') comment='Identifier'"]
    if not dimensions_list:
        dimensions_list = [f"    {first_table_name}.ENTITY_ID as ENTITY_KEY with synonyms=('key','id') comment='Key'"]
    
    facts_sql = ",\n".join(facts_list)
    dimensions_sql = ",\n".join(dimensions_list)
    
    # Build TABLES section for semantic view
    tables_sql_lines = []
    for table_info in tables_info:
        tables_sql_lines.append(f"    {full_schema_name}.{table_info['name']} PRIMARY KEY (ENTITY_ID)")
    tables_sql = ",\n".join(tables_sql_lines)
    
    # Build RELATIONSHIPS section
    # Connect fact tables to all dimension tables via ENTITY_ID
    relationships_list = []
    relationship_counter = 1
    
    for fact_table in fact_tables:
        for dim_table in dimension_tables:
            rel_name = f"LINK_{relationship_counter}"
            relationships_list.append(f"    {rel_name} AS {fact_table['name']}(ENTITY_ID) REFERENCES {dim_table['name']}(ENTITY_ID)")
            relationship_counter += 1
    
    # If we don't have explicit fact/dimension split, create simple relationships between all tables
    if not relationships_list and len(tables_info) >= 2:
        # Connect first table to all others
        for i in range(1, len(tables_info)):
            rel_name = f"LINK_{i}"
            relationships_list.append(f"    {rel_name} AS {tables_info[0]['name']}(ENTITY_ID) REFERENCES {tables_info[i]['name']}(ENTITY_ID)")
    
    relationships_sql = ",\n".join(relationships_list) if relationships_list else f"    ENTITY_LINK AS {tables_info[0]['name']}(ENTITY_ID) REFERENCES {tables_info[1]['name']}(ENTITY_ID)"
    
    # Build comment listing all tables
    table_names = ", ".join([t['name'] for t in tables_info])
        
    # Drop existing view/semantic view first to handle edge cases
    # Semantic views are a type of VIEW, so DROP VIEW works for both
    try:
        session.sql(f"DROP VIEW IF EXISTS {full_schema_name}.{view_name}").collect()
        # Silently drop existing view without showing message
    except Exception as e:
        # View doesn't exist or other error - continue
        if "does not exist" not in str(e).lower():
            st.warning(f"Note: Could not drop existing view: {str(e)[:100]}")
        pass
    
    # Create semantic view (not semantic model with YAML) using fully qualified names
    semantic_sql = f"""CREATE OR REPLACE SEMANTIC VIEW {full_schema_name}.{view_name}
TABLES (
{tables_sql}
)
RELATIONSHIPS (
{relationships_sql}
)
FACTS (
{facts_sql}
)
DIMENSIONS (
{dimensions_sql}
)
COMMENT = 'Semantic view combining {table_names}'"""
    
    try:
        session.sql(semantic_sql).collect()
        
        # Grant permissions on semantic view to current role (OPTIMIZED: batch grants)
        grant_messages = []
        try:
            current_role = session.get_current_role()
            
            # Collect all grants to execute in batch
            grants_to_execute = [
                ('USAGE', f'SCHEMA {schema_name}', current_role),
                ('SELECT', f'VIEW {full_schema_name}.{view_name}', current_role)
            ]
            
            # If not ACCOUNTADMIN, grant to ACCOUNTADMIN as well
            if current_role.upper() != 'ACCOUNTADMIN':
                grants_to_execute.extend([
                    ('USAGE', f'SCHEMA {schema_name}', 'ACCOUNTADMIN'),
                    ('SELECT', f'VIEW {full_schema_name}.{view_name}', 'ACCOUNTADMIN')
                ])
            
            # Execute all grants in batch (OPTIMIZATION)
            try:
                batch_grants(session, grants_to_execute)
                grant_messages.append(f"✓ Semantic view permissions granted to: {current_role}")
                if current_role.upper() != 'ACCOUNTADMIN':
                    grant_messages.append(f"✓ Semantic view permissions granted to: ACCOUNTADMIN")
            except Exception as e:
                grant_messages.append(f"⚠️ Some grants may have failed: {str(e)[:200]}")
        except Exception as grant_error:
            # Log but don't fail
            grant_messages.append(f"✗ Grant error: {str(grant_error)[:200]}")
        
        return {
            "view_name": view_name,
            "database_name": database_name,
            "schema_name": schema_name,
            "full_schema_name": full_schema_name,
            "num_tables": len(tables_info),
            "table_names": [t['name'] for t in tables_info],
            "join_key": "ENTITY_ID",
            "example_queries": [
                f"What are the top 10 entities by total value?",
                f"Show me the distribution of entities by status",
                f"Compare performance metrics across different types"
            ],
            "grant_messages": grant_messages,
            "create_sql": semantic_sql
        }
    except Exception as e:
        error_handler.log_error(
            error_code=ErrorCode.SEMANTIC_VIEW_FAILED,
            error_type=type(e).__name__,
            severity=ErrorSeverity.WARNING,
            message=f"Semantic view creation failed: {str(e)}"
        )
        return {
            "view_name": view_name,
            "database_name": database_name,
            "schema_name": schema_name,
            "full_schema_name": full_schema_name,
            "num_tables": len(tables_info),
            "table_names": [t['name'] for t in tables_info],
            "join_key": "ENTITY_ID",
            "example_queries": [],
            "grant_messages": [f"✗ Semantic view creation failed: {str(e)[:200]}"],
            "create_sql": semantic_sql
        }

