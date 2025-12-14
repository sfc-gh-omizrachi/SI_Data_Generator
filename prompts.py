"""
LLM prompt templates for SI Data Generator application.

This module contains all LLM prompt generation functions used throughout
the application. Centralizing prompts here makes them easier to maintain,
test, and version independently from business logic.
"""

from typing import List, Dict, Optional, Any


def get_company_analysis_prompt(company_url: str) -> str:
    """
    Generate prompt for analyzing company URLs to infer business context.
    
    Args:
        company_url: Company website URL to analyze
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""Analyze this company URL and provide business context: {company_url}

Based on the domain name, provide:
1. Primary industry (e.g., Finance, Healthcare, Retail, Manufacturing, Technology, E-commerce, etc.)
2. Likely business focus (2-3 keywords)
3. Suggested data analysis focus (brief)

Respond ONLY in valid JSON format:
{{
  "industry": "Industry Name",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "context": "Brief description of likely business focus and data needs"
}}"""


def get_question_generation_prompt(
    num_questions: int,
    industry: str,
    company_name: str,
    demo_data: Dict,
    url_context: Optional[Dict] = None,
    table_info: str = "",
    columns_info: str = "",
    rich_table_contexts: Optional[List[Dict]] = None
) -> str:
    """
    Generate prompt for creating analytical questions for demos.
    
    Args:
        num_questions: Number of questions to generate
        industry: Industry context
        company_name: Company name for context
        demo_data: Demo configuration dictionary
        url_context: Optional URL analysis context
        table_info: Description of available tables (deprecated, use rich_table_contexts)
        columns_info: Description of available columns (deprecated, use rich_table_contexts)
        rich_table_contexts: Optional rich context with full schema, data analysis, and cardinality
        
    Returns:
        Formatted prompt string for LLM
    """
    # Add URL context if available
    url_context_text = ""
    if url_context and url_context.get('context'):
        url_context_text = f"\n\nCompany Context (from URL analysis):\n"
        url_context_text += f"- Industry: {url_context.get('industry', 'N/A')}\n"
        url_context_text += f"- Business Focus: {url_context.get('context', 'N/A')}\n"
        if url_context.get('keywords'):
            url_context_text += f"- Key Areas: {', '.join(url_context['keywords'])}\n"
        url_context_text += f"\nUse this context to make questions more relevant to {company_name}'s specific business needs."
    
    # Build comprehensive table context section from rich contexts
    detailed_table_info = ""
    if rich_table_contexts:
        detailed_table_info = "\n\n=== AVAILABLE DATA (COMPLETE CONTEXT) ===\n"
        
        for idx, ctx in enumerate(rich_table_contexts, 1):
            detailed_table_info += f"\nTable {idx}: {ctx['name']}\n"
            detailed_table_info += f"Description: {ctx['description']}\n"
            detailed_table_info += f"Row Count: {ctx['row_count']}\n"
            detailed_table_info += "\nColumns:\n"
            
            for col in ctx['columns']:
                detailed_table_info += f"  • {col['name']} ({col['type']}): {col['description']}\n"
                
                # Add cardinality for categorical columns
                if col.get('unique_count') is not None:
                    detailed_table_info += f"    - Unique values: {col['unique_count']}\n"
                    if col.get('sample_actual_values'):
                        samples = col['sample_actual_values'][:5]
                        detailed_table_info += f"    - Examples: {', '.join(str(s) for s in samples)}\n"
                
                # Add range for numeric columns
                if col.get('numeric_range'):
                    nr = col['numeric_range']
                    detailed_table_info += f"    - Range: {nr['min']} to {nr['max']} (avg: {nr['avg']:.2f})\n"
                
                # Add date range if available
                if col.get('date_range'):
                    dr = col['date_range']
                    detailed_table_info += f"    - Date range: {dr['min']} to {dr['max']}\n"
            
            # Add sample rows
            if ctx.get('sample_rows'):
                detailed_table_info += f"\n  Sample Data (first 3 rows):\n"
                for row_idx, row in enumerate(ctx['sample_rows'][:3], 1):
                    row_str = ", ".join([f"{k}={v}" for k, v in list(row.items())[:5]])
                    detailed_table_info += f"    Row {row_idx}: {row_str}...\n"
    else:
        # Fallback to old format if rich contexts not provided
        detailed_table_info = f"\n- Available Tables:\n{table_info}{columns_info}"
    
    return f"""Generate {num_questions} natural language questions for a {industry} data analysis demo.

Context:
- Company: {company_name}
- Demo: {demo_data.get('title', 'Data Analysis')}
- Business Focus: {demo_data.get('business_value', 'Improve decision making')}
{url_context_text}{detailed_table_info}

CRITICAL QUESTION GENERATION RULES:

1. COLUMN EXISTENCE VALIDATION (MOST IMPORTANT):
   - Before generating any question, verify that ALL required columns exist in the data context above
   - Example: Don't ask about "restaurants" unless you see a column like RESTAURANT_ID or RESTAURANT_NAME
   - Example: Don't ask about "waste quantities" unless you see both a column indicating waste (like MOVEMENT_TYPE with "waste" value) AND a quantity column
   - If a concept is mentioned in table description but you don't see the actual column above, DO NOT generate questions about it
   - Only generate questions about data that is explicitly shown in the columns list above

2. DATA-GROUNDED CONSTRAINTS:
   - When column has N unique values, NEVER ask for "top M" where M >= N
   - Example: 4 unique restaurants → max is "top 3" or "top 4", NOT "top 10"
   - For aggregations, ensure grouping columns have sufficient cardinality
   - Use actual sample values shown above to understand what data looks like
   - Questions must be answerable using ONLY the columns explicitly listed above

3. NUMERIC AWARENESS:
   - Respect min/max ranges shown above
   - Don't ask for values outside observed ranges
   - Use realistic thresholds based on averages

4. QUESTION TYPES MUST MATCH DATA:
   - If low cardinality (< 10 unique): ask for "all categories" or "breakdown by X"
   - If high cardinality (> 20 unique): ask for "top N" where N < cardinality/2
   - Prefer aggregations that work with available dimensions

5. USE ACTUAL SAMPLE DATA:
   - Reference actual value examples from sample data
   - Ensure question terminology matches column descriptions
   - Make questions specific to the data characteristics shown

6. CONSERVATIVE DEFAULTS:
   - If no cardinality info: use "top 5" max
   - Prefer "show me" over "top N" when uncertain
   - Focus on trends, patterns, and comparisons over exact counts
   - When in doubt, ask simpler, more general questions that are guaranteed to work

EXAMPLES OF GOOD VS BAD QUESTIONS:

Scenario: Table has columns [ENTITY_ID, SUPPLIER_NAME (4 unique), DELIVERY_DATE, QUALITY_SCORE (1-100)]

❌ BAD: "What are the top 10 suppliers by delivery performance?"
   Why: Only 4 unique suppliers exist, can't show top 10

✓ GOOD: "What are the top 3 suppliers by average quality score?"
   Why: Only 4 suppliers, asking for 3 is safe

❌ BAD: "How do metrics vary during promotional periods?"
   Why: No PROMOTIONAL_PERIOD column exists in data

✓ GOOD: "How do quality scores vary over time by supplier?"
   Why: Uses actual columns (QUALITY_SCORE, DELIVERY_DATE, SUPPLIER_NAME)

❌ BAD: "Show me year-over-year growth"
   Why: Only 100 records, may not span multiple years

✓ GOOD: "What is the average quality score by supplier?"
   Why: Simple aggregation guaranteed to work

FOLLOW THESE PATTERNS: Use actual column names, respect cardinality, avoid assumptions.

QUESTION VALIDATION CHECKLIST (use this BEFORE finalizing each question):

For each question you generate, verify:
1. ✓ All referenced concepts have corresponding columns in data above
2. ✓ Any "top N" has N < unique_count for that column
3. ✓ All filters/groupings use actual column names shown above
4. ✓ Date ranges are realistic given the data shown
5. ✓ No assumptions about data not explicitly shown

If ANY check fails, revise the question or discard it.

Requirements:
- Mix of difficulty levels: basic (simple queries), intermediate (trends/comparisons), advanced (insights/predictions)
- Include both analytics questions (70%) and search questions (30%) for unstructured content
- Questions should be realistic for {industry}
- Use business terminology, not technical database terms
- All tables join on ENTITY_ID column

Return ONLY a JSON array in this exact format:
[
  {{"text": "What are the top 5 entities by total value?", "difficulty": "basic", "category": "analytics"}},
  {{"text": "Show me trends over time for key metrics", "difficulty": "intermediate", "category": "analytics"}},
  {{"text": "Find best practices for improving performance", "difficulty": "basic", "category": "search"}}
]
"""


def get_follow_up_questions_prompt(primary_question: str) -> str:
    """
    Generate prompt for creating follow-up questions.
    
    Args:
        primary_question: Primary question to generate follow-ups for
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""Given this analytical question: "{primary_question}"

Generate 2-3 natural follow-up questions that:
- Build on insights from the primary question
- Explore the "why" behind patterns
- Suggest next steps or actions
- Are conversational and business-focused

Return ONLY a JSON array of question strings:
["follow-up question 1", "follow-up question 2", "follow-up question 3"]
"""


def get_target_question_analysis_prompt(questions: List[str]) -> str:
    """
    Generate prompt for analyzing target questions to extract requirements.
    
    Args:
        questions: List of target questions to analyze
        
    Returns:
        Formatted prompt string for LLM
    """
    questions_text = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
    
    return f"""Analyze these target questions to determine what data structures, dimensions, and metrics are needed to answer them:

Target Questions:
{questions_text}

For each question, identify:
1. **Required Dimensions**: Data attributes/fields needed (e.g., age_range, product_category, time_period, customer_segment, geographic_region)
2. **Metrics Needed**: Calculations or measurements (e.g., percentage, count, average, sum, growth_rate, ratio)
3. **Data Characteristics**: Specific requirements for the data (e.g., "need age field with values 18-65", "need timestamp field for time-based analysis", "need numeric revenue field for aggregation")

Return ONLY a JSON object in this exact format:
{{
  "required_dimensions": ["dimension1", "dimension2", "dimension3"],
  "metrics_needed": ["metric1", "metric2"],
  "data_characteristics": {{
    "numeric_fields": ["list of numeric fields needed"],
    "categorical_fields": ["list of categorical fields with examples"],
    "temporal_fields": ["list of date/time fields needed"],
    "value_ranges": {{"field_name": "description of range needed"}},
    "special_requirements": ["any other specific requirements"]
  }},
  "question_types": ["analytical", "aggregation", "trend", "comparison"]
}}

Be specific and comprehensive. Extract all implicit and explicit requirements."""


def get_agent_system_prompt(
    demo_data: Dict,
    company_name: str
) -> str:
    """
    Generate comprehensive system prompt for Snowflake Intelligence agent.
    
    Args:
        demo_data: Demo configuration dictionary
        company_name: Company name for context
        
    Returns:
        Formatted system prompt string
    """
    industry = demo_data.get(
        'industry_focus',
        demo_data.get('industry', 'Business Intelligence')
    )
    description = demo_data.get('description', '')
    business_value = demo_data.get(
        'business_value',
        'Improve operational efficiency and decision making'
    )
    
    return f"""You are an AI-powered data analyst assistant for {company_name}.

Your expertise includes:
- {industry} domain knowledge and industry best practices
- Advanced data analysis and pattern recognition
- SQL query generation and optimization
- Natural language insights and recommendations
- Data visualization and reporting

You have access to:
- Structured data tables via Cortex Analyst for quantitative analysis
- Unstructured documents via Cortex Search for qualitative insights

Context for this demo:
{description}

Business Value Focus:
{business_value}

Your goal is to help users understand their data through:
- Answering analytical questions with data-driven insights
- Generating actionable recommendations based on analysis
- Finding relevant information in documents and reports
- Explaining complex data patterns in simple terms
- Proactively suggesting follow-up analyses

Guidelines:
- Always provide context for your answers
- Use specific numbers and data points when available
- Suggest follow-up questions when appropriate
- Explain your reasoning when making recommendations
- Be concise but comprehensive in your responses"""


def get_agent_persona_prompt(
    demo_data: Dict,
    company_name: str
) -> str:
    """
    Generate prompt for creating agent persona description.
    
    Args:
        demo_data: Demo configuration dictionary
        company_name: Company name for context
        
    Returns:
        Formatted prompt string for LLM
    """
    industry = demo_data.get(
        'industry_focus',
        demo_data.get('industry', 'Business Intelligence')
    )
    
    return f"""Generate a professional persona description for an AI data analyst agent specialized in {industry} for {company_name}.

The persona should:
- Highlight relevant industry expertise
- Emphasize analytical capabilities
- Be professional and trustworthy
- Be 2-3 sentences long

Return only the persona description, no additional text."""


def get_demo_generation_prompt(
    company_name: str,
    team_members: str,
    use_cases: str,
    num_ideas: int = 3,
    target_questions: Optional[List[str]] = None,
    advanced_mode: bool = False
) -> str:
    """
    Generate comprehensive prompt for creating demo scenarios.
    
    This is the largest and most complex prompt, used to generate complete
    demo scenarios with multiple tables and realistic data structures.
    
    Args:
        company_name: Company name for context
        team_members: Target audience description
        use_cases: Specific use cases to address
        num_ideas: Number of demo scenarios to generate
        target_questions: Optional list of questions demos should answer
        advanced_mode: Whether to generate advanced schemas (3-5 tables)
        
    Returns:
        Formatted prompt string for LLM
    """
    use_case_context = (
        f"\n- Use Cases: {use_cases}"
        if use_cases
        else "\n- Use Cases: Not specified"
    )
    
    # Add target questions context if provided
    target_questions_context = ""
    if target_questions and len(target_questions) > 0:
        questions_list = "\n".join(
            [f"  {i+1}. {q}" for i, q in enumerate(target_questions)]
        )
        target_questions_context = f"""

CRITICAL REQUIREMENT - Target Questions:
The demo MUST be designed to answer these specific questions:
{questions_list}

Data Generation Strategy:
PRIMARY GOAL (70%): Ensure the data can answer the target questions above
SECONDARY GOAL (30%): Include additional varied data for general analytics

Ensure that:
- Table descriptions explicitly include EXACT data fields necessary to answer these questions
- Demo scenarios naturally support these analytical goals
- Column types and structures enable the required calculations and aggregations
- Data distributions are realistic (not 100% or 0% for percentages, sufficient variety for "top N" queries)
"""
    
    # Advanced mode specifications
    if advanced_mode:
        table_spec = """2. 3-5 structured data tables forming a realistic data model:
   - 1-2 FACT tables (transactional/event data with metrics and foreign keys)
   - 2-3 DIMENSION tables (descriptive attributes, lookups, reference data)
   - Ensure proper foreign key relationships between tables (fact tables reference dimension tables)
   - Design for realistic star or snowflake schema patterns
3. 1-2 unstructured data tables (for Cortex Search)

Table Structure Requirements for Advanced Mode:
- Primary fact table(s) should have foreign keys to dimension tables
- Dimension tables should have clear business meaning (customers, products, dates, categories, locations)
- Include both granular and aggregated data opportunities
- Design for realistic business analytics scenarios with joins"""
        schema_example = """
{
  "demos": [
    {
      "title": "Demo Title",
      "description": "Detailed description",
      "industry_focus": "Industry",
      "business_value": "Business value",
      "tables": {
        "structured_1": {"name": "FACT_TABLE", "description": "Main fact table with metrics and FKs", "purpose": "Primary analytics", "table_type": "fact"},
        "structured_2": {"name": "DIM_TABLE_1", "description": "First dimension table", "purpose": "Descriptive attributes", "table_type": "dimension"},
        "structured_3": {"name": "DIM_TABLE_2", "description": "Second dimension table", "purpose": "Lookup data", "table_type": "dimension"},
        "structured_4": {"name": "DIM_TABLE_3", "description": "Optional third dimension", "purpose": "Additional context", "table_type": "dimension"},
        "structured_5": {"name": "OPTIONAL_FACT_2", "description": "Optional second fact table", "purpose": "Secondary metrics", "table_type": "fact"},
        "unstructured": {"name": "CONTENT_CHUNKS", "description": "Unstructured text data", "purpose": "Semantic search"},
        "unstructured_2": {"name": "OPTIONAL_DOCS", "description": "Optional second unstructured table", "purpose": "Additional semantic search content"}
      }
    }
  ]
}

NOTE: Include 3-5 structured tables. structured_4 and structured_5 are optional but recommended for richer models. unstructured_2 is optional for additional search content."""
    else:
        table_spec = """2. 3 structured data tables forming a realistic data model:
   - 1 FACT table (transactional/event data with metrics and foreign keys)
   - 2 DIMENSION tables (descriptive attributes, lookups, reference data)
   - Ensure proper foreign key relationships between tables (fact table references dimension tables)
   - Design for realistic star schema pattern
3. 1 unstructured data table (for Cortex Search)

Table Structure Requirements for Standard Mode:
- Primary fact table should have foreign keys to dimension tables
- Dimension tables should have clear business meaning (customers, products, dates, categories, locations)
- Include both granular and aggregated data opportunities
- Design for realistic business analytics scenarios with joins"""
        schema_example = """
{
  "demos": [
    {
      "title": "Demo Title",
      "description": "Detailed description",
      "industry_focus": "Industry",
      "business_value": "Business value",
      "tables": {
        "structured_1": {"name": "FACT_TABLE", "description": "Main fact table with metrics and FKs", "purpose": "Primary analytics", "table_type": "fact"},
        "structured_2": {"name": "DIM_TABLE_1", "description": "First dimension table", "purpose": "Descriptive attributes", "table_type": "dimension"},
        "structured_3": {"name": "DIM_TABLE_2", "description": "Second dimension table", "purpose": "Lookup data", "table_type": "dimension"},
        "unstructured": {"name": "CONTENT_CHUNKS", "description": "Unstructured text data", "purpose": "Semantic search"}
      }
    }
  ]
}

NOTE: Include exactly 3 structured tables (1 fact + 2 dimensions) and 1 unstructured table."""
    
    return f"""You are a Snowflake solutions architect creating tailored demo scenarios for a customer. 

IMPORTANT: You MUST generate exactly {num_ideas} complete, distinct demo scenarios. Each demo should be fully detailed and different from the others.

Customer Information:
- Company: {company_name}
- Team/Audience: {team_members}
{use_case_context}{target_questions_context}

For EACH of the {num_ideas} demos, provide:
1. A compelling title and detailed description
{table_spec}

Requirements:
- Make demos relevant to the company's likely industry/domain
- Consider the audience when designing complexity
- Focus on business value and real-world scenarios
- Ensure table names are SQL-friendly (uppercase, underscores)
- Provide DETAILED, SPECIFIC table descriptions that explain what data fields and metrics are included (e.g., "Customer touchpoint data including product usage, support tickets, timestamps, interaction types, and feature adoption metrics")
- Generate {num_ideas} COMPLETE demo objects - do not stop after just one!

Return ONLY a JSON object with this exact structure (with {num_ideas} complete demo objects in the array):
{schema_example}"""


def get_schema_generation_prompt(
    table_name: str,
    table_description: str,
    company_name: str,
    target_questions: Optional[List[str]] = None,
    question_analysis: Optional[Dict] = None,
    required_fields: Optional[List[Dict]] = None
) -> str:
    """
    Generate prompt for creating realistic table schemas.
    
    Args:
        table_name: Name of table to generate schema for
        table_description: Description of table purpose and contents
        company_name: Company name for context
        target_questions: Optional list of questions schema should support
        question_analysis: Optional analysis of question requirements
        required_fields: Optional list of mandatory fields extracted from description
        
    Returns:
        Formatted prompt string for LLM
    """
    # Add target questions context if provided
    target_questions_context = ""
    if target_questions and len(target_questions) > 0:
        questions_list = "\n".join([f"  - {q}" for q in target_questions])
        target_questions_context = f"""

CRITICAL - Target Questions Support:
This table MUST support answering these questions:
{questions_list}
"""
    
    # Add question analysis context if provided
    analysis_context = ""
    if question_analysis and question_analysis.get('has_target_questions'):
        required_dims = question_analysis.get('required_dimensions', [])
        metrics_needed = question_analysis.get('metrics_needed', [])
        data_chars = question_analysis.get('data_characteristics', {})
        
        if required_dims:
            analysis_context += f"\nRequired Dimensions: {', '.join(required_dims)}"
        if metrics_needed:
            analysis_context += f"\nMetrics to Support: {', '.join(metrics_needed)}"
        if data_chars:
            numeric_fields = data_chars.get('numeric_fields', [])
            categorical_fields = data_chars.get('categorical_fields', [])
            temporal_fields = data_chars.get('temporal_fields', [])
            
            if numeric_fields:
                analysis_context += (
                    f"\nNumeric Fields Needed: {', '.join(numeric_fields)}"
                )
            if categorical_fields:
                analysis_context += (
                    f"\nCategorical Fields Needed: "
                    f"{', '.join(categorical_fields)}"
                )
            if temporal_fields:
                analysis_context += (
                    f"\nTemporal Fields Needed: {', '.join(temporal_fields)}"
                )
        
        if analysis_context:
            target_questions_context += f"\nBased on analysis:{analysis_context}\n"
    
    # Build mandatory fields section
    mandatory_fields_section = ""
    if required_fields and len(required_fields) > 0:
        mandatory_fields_section = "\n\nMANDATORY FIELDS - MUST INCLUDE:\n"
        for field in required_fields:
            sample_vals = field.get('sample_values', [])
            sample_str = f"Sample values: {sample_vals}" if sample_vals else "No sample values provided"
            mandatory_fields_section += f"- {field['field_name']} ({field['suggested_type']}): {field.get('description', 'Required field')}\n  {sample_str}\n"
        
        mandatory_fields_section += """
CRITICAL: The fields listed above are MANDATORY. You MUST include every single one in your schema.
Failure to include any mandatory field will result in regeneration.

You may add 3-5 ADDITIONAL columns beyond the mandatory ones to enrich the schema.
"""
    
    return f"""Generate a realistic database schema for a table named {table_name}.

Description: {table_description}
Company Context: {company_name}{target_questions_context}{mandatory_fields_section}

CRITICAL - Field Name Extraction from Description:
The table description above mentions SPECIFIC field names and their expected values. You MUST:
1. Parse the description carefully to identify all explicitly mentioned field names
2. Create columns with those EXACT field names (convert to UPPERCASE with underscores)
3. When the description mentions field values in parentheses, use those as sample_values
4. If the description says "including X, Y, Z", those become required columns

Examples of field extraction:
- "movement_type (receipt, usage, waste, transfer)" → column "MOVEMENT_TYPE" with sample_values: ["receipt", "usage", "waste", "transfer", ...]
- "including restaurant_id, supplier_id, product_id" → columns "RESTAURANT_ID", "SUPPLIER_ID", "PRODUCT_ID"
- "quality_score" → column "QUALITY_SCORE"

Requirements:
- First column should always be ENTITY_ID (NUMBER) as the primary key
- Include ALL mandatory fields listed above (if any) - these are non-negotiable
- Extract and include ALL field names explicitly mentioned in the table description
- Add 3-5 additional meaningful columns for enrichment beyond mandatory fields
- Use realistic column names and types (NUMBER, STRING, DATE, TIMESTAMP, BOOLEAN, FLOAT)
- Make it practical for analytics and business intelligence
- For each column, provide 15-30 diverse and realistic sample values to ensure data variety
- For mandatory fields with sample_values, include ALL those values plus 10-20 more variations
- Sample values should represent the full range of expected variation
- When the description specifies values in parentheses, include ALL of them plus additional variations
- Ensure columns support the required dimensions, metrics, and analytical capabilities mentioned above

Format as JSON:
{{
  "columns": [
    {{"name": "ENTITY_ID", "type": "NUMBER", "description": "Unique identifier", "sample_values": [1, 2, 3, 4, 5, ...]}},
    {{"name": "FIELD_FROM_DESCRIPTION", "type": "STRING", "description": "Description text", "sample_values": ["value1", "value2", "value3", ...]}},
    {{"name": "ANOTHER_FIELD", "type": "TYPE", "description": "Column purpose", "sample_values": ["example1", "example2", ...]}}
  ]
}}

IMPORTANT: 
- Strictly honor the field names mentioned in the table description - these are NOT optional suggestions
- Provide realistic, business-appropriate sample_values based on the table description
- For categorical fields mentioned with specific values (like movement_type with "receipt, usage, waste"), include those exact values plus variations
- Ensure sample values have appropriate distributions to answer the target questions if specified"""


def get_collective_validation_prompt(
    tables_text: str,
    questions_text: str
) -> str:
    """
    Generate prompt for validating that tables collectively answer questions.
    
    Args:
        tables_text: Formatted description of all available tables
        questions_text: Formatted list of target questions
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""Validate if these tables TOGETHER can answer the target questions.

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


def get_single_table_validation_prompt(
    table_name: str,
    columns_text: str,
    sample_data_text: str,
    questions_text: str
) -> str:
    """
    Generate prompt for validating single table against questions.
    
    Args:
        table_name: Name of table being validated
        columns_text: Formatted description of table columns
        sample_data_text: Sample data from table
        questions_text: Formatted list of target questions
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""Validate if this SINGLE table can contribute to answering the target questions.

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


def get_unstructured_data_generation_prompt(
    table_name: str,
    table_description: str,
    company_name: str,
    num_chunks: int = 5
) -> str:
    """
    Generate prompt for creating unstructured text content.
    
    Args:
        table_name: Name of unstructured table
        table_description: Description of content type
        company_name: Company name for context
        num_chunks: Number of text chunks to generate
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""Generate {min(num_chunks, 5)} realistic text chunks for this unstructured data:

Type: {table_name}
Description: {table_description}
Company: {company_name}

Each chunk should be 2-3 paragraphs of realistic business content.
Format as JSON array:
[
  {{"chunk_text": "Content here...", "document_type": "type", "source_system": "system"}}
]"""


def get_table_relationships_analysis_prompt(
    demo_data: Dict,
    tables_text: str,
    questions_context: str = ""
) -> str:
    """
    Generate prompt for analyzing table relationships and join patterns.
    
    Args:
        demo_data: Demo configuration dictionary
        tables_text: Formatted description of all tables
        questions_context: Optional context about target questions
        
    Returns:
        Formatted prompt string for LLM
    """
    return f"""You are a data architect analyzing a data model for a demo scenario.

## Demo Context:
**Title:** {demo_data.get('title', 'N/A')}
**Description:** {demo_data.get('description', 'N/A')}

## Tables in the Data Model:
{tables_text}

## Technical Details:
- All structured tables have ENTITY_ID as the primary key
- Tables are designed with 70% join overlap on ENTITY_ID
- This is a star or snowflake schema pattern with fact and dimension tables
{questions_context}

## Your Task:
Analyze this data model and provide:

1. **Model Type**: Identify if this is a Star Schema, Snowflake Schema, or other pattern
2. **Relationships**: Describe how each table connects to others (which are fact tables, which are dimensions, what joins are used)
3. **Join Paths**: Provide 2-3 example SQL JOIN queries showing how to combine tables for analysis
4. **Question Mapping**: For each target question (if provided), explain which tables and joins are needed

Return ONLY a JSON object with this structure:
{{
  "model_type": "Star Schema" or "Snowflake Schema",
  "relationships": [
    "Fact table X connects to Dimension table Y via ENTITY_ID for detailed analysis",
    "Dimension table A provides descriptive attributes for entities in fact tables"
  ],
  "join_paths": [
    {{"description": "Purpose of join", "sql": "SELECT ... FROM ... JOIN ... ON ..."}}
  ],
  "insights": [
    "This model enables X type of analysis",
    "Users can correlate Y with Z across tables"
  ]
}}"""

