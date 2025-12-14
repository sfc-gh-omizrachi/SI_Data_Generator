-- ================================================================================
-- SI Data Generator Setup Script
-- 
-- This script sets up the complete environment for the SI Data Generator
-- Streamlit application including database, permissions, stage for files,
-- and the Streamlit app itself.
--
-- Requirements:
-- - ACCOUNTADMIN role to create roles, databases, and integrations
-- - Application files uploaded to the stage (instructions below)
-- ================================================================================

USE ROLE ACCOUNTADMIN;

-- ================================================================================
-- 1. CREATE DATABASES AND SCHEMAS
-- ================================================================================

-- Create main demo database
CREATE DATABASE IF NOT EXISTS SI_DEMOS
    COMMENT = 'Database for Snowflake Intelligence demo data and applications';

-- Create schema for the application
CREATE SCHEMA IF NOT EXISTS SI_DEMOS.APPLICATIONS
    COMMENT = 'Schema for Streamlit applications and notebooks';

-- Create schema for demo data (this will be populated by the app)
CREATE SCHEMA IF NOT EXISTS SI_DEMOS.DEMO_DATA
    COMMENT = 'Schema for generated demo data tables and semantic views';

-- ================================================================================
-- 1.5. CREATE HISTORY TRACKING TABLE
-- ================================================================================

-- Create history table to track all demo generations
CREATE TABLE IF NOT EXISTS SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY (
    HISTORY_ID VARCHAR(50) PRIMARY KEY,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP,
    COMPANY_NAME VARCHAR(500),
    COMPANY_URL VARCHAR(1000),
    DEMO_TITLE VARCHAR(500),
    DEMO_DESCRIPTION VARCHAR(5000),
    SCHEMA_NAME VARCHAR(500),
    NUM_RECORDS INTEGER,
    LANGUAGE_CODE VARCHAR(10),
    TEAM_MEMBERS VARCHAR(1000),
    USE_CASES VARCHAR(5000),
    ENABLE_SEMANTIC_VIEW BOOLEAN,
    ENABLE_SEARCH_SERVICE BOOLEAN,
    ENABLE_AGENT BOOLEAN,
    ADVANCED_MODE BOOLEAN,
    TABLE_NAMES VARIANT,
    TARGET_QUESTIONS VARIANT,
    GENERATED_QUESTIONS VARIANT,
    DEMO_DATA_JSON VARIANT
)
COMMENT = 'Tracks all demo generations for history and re-use';



-- Create Snowflake Intelligence database and schema for agent discovery
CREATE DATABASE IF NOT EXISTS SNOWFLAKE_INTELLIGENCE
    COMMENT = 'Database for Snowflake Intelligence agents - enables UI discoverability';

CREATE SCHEMA IF NOT EXISTS SNOWFLAKE_INTELLIGENCE.AGENTS
    COMMENT = 'Schema for AI agents - agents here appear in Snowsight under AI & ML » Agents';

-- Grant usage to allow agent creation and discovery
GRANT USAGE ON DATABASE SNOWFLAKE_INTELLIGENCE TO ROLE ACCOUNTADMIN;
GRANT USAGE ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;
GRANT CREATE AGENT ON SCHEMA SNOWFLAKE_INTELLIGENCE.AGENTS TO ROLE ACCOUNTADMIN;

-- ================================================================================
-- 2. CREATE COMPUTE RESOURCES
-- ================================================================================

-- Create warehouse for the application
CREATE WAREHOUSE IF NOT EXISTS SI_DEMO_WH
    WITH 
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    COMMENT = 'Warehouse for SI Data Generator application';

-- Create compute warehouse for Cortex operations
CREATE WAREHOUSE IF NOT EXISTS compute_wh
    WITH 
    WAREHOUSE_SIZE = 'X-SMALL'
    AUTO_SUSPEND = 300
    AUTO_RESUME = TRUE
    COMMENT = 'Compute warehouse for Cortex LLM operations';

-- ================================================================================
-- 3. ENABLE CORTEX AND VERIFY SETUP
-- ================================================================================
ALTER ACCOUNT SET CORTEX_ENABLED_CROSS_REGION = 'ANY_REGION';

-- ================================================================================
-- 4. CREATE STAGE FOR APPLICATION FILES
-- ================================================================================

-- Switch to the application schema
USE SCHEMA SI_DEMOS.APPLICATIONS;

-- Create internal stage for Streamlit application files
CREATE OR REPLACE STAGE SI_DATA_GENERATOR_STAGE
    DIRECTORY = (ENABLE = TRUE)
    COMMENT = 'Stage for SI Data Generator Streamlit application files';

-- Grant privileges on the stage
GRANT READ, WRITE ON STAGE SI_DATA_GENERATOR_STAGE TO ROLE ACCOUNTADMIN;

-- ================================================================================
-- 5. UPLOAD APPLICATION FILES TO STAGE
-- ================================================================================

/*
IMPORTANT: Before creating the Streamlit app, upload your application files to the stage.

METHOD 1: Using SnowSQL (Command Line)
--------------------------------------
snowsql -a <account> -u <username> -r ACCOUNTADMIN

-- Upload all files from your local directory to the stage
PUT file:///path/to/SI_Data_Generator-main/*.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- If you have subdirectories, upload them too (e.g., for any assets or configs)
-- PUT file:///path/to/SI_Data_Generator-main/assets/* @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE/assets AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

METHOD 2: Using Snowsight UI (Web Interface)
--------------------------------------------
1. Navigate to: Data » Databases » SI_DEMOS » APPLICATIONS » Stages
2. Click on: SI_DATA_GENERATOR_STAGE
3. Click: "+ Files" button in the top-right
4. Upload all your .py files from the SI_Data_Generator-main directory
   Required files:
   - SI_Generator.py (this is your main file)
   - demo_content.py
   - errors.py
   - infrastructure.py
   - metrics.py
   - prompts.py
   - styles.py
   - utils.py
   - environment.yml (if needed for dependencies)

METHOD 3: Using Python Connector
--------------------------------
import snowflake.connector
from pathlib import Path

conn = snowflake.connector.connect(
    user='<username>',
    password='<password>',
    account='<account>',
    role='ACCOUNTADMIN'
)

cursor = conn.cursor()
cursor.execute("USE SCHEMA SI_DEMOS.APPLICATIONS")

# Upload each file
files = Path('/path/to/SI_Data_Generator-main').glob('*.py')
for file in files:
    cursor.execute(f"PUT file://{file} @SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE")

conn.close()

VERIFICATION: List files in stage
----------------------------------
*/

-- After uploading, verify files are in the stage:
LIST @SI_DATA_GENERATOR_STAGE;

-- ================================================================================
-- 6. CREATE STREAMLIT APPLICATION
-- ================================================================================

-- Switch to the application schema and warehouse
USE SCHEMA SI_DEMOS.APPLICATIONS;
USE WAREHOUSE SI_DEMO_WH;

-- Create the Streamlit application from stage
CREATE OR REPLACE STREAMLIT SI_DATA_GENERATOR_APP
    ROOT_LOCATION = '@SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE'
    MAIN_FILE = 'SI_Generator.py'
    QUERY_WAREHOUSE = SI_DEMO_WH
    COMMENT = 'SI Data Generator Streamlit Application'
    TITLE = 'Snowflake Agent Demo Data Generator';

-- Grant usage on the Streamlit app to ACCOUNTADMIN
GRANT USAGE ON STREAMLIT SI_DATA_GENERATOR_APP TO ROLE ACCOUNTADMIN;

-- ================================================================================
-- 7. ADDITIONAL SETUP FOR CORTEX SEARCH
-- ================================================================================

-- NOTE: No additional setup required for Cortex Search
-- Cortex Search services are created dynamically by the application when:
--   1. Users generate demo data through the Streamlit app
--   2. Enable the "Create Cortex Search Service" option
-- ACCOUNTADMIN role has all necessary privileges to create services
-- Cortex is already enabled via CORTEX_ENABLED_CROSS_REGION setting above

-- ================================================================================
-- 8. SETUP VALIDATION AND INFORMATION
-- ================================================================================

-- Set context for testing
USE ROLE ACCOUNTADMIN;
USE WAREHOUSE SI_DEMO_WH;
USE DATABASE SI_DEMOS;

-- Show created objects
SHOW DATABASES LIKE 'SI_DEMOS';
SHOW SCHEMAS IN DATABASE SI_DEMOS;
SHOW WAREHOUSES LIKE 'SI_DEMO_WH';
SHOW STAGES IN SCHEMA SI_DEMOS.APPLICATIONS;

-- Test Cortex function access
SELECT SNOWFLAKE.CORTEX.COMPLETE(
    'claude-4-sonnet', 
    'Say "SI Data Generator setup is working!" and nothing else.'
) AS test_result;

-- ================================================================================
-- 9. POST-SETUP INSTRUCTIONS
-- ================================================================================

/*
================================================================================
POST-SETUP INSTRUCTIONS
================================================================================

1. FILE UPLOAD (REQUIRED BEFORE FIRST USE):
   - Upload all Python files to: @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE
   - Required files: SI_Generator.py, demo_content.py, errors.py, infrastructure.py,
                    metrics.py, prompts.py, styles.py, utils.py
   - Use one of the methods shown in section 5 above
   - Verify with: LIST @SI_DATA_GENERATOR_STAGE;

2. STREAMLIT APP ACCESS:
   - The Streamlit app is available at: SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_APP
   - Access URL: Go to Snowsight » Streamlit » SI_DATA_GENERATOR_APP
   - Or navigate directly to: https://<account>.snowflakecomputing.com/streamlit/SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_APP

3. UPDATING THE APPLICATION:
   - To update the app, simply re-upload the modified files to the stage:
     PUT file:///path/to/updated_file.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
   - The Streamlit app will automatically pick up the changes on next load
   - No need to recreate the Streamlit object

4. USER ACCESS:
   - All operations run with ACCOUNTADMIN privileges
   - No additional role grants needed for basic functionality

5. DEMO USAGE:
   - The app will create schemas under SI_DEMOS with pattern: SI_DEMOS.[COMPANY]_DEMO_[DATE]
   - Each demo creates tables, semantic views, and Cortex Search services
   - All demo data is isolated by schema for easy cleanup
   
6. AGENT CREATION:
   - Agents are created in SNOWFLAKE_INTELLIGENCE.AGENTS schema
   - Agent naming: [COMPANY]_[DATE]_AGENT (e.g., ACMECORP_20250131_AGENT)
   - Agents appear in Snowsight UI under: AI & ML » Agents
   - Agents persist after demo schema cleanup (manual deletion required if needed)
   - To delete an agent: DROP AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.[AGENT_NAME];

7. CUSTOMIZATION:
   - Modify the Streamlit app by editing Python files locally
   - Re-upload to stage to deploy changes
   - Add new demo templates by modifying the fallback demo ideas
   - Customize warehouse size based on expected usage

8. MONITORING:
   - Monitor warehouse usage and costs
   - Review generated schemas periodically for cleanup
   - Check Cortex function usage for cost optimization

9. PERMISSIONS:
   - All permissions handled by ACCOUNTADMIN role
   - No additional role management required

10. CLEANUP:
    To remove demo data, drop the individual demo schemas:
    DROP SCHEMA SI_DEMOS.[COMPANY]_DEMO_[DATE];
    
    To remove agents (if needed):
    DROP AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.[COMPANY]_[DATE]_AGENT;
    
    To remove the stage and all files:
    DROP STAGE SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;

================================================================================
SETUP COMPLETE - NEXT STEP: UPLOAD FILES TO STAGE (see section 5)
================================================================================
*/

-- Final verification
SELECT 'SI Data Generator setup completed successfully! 🎉' AS status,
       'Next step: Upload application files to @SI_DATA_GENERATOR_STAGE' AS next_action,
       CURRENT_ROLE() AS current_role,
       CURRENT_WAREHOUSE() AS current_warehouse,
       CURRENT_DATABASE() AS current_database,
       CURRENT_SCHEMA() AS current_schema;
