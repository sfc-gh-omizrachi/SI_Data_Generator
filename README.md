## ❄️ Snowflake Intelligence Data Generator\n
\n
Generate complete, AI-powered demo environments for Snowflake Cortex Analyst, Cortex Search, and Snowflake Intelligence in minutes. This Streamlit app creates realistic structured and unstructured data, semantic views, and optional agent tooling tailored to a customer’s industry and use cases.\n
\n
### What this app creates\n
- **AI-tailored demo scenarios** from customer context (company URL, audience, use cases)\n
- **Structured tables** with realistic business data and join-ready keys (ENTITY_ID)\n
- **Unstructured content** optimized for semantic search\n
- **Semantic views** ready for Cortex Analyst (joins, synonyms, example queries)\n
- **Optional**: Cortex Search service and Snowflake Intelligence agent\n
- **History tracking** of demo generations for reuse\n
\n
### Repository layout\n
- `SI_Generator.py` — Streamlit UI and end-to-end flow\n
- `demo_content.py` — demo ideas, schema generation, data generation, validation\n
- `infrastructure.py` — semantic view, search service, and agent automation\n
- `prompts.py` — LLM prompt builders\n
- `utils.py` — Snowflake session, LLM helpers, language support, parallelism\n
- `errors.py` — error handling, retry utilities, decorators\n
- `metrics.py` — timing and progress tracking utilities\n
- `styles.py` — Streamlit CSS and rendering helpers\n
- `Setup.sql` — full Snowflake environment setup (DBs, stage, Streamlit app)\n
- `environment.yml` — optional local conda environment\n
\n
---\n
\n
## 🚀 Quick start (Snowflake)\n
\n
Use this path if you want the app running in Snowsight with one setup.\n
\n
### Prerequisites\n
- Snowflake account with privileges to run the setup (recommended: `ACCOUNTADMIN` for initial setup)\n
- Cortex access enabled in your account\n
- Streamlit in Snowflake enabled\n
\n
### 1) Run the setup script\n
Run the full script in Snowsight Worksheets as `ACCOUNTADMIN` to create databases, schemas, warehouses, stage, history table, and the Streamlit app:\n
\n
```sql\n
-- Open and run this file from the repo\n
-- File: Setup.sql\n
-- Creates:\n
-- - SI_DEMOS database and schemas (APPLICATIONS, DEMO_DATA)\n
-- - SNOWFLAKE_INTELLIGENCE.AGENTS schema for agent discoverability\n
-- - SI_DEMO_WH and compute_wh warehouses\n
-- - SI_DATA_GENERATOR_STAGE stage for app files\n
-- - SI_GENERATOR_HISTORY tracking table\n
-- - SI_DATA_GENERATOR_APP Streamlit application\n
```\n
\n
What the script provisions (high level):\n
- Database: `SI_DEMOS`\n
- Schemas: `SI_DEMOS.APPLICATIONS`, `SI_DEMOS.DEMO_DATA`\n
- History table: `SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY`\n
- Stage: `SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE`\n
- Warehouses: `SI_DEMO_WH`, `compute_wh`\n
- Intelligence DB/Schema: `SNOWFLAKE_INTELLIGENCE.AGENTS`\n
- Streamlit app: `SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_APP` pointing to `SI_Generator.py`\n
\n
### 2) Upload application files to the stage (required before first use)\n
Upload all app files to the stage created by the setup script. From Snowsight or SnowSQL:\n
\n
```sql\n
-- Replace /local/path with the path where these files live locally\n
PUT file:///local/path/SI_Generator.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/demo_content.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/errors.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/infrastructure.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/metrics.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/prompts.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/styles.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
PUT file:///local/path/utils.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
\n
-- Verify upload\n
LIST @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;\n
```\n
\n
Note: The Streamlit object is already created by `Setup.sql`. Re-upload files to this stage anytime you update the code; the app will pick them up on reload.\n
\n
### 3) Open the Streamlit app\n
- In Snowsight, navigate to: Data → Streamlit → `SI_DATA_GENERATOR_APP`\n
- The app runs in the role/warehouse configured by the setup script\n
\n
---\n
\n
## 🎮 How to use the app (Snowsight)\n
\n
1. **Customer info**\n
   - Enter Company URL, Team Members/Audience, optional Use Cases, and Records per table\n
   - Choose content language\n
\n
2. **Generate demo ideas**\n
   - Click “Generate Demo Ideas”\n
   - The app uses Cortex LLM to produce 3 tailored scenarios (fallback templates if Cortex is unavailable)\n
\n
3. **Select a demo**\n
   - Review each scenario: structured tables (facts/dimensions), unstructured content, purposes\n
   - Select the best fit\n
\n
4. **Configure & create infrastructure**\n
   - Provide a schema name (timestamped default is suggested)\n
   - Optional toggles: Semantic View, Cortex Search Service, AI Agent\n
   - Click “Create Demo Infrastructure”\n
\n
5. **Done!**\n
   - The app creates:\n
     - Structured tables with `ENTITY_ID` PK and realistic data\n
     - Unstructured chunks table for search\n
     - Optional semantic view, search service, and agent\n
   - History is stored in `SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY`\n
\n
---\n
\n
## 💻 Local development (optional)\n
\n
Run locally if you want to iterate on UI/logic before staging the files.\n
\n
### Using pip\n
```bash\n
python -m pip install -U pip\n
pip install streamlit snowflake-snowpark-python pandas cryptography\n
streamlit run SI_Generator.py\n
```\n
\n
### Using conda (environment.yml)\n
```bash\n
conda env create -f environment.yml\n
conda activate si-data-generator\n
streamlit run SI_Generator.py\n
```\n
\n
Configure your local Snowflake connection as needed (e.g., environment variables, connection profile) so the app can create schemas/tables/views in your account.\n
\n
---\n
\n
## 🧱 Architecture (at a glance)\n
\n
```\n
SI_DEMOS (Database)\n
├── APPLICATIONS (Schema)\n
│   ├── SI_DATA_GENERATOR_STAGE (Stage for app files)\n
│   ├── SI_GENERATOR_HISTORY (History table)\n
│   └── SI_DATA_GENERATOR_APP (Streamlit App)\n
├── DEMO_DATA (Schema for generated data)\n
└── [COMPANY]_DEMO_[YYYYMMDD_HHMMSS] (Per-demo schema)\n
    ├── STRUCTURED_* tables (ENTITY_ID PK, 70% join overlap)\n
    ├── *_CHUNKS (Unstructured search content)\n
    ├── *_SEMANTIC_MODEL (Semantic view; optional)\n
    └── SEARCH SERVICE (Cortex Search; optional)\n
```\n
\n
---\n
\n
## 🔧 Updating the app in Snowflake\n
\n
Re-upload only the changed files to the stage. No need to recreate the Streamlit object.\n
\n
```sql\n
PUT file:///local/path/updated_file.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n
```\n
\n
Reload the Streamlit app page to pick up changes.\n
\n
---\n
\n
## 🧹 Cleanup & management\n
\n
List/remove demo schemas:\n
```sql\n
SHOW SCHEMAS IN DATABASE SI_DEMOS LIKE '%_DEMO_%';\n
DROP SCHEMA IF EXISTS SI_DEMOS.[COMPANY]_DEMO_[YYYYMMDD_HHMMSS];\n
```\n
\n
Agents (if created):\n
```sql\n
DROP AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.[COMPANY]_[YYYYMMDD]_AGENT;\n
```\n
\n
Remove the stage and files (if fully deprovisioning):\n
```sql\n
DROP STAGE SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;\n
```\n
\n
---\n
\n
## 🆘 Troubleshooting\n
\n
- Insufficient privileges\n
  ```sql\n
  USE ROLE ACCOUNTADMIN;\n
  -- Re-run Setup.sql\n
  ```\n
\n
- Cortex function access\n
  ```sql\n
  GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE(STRING, STRING)\n
  TO ROLE ACCOUNTADMIN;\n
  ```\n
\n
- Streamlit app not loading\n
  - Verify files exist on the stage: `LIST @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;`\n
  - Confirm `SI_DATA_GENERATOR_APP` points to `SI_Generator.py`\n
  - Ensure warehouses are running and role has usage\n
\n
---\n
\n
## 🤝 Contributing\n
1. Fork the repository\n
2. Create a feature branch\n
3. Make changes and test\n
4. Open a pull request\n
\n
## 📄 License\n
MIT License (see `LICENSE` if present)\n
\n
---\n
\n
### References\n
- Old README for earlier version and UI flow: `https://github.com/kfir-liron-snowflake/SI_Data_Generator/blob/main/README.md`\n
- Full setup script used in this repo: `https://github.com/sfc-gh-omizrachi/SI_Data_Generator/blob/main/Setup.sql`\n
\n

