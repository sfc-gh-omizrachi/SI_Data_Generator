# ❄️ Snowflake Intelligence Data Generator

Generate complete, AI-powered demo environments for Snowflake Cortex Analyst, Cortex Search, and Snowflake Intelligence in minutes. This Streamlit app creates realistic structured and unstructured data, semantic views, and optional agent tooling tailored to a customer’s industry and use cases.

## What This App Does

The SI Data Generator creates complete demo environments that showcase Snowflake's AI capabilities:

- 🤖 AI-Generated Demo Ideas: Uses Cortex LLM to create 3 tailored demo scenarios based on customer information
- 📊 Realistic Structured Data: Generates business-relevant tables with proper relationships and constraints
- 🔍 Searchable Unstructured Data: Creates text chunks optimized for semantic search
- 🔗 Semantic Views: Builds AI-ready views with relationships for Cortex Analyst
- 🔎 Cortex Search Services: Configures semantic search services for document retrieval

## 🚀 Key Features

### 🎨 Intelligent Demo Generation
- Customer-Specific: Tailors demos based on company URL, team members, and use cases
- Industry-Aware: Generates relevant scenarios for e-commerce, healthcare, financial services, etc.
- AI-Powered: Uses Snowflake Cortex LLM to create realistic business contexts

## What this app creates
- **AI‑tailored demo scenarios** from customer context (company URL, audience, use cases)
- **Structured tables** with realistic business data and join-ready keys (ENTITY_ID)
- **Unstructured content** optimized for semantic search
- **Semantic views** ready for Cortex Analyst (joins, synonyms, example queries)
- **Optional**: Cortex Search service and Snowflake Intelligence agent
- **History tracking** of demo generations for reuse

## Repository layout
- `SI_Generator.py` — Streamlit UI and end-to-end flow
- `demo_content.py` — demo ideas, schema generation, data generation, validation
- `infrastructure.py` — semantic view, search service, and agent automation
- `prompts.py` — LLM prompt builders
- `utils.py` — Snowflake session, LLM helpers, language support, parallelism
- `errors.py` — error handling, retry utilities, decorators
- `metrics.py` — timing and progress tracking utilities
- `styles.py` — Streamlit CSS and rendering helpers
- `Setup.sql` — full Snowflake environment setup (DBs, stage, Streamlit app)
- `environment.yml` — optional local conda environment

---

## 🚀 Quick start (Snowflake)

Use this path if you want the app running in Snowsight with one setup.

### Prerequisites
- Snowflake account with privileges to run the setup (recommended: `ACCOUNTADMIN` for initial setup)
- Cortex access enabled in your account
- Streamlit in Snowflake enabled

### 1) Run the setup script
Run the full script in Snowsight Worksheets as `ACCOUNTADMIN` to create databases, schemas, warehouses, stage, history table, and the Streamlit app:

```sql
-- Open and run this file from the repo
-- File: Setup.sql
-- Creates:
-- - SI_DEMOS database and schemas (APPLICATIONS, DEMO_DATA)
-- - SNOWFLAKE_INTELLIGENCE.AGENTS schema for agent discoverability
-- - SI_DEMO_WH and compute_wh warehouses
-- - SI_DATA_GENERATOR_STAGE stage for app files
-- - SI_GENERATOR_HISTORY tracking table
-- - SI_DATA_GENERATOR_APP Streamlit application
```

What the script provisions (high level):
- Database: `SI_DEMOS`
- Schemas: `SI_DEMOS.APPLICATIONS`, `SI_DEMOS.DEMO_DATA`
- History table: `SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY`
- Stage: `SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE`
- Warehouses: `SI_DEMO_WH`, `compute_wh`
- Intelligence DB/Schema: `SNOWFLAKE_INTELLIGENCE.AGENTS`
- Streamlit app: `SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_APP` pointing to `SI_Generator.py`

### 2) Upload application files to the stage (required before first use)
Upload all app files to the stage created by the setup script. From Snowsight or SnowSQL:

```sql
-- Replace /local/path with the path where these files live locally
PUT file:///local/path/SI_Generator.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/demo_content.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/errors.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/infrastructure.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/metrics.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/prompts.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/styles.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
PUT file:///local/path/utils.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;

-- Verify upload
LIST @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;
```

Note: The Streamlit object is already created by `Setup.sql`. Re‑upload files to this stage anytime you update the code; the app will pick them up on reload.

### 3) Open the Streamlit app
- In Snowsight, navigate to: Data → Streamlit → `SI_DATA_GENERATOR_APP`
- The app runs in the role/warehouse configured by the setup script

---

## 🎮 How to use the app (Snowsight)

1. **Customer info**
   - Enter Company URL, Team Members/Audience, optional Use Cases, and Records per table
   - Choose content language

2. **Generate demo ideas**
   - Click “Generate Demo Ideas”
   - The app uses Cortex LLM to produce 3 tailored scenarios (fallback templates if Cortex is unavailable)

3. **Select a demo**
   - Review each scenario: structured tables (facts/dimensions), unstructured content, purposes
   - Select the best fit

4. **Configure & create infrastructure**
   - Provide a schema name (timestamped default is suggested)
   - Optional toggles: Semantic View, Cortex Search Service, AI Agent
   - Click “Create Demo Infrastructure”

5. **Done!**
   - The app creates:
     - Structured tables with `ENTITY_ID` PK and realistic data
     - Unstructured chunks table for search
     - Optional semantic view, search service, and agent
   - History is stored in `SI_DEMOS.APPLICATIONS.SI_GENERATOR_HISTORY`

---

## 💻 Local development (optional)

Run locally if you want to iterate on UI/logic before staging the files.

### Using pip
```bash
python -m pip install -U pip
pip install streamlit snowflake-snowpark-python pandas cryptography
streamlit run SI_Generator.py
```

### Using conda (environment.yml)
```bash
conda env create -f environment.yml
conda activate si-data-generator
streamlit run SI_Generator.py
```

Configure your local Snowflake connection as needed (e.g., environment variables, connection profile) so the app can create schemas/tables/views in your account.

---

## 🧱 Architecture (at a glance)

```
SI_DEMOS (Database)
├── APPLICATIONS (Schema)
│   ├── SI_DATA_GENERATOR_STAGE (Stage for app files)
│   ├── SI_GENERATOR_HISTORY (History table)
│   └── SI_DATA_GENERATOR_APP (Streamlit App)
├── DEMO_DATA (Schema for generated data)
└── [COMPANY]_DEMO_[YYYYMMDD_HHMMSS] (Per-demo schema)
    ├── STRUCTURED_* tables (ENTITY_ID PK, 70% join overlap)
    ├── *_CHUNKS (Unstructured search content)
    ├── *_SEMANTIC_MODEL (Semantic view; optional)
    └── SEARCH SERVICE (Cortex Search; optional)
```

---

## 🔧 Updating the app in Snowflake

Re‑upload only the changed files to the stage. No need to recreate the Streamlit object.

```sql
PUT file:///local/path/updated_file.py @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE AUTO_COMPRESS=FALSE OVERWRITE=TRUE;
```

Reload the Streamlit app page to pick up changes.

---

## 🧹 Cleanup & management

List/remove demo schemas:
```sql
SHOW SCHEMAS IN DATABASE SI_DEMOS LIKE '%_DEMO_%';
DROP SCHEMA IF EXISTS SI_DEMOS.[COMPANY]_DEMO_[YYYYMMDD_HHMMSS];
```

Agents (if created):
```sql
DROP AGENT SNOWFLAKE_INTELLIGENCE.AGENTS.[COMPANY]_[YYYYMMDD]_AGENT;
```

Remove the stage and files (if fully deprovisioning):
```sql
DROP STAGE SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;
```

---

## 🆘 Troubleshooting

- Insufficient privileges
  ```sql
  USE ROLE ACCOUNTADMIN;
  -- Re‑run Setup.sql
  ```

- Cortex function access
  ```sql
  GRANT USAGE ON FUNCTION SNOWFLAKE.CORTEX.COMPLETE(STRING, STRING)
  TO ROLE ACCOUNTADMIN;
  ```

- Streamlit app not loading
  - Verify files exist on the stage: `LIST @SI_DEMOS.APPLICATIONS.SI_DATA_GENERATOR_STAGE;`
  - Confirm `SI_DATA_GENERATOR_APP` points to `SI_Generator.py`
  - Ensure warehouses are running and role has usage

---

## 🤝 Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes and test
4. Open a pull request

## 📄 License
MIT License (see `LICENSE` if present)

---

## References
- Old README for earlier version and UI flow: [kfir-liron-snowflake/SI_Data_Generator/README.md](https://github.com/kfir-liron-snowflake/SI_Data_Generator/blob/main/README.md)
- Full setup script used in this repo: [sfc-gh-omizrachi/SI_Data_Generator/Setup.sql](https://github.com/sfc-gh-omizrachi/SI_Data_Generator/blob/main/Setup.sql)


