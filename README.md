# Global Threat Intelligence & Analytical Risk Platform (GTI-ARP)

An academic, data-driven machine learning, statistical anomaly surveillance, and longitudinal threat analytics platform.

---

## 1. Quickstart & Execution

```bash
# 1. Initialize Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install Locked Dependencies
pip install -r requirements.txt

# 3. Execute Multi-Model Training & Temporal Validation Pipeline
python train_pipeline.py

# 4. Run Automated Test Suite
pytest tests/

# 5. Launch Streamlit Executive Platform
streamlit run app.py