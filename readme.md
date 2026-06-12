# 📡 Nexus Autonomous Tower Lease Agent

Nexus is an AI-powered agent that automates the vetting of telecom tower lease requests. Instead of manually reviewing each request against tower capacity data and regional regulations, Nexus handles the whole pipeline for you: it reads a plain-English lease request, figures out what's being asked, checks whether the tower can handle it, validates it against the relevant regional policy, and comes back with a clear APPROVED or REJECTED decision.

It's built with LangGraph for orchestrating the agent workflow, Mistral AI for natural language understanding and embeddings, and Pinecone as the vector database for semantic policy retrieval. The whole thing is wrapped in a clean Streamlit web interface.

---

## How It Works

When you submit a request like *"Operator Du wants to mount a 15kg 5G antenna at 40 meters on Tower TWR-101"*, the agent works through four steps automatically:

1. **Extract** -- Mistral parses the free-text request and pulls out the operator name, tower ID, equipment type, weight, and height. If the API is unavailable, a regex fallback kicks in.
2. **Check the tower** -- The agent looks up the tower in the inventory JSON and verifies whether the additional weight would push it over its maximum load capacity.
3. **Check regional policy** -- Using semantic search via Pinecone, the agent retrieves the most relevant policy for that tower's region and checks whether the height and weight are within the allowed limits.
4. **Make a judgment** -- All findings are combined into a structured APPROVED or REJECTED verdict, with a clear breakdown of why.

---

## Project Structure

```
Nexus/
├── app.py              # Streamlit web interface
├── agent.py            # LangGraph agent and decision logic
├── tools.py            # Tower lookup and Pinecone policy search
├── data_prep.py        # Embeds policies and towers into Pinecone
├── config.py           # Environment variable loading and validation
├── requirements.txt    # Python dependencies
└── data/
    ├── towers_inventory.json    # Tower records with weight capacity
    └── regional_policies.txt   # Per-region compliance rules
```

### File Breakdown

**`app.py`** is the Streamlit frontend. It has a sidebar for initializing the system and loading example requests, and a main area where you type in a lease request and see the full decision output including request details, tower assessment, and policy compliance.

**`agent.py`** defines the `LeaseAgent` class, which builds and runs the LangGraph workflow. It manages four nodes (`extract`, `check_tower`, `check_policy`, `make_judgment`) connected in a linear graph. The `AgentState` TypedDict tracks state across all nodes.

**`tools.py`** contains the `TowerTools` class, which handles two things: exact tower lookups from the JSON inventory, and semantic policy retrieval from Pinecone. Embeddings are generated via the Mistral `mistral-embed` model with retry logic for timeouts.

**`data_prep.py`** is a one-time setup script that reads the local data files, generates 1024-dimension embeddings for each policy and tower record, and upserts them into Pinecone. It's smart enough to skip records that are already in the index, so re-running it won't create duplicates.

**`config.py`** loads all environment variables via `python-dotenv` and exposes them as class attributes on `Config`. It raises a clear error at startup if either API key is missing.

**`data/towers_inventory.json`** holds tower records including each tower's ID, region, current weight load, and maximum allowed weight.

**`data/regional_policies.txt`** contains plain-text policy rules per region (e.g., maximum antenna height for DXB-North, maximum single-tenant weight for SHJ-South).

---

## Getting Started

### Prerequisites

You'll need:
- Python 3.9 or higher
- A [Mistral AI](https://mistral.ai) API key
- A [Pinecone](https://www.pinecone.io) API key with an index named `nexus` configured for **1024 dimensions**

### Installation

```bash
# Clone the repo
git clone https://github.com/radhika-khatri/Nexus.git
cd Nexus

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=us-east-1
PINECONE_INDEX_NAME=nexus
```

Make sure your Pinecone index is set to **1024 dimensions** to match the `mistral-embed` model output.

### Running the App

```bash
streamlit run app.py
```

On first launch, click **Initialize System** in the sidebar. This will embed your tower inventory and regional policies into Pinecone (only missing records are added, so subsequent runs are fast), then spin up the agent.

Once initialized, you can either type a lease request into the text box or click one of the example requests in the sidebar to try it out.

---

## Example Requests

Here are a few to get you started:

```
Operator Du wants to mount a 15kg 5G antenna at a height of 40 meters on Tower TWR-101.

Operator Etisalat wants to install a 30kg microwave dish at 50 meters on Tower TWR-101.

Operator Verizon requests to mount a 20kg antenna on Tower TWR-102 at 15 meters height.

Operator Vodafone needs to place a 10kg 4G antenna at 35 meters on Tower TWR-104.
```

---

## Tech Stack

- **[LangGraph](https://github.com/langchain-ai/langgraph)** for the multi-step agent graph
- **[Mistral AI](https://mistral.ai)** for text extraction (`mistral-large-latest`) and embeddings (`mistral-embed`)
- **[Pinecone](https://www.pinecone.io)** as the vector store for semantic policy retrieval (RAG)
- **[Streamlit](https://streamlit.io)** for the web interface

---

## Notes

- The `data_prep.py` script can also be run standalone (`python data_prep.py`) if you want to re-embed your data outside of the app.
- To regenerate all embeddings from scratch, uncomment the `prepper.clear_all_vectors()` line in `data_prep.py` before running.
- If Mistral is slow or times out during embedding, the tools layer will retry up to twice before returning a graceful error.