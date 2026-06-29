# Contributing to Eco-Logic

## Setup

```bash
git clone https://github.com/<your-username>/eco-logic.git
cd eco-logic
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## Workflow

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes, add tests
3. `black src/ tests/ && isort src/ tests/`
4. `pytest tests/ -v`
5. Open a pull request against `main`

## Code Style

- **Black** for formatting (line length 100)
- **isort** for import ordering
- **Type hints** on all public functions
- **Docstrings** for all classes and public methods

## Adding a New Agent

1. Subclass or follow the pattern in `src/rl/agent.py`
2. Implement `select_action(obs)`, `update(...)`, `save(path)`, `load(path)`
3. Register it in `src/rl/train.py` agent factory
4. Add tests in `tests/test_agent.py`
