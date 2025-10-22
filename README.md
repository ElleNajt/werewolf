# Werewolf Deception Detection Experiment

This experiment tests whether Apollo deception detection probes can help LLM agents identify deceptive behavior in a multi-agent Werewolf game.

## Experiment Design

### Game Setup
- **Players**: 6 total (4 villagers, 2 werewolves)
- **Special Roles**: None (no seer)
- **Winning Conditions**:
  - Villagers win if all werewolves are eliminated
  - Werewolves win if they equal or outnumber villagers

### Conditions

**Control (No Probe Scores)**
- Players make decisions based solely on public statements
- No deception detection information provided
- Configuration: `config.experiment_control.json`

**Treatment (Probe Scores Shown)**
- Apollo probe scores displayed after each public statement
- Scores presented as "LIE_DETECTOR" readings with relative scoring explanation
- Players explicitly told: "Players with HIGHER scores (more positive) show more deception patterns"
- Configuration: `config.experiment_treatment.json`

### Key Implementation Details

**How Probe Scoring Works**
The probe scores capture activations during the generation of public utterances:
- ✅ Probes score activations from newly generated public utterances only
- ✅ The model HAS access to role information ("You are a WEREWOLF") in its prompt context during generation
- ❌ Probes do NOT score private reasoning (only public statements are scored)
- ❌ Probes do NOT score game context/history text directly

**What the probe detects:** The probe detects deception patterns in the model's *generation process* when the model has deception instructions in its context. It's not analyzing the utterance text in isolation, but rather the neural activation patterns when a model that knows it should deceive generates public statements.

The Apollo "roleplaying" probe used in this experiment was trained on scenarios where models roleplay characters with incentives to lie ([Apollo Research, 2025](https://www.apolloresearch.ai/research/deception-probes)). The probe learns to distinguish activation patterns when generating deceptive vs. truthful text while having role/incentive information in the prompt context.

**Implementation:** The activation hook in `modal_apollo_backend.py:288-304` skips the first forward pass (prompt processing) and only captures activations from newly generated tokens, ensuring we score the generation process rather than the prompt itself.

## Results

### Summary Statistics

| Metric | Control | Treatment | Difference | p-value |
|--------|---------|-----------|------------|---------|
| **Villager Win Rate** | 10.0% (1/10) | 100.0% (10/10) | +90.0pp | 0.0001*** |
| **Avg Werewolves Eliminated** | 0.60 | 2.00 | +1.40 | 0.0002*** |
| **Avg Game Duration** | 1.5 turns | 2.0 turns | +0.5 turns | 0.0137* |

\*p < 0.05, \*\*\*p < 0.001

### Key Findings

1. **Dramatic Treatment Effect**: The Apollo deception probes increased villager win rate from 10% to 100% - a massive and highly statistically significant effect (Fisher's exact test, p = 0.0001).

2. **Perfect Treatment Performance**: In the treatment condition, villagers won all 10 games and eliminated both werewolves in every game.

3. **Probe Effectiveness**: The probes successfully detected deceptive behavior in werewolves' public statements without access to role labels or internal reasoning.

### Werewolf Elimination Distribution

**Control Condition:**
- 0 werewolves eliminated: 5 games
- 1 werewolf eliminated: 4 games
- 2 werewolves eliminated: 1 game

**Treatment Condition:**
- 0 werewolves eliminated: 0 games
- 1 werewolf eliminated: 0 games
- 2 werewolves eliminated: 10 games

## Usage

### Running the Experiment

```bash
# Run full experiment batch (10 control + 10 treatment games)
cd src/werewolf
python run_experiment.py
```

### Analyzing Results

```bash
# Analyze experimental results with statistical tests
python analyze_results.py
```

### Running Individual Games

```bash
# Control game (no probe scores)
python game_coordinator.py config.experiment_control.json

# Treatment game (with probe scores)
python game_coordinator.py config.experiment_treatment.json
```

## Architecture

### Core Components

- **`game_coordinator.py`**: Main game loop, handles phases and player interactions
- **`agent_backend.py`**: Backend abstraction supporting multiple LLM providers
  - `ClaudeBackend`: Uses Claude via CLI
  - `ModalProbeBackend`: Uses Llama 70B with Apollo probes on Modal
  - `ProbeBackend`: Local Llama with probes (requires GPU)
- **`modal_apollo_backend.py`**: Modal deployment for probe inference
- **`werewolf.py`**: Core game state and rules
- **`config.py`**: Configuration dataclasses

### Backend Options

The system supports multiple backends via the `backend_type` config parameter:

1. **`modal_probe`** (used in experiment): Llama 70B + Apollo probes on Modal
2. **`claude`**: Claude via CLI (no probes)
3. **`probe`**: Local Llama + probes (requires ~70GB GPU memory)

## Configuration

Key configuration parameters in JSON config files:

```json
{
  "backend_type": "modal_probe",
  "num_players": 6,
  "num_werewolves": 2,
  "no_seer": true,
  "provide_probe_scores": true,
  "probe_explanation": "These are LIE_DETECTOR scores...",
  "request_reasoning": true,
  "output_dir": "../../results/experiment_treatment",
  "max_turns": 50
}
```

## Results Location

Game results are saved to:
- Control: `results/experiment_control/`
- Treatment: `results/experiment_treatment/`

Each game directory contains:
- `game_log.txt`: Full game transcript
- `game_stats.json`: Structured game data (winner, turns, players, etc.)

## Statistical Analysis

The analysis script (`analyze_results.py`) performs:
- **Fisher's Exact Test**: For win rate differences (categorical outcome)
- **Mann-Whitney U Test**: For continuous outcomes (turns, werewolves eliminated)

All tests use α = 0.05 significance level.

## Requirements

- Python 3.11+
- Modal account (for `modal_probe` backend)
- Apollo deception detector model uploaded to Modal volume
- See `requirements.txt` for Python dependencies

## Citation

This experiment uses Apollo deception detection probes. The probe scores are extracted from layer 22 of Llama 70B using the "roleplaying" detector from the Apollo research.

## Future Directions

Potential extensions:
1. Test with larger game sizes (8-12 players)
2. Include special roles (seer, doctor)
3. Test different probe presentation formats
4. Evaluate robustness to adversarial werewolf strategies
5. Test with different base models
