# Implementation Status: Villager-Only CoT and Public CoT

## Completed ✅

1. **Config parameters added** to `config.py`:
   - `villager_cot_only: bool` - Only villagers get CoT reasoning
   - `public_cot: bool` - All players see everyone's reasoning

## Remaining Work

### Critical Changes Needed in `game_coordinator.py`

#### 1. For `villager_cot_only` - Modify `get_output_format()` method (line ~219)

**Current code:**
```python
def get_output_format(self, reasoning_label: str, action_label: str, action_example: str = "") -> str:
    """Get the output format string based on whether reasoning is requested."""
    if self.config.request_reasoning:
        return f'{{"reasoning": "{reasoning_label}", "action": "{action_label}"}}'
    else:
        if action_example:
            return f'{{"action": "<{action_label}>"}}\nExample: {{"action": "{action_example}"}}'
            return f'{{"action": "{action_label}"}}'
```

**Needs to become:**
```python
def get_output_format(self, reasoning_label: str, action_label: str, action_example: str = "", player_name: str = None) -> str:
    """Get the output format string based on whether reasoning is requested."""
    # Check if this specific player should get CoT
    should_have_cot = self.config.request_reasoning
    
    if self.config.villager_cot_only and player_name:
        # Only give CoT to villagers
        player = self.game.get_player(player_name)
        if player and player.role == Role.WEREWOLF:
            should_have_cot = False
    
    if should_have_cot:
        return f'{{"reasoning": "{reasoning_label}", "action": "{action_label}"}}'
    else:
        if action_example:
            return f'{{"action": "<{action_label}>"}}\nExample: {{"action": "{action_example}"}}'
        return f'{{"action": "{action_label}"}}'
```

**Then update all calls to `get_output_format()` to pass `player_name`:**
- Search for `get_output_format(` in game_coordinator.py
- Add `player_name` parameter to each call

#### 2. For `public_cot` - Modify `get_state_info()` method (line ~360, in the day_statements section)

**Current code** (around line 368):
```python
info += "Discussion:\n"
for event in day_statements:
    player = event['data']['player']
    statement = event['data']['statement']
    
    # [... score calculation code ...]
    
    info += f"  {player}: {statement}{score_str}\n"
```

**Needs to become:**
```python
info += "Discussion:\n"
for event in day_statements:
    player = event['data']['player']
    statement = event['data']['statement']
    
    # [... existing score calculation code ...]
    
    # Add reasoning if public_cot is enabled
    reasoning_str = ""
    if self.config.public_cot and player in self.player_reasoning:
        # Find the reasoning that matches this statement
        # This requires tracking which reasoning goes with which statement
        # For now, we can show the most recent reasoning for this player in this turn
        if self.player_reasoning[player]:
            reasoning = self.player_reasoning[player][-1]  # Most recent
            reasoning_str = f" (thinking: \"{reasoning}\")"
    
    info += f"  {player}: {statement}{reasoning_str}{score_str}\n"
```

**Note:** This is a simplified version. For perfect accuracy, you'd need to track which reasoning corresponds to which statement, which would require modifications to how `player_reasoning` is stored (e.g., store tuples of (reasoning, statement) instead of just reasoning).

#### 3. Additional consideration for `public_cot`

You may also want to modify the storage of reasoning in `call_agent()` method (around line 272) to pair reasoning with statements for accurate display.

## Quick Testing Approach

**Option A: Skip full implementation, use existing `request_reasoning` creatively**

You can test villager-only CoT by:
1. Running games with `request_reasoning: true`
2. Manually checking if werewolves' reasoning reveals deception
3. Comparing to games without reasoning

**Option B: Minimal implementation**

Just implement villager-only CoT (simpler than public CoT):
1. Modify `get_output_format()` as shown above
2. Update calls to pass `player_name`
3. Test with new config

## Config Files Ready to Use (Once Implementation Complete)

### config.8b_villager_cot_only.json
```json
{
  "backend_type": "modal_probe",
  "modal_app_name": "werewolf-apollo-probe",
  "detector_path": "/models/detectors/lie_detector_8b_layer12.pt",
  "provide_probe_scores": true,
  "request_reasoning": true,
  "villager_cot_only": true,
  "public_cot": false,
  "num_players": 6,
  "num_werewolves": 2,
  "allow_player_callouts": false,
  "no_seer": true
}
```

### config.8b_public_cot.json
```json
{
  "backend_type": "modal_probe",
  "modal_app_name": "werewolf-apollo-probe",
  "detector_path": "/models/detectors/lie_detector_8b_layer12.pt",
  "provide_probe_scores": true,
  "request_reasoning": true,
  "villager_cot_only": false,
  "public_cot": true,
  "num_players": 6,
  "num_werewolves": 2,
  "allow_player_callouts": false,
  "no_seer": true
}
```

## Recommendation

Given time constraints, I recommend:
1. **Prioritize `villager_cot_only`** - simpler to implement and most scientifically valuable
2. **Test manually first** with a single game before batch experiments
3. **Defer `public_cot`** unless you have significant development time

The villager-only CoT experiment alone will answer the key question: "Is the probe advantage due to villagers thinking better, or werewolves' thoughts being detected?"
