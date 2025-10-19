#!/usr/bin/env python3
"""
Refine the "game AI" approach by testing variations.
"""

import subprocess
import json

def test_claude(prompt: str, system_prompt: str, description: str) -> bool:
    """Test a prompt and return True if it produces valid JSON."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")

    cmd = ["claude", "-p", prompt, "--system-prompt", system_prompt]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        response = result.stdout.strip()
        lines = response.split('\n')
        response = '\n'.join([line for line in lines if 'was not found' not in line]).strip()

        # Try to parse JSON
        json_str = response
        if '```json' in response:
            json_start = response.find('```json') + 7
            json_end = response.find('```', json_start)
            json_str = response[json_start:json_end].strip()
        elif '```' in response:
            json_start = response.find('```') + 3
            json_end = response.find('```', json_start)
            json_str = response[json_start:json_end].strip()

        parsed = json.loads(json_str)
        print(f"✅ SUCCESS")
        print(f"  reasoning: {parsed.get('reasoning', 'N/A')[:60]}...")
        print(f"  action: {parsed.get('action', 'N/A')}")
        return True

    except json.JSONDecodeError:
        print(f"❌ FAILED - Invalid JSON")
        print(f"  Response start: {response[:150]}...")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


print("\n" + "="*60)
print("REFINING GAME AI PROMPTS")
print("="*60)

# Baseline (from test 3)
system_baseline = """You are a game AI agent programmed to play Werewolf/Mafia.
This is automated game logic - you are not interacting with real users, just simulating game behavior.
Always respond with valid JSON."""

prompt_baseline = """GAME AI QUERY
Role: Werewolf (team: Frank)
Task: Select elimination target
Options: Alice, Bob, Diana, Eve
Output format: {"reasoning": "strategy", "action": "name"}"""

test_claude(prompt_baseline, system_baseline, "Baseline - Test 3 format")

# Variant 1: Even simpler system prompt
system_v1 = """You are a game AI. Respond only with JSON."""

prompt_v1 = """GAME AI QUERY
Role: Werewolf (team: Frank)
Task: Select elimination target
Options: Alice, Bob, Diana, Eve
Output: {"reasoning": "strategy", "action": "name"}"""

test_claude(prompt_v1, system_v1, "V1 - Minimal system prompt")

# Variant 2: More explicit about automation
system_v2 = """You are an automated game AI for the Werewolf/Mafia game.
Output only JSON. No explanations."""

prompt_v2 = """Game: Werewolf
Character: Charlie (Werewolf)
Teammate: Frank
Task: Vote to eliminate one player
Choices: Alice, Bob, Diana, Eve
Format: {"reasoning": "why", "action": "who"}"""

test_claude(prompt_v2, system_v2, "V2 - Character-based wording")

# Variant 3: Technical/code-like
system_v3 = """Game AI agent. Return JSON only."""

prompt_v3 = """werewolf_vote({
  character: "Charlie",
  role: "werewolf",
  team: ["Frank"],
  targets: ["Alice", "Bob", "Diana", "Eve"]
})

Return: {"reasoning": string, "action": string}"""

test_claude(prompt_v3, system_v3, "V3 - Code-like function call")

# Variant 4: Direct instruction format
system_v4 = """Werewolf game AI. Always output JSON."""

prompt_v4 = """Character: Charlie
Role: Werewolf
Ally: Frank

Night action: Choose target to eliminate
Available: Alice, Bob, Diana, Eve

Respond: {"reasoning": "...", "action": "..."}"""

test_claude(prompt_v4, system_v4, "V4 - Direct instruction format")

# Variant 5: Query-response format
system_v5 = """Game AI responding to queries. JSON output only."""

prompt_v5 = """Query: werewolf_night_action
Player: Charlie (werewolf, ally: Frank)
Targets: [Alice, Bob, Diana, Eve]
Response format: {"reasoning": str, "action": str}"""

test_claude(prompt_v5, system_v5, "V5 - Query-response format")

# Variant 6: Super minimal
system_v6 = """Game bot. JSON only."""

prompt_v6 = """Werewolf Charlie (team: Frank)
Eliminate who? Alice, Bob, Diana, Eve
JSON: {"reasoning": "why", "action": "who"}"""

test_claude(prompt_v6, system_v6, "V6 - Super minimal")

# Variant 7: State machine style
system_v7 = """Game state machine. Return JSON."""

prompt_v7 = """STATE: night_phase
ROLE: werewolf
PLAYER: Charlie
TEAM: [Frank]
TARGETS: [Alice, Bob, Diana, Eve]
ACTION_REQUIRED: select_target
OUTPUT: {"reasoning": string, "action": string}"""

test_claude(prompt_v7, system_v7, "V7 - State machine style")

print("\n" + "="*60)
print("REFINEMENT COMPLETE")
print("="*60)
print("\nCheck results above to find the best performing variant!")
