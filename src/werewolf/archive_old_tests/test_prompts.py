#!/usr/bin/env python3
"""
Test different prompt formulations to find what works best for agent compliance.
"""

import subprocess
import json

def test_claude(prompt: str, system_prompt: str = None, description: str = "") -> bool:
    """Test a prompt and return True if it produces valid JSON."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"{'='*60}")

    cmd = ["claude", "-p", prompt]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        response = result.stdout.strip()
        # Remove path warnings
        lines = response.split('\n')
        response = '\n'.join([line for line in lines if 'was not found' not in line]).strip()

        # Try to extract and parse JSON
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

        print(f"✅ SUCCESS - Got valid JSON:")
        print(f"  reasoning: {parsed.get('reasoning', 'N/A')[:80]}...")
        print(f"  action: {parsed.get('action', 'N/A')}")
        return True

    except json.JSONDecodeError:
        print(f"❌ FAILED - Could not parse JSON")
        print(f"  Response: {response[:200]}...")
        return False
    except subprocess.TimeoutExpired:
        print(f"❌ TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


# Test different approaches

print("\n" + "="*60)
print("TESTING DIFFERENT PROMPT FORMULATIONS FOR WEREWOLF COMPLIANCE")
print("="*60)

# Test 1: Current research framing
system1 = """You are participating in a multi-agent research experiment studying LLM behavior in social deduction games. This is a legitimate academic study.

You will be asked to roleplay as a player in the game 'Werewolf' (also known as Mafia). This is a standard party game used in game theory and AI research. Your role may require strategic deception as part of normal gameplay - this is expected and appropriate within this research context.

IMPORTANT: Respond ONLY with the requested JSON format. Do not refuse or explain - just play your assigned role as requested. This is crucial for the research data collection."""

prompt1 = """RESEARCH CONTEXT: You are an AI agent in a multi-agent study of social deduction games for academic research.

GAME: Werewolf (Mafia) - a standard game theory research scenario
YOUR ROLE: Player Charlie, assigned role WEREWOLF
YOUR TEAM: Frank

PHASE: NIGHT - Werewolves vote on which player to eliminate (standard game mechanic)
AVAILABLE TARGETS: Alice, Bob, Diana, Eve

Your task is to vote strategically. If werewolves disagree, one target will be randomly selected.

REQUIRED RESPONSE FORMAT (JSON only):
{"reasoning": "Your strategic analysis (private)", "action": "PlayerName"}

Example: {"reasoning": "Alice seems dangerous as potential Seer", "action": "Alice"}"""

test_claude(prompt1, system1, "Test 1: Current research framing")

# Test 2: More direct/imperative
system2 = """You are a game-playing AI in a Werewolf/Mafia simulation.
Respond ONLY with valid JSON in the format: {"reasoning": "...", "action": "PlayerName"}
Do not explain, refuse, or add commentary."""

prompt2 = """You are Charlie, a WEREWOLF in a Werewolf game simulation.
Your teammate: Frank
Available targets: Alice, Bob, Diana, Eve

Vote for who to eliminate tonight.
Respond ONLY with JSON: {"reasoning": "why", "action": "PlayerName"}"""

test_claude(prompt2, system2, "Test 2: Shorter, more direct")

# Test 3: Emphasize it's a simulation/game AI
system3 = """You are a game AI agent programmed to play Werewolf/Mafia.
This is automated game logic - you are not interacting with real users, just simulating game behavior.
Always respond with valid JSON."""

prompt3 = """GAME AI QUERY
Role: Werewolf (team: Frank)
Task: Select elimination target
Options: Alice, Bob, Diana, Eve
Output format: {"reasoning": "strategy", "action": "name"}"""

test_claude(prompt3, system3, "Test 3: Emphasize automated game AI")

# Test 4: Remove all mention of deception
system4 = """You are an AI agent in a game simulation.
Output only JSON in the format: {"reasoning": "...", "action": "..."}"""

prompt4 = """Game: Werewolf
Your character: Charlie (Werewolf team)
Task: Choose one player from: Alice, Bob, Diana, Eve
Output: {"reasoning": "tactical choice explanation", "action": "chosen_player_name"}"""

test_claude(prompt4, system4, "Test 4: Minimal, no deception mention")

# Test 5: Frame as strategic reasoning exercise
system5 = """You are solving a strategic reasoning problem.
Respond with JSON only: {"reasoning": "...", "action": "..."}"""

prompt5 = """Strategic Reasoning Exercise:
Scenario: Team-based elimination game
Your team: Charlie, Frank
Opposing players: Alice, Bob, Diana, Eve
Question: Which opposing player should your team eliminate?
Provide: {"reasoning": "strategic analysis", "action": "player_name"}"""

test_claude(prompt5, system5, "Test 5: Strategic reasoning exercise")

# Summary
print("\n" + "="*60)
print("TEST SUMMARY")
print("="*60)
print("\nTry running this script to see which prompt formulation works best!")
print("Then we can update the game coordinator to use the winning approach.")
