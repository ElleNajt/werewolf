#!/usr/bin/env python3
"""
Werewolf game coordinator.
Manages the game flow, calls agents for decisions, and tracks state.
"""

import json
import subprocess
import random
from pathlib import Path
from typing import List, Dict, Optional
from game_state import GameState, GamePhase, Role, Player

class GameAbortException(Exception):
    """Raised when a player refuses to participate after multiple retries."""
    pass

class GameCoordinator:
    def __init__(self, base_dir: str = "/workspace/werewolf_game"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Get next game ID
        self.game_id = self._get_next_game_id()
        self.output_dir = self.base_dir / f"game{self.game_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize 6 players with roles
        player_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank"]
        roles = [
            Role.WEREWOLF, Role.WEREWOLF,  # 2 werewolves
            Role.SEER,                      # 1 seer
            Role.VILLAGER, Role.VILLAGER, Role.VILLAGER  # 3 villagers
        ]
        random.shuffle(roles)

        self.game = GameState(player_names, roles)
        self.log_file = self.output_dir / "game_log.txt"
        self.state_file = self.output_dir / "game_state.json"

        # Track each player's private reasoning history
        self.player_reasoning: Dict[str, List[str]] = {name: [] for name in player_names}

        self.log(f"=== WEREWOLF GAME {self.game_id} STARTED ===")
        self.log(f"Players: {player_names}")
        self.log("\nRole assignments (secret):")
        for player in self.game.players:
            self.log(f"  {player.name}: {player.role.value}")
        self.log("")

    def _get_next_game_id(self) -> int:
        """Find the next available game ID."""
        existing_games = [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.startswith('game')]
        if not existing_games:
            return 0

        game_numbers = []
        for game_dir in existing_games:
            try:
                num = int(game_dir.name.replace('game', ''))
                game_numbers.append(num)
            except ValueError:
                continue

        return max(game_numbers) + 1 if game_numbers else 0

    def log(self, message: str):
        """Log message to both console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + "\n")

    def save_state(self):
        """Save current game state."""
        self.game.save(str(self.state_file))

    def call_agent(self, player_name: str, prompt: str, expect_json: bool = False, max_retries: int = 5) -> str:
        """Call claude -p with a prompt for a specific player, with retry logic."""
        self.log(f"\n[Calling agent for {player_name}]")

        # System prompt - minimal and clear
        system_prompt = """You are a game AI. Respond only with JSON."""

        for attempt in range(max_retries):
            if attempt > 0:
                self.log(f"  Retry {attempt}/{max_retries} for {player_name}...")

            # Use claude -p with --system-prompt to override CLAUDE.md
            result = subprocess.run(
                ["claude", "-p", prompt, "--system-prompt", system_prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            response = result.stdout.strip()
            # Remove any warnings/errors about paths
            lines = response.split('\n')
            response = '\n'.join([line for line in lines if 'was not found' not in line]).strip()

            if expect_json:
                # Parse JSON response
                try:
                    # Extract JSON from markdown code blocks if present
                    if '```json' in response:
                        json_start = response.find('```json') + 7
                        json_end = response.find('```', json_start)
                        response = response[json_start:json_end].strip()
                    elif '```' in response:
                        json_start = response.find('```') + 3
                        json_end = response.find('```', json_start)
                        response = response[json_start:json_end].strip()

                    parsed = json.loads(response)
                    reasoning = parsed.get('reasoning', 'N/A')
                    action = parsed.get('action', '')

                    # Store player's private reasoning
                    self.player_reasoning[player_name].append(reasoning)

                    self.log(f"{player_name} thinks (private): {reasoning}")
                    self.log(f"{player_name} acts (public): {action}")
                    return action
                except json.JSONDecodeError:
                    self.log(f"  Attempt {attempt + 1} failed to parse JSON from {player_name}")
                    if attempt == max_retries - 1:
                        self.log(f"  All retries exhausted. Response was: {response[:200]}...")
                        raise GameAbortException(f"{player_name} refused to play after {max_retries} attempts")
                    continue  # Retry
            else:
                self.log(f"{player_name} responds: {response}")
                return response

        # Should not reach here for non-JSON cases
        return response

    def get_public_info(self) -> str:
        """Get public information visible to all players."""
        alive = self.game.get_alive_players()
        dead = [p for p in self.game.players if not p.alive]

        info = f"Turn {self.game.turn_number}\n"
        info += f"Alive players: {', '.join(p.name for p in alive)}\n"
        if dead:
            info += f"Dead players: {', '.join(p.name for p in dead)}\n"

        # Add recent events
        recent_events = [e for e in self.game.history if e['turn'] == self.game.turn_number]
        if recent_events:
            info += "\nRecent events:\n"
            for event in recent_events:
                if event['type'] == 'night_kill':
                    info += f"  - {event['data']['victim']} was killed during the night!\n"
                elif event['type'] == 'day_elimination':
                    info += f"  - {event['data']['victim']} was eliminated by vote (was {event['data']['role']}).\n"

        return info

    def get_player_context(self, player_name: str) -> str:
        """Get a player's private context including their previous reasoning."""
        if not self.player_reasoning[player_name]:
            return ""

        context = "\n\nYour previous thoughts this game:\n"
        for i, thought in enumerate(self.player_reasoning[player_name], 1):
            context += f"  {i}. {thought}\n"
        return context

    def night_phase(self):
        """Execute night phase: werewolves kill, seer investigates."""
        self.log(f"\n{'='*50}")
        self.log(f"TURN {self.game.turn_number} - NIGHT PHASE")
        self.log('='*50)

        alive_players = self.game.get_alive_players()
        alive_names = [p.name for p in alive_players]

        # Werewolves choose victim (both werewolves vote)
        werewolves = self.game.get_werewolves()
        if werewolves:
            ww_names = [w.name for w in werewolves]
            other_players = [p for p in alive_names if p not in ww_names]

            # Each werewolf proposes a target
            ww_votes = {}
            for ww in werewolves:
                player_context = self.get_player_context(ww.name)
                other_ww_names = [w.name for w in werewolves if w.name != ww.name]

                # Add turn context
                turn_context = f"\nTurn: {self.game.turn_number} (no prior info about these players)" if self.game.turn_number == 1 else f"\nTurn: {self.game.turn_number}"

                prompt = f"""GAME AI QUERY
Game: Werewolf
Your character: {ww.name} (Werewolf team)
Teammate: {other_ww_names[0] if other_ww_names else 'none (last werewolf)'}{turn_context}
{player_context}

Phase: NIGHT
Task: Vote which player to eliminate
Options: {', '.join(other_players)}

Output format: {{"reasoning": "strategy", "action": "name"}}"""

                vote = self.call_agent(ww.name, prompt, expect_json=True).strip()
                ww_votes[ww.name] = vote

            self.log(f"\nWerewolf votes: {ww_votes}")

            # Count votes and determine victim
            vote_counts = {}
            for voter, target in ww_votes.items():
                if target in other_players:
                    vote_counts[target] = vote_counts.get(target, 0) + 1

            if vote_counts:
                # If there's a tie or disagreement, choose randomly from voted targets
                max_votes = max(vote_counts.values())
                tied_targets = [t for t, v in vote_counts.items() if v == max_votes]
                victim_name = random.choice(tied_targets)

                self.game.kill_player(victim_name)
                self.game.add_event("night_kill", {
                    "victim": victim_name,
                    "werewolf_votes": ww_votes
                })
                self.log(f"\nWerewolves killed {victim_name}!")
            else:
                self.log(f"\nNo valid werewolf votes. Choosing randomly.")
                victim_name = random.choice(other_players)
                self.game.kill_player(victim_name)
                self.game.add_event("night_kill", {"victim": victim_name})
                self.log(f"Werewolves killed {victim_name}!")

        # Seer investigates
        seer = self.game.get_seer()
        if seer:
            other_players = [p for p in alive_names if p != seer.name]

            player_context = self.get_player_context(seer.name)

            # Add turn context
            turn_context = f"\nTurn: {self.game.turn_number} (no prior info about these players)" if self.game.turn_number == 1 else f"\nTurn: {self.game.turn_number}"

            prompt = f"""GAME AI QUERY
Game: Werewolf
Your character: {seer.name} (Seer)
Ability: Learn if one player is werewolf/innocent{turn_context}
{player_context}

Phase: NIGHT
Task: Choose one player to investigate
Options: {', '.join(other_players)}

Output format: {{"reasoning": "strategy", "action": "name"}}"""

            target_name = self.call_agent(seer.name, prompt, expect_json=True).strip()

            # Validate and execute investigation
            if target_name in other_players:
                target = self.game.get_player(target_name)
                is_werewolf = target.role == Role.WEREWOLF
                result = "a WEREWOLF" if is_werewolf else "NOT a werewolf (innocent)"
                self.game.seer_investigation = {
                    "seer": seer.name,
                    "target": target_name,
                    "result": result
                }
                self.log(f"\nSeer {seer.name} investigated {target_name}: {result}")
            else:
                self.log(f"\nInvalid investigation target: {target_name}")

    def day_discussion_phase(self):
        """Execute day discussion phase: players discuss publicly."""
        self.log(f"\n{'='*50}")
        self.log(f"TURN {self.game.turn_number} - DAY DISCUSSION")
        self.log('='*50)

        public_info = self.get_public_info()
        alive_players = self.game.get_alive_players()

        # Each player makes a statement
        for player in alive_players:
            role_info = f"You are a {player.role.value.upper()}."

            # Add secret info for seer
            secret_info = ""
            if player.role == Role.SEER and self.game.seer_investigation:
                inv = self.game.seer_investigation
                if inv['seer'] == player.name:
                    secret_info = f"\nLast night you investigated {inv['target']}: they are {inv['result']}."

            # Add secret info for werewolves
            if player.role == Role.WEREWOLF:
                ww_names = [w.name for w in self.game.get_werewolves() if w.name != player.name]
                if ww_names:
                    secret_info = f"\nYour fellow werewolf is: {ww_names[0]}"

            player_context = self.get_player_context(player.name)

            prompt = f"""GAME AI QUERY
Game: Werewolf
Your character: {player.name} - {role_info}{secret_info}

{public_info}
{player_context}

Phase: DAY DISCUSSION
Task: Make a public statement (1-2 sentences)

Output format: {{"reasoning": "private thoughts", "action": "public statement"}}"""

            statement = self.call_agent(player.name, prompt, expect_json=True)
            self.game.add_event("day_statement", {"player": player.name, "statement": statement})

    def day_vote_phase(self):
        """Execute day vote phase: players vote to eliminate someone."""
        self.log(f"\n{'='*50}")
        self.log(f"TURN {self.game.turn_number} - DAY VOTE")
        self.log('='*50)

        public_info = self.get_public_info()
        alive_players = self.game.get_alive_players()
        alive_names = [p.name for p in alive_players]

        # Get recent statements
        recent_statements = [e for e in self.game.history
                           if e['turn'] == self.game.turn_number
                           and e['type'] == 'day_statement']
        statements_text = "\n".join([f"  {s['data']['player']}: {s['data']['statement']}"
                                    for s in recent_statements])

        votes = {}
        for player in alive_players:
            other_players = [p for p in alive_names if p != player.name]

            role_info = f"You are a {player.role.value.upper()}."
            secret_info = ""
            if player.role == Role.SEER and self.game.seer_investigation:
                inv = self.game.seer_investigation
                if inv['seer'] == player.name:
                    secret_info = f"\nYou investigated {inv['target']}: they are {inv['result']}."

            if player.role == Role.WEREWOLF:
                ww_names = [w.name for w in self.game.get_werewolves() if w.name != player.name]
                if ww_names:
                    secret_info = f"\nYour fellow werewolf is: {ww_names[0]}"

            player_context = self.get_player_context(player.name)

            prompt = f"""GAME AI QUERY
Game: Werewolf
Your character: {player.name} - {role_info}{secret_info}

{public_info}

Statements:
{statements_text}
{player_context}

Phase: DAY VOTE
Task: Vote to eliminate one player
Options: {', '.join(other_players)}

Output format: {{"reasoning": "vote analysis", "action": "name"}}"""

            vote = self.call_agent(player.name, prompt, expect_json=True).strip()
            votes[player.name] = vote
            self.log(f"{player.name} votes for: {vote}")

        # Count votes
        vote_counts = {}
        for voter, target in votes.items():
            vote_counts[target] = vote_counts.get(target, 0) + 1

        # Find player with most votes
        if vote_counts:
            eliminated = max(vote_counts.items(), key=lambda x: x[1])[0]
            self.log(f"\nVote results: {vote_counts}")
            self.log(f"{eliminated} is eliminated with {vote_counts[eliminated]} votes!")

            # Reveal role on elimination
            eliminated_player = self.game.get_player(eliminated)
            if eliminated_player:
                self.log(f"{eliminated} was a {eliminated_player.role.value.upper()}!")
                self.game.kill_player(eliminated)
                self.game.add_event("day_elimination", {
                    "victim": eliminated,
                    "role": eliminated_player.role.value,
                    "votes": vote_counts
                })

    def run_turn(self):
        """Run one complete turn (night + day)."""
        self.night_phase()

        # Check if game over after night kill
        winner = self.game.check_game_over()
        if winner:
            return winner

        self.day_discussion_phase()
        self.day_vote_phase()

        # Check if game over after day elimination
        winner = self.game.check_game_over()
        if winner:
            return winner

        self.game.turn_number += 1
        self.save_state()
        return None

    def save_game_stats(self, winner: str):
        """Save game statistics and history."""
        stats = {
            "game_id": self.game_id,
            "winner": winner,
            "total_turns": self.game.turn_number,
            "players": [
                {
                    "name": p.name,
                    "role": p.role.value,
                    "survived": p.alive
                }
                for p in self.game.players
            ],
            "history": self.game.history,
            "player_reasoning": {
                name: thoughts
                for name, thoughts in self.player_reasoning.items()
            }
        }

        stats_file = self.output_dir / "game_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        self.log(f"\nGame stats saved to {stats_file}")

    def run_game(self, max_turns: int = 5):
        """Run the complete game."""
        self.save_state()

        try:
            for turn in range(max_turns):
                winner = self.run_turn()
                if winner:
                    self.log(f"\n{'='*50}")
                    self.log(f"GAME OVER! {winner.upper()} WINS!")
                    self.log('='*50)
                    self.game.phase = GamePhase.GAME_OVER
                    self.save_state()
                    self.save_game_stats(winner)
                    return winner

            self.log("\nGame reached maximum turns without a winner.")
            self.save_state()
            self.save_game_stats("Draw")
            return "Draw"

        except GameAbortException as e:
            self.log(f"\n{'='*50}")
            self.log(f"GAME ABORTED: {e}")
            self.log('='*50)
            self.game.phase = GamePhase.GAME_OVER
            self.save_state()
            self.save_game_stats("Aborted")
            return "Aborted"

if __name__ == "__main__":
    coordinator = GameCoordinator()
    coordinator.run_game(max_turns=5)
