"""
Werewolf game state manager.
Tracks players, roles, eliminations, and game progression.
"""

import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class Role(Enum):
    WEREWOLF = "werewolf"
    SEER = "seer"
    VILLAGER = "villager"

class GamePhase(Enum):
    NIGHT = "night"
    DAY_DISCUSSION = "day_discussion"
    DAY_VOTE = "day_vote"
    GAME_OVER = "game_over"

@dataclass
class Player:
    name: str
    role: Role
    alive: bool = True

    def to_dict(self):
        return {
            "name": self.name,
            "role": self.role.value,
            "alive": self.alive
        }

class GameState:
    def __init__(self, player_names: List[str], roles: List[Role]):
        assert len(player_names) == len(roles)
        self.players = [Player(name, role) for name, role in zip(player_names, roles)]
        self.phase = GamePhase.NIGHT
        self.turn_number = 1
        self.history: List[Dict] = []
        self.night_kill: Optional[str] = None
        self.seer_investigation: Optional[Dict] = None

    def get_alive_players(self) -> List[Player]:
        return [p for p in self.players if p.alive]

    def get_player(self, name: str) -> Optional[Player]:
        for p in self.players:
            if p.name == name:
                return p
        return None

    def kill_player(self, name: str):
        player = self.get_player(name)
        if player:
            player.alive = False

    def get_werewolves(self) -> List[Player]:
        return [p for p in self.players if p.role == Role.WEREWOLF and p.alive]

    def get_seer(self) -> Optional[Player]:
        for p in self.players:
            if p.role == Role.SEER and p.alive:
                return p
        return None

    def check_game_over(self) -> Optional[str]:
        """Returns winning team if game is over, None otherwise."""
        alive = self.get_alive_players()
        werewolves = [p for p in alive if p.role == Role.WEREWOLF]
        villagers = [p for p in alive if p.role != Role.WEREWOLF]

        if len(werewolves) == 0:
            return "Village"
        if len(werewolves) >= len(villagers):
            return "Werewolves"
        return None

    def add_event(self, event_type: str, data: Dict):
        """Add an event to the game history."""
        self.history.append({
            "turn": self.turn_number,
            "phase": self.phase.value,
            "type": event_type,
            "data": data
        })

    def to_dict(self):
        return {
            "players": [p.to_dict() for p in self.players],
            "phase": self.phase.value,
            "turn_number": self.turn_number,
            "history": self.history
        }

    def save(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct players
        players_data = data['players']
        player_names = [p['name'] for p in players_data]
        roles = [Role(p['role']) for p in players_data]

        game = cls(player_names, roles)

        # Restore state
        for i, p_data in enumerate(players_data):
            game.players[i].alive = p_data['alive']

        game.phase = GamePhase(data['phase'])
        game.turn_number = data['turn_number']
        game.history = data['history']

        return game
