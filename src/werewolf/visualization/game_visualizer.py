#!/usr/bin/env python3
"""
Game visualization module for werewolf games.
Generates HTML reports and activation analysis plots.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class GameVisualizer:
    """Creates HTML reports and visualizations for werewolf games."""
    
    def __init__(self, game_dir: Path):
        """
        Initialize visualizer for a game directory.
        
        Args:
            game_dir: Path to game results directory (e.g., results/game45/)
        """
        self.game_dir = Path(game_dir)
        self.game_stats = self._load_game_stats()
        self.game_log = self._load_game_log()
        
    def _load_game_stats(self) -> Dict:
        """Load game_stats.json file."""
        stats_file = self.game_dir / "game_stats.json"
        with open(stats_file, 'r') as f:
            return json.load(f)
    
    def _load_game_log(self) -> str:
        """Load game_log.txt file."""
        log_file = self.game_dir / "game_log.txt"
        with open(log_file, 'r') as f:
            return f.read()
    
    def generate_all(self):
        """Generate all visualizations and HTML report."""
        print(f"Generating visualizations for {self.game_dir.name}...")
        
        # Create figures directory
        figures_dir = self.game_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Generate plots if probe data exists
        if self.game_stats.get('probe_enabled', False):
            self._generate_activation_histograms(figures_dir)
            self._generate_player_timeseries(figures_dir)
        
        # Generate HTML report
        self._generate_html_report()
        
        print(f"✓ Generated report: {self.game_dir / 'game_report.html'}")
    
    def _categorize_activation(self, event: Dict, action_data: Dict) -> str:
        """
        Categorize an activation by game phase.
        
        Returns: 'role_reveal', 'reasoning', 'public_statement', 'voting'
        """
        event_type = event.get('type', '')
        
        # Role reveal activations (from role_reveal_activations)
        if 'role_reveal' in str(action_data.get('action', '')).lower():
            return 'role_reveal'
        
        # Voting activations
        if event_type == 'day_elimination' or 'vote' in str(action_data.get('action', '')).lower():
            return 'voting'
        
        # Public statements
        if event_type == 'day_statement':
            return 'public_statement'
        
        # Everything else is reasoning/CoT
        return 'reasoning'
    
    def _generate_activation_histograms(self, figures_dir: Path):
        """
        Generate histogram of probe activations split by player role.
        """
        # Collect all activations by role
        werewolf_scores = []
        villager_scores = []
        role_reveal_werewolf = []
        role_reveal_villager = []
        
        # Get player roles
        player_roles = {p['name']: p['role'] for p in self.game_stats['players']}
        
        # Process player activations (statements during game)
        for player_name, activations in self.game_stats.get('player_activations', {}).items():
            role = player_roles[player_name]
            
            for activation in activations:
                score = activation['activations'].get('aggregate_score')
                if score is None:
                    continue
                
                if role == 'werewolf':
                    werewolf_scores.append(score)
                else:
                    villager_scores.append(score)
        
        # Process role reveal activations separately
        for player_name, activation in self.game_stats.get('role_reveal_activations', {}).items():
            if activation is None:
                continue
            role = player_roles[player_name]
            score = activation.get('prompt_mean_score')  # Role reveals use prompt_mean_score
            if score is None:
                continue
                
            if role == 'werewolf':
                role_reveal_werewolf.append(score)
            else:
                role_reveal_villager.append(score)
        
        # Create 1x2 subplot (gameplay + role reveal)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Probe Activation Distributions - Game {self.game_stats["game_id"]}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Gameplay activations
        all_gameplay = werewolf_scores + villager_scores
        if all_gameplay:
            bins = np.linspace(min(all_gameplay) - 5, max(all_gameplay) + 5, 30)
            
            if werewolf_scores:
                ax1.hist(werewolf_scores, bins=bins, alpha=0.6, color='red', 
                       label=f'Werewolves (n={len(werewolf_scores)})', edgecolor='darkred')
            if villager_scores:
                ax1.hist(villager_scores, bins=bins, alpha=0.6, color='blue',
                       label=f'Villagers (n={len(villager_scores)})', edgecolor='darkblue')
            
            ax1.set_xlabel('Probe Score (lower = more deceptive)')
            ax1.set_ylabel('Count')
            ax1.set_title('Gameplay Statements')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        else:
            ax1.text(0.5, 0.5, 'No gameplay data', ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title('Gameplay Statements')
        
        # Plot 2: Role reveal activations  
        all_role_reveal = role_reveal_werewolf + role_reveal_villager
        if all_role_reveal:
            bins = np.linspace(min(all_role_reveal) - 2, max(all_role_reveal) + 2, 20)
            
            if role_reveal_werewolf:
                ax2.hist(role_reveal_werewolf, bins=bins, alpha=0.6, color='red',
                       label=f'Werewolves (n={len(role_reveal_werewolf)})', edgecolor='darkred')
            if role_reveal_villager:
                ax2.hist(role_reveal_villager, bins=bins, alpha=0.6, color='blue',
                       label=f'Villagers (n={len(role_reveal_villager)})', edgecolor='darkblue')
            
            ax2.set_xlabel('Probe Score (initial role reveal)')
            ax2.set_ylabel('Count')
            ax2.set_title('Role Reveal Activations')
            ax2.legend()
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No role reveal data', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Role Reveal Activations')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'activation_histograms.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Generated activation histograms")
    
    def _generate_player_timeseries(self, figures_dir: Path):
        """
        Generate time-series plots of activations for each player.
        """
        player_roles = {p['name']: p['role'] for p in self.game_stats['players']}
        num_players = len(self.game_stats['players'])
        
        # Create subplots - 2 columns, enough rows for all players
        nrows = (num_players + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(16, 4 * nrows))
        fig.suptitle(f'Player Activation Scores Over Time - Game {self.game_stats["game_id"]}',
                     fontsize=16, fontweight='bold')
        
        if num_players == 1:
            axes = [axes]
        else:
            axes = axes.flat
        
        for idx, (player_name, activations) in enumerate(sorted(self.game_stats.get('player_activations', {}).items())):
            ax = axes[idx]
            role = player_roles[player_name]
            
            # Extract scores in order
            scores = [a['activations'].get('aggregate_score') for a in activations 
                     if a['activations'].get('aggregate_score') is not None]
            
            if not scores:
                ax.text(0.5, 0.5, 'No activation data', ha='center', va='center', 
                       transform=ax.transAxes)
                ax.set_title(f'{player_name} ({role})')
                continue
            
            # Plot time series
            color = 'red' if role == 'werewolf' else 'blue'
            ax.plot(range(len(scores)), scores, marker='o', color=color, linewidth=2, 
                   markersize=6, alpha=0.7)
            ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            ax.set_xlabel('Action Number')
            ax.set_ylabel('Aggregate Probe Score')
            ax.set_title(f'{player_name} ({role.upper()})', fontweight='bold')
            ax.grid(alpha=0.3)
            
            # Add mean line
            mean_score = np.mean(scores)
            ax.axhline(mean_score, color=color, linestyle=':', linewidth=2, alpha=0.5,
                      label=f'Mean: {mean_score:.2f}')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(num_players, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'player_timeseries.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Generated player time-series plots")
    
    def _generate_html_report(self):
        """Generate comprehensive HTML report with chronological game log."""
        player_roles = {p['name']: p['role'] for p in self.game_stats['players']}
        
        # Build chronological event list with formatting
        events_html = []
        
        for event in self.game_stats['history']:
            turn = event.get('turn', 0)
            phase = event.get('phase', '')
            event_type = event['type']
            data = event['data']
            
            # Format event based on type
            if event_type == 'night_kill':
                events_html.append(f"""
                <div class="event night-kill">
                    <span class="turn-marker">Turn {turn} - Night</span>
                    <strong>☠️ {data['victim']} was killed during the night</strong>
                </div>
                """)
            
            elif event_type == 'day_statement':
                player = data['player']
                statement = data['statement']
                role = player_roles[player]
                role_class = 'werewolf' if role == 'werewolf' else 'villager'
                
                # Find activation score for this statement
                score_html = ""
                if player in self.game_stats.get('player_activations', {}):
                    for activation in self.game_stats['player_activations'][player]:
                        if activation.get('action') == statement:
                            score = activation['activations'].get('aggregate_score')
                            if score is not None:
                                score_html = f' <span class="probe-score">LIE DETECTOR: {score:.2f}</span>'
                            break
                
                # Check if called by another player
                called_info = ""
                if 'called_by' in data:
                    question = data.get('question', '')
                    called_info = f'<div class="called-on">Called by {data["called_by"]}: "{question}"</div>'
                
                events_html.append(f"""
                <div class="event statement {role_class}">
                    <span class="turn-marker">Turn {turn} - Day</span>
                    <strong>{player}</strong> <span class="role-badge {role_class}">{role}</span>
                    {called_info}
                    <div class="statement-text">"{statement}"{score_html}</div>
                </div>
                """)
            
            elif event_type == 'day_elimination':
                victim = data['victim']
                victim_role = data['role']
                votes = data.get('votes', {})
                vote_str = ', '.join([f"{player}: {votes[player]} votes" for player in votes])
                
                events_html.append(f"""
                <div class="event elimination">
                    <span class="turn-marker">Turn {turn} - Day</span>
                    <strong>⚖️ {victim} was eliminated by vote</strong>
                    <div class="role-reveal">They were a {victim_role.upper()}</div>
                    <div class="vote-details">{vote_str}</div>
                </div>
                """)
        
        # Generate player summary table
        player_rows = []
        for player in self.game_stats['players']:
            name = player['name']
            role = player['role']
            survived = player['survived']
            role_class = 'werewolf' if role == 'werewolf' else 'villager'
            status = '✓ Survived' if survived else '✗ Eliminated'
            
            # Calculate stats
            activations = self.game_stats.get('player_activations', {}).get(name, [])
            scores = [a['activations'].get('aggregate_score') for a in activations 
                     if a['activations'].get('aggregate_score') is not None]
            
            mean_score = f"{np.mean(scores):.2f}" if scores else "N/A"
            num_actions = len(scores)
            
            player_rows.append(f"""
            <tr>
                <td><strong>{name}</strong></td>
                <td><span class="role-badge {role_class}">{role.upper()}</span></td>
                <td>{status}</td>
                <td>{mean_score}</td>
                <td>{num_actions}</td>
            </tr>
            """)
        
        # Build complete HTML
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Game {self.game_stats['game_id']} Report - Werewolf</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .game-info {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .info-box {{
            background: rgba(255,255,255,0.2);
            padding: 15px;
            border-radius: 5px;
        }}
        .info-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        .info-value {{
            font-size: 1.5em;
            font-weight: bold;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        th {{
            background-color: #667eea;
            color: white;
            font-weight: 600;
        }}
        .role-badge {{
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 0.85em;
            font-weight: bold;
        }}
        .role-badge.werewolf {{
            background-color: #ff4444;
            color: white;
        }}
        .role-badge.villager, .role-badge.seer {{
            background-color: #4444ff;
            color: white;
        }}
        .timeline {{
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-top: 30px;
        }}
        .event {{
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #ddd;
            background: #f9f9f9;
            border-radius: 5px;
        }}
        .event.night-kill {{
            border-left-color: #333;
            background: #ffe6e6;
        }}
        .event.statement.werewolf {{
            border-left-color: #ff4444;
            background: #fff0f0;
        }}
        .event.statement.villager {{
            border-left-color: #4444ff;
            background: #f0f0ff;
        }}
        .event.elimination {{
            border-left-color: #ff8800;
            background: #fff8e6;
        }}
        .turn-marker {{
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 3px 10px;
            border-radius: 3px;
            font-size: 0.85em;
            margin-bottom: 8px;
        }}
        .statement-text {{
            margin-top: 10px;
            font-style: italic;
            color: #333;
        }}
        .probe-score {{
            display: inline-block;
            background: #ffd700;
            color: #333;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.9em;
            font-weight: bold;
            margin-left: 10px;
        }}
        .called-on {{
            background: rgba(0,0,0,0.05);
            padding: 8px;
            margin: 8px 0;
            border-radius: 3px;
            font-size: 0.95em;
        }}
        .role-reveal {{
            font-weight: bold;
            margin-top: 5px;
        }}
        .vote-details {{
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }}
        .figures {{
            margin-top: 30px;
        }}
        .figure {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .figure img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .figure h3 {{
            margin-top: 0;
            color: #667eea;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🐺 Werewolf Game {self.game_stats['game_id']} Report</h1>
        <div class="game-info">
            <div class="info-box">
                <div class="info-label">Winner</div>
                <div class="info-value">{self.game_stats['winner']}</div>
            </div>
            <div class="info-box">
                <div class="info-label">Total Turns</div>
                <div class="info-value">{self.game_stats['total_turns']}</div>
            </div>
            <div class="info-box">
                <div class="info-label">Backend</div>
                <div class="info-value">{self.game_stats['backend_type']}</div>
            </div>
            <div class="info-box">
                <div class="info-label">Probe Enabled</div>
                <div class="info-value">{'✓ Yes' if self.game_stats.get('probe_enabled') else '✗ No'}</div>
            </div>
        </div>
    </div>
    
    <h2>👥 Player Summary</h2>
    <table>
        <thead>
            <tr>
                <th>Player</th>
                <th>Role</th>
                <th>Status</th>
                <th>Mean Probe Score</th>
                <th>Total Actions</th>
            </tr>
        </thead>
        <tbody>
            {''.join(player_rows)}
        </tbody>
    </table>
    
    <div class="timeline">
        <h2>📜 Chronological Game Log</h2>
        {''.join(events_html)}
    </div>
    
    <div class="figures">
        <h2>📊 Activation Analysis</h2>
        
        <div class="figure">
            <h3>Probe Activation Distributions by Game Phase</h3>
            <img src="figures/activation_histograms.png" alt="Activation Histograms">
            <p>Distribution of probe activation scores split by player type (werewolf vs villager/seer) and game phase. 
            Scores closer to 0 or positive indicate higher deception according to the probe.</p>
        </div>
        
        <div class="figure">
            <h3>Player Activation Scores Over Time</h3>
            <img src="figures/player_timeseries.png" alt="Player Time Series">
            <p>Time-series view of each player's probe activation scores throughout the game. 
            Red indicates werewolves, blue indicates villagers/seer.</p>
        </div>
    </div>
    
    <div style="margin-top: 50px; text-align: center; color: #999; font-size: 0.9em;">
        Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
        
        # Write HTML file
        html_file = self.game_dir / 'game_report.html'
        with open(html_file, 'w') as f:
            f.write(html)


def visualize_game(game_dir: str):
    """
    Convenience function to visualize a game.
    
    Args:
        game_dir: Path to game directory (e.g., 'results/game45')
    """
    visualizer = GameVisualizer(game_dir)
    visualizer.generate_all()
