# Banshi (办事) Quest — Multiplayer AI Dungeon Master in Your Terminal

A real-time multiplayer terminal RPG where an AI dungeon master narrates your fate.
Friends connect, pick characters, and descend into chaos together.

## The Experience

```
$ banshi connect banshi.sukaseven.com

Welcome, traveler. Enter your name: poop_head
Choose your class: [1] Warrior  [2] Rogue  [3] Mage
> 2

Waiting for other players... (2/4 connected)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 BANSHI — The Cave of Eternal Suffering
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DM: You and your party stand at the entrance of a cave.
    The air smells of death and bad decisions.
    poop_head notices something glinting in the dark.

[poop_head]: I stab it
[brocoli_king]: I run away
[DM]: poop_head lunges forward — it's a mirror. You stab
      yourself. Take 3 damage. brocoli_king trips while
      fleeing and also takes 2 damage. Classic.
```

## Architecture

```
Player 1 (Rust TUI) ──┐
Player 2 (Rust TUI) ──┤──► Go Server on Railway ──► Anthropic API
Player 3 (Rust TUI) ──┘         │
                                └──► Postgres (game state, history)
```

- **Rust client** — TUI that each player runs locally, connects via WebSocket
- **Go server** — manages rooms, players, turns, and calls the AI
- **Anthropic API** — the DM brain, fed full game context each turn
- **Postgres** — persists game history

## Phases

1. **Phase 1** — Connect + chat between players, no AI yet
2. **Phase 2** — AI DM narrates player actions
3. **Phase 3** — HP, classes, death, game over screen
4. **Phase 4** — Multiple rooms via room codes