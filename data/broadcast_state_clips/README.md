# Broadcast State Clips

Use this dataset to train a simple broadcast-state filter before event classification.

Labels:

- `live`: normal live gameplay, usually with the scorebug/scoreboard graphic visible.
- `replay`: replay, alternate angle, broadcast cutaway, or any gameplay-looking clip where the scorebug is missing because the broadcast is showing a replay.

Important: a replay can still use the normal live camera view. Do not label only by camera angle. The scorebug is the main clue.

Suggested structure:

```text
data/broadcast_state_clips/
  train/
    live/
    replay/
  val/
    live/
    replay/
```

Start with 20-30 clips per class if possible. Clips can be 2-5 seconds.

Use this filter before pass/shot/save/goal classification:

```text
video window -> live vs replay
live -> event classifier
replay -> skip or label as replay
```
