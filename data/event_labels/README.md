# Event Labels

This folder stores timestamp labels for longer game clips.

Each row in `event_labels.csv` marks one event segment:

```text
video,start,end,label,source,notes
```

Use rough timestamps at first. The goal is to build enough labeled windows to
compare tracking-motion features against real lacrosse events.

Recommended labels:

- `pass`
- `shot_goal`
- `shot_save`
- `shot_miss`
- `other`
- `replay`

