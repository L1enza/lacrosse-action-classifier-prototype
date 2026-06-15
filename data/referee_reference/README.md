# Referee Reference Images

Small broadcast crops used to tune lacrosse referee filtering.

Folders:

- `referee/`: positive examples of officials.
- `non_referee_black_white/`: player examples that include black/white uniforms or dark gear, but should not be classified as officials.

These images are not enough to train a new model. They are meant to tune and test simple visual rules:

- officials usually have a vertical black/white striped torso
- officials usually have black pants
- officials often have dark sleeves/shoulders/head area
- players may have black/white uniforms, but usually do not have repeated vertical torso stripes

The current plan is to use these as reference examples for improving `scripts/review_sports_reid_roles.py`.
