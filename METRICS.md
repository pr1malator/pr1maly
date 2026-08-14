# What the numbers mean

Every figure this app shows you, what it measures, how it is worked out,
and — where it grades you — where that grade came from.

*Generated from `src/domain/metrics/catalogue.py` by
`tools/generate_metric_reference.py`. Edit the catalogue, not this file.*

## How to read a grade

**Measured.** Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

**Published elsewhere.** A formula published by someone else, reproduced as specified. Not ours to change, and not calibrated against this player.

**Hand-set.** These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

Most of the tiers in this app are the third kind. That is worth knowing
before reading a verdict as a comparison against other players.

## Every figure carries the same statistics

| | |
|---|---|
| `median` | The headline figure. Used because one outlier round cannot move it. |
| `avg` | The mean, kept alongside the median: a wide gap between them means a skewed match. |
| `min` | The best single sample in the match. |
| `max` | The worst single sample in the match. |
| `n` | How many samples the figure is built from. |
| `confidence` | How much weight the sample count supports — low, medium or high. A figure from four duels is not the same claim as one from forty. |

## The scoreboard

Computed while the match is being read, alongside the scoreboard. Not separately versioned.

### HLTV 2.0 Rating

`hltv_rating`

Overall contribution across the match, on the scale where 1.00 is par.

**How:** HLTV's published 2.0 formula over kills, deaths, KAST, ADR and impact, per round.

**Worth knowing:** The coefficients are HLTV's, applied to matchmaking demos rather than the professional matches they were fitted on.

**Graded against:** published elsewhere thresholds. A formula published by someone else, reproduced as specified. Not ours to change, and not calibrated against this player.

### Average Damage per Round (damage)

`adr`

Damage dealt to enemies per round played.

**How:** Damage capped at the health the victim actually had, so overkill does not inflate it, then divided by rounds.

### KAST (%)

`kast`

Share of rounds in which you got a kill, an assist, survived, or were traded.

**How:** Rounds meeting any of the four conditions, over rounds played.

### Kills per Round

`kpr`

Kills divided by rounds played.

**How:** Total kills over total rounds.

### Deaths per Round

`dpr`

Deaths divided by rounds played.

**How:** Total deaths over total rounds.

### Impact

`impact`

HLTV's impact term: opening duels and multi-kills weighted above ordinary trades.

**How:** The published impact expression over kills per round, opening kills and multi-kill rounds.

**Graded against:** published elsewhere thresholds. A formula published by someone else, reproduced as specified. Not ours to change, and not calibrated against this player.

### Round timeline

`round_stats`

Your kills, deaths, assists, damage and survival, round by round.

**How:** One row per round, taken from the events attributed to you in that round.

**Worth knowing:** The source every match-level figure above is aggregated from, and what the round timeline in the interface draws.

## Aim and movement

`aim.stats` v1 — Shot speed, crosshair placement, time to damage, reaction, accuracy and counter-strafe quality, with the sample behind each figure. It can be recomputed from the database alone, without the demo.

### Aim Rating (/100)

`aim_stats.aim_rating`

A single figure for mechanical quality in duels.

**How:** Crosshair placement 40%, shot speed 30%, engagement time 30%, each scored against its own tier bounds.

**Worth knowing:** Reaction time and counter-strafe rate are deliberately not inputs. Reaction has too few samples per match to grade, and counter-strafe would score movement twice — shot speed already measures the outcome.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Shot Speed (u/s)

`aim_stats.movement`

How fast you were moving at the moment you fired the killing shot.

**How:** Your own velocity on the tick of the shot, taken across every kill.

**Worth knowing:** Rifles lose accuracy above roughly a third of running speed, which is where the tiers sit.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Peek Speed (u/s)

`aim_stats.peek`

How fast you were travelling in the half-second before firing.

**How:** Median velocity over the 0.5s window preceding the shot.

**Worth knowing:** Deliberately ungraded. The bands name what kind of peek it was — held, walk, half speed, full speed — rather than ranking them; a held angle and a fast peek are different choices, not better and worse ones.

### Crosshair Placement (°)

`aim_stats.preaim`

How far your crosshair was from the enemy when the duel began.

**How:** Angle between your view direction and the enemy at the moment they became engageable.

**Worth knowing:** The largest single input to the aim rating, because it is the one most under your control.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Engagement Time (s)

`aim_stats.ttk`

How long a duel took from your first damage to the kill.

**How:** Time between first damage dealt and the killing blow, outliers excluded and counted separately.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Reaction Time (ms)

`aim_stats.reaction`

How quickly you fired once your crosshair was on the enemy.

**How:** Walks back from the first shot to the last tick your crosshair was off the enemy; the gap between that and the shot is the reaction.

**Worth knowing:** Diagnostic only, and never an input to the aim rating. Two to twenty samples a match, and what is measured is the gap between your crosshair arriving on the enemy and your shot — which is not the same as how fast you reacted, because an enemy can walk into a crosshair you were already holding.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Accuracy (%)

`aim_stats.accuracy`

Share of your shots that hit, with the head and lower-body split.

**How:** Hits over shots fired in duels, broken down by hitgroup.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Duels

`aim_stats.encounters`

The individual duels every aim figure above is built from.

**How:** One record per engagement, carrying its weapon, distance, and each measurement taken.

**Worth knowing:** This is the raw material: any aim figure can be traced back to the duels behind it.

### Aim tier bounds

`aim_stats.thresholds`

The cut-offs each aim figure is graded against, and which direction is better.

**How:** Hand-set bounds, shipped with the analysis so the interface grades against the same lines the backend used.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Engagement time benchmark (ms)

`benchmarks.engagement_ttk`

Your engagement time placed in a tier.

**How:** The median engagement time compared against hand-set cut-offs.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

## Utility

`utility.stats` v1 — Grenades thrown and what they achieved, flash effectiveness, and how much of the grenade spend was wasted. It needs the demo file to recompute.

### Utility Rating (/100)

`utility_data.utility_rating`

A single figure for how well your grenades were spent.

**How:** Combines flash effectiveness, damage from HE and molotov, and how much utility went unused.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Flashes

`utility_data.flash`

Enemies blinded, for how long, how often a flash achieved anything, and flash assists.

**How:** Blind events attributed to your flashes, with duration and whether a kill followed.

### HE Grenades (damage)

`utility_data.he`

Damage dealt by your HE grenades and how often they landed.

**How:** Damage attributed to HE detonations, over grenades thrown.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Molotovs (damage)

`utility_data.molotov`

Damage dealt by your incendiaries and how often they landed.

**How:** Damage attributed to fire you started, over molotovs thrown.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Smokes

`utility_data.smoke`

Where you smoke, and how often a smoke put out a molotov.

**How:** Smoke detonation positions resolved to callouts, plus extinguish events.

### Utility Economy ($)

`utility_data.economics`

What you spent on grenades and how much of it was never thrown.

**How:** Buy value of utility against what was used, per match.

**Worth knowing:** Utility you died holding counts as wasted — the money bought nothing.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Team Support

`utility_data.teamplayer`

Flashes and drops that helped teammates, and damage you did to them.

**How:** Flash assists and weapon drops for teammates, against team damage and team flashes.

### Utility by round

`utility_data.per_round`

The same utility figures, round by round.

**How:** One entry per round, so a single expensive round can be told from a pattern.

### Enemies flashed benchmark (per 24 rounds)

`benchmarks.enemies_flashed`

How many enemies you blinded, placed in a tier.

**How:** Enemies flashed scaled to a 24-round match, against per-map cut-offs.

**Worth knowing:** The cut-offs differ by map because the maps do — Dust2 rewards flashes Inferno does not.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Utility damage benchmark (damage)

`benchmarks.utility_damage`

Damage from HE and molotov, placed in a tier.

**How:** Total utility damage scaled to a 24-round match, against hand-set cut-offs.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

### Utility waste benchmark (%)

`benchmarks.utility_waste_pct`

Share of bought utility you never threw, placed in a tier.

**How:** Unused utility value over utility bought, against hand-set cut-offs.

**Graded against:** hand-set thresholds. These are sensible targets, not percentiles of a player population. A tier here says where you fall against a line somebody chose, not against other players.

## Round impact

`impact.stats` v1 — Win-probability swing attributed to each kill and death. It needs the demo file to recompute.

### Net Impact per Round (win probability)

`impact_stats.net_swing_per_round`

How much your kills and deaths moved your team's chance of winning, per round.

**How:** Every kill and death is priced by how the round's win probability changes with the player count, and the two are netted off.

**Worth knowing:** The win-probability table is measured from real matches and its observation count is recorded per cell; thinly observed states fall back to a formula.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

### Impact won (win probability)

`impact_stats.kill_swing_total`

Total win probability your kills added across the match.

**How:** Sum of the swing attributed to each kill.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

### Impact lost (win probability)

`impact_stats.death_swing_total`

Total win probability your deaths cost across the match.

**How:** Sum of the swing attributed to each death.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

### Best kill (win probability)

`impact_stats.best_kill_swing`

The single kill that moved the round most.

**How:** Largest win-probability swing among your kills.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

### Typical kill (win probability)

`impact_stats.median_kill_swing`

What a normal kill of yours was worth this match.

**How:** Median swing across your kills, so one clutch does not set the figure.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

### Impact by round

`impact_stats.per_round`

The same swing figures, round by round.

**How:** One entry per round, with the kills and deaths that produced it.

**Graded against:** measured thresholds. Taken from a real corpus of matches. The numbers behind it can be regenerated, and the observation count for each cell is recorded.

## Roles

`roles.positional` v2 — The position played on each side, from where the player fights and dies. It needs the demo file to recompute.

### Primary CT role

`role_data.ct_primary`

Where you actually played on defence.

**How:** Your position at round start is resolved to a map callout, and the callouts are scored against the role definitions for that map.

**Worth knowing:** Positional, not tactical: it says where you stood, which is not the same as what your team asked you to do.

### Primary T role

`role_data.t_primary`

Where you actually played on attack.

**How:** The same zone scoring, against the attacking role definitions for that map.

### CT role split (rounds)

`role_data.roles_ct`

How your defensive rounds divided between roles.

**How:** Rounds attributed to each role, in order of frequency.

### T role split (rounds)

`role_data.roles_t`

How your attacking rounds divided between roles.

**How:** Rounds attributed to each role, in order of frequency.
