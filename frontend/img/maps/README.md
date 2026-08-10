# CS2 Map Icons

Valve's map badges, shown in the Trend map picker, the match history list, and
the match detail banner.

Files are named exactly as the map appears in the database, so no lookup table
is needed:

- `de_mirage.png`
- `de_dust2.png`
- `cs_office.png`

512×512 PNG with transparency. The largest place one is drawn is the match
detail banner at ~192 px, so 512 leaves room for high-DPI screens without
needing a second size.

## Coverage

The set here mirrors `../radar/` — every map the app already supports:

`de_ancient` `de_anubis` `de_cache` `de_dust2` `de_inferno` `de_mirage`
`de_nuke` `de_overpass` `de_train` `de_vertigo` `cs_office`

Coverage does not have to be complete. Every place an icon is drawn falls back
to a text abbreviation (`MIR`, `D2`) when the file is missing, so a map from a
rotated Premier pool degrades to what the UI showed before rather than breaking.
Adding one is just dropping in a correctly-named PNG — no code change.

## Where they come from

Extracted from the CS2 game files at:
`steamapps/common/Counter-Strike Global Offensive/game/csgo/panorama/images/map_icons/`

These files were taken from <https://github.com/MurkyYT/cs2-map-icons>, which
extracts them from Valve's depot automatically, so newer maps can be picked up
from there without a CS2 install.

The icons are the property of Valve Corporation — see `THIRD-PARTY-NOTICES` in
the repository root.
