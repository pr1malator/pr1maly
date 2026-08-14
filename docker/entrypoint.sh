#!/bin/sh
# Make /app/data writable by the app user, then become that user.
#
# The container runs the application as uid 1000, not root. That is worth
# doing, but it broke every existing install: a database written by the
# earlier root container is -rw-r--r-- root root, and uid 1000 can read it
# and not write it. SQLite reports that as "attempt to write a readonly
# database", which says nothing about ownership and sends you looking in the
# wrong place. Ask the person who spent an evening on it.
#
# So: if we start as root, hand the data directory over first. If we do not —
# because someone passed `user:` in compose or `--user` on the command line —
# there is nothing to hand over and nothing to drop, so just run.

set -e

APP_UID=1000
APP_GID=1000
DATA_DIR="${DATA_DIR:-/app/data}"

if [ "$(id -u)" = "0" ]; then
    if [ -d "$DATA_DIR" ]; then
        # Only what is actually wrong. The demo folder can hold hundreds of
        # files of a few hundred MB each, and chowning all of it on every
        # start is a slow no-op across a bind mount.
        find "$DATA_DIR" \( ! -user "$APP_UID" -o ! -group "$APP_GID" \) \
            -exec chown "$APP_UID:$APP_GID" {} + 2>/dev/null || true
    fi
    exec setpriv --reuid="$APP_UID" --regid="$APP_GID" --init-groups "$@"
fi

exec "$@"
