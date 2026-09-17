#!/bin/sh
set -eu

if [ "${1:-}" = "postgres" ]; then
    if [ ! -s "${PGDATA}/PG_VERSION" ]; then
        initdb \
            --pgdata="${PGDATA}" \
            --username=postgres \
            --auth-local=trust \
            --auth-host=trust
    fi

    exec postgres \
        -D "${PGDATA}" \
        -c "config_file=${PGDATA}/postgresql.conf" \
        -c listen_addresses='*'
fi

exec "$@"
