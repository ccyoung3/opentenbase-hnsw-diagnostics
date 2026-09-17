# syntax=docker/dockerfile:1.7

FROM debian:bookworm AS build

ARG BUILD_JOBS=6

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bison \
        build-essential \
        ca-certificates \
        flex \
        git \
        libreadline-dev \
        libssl-dev \
        libxml2-dev \
        perl \
        pkg-config \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src/opentenbase
COPY OpenTenBase/ ./

RUN ./configure \
        --prefix=/opt/opentenbase \
        --without-icu \
        --with-openssl \
        --with-libxml \
    && make -j"${BUILD_JOBS}" \
    && make install

WORKDIR /src/pgvector
COPY pgvector/ ./

ARG PGVECTOR_CPPFLAGS=""

RUN make -j"${BUILD_JOBS}" \
        PG_CONFIG=/opt/opentenbase/bin/pg_config \
        PG_CPPFLAGS="${PGVECTOR_CPPFLAGS}" \
    && make PG_CONFIG=/opt/opentenbase/bin/pg_config install

FROM build AS test

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        libipc-run-perl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home pgtest \
    && install -d /opt/opentenbase/lib/postgresql/pgxs/src/test/perl \
    && cp -R /src/opentenbase/src/test/perl/PostgreSQL \
        /opt/opentenbase/lib/postgresql/pgxs/src/test/perl/ \
    && chown -R pgtest:pgtest /src/pgvector

ENV PATH=/opt/opentenbase/bin:${PATH}
ENV PG_CONFIG=/opt/opentenbase/bin/pg_config
USER pgtest
WORKDIR /src/pgvector

FROM debian:bookworm-slim AS runtime

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        ca-certificates \
        libreadline8 \
        libssl3 \
        libxml2 \
        zlib1g \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system postgres \
    && useradd --system --gid postgres --home-dir /var/lib/postgresql postgres \
    && install -d -o postgres -g postgres /var/lib/postgresql/data

COPY --from=build /opt/opentenbase /opt/opentenbase
COPY scripts/docker-entrypoint.sh /usr/local/bin/opentenbase-entrypoint

RUN chmod 0755 /usr/local/bin/opentenbase-entrypoint

ENV PATH=/opt/opentenbase/bin:${PATH}
ENV PGDATA=/var/lib/postgresql/data

USER postgres
VOLUME ["/var/lib/postgresql/data"]
EXPOSE 5432

ENTRYPOINT ["opentenbase-entrypoint"]
CMD ["postgres"]
