"""Private, source-attested local fixtures for the overhead acceptance protocol."""

import ipaddress
import json
import os
from pathlib import Path
import re
import secrets
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

from tools.hnsw import observe

START = """set -eu
initdb -D /var/lib/postgresql/data -U postgres --auth-local=trust --auth-host=scram-sha-256 >/dev/null
exec postgres -D /var/lib/postgresql/data -c listen_addresses='*' \
 -c shared_buffers=256MB -c autovacuum=off -c checkpoint_timeout=1h -c max_wal_size=4GB \
 -c max_parallel_workers=4 -c max_worker_processes=8 -c jit=off -c track_io_timing=off
"""
COUNTERS = "for f in cpu.stat cpu.pressure memory.current memory.events io.stat; do echo FILE:$f; cat /sys/fs/cgroup/$f; done"


def docker(*args, input=None, timeout=60):
    p = subprocess.run(["docker", *args], input=input, capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        # Arguments/output may contain an ephemeral credential; do not echo them.
        raise RuntimeError("owned-fixture Docker operation failed")
    return p.stdout


def power_source():
    if sys.platform != 'darwin':
        supplies = list(Path('/sys/class/power_supply').glob('*/online'))
        if not supplies:
            return 'unavailable'
        return 'AC' if any(path.read_text().strip() == '1' for path in supplies) else 'Battery'
    result = subprocess.run(['pmset', '-g', 'batt'], capture_output=True, text=True, timeout=5)
    if result.returncode or not result.stdout:
        raise RuntimeError('Mac power source unavailable')
    return 'AC' if "Now drawing from 'AC Power'" in result.stdout else 'Battery'


@contextmanager
def keep_awake():
    if sys.platform != 'darwin':
        yield
        return
    # Scoped idle-sleep assertion only. Does not change settings or prevent lid sleep.
    process = subprocess.Popen(['caffeinate', '-i', '-w', str(os.getpid())],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)


def cpu_set(value):
    """Parse Docker CPU indices without accepting duplicate or reversed ranges."""
    indices = set()
    for part in value.split(','):
        if not re.fullmatch(r'\d+(?:-\d+)?', part):
            raise ValueError('CPU set must contain indices or ranges, e.g. 0-3')
        bounds = [int(number) for number in part.split('-')]
        start, end = bounds[0], bounds[-1]
        if end < start or end > 65535:
            raise ValueError('invalid CPU range')
        group = set(range(start, end + 1))
        if group & indices:
            raise ValueError('duplicate CPU index')
        indices.update(group)
    return indices


class Replica:
    def __init__(self, folder, image, port, nonce, affinity):
        self.folder, self.image, self.port, self.nonce = folder, image, port, nonce
        self.affinity = affinity
        self.name = f"hnsw-acceptance-{nonce}-{port}"
        self.volume = self.name + "-data"
        self.password, self.watcher_password = secrets.token_hex(24), secrets.token_hex(24)
        self.container_id = None
        self.volume_created = False
        self.run_attempted = False
        self.connections = []

    def connect(self, watcher=False):
        import psycopg
        conn = psycopg.connect(host="127.0.0.1", port=self.port, dbname="postgres",
            user="hnsw_watcher" if watcher else "postgres", password=self.watcher_password if watcher else self.password,
            autocommit=True, client_encoding="UTF8", connect_timeout=5, prepare_threshold=None,
            application_name="hnsw_readonly_observer" if watcher else "hnsw_acceptance_builder",
            options=observe.OPTIONS if watcher else "-c statement_timeout=180000 -c lock_timeout=2000")
        self.connections.append(conn)
        return conn

    def start(self, large_rows, small_rows):
        self.folder.mkdir()
        docker("volume", "create", "--label", "hnsw.acceptance.run=" + self.nonce, self.volume)
        self.volume_created = True
        self.run_attempted = True
        self.container_id = docker("run", "-d", "--name", self.name, "--label", "hnsw.acceptance.run=" + self.nonce,
            "--memory=2g", "--memory-swap=2g", "--shm-size=1g", "--cpuset-cpus=" + self.affinity,
            "-p", f"127.0.0.1:{self.port}:5432", "-v", self.volume + ":/var/lib/postgresql/data",
            "--entrypoint", "/bin/sh", self.image, "-c", START).strip()
        for _ in range(100):
            p = subprocess.run(["docker", "exec", self.name, "/opt/opentenbase/bin/pg_isready", "-U", "postgres"],
                               capture_output=True, timeout=5)
            if p.returncode == 0:
                break
            time.sleep(.1)
        else:
            raise RuntimeError("owned database did not become ready")
        script = (f"ALTER USER postgres PASSWORD '{self.password}';\n"
                  f"CREATE ROLE hnsw_watcher LOGIN PASSWORD '{self.watcher_password}';\n"
                  "GRANT pg_read_all_stats TO hnsw_watcher;\n")
        docker("exec", "-i", self.name, "/opt/opentenbase/bin/psql", "-X", "-v", "ON_ERROR_STOP=1",
               "-U", "postgres", "-d", "postgres", input=script)
        details = json.loads(docker("inspect", self.name))[0]
        assert details["Image"] == self.image and details["Config"]["Labels"]["hnsw.acceptance.run"] == self.nonce
        assert details["HostConfig"]["Memory"] == details["HostConfig"]["MemorySwap"] == 2 * 2**30
        assert details["HostConfig"]["CpusetCpus"] == self.affinity
        gateway = ipaddress.ip_address(details["NetworkSettings"]["Networks"]["bridge"]["Gateway"])
        if gateway.version != 4:
            raise RuntimeError("IPv4 task-local bridge required")
        hba = self.folder / "pg_hba.conf"
        hba.write_text("local all all trust\nhost all all 127.0.0.1/32 scram-sha-256\n"
                       "host all all ::1/128 scram-sha-256\n"
                       f"host postgres postgres,hnsw_watcher {gateway}/32 scram-sha-256\n")
        docker("cp", str(hba), self.name + ":/var/lib/postgresql/data/pg_hba.conf")
        docker("exec", self.name, "/opt/opentenbase/bin/psql", "-X", "-U", "postgres",
               "-c", "SELECT pg_reload_conf()")
        time.sleep(.2)
        with self.connect() as conn:
            actual = str(conn.execute("SELECT system_identifier FROM pg_control_system()").fetchone()[0])
            local = docker("exec", self.name, "/opt/opentenbase/bin/psql", "-X", "-At", "-U", "postgres",
                           "-c", "SELECT system_identifier FROM pg_control_system()").strip()
            assert actual == local
            conn.execute(f"""CREATE EXTENSION vector; CREATE SCHEMA hnsw_accept;
                SELECT setseed(.42);
                CREATE TABLE hnsw_accept.large (id bigint PRIMARY KEY, embedding vector(32) NOT NULL);
                INSERT INTO hnsw_accept.large SELECT i, ARRAY(SELECT random()::real
                    FROM generate_series(1,32) d WHERE i IS NOT NULL ORDER BY d)::vector
                    FROM generate_series(1,{large_rows}) i;
                CREATE TABLE hnsw_accept.small AS SELECT * FROM hnsw_accept.large WHERE id <= {small_rows};
                ANALYZE hnsw_accept.large; ANALYZE hnsw_accept.small; CHECKPOINT;""")
            data = {}
            for table in ("large", "small"):
                count, digest = conn.execute(f"SELECT count(*), md5(string_agg(id::text || ':' || embedding::text, '|' ORDER BY id)) FROM hnsw_accept.{table}").fetchone()
                data[table] = dict(rows=count, digest=digest)
            assert data["large"]["rows"] == large_rows and data["small"]["rows"] == small_rows
        self.identity = dict(container_id=self.container_id, image=self.image, system_identifier=actual,
            port=self.port, affinity=self.affinity, memory_limit=2*2**30, data=data,
            source_attestation=docker("exec", self.name, "cat", "/opt/hnsw-acceptance-identity.txt"))
        (self.folder / "identity.json").write_text(json.dumps(self.identity, indent=2) + "\n")

    def counters(self):
        return dict(host_load_average=os.getloadavg(), power_source=power_source(),
                    cgroup=docker("exec", self.name, "/bin/sh", "-c", COUNTERS))

    def close(self):
        errors = []
        for conn in self.connections:
            try:
                conn.close()
            except Exception as error:
                errors.append(type(error).__name__)
        if self.run_attempted and not self.container_id:
            probe = subprocess.run(["docker", "inspect", self.name], capture_output=True, text=True, timeout=10)
            if probe.returncode == 0:
                details = json.loads(probe.stdout)[0]
                if details["Config"]["Labels"].get("hnsw.acceptance.run") == self.nonce:
                    self.container_id = details["Id"]
                else:
                    errors.append("container ownership mismatch; preserved")
        if self.container_id:
            try:
                details = json.loads(docker("inspect", self.name))[0]
                assert details["Id"] == self.container_id and details["Config"]["Labels"]["hnsw.acceptance.run"] == self.nonce
                docker("stop", "-t", "10", self.container_id)
                docker("rm", "-v", self.container_id)
            except Exception as error:
                errors.append(type(error).__name__)
        if self.volume_created:
            try:
                details = json.loads(docker("volume", "inspect", self.volume))[0]
                assert details["Labels"]["hnsw.acceptance.run"] == self.nonce
                docker("volume", "rm", self.volume)
            except Exception as error:
                errors.append(type(error).__name__)
        self.folder.mkdir(exist_ok=True)
        (self.folder / "cleanup.json").write_text(json.dumps(dict(status="incomplete" if errors else "complete", errors=errors), indent=2) + "\n")
        return errors


@contextmanager
def panel(folder, images, *, large_rows, small_rows, affinity="0-3"):
    if not 10000 <= small_rows <= large_rows <= 200000:
        raise ValueError("bounded task-local data sizes required")
    if len(cpu_set(affinity)) < 3 or not 2 <= len(images) <= 4:
        raise ValueError("unsupported fixture configuration")
    folder.mkdir()
    for port in range(55432, 55432 + len(images)):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                raise RuntimeError("reserved port occupied; no existing database will be touched")
        except ConnectionRefusedError:
            pass
    nonce = secrets.token_hex(6)
    replicas = {}
    try:
        for offset, (label, image) in enumerate(images.items()):
            replica = Replica(folder / label, image, 55432+offset, nonce, affinity)
            replicas[label] = replica
            replica.start(large_rows, small_rows)
        assert len({json.dumps(r.identity["data"], sort_keys=True) for r in replicas.values()}) == 1
        yield replicas
    finally:
        errors = [error for replica in replicas.values() for error in replica.close()]
        if errors:
            raise RuntimeError("owned fixture cleanup incomplete")
