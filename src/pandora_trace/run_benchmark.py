import json
import math
import os
import random
import subprocess
from contextlib import contextmanager
from enum import Enum
from time import sleep
from typing import Set, NamedTuple, List, Literal

import docker
from docker.models.containers import Container

from src.pandora_trace.jaeger_to_gent import (download_traces_from_jaeger_for_all_services,
                                                  translate_jaeger_to_gent,
                                                  translate_jaeger_to_gent_from_list)

FUZZLER_COMPILE_COMMAND = "/RESTler/restler/Restler compile --api_spec ./swagger.json"
EXEC_FUZZ_LEAN_COMMAND = "/RESTler/restler/Restler fuzz-lean --grammar_file ./Compile/grammar.py --dictionary_file ./Compile/dict.json --settings ./Compile/engine_settings.json --no_ssl"

BENCHMARK_DIR = "/Users/saart/cmu/GenT/src/paper/benchmark/data/restler"
DEATHSTAR_DIR = "/Users/saart/cmu/DeathStarBench"
SOCIAL_NETWORK_APP = "socialNetwork"


class AppName(Enum):
    socialNetwork = "socialNetwork"
    hotelReservation = "hotelReservation"
    mediaMicroservices = "mediaMicroservices"


class Incident(NamedTuple):
    command: str
    ratio: float
    incident_name: str
    apt_dependencies: List[str]


RAW_TRAIN_TICKET_INCIDENTS = [
    ("tc qdisc add dev eth0 root netem loss 50%", "packet_loss", ["iproute2"]),
    ("tc qdisc add dev eth0 root netem delay 100ms 20ms distribution normal", "latency", ["iproute2"]),
    ("shutdown", "crush", []),
    ("stress --cpu 1 --timeout 60s", "cpu_load", ["stress"]),
    ("stress --vm 4 --vm-bytes 256M --timeout 60s", "memory_stress", ["stress"]),
    ("stress --io 4 --timeout 60s", "disk_io_stress", ["stress"]),
]
INCIDENTS: List[Incident] = [
    Incident(command=cmd, ratio=p / 10, incident_name=f"{name}-{p / 10}", apt_dependencies=deps)
    for p in range(1, 11) for cmd, name, deps in RAW_TRAIN_TICKET_INCIDENTS
]


@contextmanager
def setup_test(app: AppName):
    before_containers = set(docker.from_env().containers.list())
    subprocess.check_output("docker-compose up -d --wait", cwd=f"{DEATHSTAR_DIR}/{app.value}",
                            shell=True, stderr=subprocess.STDOUT)
    app_containers = set(docker.from_env().containers.list()) - before_containers
    restler_container: Container = docker.from_env().containers.run(
        "restler", stdin_open=True, tty=True, detach=True, network_mode="host", auto_remove=True,
        working_dir=f"/code/{app.value}",
        volumes={BENCHMARK_DIR: {"bind": "/code", "mode": "rw"}},
    )
    restler_container.exec_run(FUZZLER_COMPILE_COMMAND)
    try:
        yield restler_container, app_containers
    finally:
        try:
            print("docker compose down...", end=" ", flush=True)
            subprocess.check_output("docker-compose down", cwd=f"{DEATHSTAR_DIR}/{app.value}",
                                    shell=True, stderr=subprocess.STDOUT)
            print("done")
        except Exception as e:
            print("Failed to kill app", e)
        try:
            restler_container.kill()
        except Exception as e:
            print("Failed to kill fuzzler", e)


def print_result(app: str):
    results_dir = f"{BENCHMARK_DIR}/{app}/FuzzLean/RestlerResults/"
    for experiment in os.listdir(results_dir):
        print(f"Experiment {experiment}")
        with open(f"{results_dir}/{experiment}/logs/main.txt") as f:
            print(f.read())


def run(container: Container, cmd: str, **kwargs):
    for i in range(5):
        try:
            res = container.exec_run(cmd, privileged=True, user='root', **kwargs)
            if res.exit_code == 0 or kwargs.get("detach"):
                return res
            raise Exception(f"Exit status {res}, output: {res.output[:100]}")
        except Exception as e:
            print(f"Failed to add chaos (attempt {i}/5)", e)
            sleep(5)
    raise Exception(f"Failed to execute command {cmd}")


def add_chaos(app_containers: Set[Container], incident: Incident):
    chosen_containers = random.sample(list(app_containers), math.ceil(len(app_containers) * incident.ratio))
    print(f"Adding incident {incident.incident_name} to containers {[c.name for c in chosen_containers]}", end='', flush=True)
    for container in chosen_containers:
        print('.', end='', flush=True)
        try:
            if incident.apt_dependencies:
                run(container, "apt update --allow-insecure-repositories")
                [run(container, f"apt install -y {dep}") for dep in incident.apt_dependencies]
            if "crush" in incident.incident_name:
                container.kill()
            else:
                run(container, incident.command, tty=True, demux=False, detach=True)
        except Exception as e:
            print(f"Could not add incident {incident.incident_name} to container {container.name} due to {e}")
    print()


def run_restler(restler_container: Container):
    fuzz_lean = restler_container.exec_run(EXEC_FUZZ_LEAN_COMMAND)
    return [l for l in fuzz_lean.output.decode().splitlines() if "Attempted requests" in l][0]


def run_test(app: AppName, incidents: List[Incident], target_count: int = 3000):
    for incident in incidents:
        target_dir = f"data/{app.value}/{incident.incident_name}/raw_jaeger/"
        incident_traces = []
        if os.path.exists(target_dir):
            for f in os.listdir(target_dir):
                incident_traces.extend(json.load(open(os.path.join(target_dir, f))))
        if len(incident_traces) > target_count:
            print(f"Skipping incident {incident.incident_name} as it already has enough traces")
            continue
        else:
            print(f"Running incident {incident.incident_name}")
        with setup_test(app) as (restler_container, app_containers):
            add_chaos(app_containers, incident)
            for _ in range(15):
                run_restler(restler_container)
                traces = download_traces_from_jaeger_for_all_services(target_dir=target_dir)
                if traces > target_count:
                    print(f"Collected {traces} traces for incident {incident.incident_name}")
                    break
                print(f"Collected {traces}/{target_count} traces for incident {incident.incident_name}. Keep trying...")
            else:
                print(f"Failed to collect enough traces for incident {incident.incident_name}")


def create_baseline(app: AppName):
    with setup_test(app) as (restler_container, app_containers):
        for i in range(100):
            run_restler(restler_container)
            traces = download_traces_from_jaeger_for_all_services(target_dir=f"data/{app.value}/baseline/raw_jaeger/")
            print(f"Baseline collected {traces} traces in the {i}th iteration")
    translate_jaeger_to_gent(from_dir=f"data/{app.value}/baseline/raw_jaeger/")


def merge_with_exp(benign_traces: List[dict], incident_traces: List[dict], exp_lambda: float) -> List[dict]:
    merged_traces = []
    time_until_next_incident = random.expovariate(exp_lambda)

    for trace in benign_traces:
        if time_until_next_incident <= 0:
            if not incident_traces:
                if len(merged_traces) > 10_000:
                    break
                raise Exception(f"Ran out of incidents after {len(merged_traces)} / 10000. Shouldn't happen. "
                                f"Create more incidents.")
            merged_traces.append(incident_traces.pop())
            time_until_next_incident = random.expovariate(exp_lambda)
        else:
            merged_traces.append(trace)
            time_until_next_incident -= 1

    return merged_traces


def prepare_merged_traces(app: AppName, incident: Incident, exp_lambda: float):
    incident_traces = []
    incident_dir = f"data/{app.value}/{incident.incident_name}/raw_jaeger"
    if not os.path.exists(incident_dir):
        print(f"Skipping incident {incident.incident_name}")
        return
    for f in os.listdir(incident_dir):
        incident_traces.extend(json.load(open(os.path.join(incident_dir, f))))

    benign_dir = f"data/{app.value}/baseline/raw_jaeger"
    benign_traces = sum((json.load(open(os.path.join(benign_dir, f))) for f in os.listdir(benign_dir)), [])

    merged_traces = merge_with_exp(benign_traces, incident_traces, exp_lambda)

    target_dir = f"/Users/saart/cmu/GenT/traces/{app.value}_{incident.incident_name}_{exp_lambda}"
    os.makedirs(target_dir, exist_ok=True)
    translate_jaeger_to_gent_from_list(merged_traces, f"{target_dir}/txs.json")


if __name__ == '__main__':
    # create_baseline(AppName.socialNetwork)
    # run_test(AppName.socialNetwork, incidents=INCIDENTS)
    for incident in INCIDENTS:
        for exp_lambda in [0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]:
            prepare_merged_traces(AppName.socialNetwork, incident, exp_lambda=exp_lambda)
