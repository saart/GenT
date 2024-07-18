import sqlite3
from collections import OrderedDict
from string import Formatter
from typing import List

import numpy
import pandas
import scipy

BENCHMARK_QUERIES = [
    ("Find traces of a service that having an error", """
    SELECT S1.startTime / 3600 as f, COUNT(*) as c
    FROM {table_name} as S1, {table_name} as S2
    WHERE S1.serviceName = '{entry_point}'
      AND S1.traceId = S2.traceId
      AND S2.status = 1
    GROUP BY S1.startTime / 3600;
    """, ["crush", "packet_loss"]),
    ("Find traces that have a particular attribute", """
    SELECT S2.{attr_name} as f, COUNT(*) as c
    FROM {table_name} as S1, {table_name} as S2
    WHERE S1.serviceName = '{entry_point}'
      AND S1.traceId = S2.traceId
    GROUP BY S2.{attr_name};
    """, ["crush", "packet_loss"]),
    ("Discover architecture of the whole system", """
     SELECT S1.startTime / 3600 as f, COUNT(*) as c
    FROM {table_name} as S1, {table_name} as S2
    Where S1.spanId = S2.parentId
        AND S1.serviceName = '{service_name}'
        AND S2.serviceName = '{service_name2}'
    GROUP BY S1.startTime / 3600;
    """, ["crush", "packet_loss"]),
    ("Find bottlenecks", """
    SELECT ROUND((S2.endTime - S2.startTime) / (S1.endTime - S1.startTime), 1) AS f, count(*) as c
    FROM {table_name} as S1, {table_name} as S2
    WHERE S1.serviceName = '{entry_point}'
      AND S2.serviceName = '{service_name}'
      AND S1.traceId = S2.traceId
    GROUP BY f;
    """, ["cpu_load", "disk_io_stress", "latency", "memory_stress"]),
    ("RED metrics - rate", """
    SELECT startTime / 3600 as f, COUNT(*) as c
    FROM {table_name}
    WHERE serviceName = '{service_name}'
    GROUP BY startTime / 3600;
    """, ["crush", "packet_loss", "cpu_load", "disk_io_stress", "latency", "memory_stress"]),
    ("RED metrics - error", """
    SELECT endTime - startTime as f, COUNT(*) as c
    FROM {table_name}
    WHERE serviceName = '{service_name}'
        AND status = 1
    GROUP BY endTime - startTime;
    """, ["crush", "packet_loss", "cpu_load", "disk_io_stress", "latency", "memory_stress"]),
    ("RED metrics - duration", """
    SELECT endTime - startTime as f, COUNT(*) as c
    FROM {table_name}
    WHERE serviceName = '{service_name}'
    GROUP BY endTime - startTime;
    """, ["crush", "packet_loss", "cpu_load", "disk_io_stress", "latency", "memory_stress"]),
    ("Frequency of an attribute", """
    SELECT {attr_name} as f, count(*) as c
    FROM {table_name}
    GROUP BY {attr_name};
    """, ["crush", "packet_loss", "cpu_load", "disk_io_stress", "latency", "memory_stress"]),
    ("Max value of an attribute for every 5 minute window", """
    SELECT startTime / 3600 as f, MAX({int_attr_name}) as c
    FROM {table_name}
    Where serviceName = '{service_name}'
    GROUP BY startTime / 3600;
    """, ["crush", "packet_loss"]),
    ("Frequency of an attribute after filtering by another attribute", """
    SELECT {attr_name} as f, count(*) as c
    FROM {table_name}
    Where serviceName = '{service_name}'
    GROUP BY {attr_name};
    """, ["crush", "packet_loss"]),
]


def get_wasserstein_distance(syn, real):
    syn["c"] = syn["c"] / syn["c"].sum()
    real["c"] = real["c"] / real["c"].sum()
    all_features = set(syn["f"].values) | set(real["f"].values)
    syn = syn.set_index("f").reindex(all_features).fillna(0)
    real = real.set_index("f").reindex(all_features).fillna(0)
    return scipy.stats.wasserstein_distance(syn["c"], real["c"])


def iterate_template_parameters(query: str, parameters_values: OrderedDict[str, List[str]]):
    def inner(partial_dict: dict, parameters_to_add: List[str]):
        if not parameters_to_add:
            yield partial_dict
            return
        name = parameters_to_add[0]
        for value in parameters_values[name]:
            yield from inner(partial_dict | {name: value}, parameters_to_add[1:])

    query_parameters = {t[1] for t in Formatter().parse(query)}
    query_parameters = [p for p in parameters_values if p in query_parameters]
    return inner({}, query_parameters)


def run_templates(conn: sqlite3.Connection, parameters_values: OrderedDict[str, List[str]], table1: str, table2: str, queries=tuple(BENCHMARK_QUERIES)):
    avg_query = []
    for query_name, query, relevant_incidents in queries:
        avg_params = []
        for index, params in enumerate(iterate_template_parameters(query, parameters_values)):
            syn = pandas.read_sql_query(query.format(table_name=table1, **params), conn)
            real = pandas.read_sql_query(query.format(table_name=table2, **params), conn)
            if len(syn) == 0 or len(real) == 0:
                continue
            distance = get_wasserstein_distance(syn, real)
            if distance:
                avg_params.append(distance)
        if avg_params:
            avg_query.append(numpy.average(avg_params))
    if avg_query:
        print(f'"{table1}<->{table2}": {numpy.average(avg_query)},')
    return avg_query
