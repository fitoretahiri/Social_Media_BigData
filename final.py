# import pandas as pd
import random
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from graphframes import GraphFrame

# Start Spark Session
spark = SparkSession.builder.appName("SocialNetworkAnalysis").master("local[*]").config("spark.executor.memory", "4g").config("spark.driver.memory", "4g").config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12").getOrCreate()

# Generate Users (Vertices)
num_users = 10
users = [(f"U{i}", f"User_{i}", random.randint(10, 1000)) for i in range(1, num_users + 1)]
vertices_schema = StructType([
    StructField("id", StringType(), False),
    StructField("name", StringType(), True),
    StructField("followers", IntegerType(), True)
])
vertices_df = spark.createDataFrame(users, schema=vertices_schema)

# Generate Connections (Edges) efficiently
def generate_edges(num_users, max_edges=5):
    edges = []
    for i in range(1, num_users + 1):
        targets = random.sample(range(1, num_users + 1), random.randint(1, max_edges))
        for target in targets:
            if target != i:
                weight = round(random.uniform(0.1, 1.0), 2)
                edges.append((f"U{i}", f"U{target}", weight))
    return edges

edges = generate_edges(num_users)
edges_schema = StructType([
    StructField("src", StringType(), False),
    StructField("dst", StringType(), False),
    StructField("weight", FloatType(), True)
])
edges_df = spark.createDataFrame(edges, schema=edges_schema)

# Create GraphFrame
graph = GraphFrame(vertices_df, edges_df)

# Answering the questions

# (a) Identifying influential users using PageRank
pagerank_results = graph.pageRank(resetProbability=0.15, tol=0.01)
pagerank_results.vertices.orderBy("pagerank", ascending=False).show(5)

# (b) Finding similar groups using Connected Components
graph.vertices.persist()
graph.edges.persist()
spark.sparkContext.setCheckpointDir("C:/Master-Projects/Social_Media_BigData/checkpoints")
components = graph.connectedComponents()
components.show(5)

# (c) Information propagation using BFS
propagation_paths = graph.bfs(fromExpr="id = 'U1'", toExpr="followers > 500", maxPathLength=2)
propagation_paths.show(5)

# (d) Impact of removing a key user
def remove_user(graph, user_id):
    new_vertices = graph.vertices.filter(f"id != '{user_id}'")
    new_edges = graph.edges.filter(f"src != '{user_id}' AND dst != '{user_id}'")
    return GraphFrame(new_vertices, new_edges)

graph_without_U1 = remove_user(graph, 'U1')
print("Graph size after removing U1:")
print("Vertices:", graph_without_U1.vertices.count())
print("Edges:", graph_without_U1.edges.count())

# (e) Evaluating influence with PageRank (already covered in (a))

# Display some data
graph.vertices.show(5)
graph.edges.show(5)