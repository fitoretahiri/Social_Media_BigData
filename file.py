from pyspark.sql import SparkSession
from graphframes import GraphFrame

spark = SparkSession.builder.appName("GraphFrames").master("local[*]")\
    .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12").getOrCreate()

print(f"Spark Version: {spark.version}")
print(f"Spark Master URL: {spark.sparkContext.master}")

vertices = spark.createDataFrame(
    [("1", "Alice"), ("2", "Bob"), ("3", "Charlie"), ("4", "David")],
    ["id", "name"]
) 

edges = spark.createDataFrame(
    [("1", "2", "friend"), ("2", "3", "friend"), ("3", "4", "friend")],
    ["src", "dst", "relationship"]
)

g = GraphFrame(vertices, edges)
edges = g.edges.filter("src != dst").dropDuplicates()

spark.sparkContext.setCheckpointDir("C:/Big Data/checkpoints")
result = g.connectedComponents(algorithm = "graphframes", checkpointInterval=10)
result.show()

# (a) Identifying influential users using PageRank
pagerank_results = g.pageRank(resetProbability=0.15, tol=0.01)
pagerank_results.vertices.orderBy("pagerank", ascending=False).show()

# (b) Finding similar groups using Connected Components
components = g.connectedComponents()
components.show()

# (c) Information propagation using BFS
propagation_paths = g.bfs(fromExpr="id = '1'", toExpr="id = '4'", maxPathLength=2)
propagation_paths.show()

# (d) Impact of removing a key user
def remove_user(graph, user_id):
    new_vertices = graph.vertices.filter(f"id != '{user_id}'")
    new_edges = graph.edges.filter(f"src != '{user_id}' AND dst != '{user_id}'")
    return GraphFrame(new_vertices, new_edges)

graph_without_1 = remove_user(g, '1')
print("Graph size after removing user 1:")
print("Vertices:", graph_without_1.vertices.count())
print("Edges:", graph_without_1.edges.count())

# (e) Evaluating influence with PageRank (already covered in (a))