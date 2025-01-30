from pyspark.sql import SparkSession
from graphframes import GraphFrame

# Initialize Spark session (connecting to the local session)
spark = SparkSession.builder.appName("GraphFrames").master("local[*]").config("spark.jars", "C:/Big Data/graphframes/graphframes-0.8.2-spark3.0-s_2.12.jar").getOrCreate()

# Check Spark version and context
print(f"Spark Version: {spark.version}")
print(f"Spark Master URL: {spark.sparkContext.master}")

# Sample data for vertices and edges
vertices = spark.createDataFrame(
    [("1", "Alice"), ("2", "Bob"), ("3", "Charlie"), ("4", "David")],
    ["id", "name"]
)

edges = spark.createDataFrame(
    [("1", "2", "friend"), ("2", "3", "friend"), ("3", "4", "friend")],
    ["src", "dst", "relationship"]
)

# Create a GraphFrame
g = GraphFrame(vertices, edges)

spark.sparkContext.setCheckpointDir("C:/Big Data/checkpoints")
# Run the connectedComponents algorithm without using checkpoints
result = g.connectedComponents()

# Show the result
result.show()

# Stop the Spark session when done
spark.stop()
