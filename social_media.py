from pyspark.sql import SparkSession
from graphframes import GraphFrame
from pyspark.sql.functions import split, col

# Initialize Spark session (connecting to the local session)
spark = SparkSession.builder.appName("SocialNetworkAnalysis").master("local[*]").config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12").getOrCreate()

# Check Spark version and context
print(f"Spark Version: {spark.version}")
print(f"Spark Master URL: {spark.sparkContext.master}")

df = spark.read.text("facebook_sample4.txt.gz")

edges = df.withColumn("src", split(col("value"), " ")[0]).withColumn("dst", split(col("value"), " ")[1]).select("src", "dst")
edges = edges.filter("src != dst").dropDuplicates()
vertices = edges.select(col("src").alias("id")).union(edges.select(col("dst").alias("id"))).distinct()

graph = GraphFrame(vertices, edges)

# Apliko PageRank
pagerank = graph.pageRank(resetProbability=0.15, maxIter=10)
pagerank.vertices.orderBy(col("pagerank").desc()).show(10, truncate=False)

spark.sparkContext.setCheckpointDir("C:/Master-Projects/Social_Media_BigData/checkpoints")
connected_components = graph.connectedComponents(algorithm = "graphframes", checkpointInterval=10)
print('------------------------------------------------------------------------------------------------------------')
connected_components.show()
print('------------------------------------------------------------------------------------------------------------')
# dframe = connected_components.toPandas()
# print(df.head(10))

# it is failing here
# connected_components.write.format("csv").save("connected_components_parquet.csv")

# Apliko algoritmin e përhapjes së informacionit
label_propagation = graph.labelPropagation(maxIter=5)
label_propagation.orderBy("label").show(10, truncate=False)

# Remove user from graph
user_to_remove = "107"
updated_edges = edges.filter((col("src") != user_to_remove) & (col("dst") != user_to_remove))
updated_vertices = vertices.filter(col("id") != user_to_remove)
updated_graph = GraphFrame(updated_vertices, updated_edges)
print('------------------------------------------------------------------------------------------------------------')
# Compute connected components again
connected_components = updated_graph.connectedComponents()
connected_components.show()
# connected_components.write.format("csv").option("header", "true").save("C:/Master-Projects/Social_Media_BigData/updated_components_csv")
print('------------------------------------------------------------------------------------------------------------')
spark.stop()
# from pyspark.sql import SparkSession
# from graphframes import GraphFrame
# from pyspark.sql.functions import split, col

# # Initialize Spark session with memory optimizations
# spark = SparkSession.builder \
#     .appName("SocialNetworkAnalysis") \
#     .master("local[*]") \
#     .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.0-s_2.12") \
#     .config("spark.executor.memory", "4g") \
#     .config("spark.driver.memory", "2g") \
#     .config("spark.sql.shuffle.partitions", "200") \
#     .getOrCreate()

# # Check Spark version and context
# print(f"Spark Version: {spark.version}")
# print(f"Spark Master URL: {spark.sparkContext.master}")

# # Load data and process in chunks if needed
# df = spark.read.text("facebook_sample4.txt.gz")

# # Split the data into edges
# edges = df.withColumn("src", split(col("value"), " ")[0]).withColumn("dst", split(col("value"), " ")[1]).select("src", "dst")
# edges = edges.filter("src != dst").dropDuplicates()

# # Repartition data to reduce memory usage
# edges = edges.repartition(100)

# # Generate vertices
# vertices = edges.select(col("src").alias("id")).union(edges.select(col("dst").alias("id"))).distinct()

# # Create graph
# graph = GraphFrame(vertices, edges)

# # Apply PageRank algorithm
# pagerank = graph.pageRank(resetProbability=0.15, maxIter=10)
# pagerank.vertices.orderBy(col("pagerank").desc()).show(10, truncate=False)

# # Checkpoint directory
# spark.sparkContext.setCheckpointDir("C:/Master-Projects/Social_Media_BigData/checkpoints")
# connected_components = graph.connectedComponents(algorithm="graphframes", checkpointInterval=10)
# # Get the connected components as a DataFrame
# connected_components_df = connected_components.toPandas()

# # Save the DataFrame to CSV
# connected_components_df.write.option("header", "true").csv("C:\\Master-Projects\\Social_Media_BigData\\updated_components_csv")


# # Apply Label Propagation
# label_propagation = graph.labelPropagation(maxIter=5)
# label_propagation.orderBy("label").show(10, truncate=False)

# # Remove user from graph
# user_to_remove = "107"
# updated_edges = edges.filter((col("src") != user_to_remove) & (col("dst") != user_to_remove))
# updated_vertices = vertices.filter(col("id") != user_to_remove)
# updated_graph = GraphFrame(updated_vertices, updated_edges)

# # Compute connected components again
# connected_components = updated_graph.connectedComponents()
# connected_components.write.format("csv").option("header", "true").save("C:\\Master-Projects\\Social_Media_BigData\\updated_components_csv")
# print('PRINTINGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')
# # Stop Spark session
# spark.stop()
