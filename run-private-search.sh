# -n 1000000: The number of data points in the dataset.
# -d 128: The dimensionality of the data points.
# -m 32: The number of clusters in the hierarchical k-means tree.
# -k 10: The number of nearest neighbors to search for.
# -q 1000: The number of queries to run.
# -input ./SIFT-dataset/bigann_base.bvecs: The path to the dataset file.
# -query ./SIFT-dataset/bigann_query.bvecs: The path to the query file.
# -output ./private-search-result.txt: The path to the output file where the search results will be saved.
# -report ./private-search-report.txt: The path to the report file where the search performance metrics will be saved.
# -gnd ./SIFT-dataset/gnd/idx_1M.ivecs: The path to the ground truth file. Change "1M" to other values if needed.
# -step 20: The maximum steps in the private graph traverse algorithm.
# -parallel 3: The number of parallel vertices to be explored in one step.
# -rtt 50: The round-trip time (RTT) in milliseconds between the client and the server.


go run private-search.go -n 1000000 -d 128 -m 32 -k 10 -q 100 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
                         -output ./private-search-result.txt -report ./private-search-report.txt -gnd ./SIFT-dataset/gnd/idx_1M.ivecs \
                         -step 20 -parallel 3 -rtt 50


#go run private-search.go -n 100000000 -d 128 -m 32 -k 10 -q 100 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
#                         -output ./private-search-result.txt -gnd ./SIFT-dataset/gnd/idx_100M.ivecs \
#                         -step 32 -parallel 4 -rtt 50 


#go run private-search.go -n 3201821 -d 192 -m 32 -k 100 -q 1000 -input ./msmarco-dataset/msmarco_embeddings.npy -query ./msmarco-dataset/msmarco_queries.npy \
#                         -step 20 -parallel 3 -rtt 50  \
#                         -output ./msmarco-dataset/msmarco_embeddings_3201821_192_32_output.txt -report ./msmarco-dataset/msmarco_embeddings_3201821_192_32_report.txt

#python testing_quality.py -input ./msmarco-dataset/msmarco_embeddings_3201821_192_32_output.txt  -output ./msmarco-dataset/msmarco_embeddings_3201821_192_32_output_docid.txt \
#                          -report ./msmarco-dataset/msmarco_embeddings_3201821_192_32_report.txt
