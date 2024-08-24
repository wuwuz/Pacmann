#go run private-search.go -n 50000000 -d 128 -m 32 -k 10 -q 1000 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs \
#                         -output ./private-search-result.txt -report ./private-search-report.txt -gnd ./SIFT-dataset/gnd/idx_50M.ivecs \
#                         -step 30 -parallel 3 -rtt 50

#mamba init
#mamba activate faiss

python cluster-search.py -n 1000000 -d 128 -k 10 -q 100 -input ./SIFT-dataset/bigann_base.bvecs -query ./SIFT-dataset/bigann_query.bvecs  \
                         -gnd ./SIFT-dataset/gnd/idx_1M.ivecs -report ./cluster-report.txt