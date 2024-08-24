import numpy as np
import faiss
import argparse
import os
import time


def load_bvecs_data(filename, n):
    """
    Taken from https://github.com/milvus-io/bootcamp/blob/5c1b1d414b9a1918a26c05d8ead1f3aeb8c318fc/benchmark_test/scripts/load.py#L39
    """
    x = np.memmap(filename, dtype='uint8', mode='r')
    d = x[:4].view('int32')[0]
    data = x.reshape(-1, d + 4)[:n, 4:] # is this correct?
    # force the datatype to be float32
    data = data.astype(np.float32).copy() # make it contiguous
    data = (data + 0.5) / 255
    return data

def load_fvecs_file(filename, n):
    """
    from https://gist.github.com/danoneata/49a807f47656fedbb389
    """

    fv = np.fromfile(filename, dtype=np.float32)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)[:n, 1:]
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv.copy() # make it contiguous
    return fv

def load_ivecs_file(filename, n):
    """
    from https://gist.github.com/danoneata/49a807f47656fedbb389
    """

    iv = np.fromfile(filename, dtype=np.uint32)
    if iv.size == 0:
        return np.zeros((0, 0))
    dim = iv.view(np.int32)[0]
    assert dim > 0
    iv = iv.reshape(-1, 1 + dim)
    if not all(iv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    iv = iv[:n, 1:].copy() # make it contiguous
    return iv

def load_vectors(filename, n, dim): 
    # load the vectors from the file

    # Step 1: identify the extensions
    ext = filename.split('.')[-1]

    print("extension", ext)
    
    if ext == 'npy':
        # Load the vectors from a numpy file
        vectors = np.load(filename) 
        if vectors.shape[0] < n:
            raise ValueError("The number of vectors in the file is less than the requested number") 
        vectors = vectors[:n].astype(np.float32)
    elif ext == 'txt':
        # Load the vectors from a text file
        vectors = np.zeros((n, dim), dtype=np.float32)
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == n:
                    break
                vectors[i] = np.array([float(x) for x in line.split()])
    elif ext == 'fvecs':
        vectors = load_fvecs_file(filename, n)
    elif ext == 'bvecs':
        vectors = load_bvecs_data(filename, n)
    elif ext == 'ivecs':
        vectors = load_ivecs_file(filename, n) # in this case the return vectors is a 2d uint32 tensor
    else: 
        raise ValueError("Unknown file extension")
    
    return vectors

# we use faiss to perform clustering
def clustering(vectors, n_clusters):
    
    print("Clustering", vectors.shape[0], "vectors into", n_clusters, "clusters")

    start = time.time()

    # verify the vector is of type float32
    if vectors.dtype != np.float32:
        raise ValueError("The vectors must be of type float32")
    kmeans = faiss.Kmeans(vectors.shape[1], int(n_clusters), niter=20, verbose=True)
    kmeans.train(vectors)

    centroids = kmeans.centroids
    # verify that the centroids are of type float32
    if centroids.dtype != np.float32:
        raise ValueError("The centroids must be of type float32")
    _, I = kmeans.index.search(vectors, 1) # the I contains the closest centroid for each vector
    I = I.flatten()
    end = time.time()
    print("Clustering time:", end - start)

    return centroids, I


class cluster_search_index:
    def __init__(self, n, dim):
        self.n = n
        self.dim = dim
        self.n_clusters = np.sqrt(n).astype(int)
        self.centroids = None
        self.I = None
    
    def build_index(self, vectors):
        if self.centroids is None:
            self.centroids, self.I = clustering(vectors, self.n_clusters)

        self.sorted_indices = np.argsort(self.I) # are the labels from 0-n_clusters-1?
        self.sorted_labels = self.I[self.sorted_indices] # the labels of the sorted indices
        self.sorted_vectors = vectors[self.sorted_indices].copy() # make it contiguous

        size_of_each_cluster = np.bincount(self.sorted_labels)
        # print the min and the max of the size of each cluster
        print("Min size of the cluster", np.min(size_of_each_cluster))
        print("Max size of the cluster", np.max(size_of_each_cluster))
        self.offset_of_each_cluster = np.cumsum(size_of_each_cluster)
        self.offset_of_each_cluster = np.concatenate(([0], self.offset_of_each_cluster))
        # now the vectors in the i-th cluster are from offset_of_each_cluster[i] to offset_of_each_cluster[i+1]


        # only for testing
        #self.faiss_index = faiss.IndexFlatL2(int(self.dim))

        # verify the type of the vectors is float32
        #if vectors.dtype != np.float32:
        #    raise ValueError("The vectors must be of type float32")
        #self.faiss_index.add(vectors)

    def save_to_file(self, savepath, dataset):
        # we only need to save the centroids and the labels
        print("Saving the index to the file", savepath + "/" + dataset + '-centroids.npy', savepath + "/" + dataset + '-labels.npy')
        # verify the centroids and labels
        if self.centroids.shape[1] != self.dim or self.centroids.shape[0] != self.n_clusters:
            raise ValueError("The centroids have the wrong shape")
        if self.I.shape[0] != self.n:
            raise ValueError("The labels have the wrong shape")

        np.save(savepath + "/" + dataset + '-centroids.npy', self.centroids)
        np.save(savepath + "/" + dataset + '-labels.npy', self.I)
    
    def load_from_file(self, savepath, dataset):
        # we first need to check if the files exist
        if not os.path.exists(savepath + "/" + dataset + '-centroids.npy') or not os.path.exists(savepath + "/" + dataset + '-labels.npy'):
            raise ValueError("The files do not exist")

        print("Loading the index from the file", savepath + "/" + dataset + '-centroids.npy', savepath + "/" + dataset + '-labels.npy')
        self.centroids = np.load(savepath + "/" + dataset + '-centroids.npy')
        self.I = np.load(savepath + "/" + dataset + '-labels.npy')

        # verify the centroids and labels
        if self.centroids.shape[1] != self.dim or self.centroids.shape[0] != self.n_clusters:
            raise ValueError("The centroids have the wrong shape")
        if self.I.shape[0] != self.n:
            raise ValueError("The labels have the wrong shape")
    
    def search(self, query, k):
        
        # step 1: find the closest centroid
        dist_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        cluster_id = np.argmin(dist_to_centroids)
        
        # step 2: find the k-nearest neighbors in the cluster
        cluster_start = self.offset_of_each_cluster[cluster_id]
        cluster_end = self.offset_of_each_cluster[cluster_id + 1]
        cluster_vectors = self.sorted_vectors[cluster_start:cluster_end]
        distance_to_vectors = np.linalg.norm(cluster_vectors - query, axis=1)
        top_k_offset = np.argsort(distance_to_vectors)
        if len(top_k_offset) < k:
            # append zeros to the top_k_offset
            top_k_offset = np.concatenate((top_k_offset, np.zeros(k - len(top_k_offset), dtype=np.int32)))
        else: 
            top_k_offset = top_k_offset[:k]
        #print("size of the cluster", cluster_vectors.shape[0])
        #print("top_k_offset", top_k_offset)
        top_k_idx = self.sorted_indices[cluster_start + top_k_offset]

        # verify that the distance is sorted in ascending order
        # may not be necessary
        #for i in range(1, k):
        #    if np.linalg.norm(cluster_vectors[top_k_offset[i]] - query) < np.linalg.norm(cluster_vectors[top_k_offset[i-1]] - query):
        #        print("Error: the distances are not sorted in ascending order", i)
        #        break
        
        return top_k_idx

    def brute_force_search(self, query, k):
        D, I = self.faiss_index.search(query.reshape(1, -1), k)
        I = I.flatten()
        return I



def calculate_recall(answers, gnd, k):
    # answers and gnd are both 2d np.array
    n = answers.shape[0]
    answers_focus = answers[:, :k]
    gnd_focus = gnd[:n, :k]
    recall = np.zeros(n)
    for i in range(n):
        # we need to find the number of common elements between the two arrays
        # we can use the numpy intersect1d function
        recall[i] = len(np.intersect1d(answers_focus[i], gnd_focus[i], assume_unique=True)) / k
    return np.mean(recall)




# arguments:
# -n: number of vectors
# -dim: dimension of the vectors
# -k: number of nearest neighbors
# -q: number of queries
# -input: the input vector file
# -query: the query vector file
# -output: the output file
# -gnd: the ground truth file
# -brute: whether to use brute force search



parser = argparse.ArgumentParser(description="tiptoe-style cluster search")
parser.add_argument("-n", type=int, help="number of vectors")
parser.add_argument("-dim", type=int, help="dimension of the vectors")
parser.add_argument("-k", type=int, help="number of nearest neighbors")
parser.add_argument("-q", type=int, help="number of queries")
parser.add_argument("-input", type=str, help="input vector file")
parser.add_argument("-query", type=str, help="query vector file")
parser.add_argument("-output", type=str, help="output file")
parser.add_argument("-report", type=str, help="report file")
parser.add_argument("-gnd", type=str, help="ground truth file")

args = parser.parse_args()

n = args.n
dim = args.dim
q = args.q
input_file = args.input
query_file = args.query
output_file = args.output
gnd_file = args.gnd
report_file = args.report
savepath = os.path.dirname(input_file)
dataset = os.path.basename(input_file) 
dataset = dataset.split('.')[0]
dataset = dataset + "_" + str(n) + "_" + str(dim)

print("savepath", savepath)
print("dataset", dataset)

if input_file is None or query_file is None:
    raise ValueError("The input and query files must be provided")


print("Loading the vectors from the input file", input_file)
vectors = load_vectors(input_file, n, dim)
# print the type of the vectors
#print("vectors type", type(vectors[0][0]))

index = cluster_search_index(n, dim)

# now we try to load the index from the file
# if the file does not exist, we build the index

try:
    print("Trying to load the index from the file")
    index.load_from_file(savepath, dataset)
    index.build_index(vectors)
except ValueError:
    print("The index does not exist, building the index")
    index.build_index(vectors)
    index.save_to_file(savepath, dataset)

# now we load the query vectors

print("Loading the query vectors from the query file", query_file)
queries = load_vectors(query_file, q, dim)
queries = queries[:q]
print("queries shape", queries.shape)

# now we perform the search
print("Performing the search")
start = time.time()
answers = np.zeros((queries.shape[0], args.k), dtype=np.int32)
for i in range(queries.shape[0]):
    answers[i] = index.search(queries[i], args.k)
    #answers[i] = index.brute_force_search(queries[i], args.k)
end = time.time()
print("Search time:", end - start)
print("Average search time:", (end - start) / queries.shape[0]) 

# we now write the answers to the output file in text format
if output_file is None:
    output_file = savepath + "/" + dataset + "-cluster-output.txt"

with open(output_file, 'w') as f:
    for i in range(queries.shape[0]):
        for j in range(args.k):
            f.write(str(answers[i, j]) + " ")
        f.write("\n")
    
# if the ground truth file is provided, we calculate the recall

recall = 0.0
if gnd_file is not None:
    gnd = load_vectors(gnd_file, n, args.k)
    recall = calculate_recall(answers, gnd, args.k)
    print("Recall:", recall)
else:
    print("No ground truth file provided")


if report_file is None:
    # we use the default report file name
    report_file = savepath + "/" + dataset + "-cluster-report.txt"

if report_file is not None:
    print("Writing the report to the file", report_file)
    with open(report_file, 'w') as f:
        f.write("Search time: " + str(end - start) + "\n")
        f.write("Average search time: " + str((end - start) / queries.shape[0]) + "\n")
        f.write("Recall: " + str(recall) + "\n")


