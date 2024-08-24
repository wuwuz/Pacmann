/*


Input: file with list of vectors
Output: 1) An HNSW index 2) A constant-degree index graph


Argument:
	-n number of vectors
	-d dimension of vectors
	-m number of neighbors in the graph
	-input file name of the list of vectors
	-output file name of the index graph
	--newgraph if true, create a new index graph
*/

package graphann

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"sync"
	"time"

	"github.com/evan176/hnswgo"
	"github.com/yahoojapan/gongt"
)

/*

var n int
var d int
var m int
var inputFile string
var outputFile string
var newGraph bool

func main() {
	// Parse arguments
	numVectors := flag.Int("n", 1000, "number of vectors")
	dimVectors := flag.Int("d", 128, "dimension of vectors")
	numNeighbors := flag.Int("m", 32, "number of neighbors in the graph")
	inputFileName := flag.String("input", "bigann_base.bvecs", "file name of the list of vectors")
	outputFileName := flag.String("output", "", "file name of the index graph")
	newGraphBool := flag.Bool("newgraph", false, "if true, create a new index graph")

	flag.Parse()

	n = *numVectors
	d = *dimVectors
	m = *numNeighbors
	inputFile = *inputFileName
	outputFile = *outputFileName
	newGraph = *newGraphBool
	dataset := inputFile[:len(inputFile)-len(filepath.Ext(inputFile))]
	dataset = dataset + "_" + strconv.Itoa(n) + "_" + strconv.Itoa(d) + "_" + strconv.Itoa(m)

	// Load vectors from file
	vectors, err := LoadBvecsFile(inputFile, n, d)
	if err != nil {
		fmt.Println(err)
		return
	}

	// First create a HNSW index
	// we first strip the file extension from input file name
	hnswIndexFileName := dataset + "_hnsw.bin"
	h := hnswgo.New(d, m, 300, 100, uint32(n), "l2")

	// only when the hnsw index file does not exist, we create a new hnsw index
	if _, err := os.Stat(hnswIndexFileName); os.IsNotExist(err) {
		for i := 0; i < n; i++ {
			h.AddPoint(vectors[i], uint32(i))
		}
		h.Save(hnswIndexFileName)
	} else {
		h = hnswgo.Load(hnswIndexFileName, d, "l2")
	}

	if newGraph {
		graph := CreateGraphBasedOnHNSW(vectors, h, m)
		if outputFile == "" {
			outputFile = dataset + "_graph.npy" // default output file name
		}
		,// save the graph to the disk with binary format
		SaveGraphToFile(outputFile, graph)
	}
}

*/

func BuildGraph(n int, dim int, m int, vectors [][]float32, savepath string, dataset string) [][]int {
	// First create a HNSW index
	// we first strip the file extension from input file name
	ngtFileName := savepath + "/" + dataset + ".ngt"
	fmt.Println("NGT index file name: ", ngtFileName)
	graph := CreateGraphBasedOnNGT(vectors, ngtFileName, m)
	EvaluateGraphQuality(vectors, graph)
	return graph
}

func L2Dist(v1, v2 []float32) float32 {
	dim := len(v1)
	remainder := dim & 7 // len(v1) % 8
	d := L2DistSIMD(v1[:dim-remainder], v2[:dim-remainder])
	for i := dim - remainder; i < dim; i++ {
		d += (v1[i] - v2[i]) * (v1[i] - v2[i])
	}
	return d
}

func L2DistanceSIMD(a, b *float32, dim int) float32

func L2DistSIMD(v1, v2 []float32) float32 {
	// this only works when the dimension is a multiple of 8
	return L2DistanceSIMD(&v1[0], &v2[0], len(v1))
}

func FindMedoid(vectors [][]float32) int {
	n := len(vectors)
	dim := len(vectors[0])
	sum := make([]float32, len(vectors[0]))
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			sum[j] += vectors[i][j]
		}
	}
	for j := 0; j < dim; j++ {
		sum[j] /= float32(n)
	}
	// find the closest vector to the sum
	minDist := float32(1e9)
	medoid := 0
	for i := 0; i < n; i++ {
		d := L2Dist(vectors[i], sum)
		if d < minDist {
			minDist = d
			medoid = i
		}
	}
	return medoid
}

type IdWithDist struct {
	id   int
	dist float32
}

// robust prune function
// for the vertex u, we prune the candidates to only m
// it's the same as the prune function in the diskann paper
func robustPrune(vectors [][]float32, u int, candidates []int, m int, alpha float32) []int {
	if len(candidates) <= m {
		return candidates
	}

	// first we compute the distance from u to all candidates
	dist2u := make([]IdWithDist, len(candidates))
	for i := 0; i < len(candidates); i++ {
		dist2u[i] = IdWithDist{
			id:   candidates[i],
			dist: L2Dist(vectors[u], vectors[candidates[i]]),
		}
	}

	sort.Slice(dist2u, func(i, j int) bool {
		return dist2u[i].dist < dist2u[j].dist
	})

	accept := make([]IdWithDist, 0)
	discarded := make([]IdWithDist, 0)
	for i := 0; i < len(dist2u); i++ {
		v := dist2u[i].id
		dist_uv := dist2u[i].dist
		ok := true
		// now we check the triangle condition:
		for j := 0; j < len(accept); j++ {
			if L2Dist(vectors[accept[j].id], vectors[v])*alpha < dist_uv {
				ok = false
				break
			}
		}
		if ok {
			accept = append(accept, dist2u[i])
			if len(accept) == m {
				break
			}
		} else {
			discarded = append(discarded, dist2u[i])
		}
	}

	// now let's see if I don't add the discarded ones, do we have enough candidates?

	// if we have not enough candidates, we add the discarded ones
	if len(accept) < m {
		for i := 0; i < len(discarded); i++ {
			accept = append(accept, discarded[i])
			if len(accept) == m {
				break
			}
		}

		// DEPRECATED: we don't need to sort the accept list
		// in this case we sort the accept list by distance to u again
		//sort.Slice(accept, func(i, j int) bool {
		//	return accept[i].dist < accept[j].dist
		//})
	}

	ret := make([]int, len(accept))
	for i := 0; i < len(accept); i++ {
		ret[i] = accept[i].id
		//if i > 0 && accept[i].dist < accept[i-1].dist {
		//	fmt.Println("Error in robust prune: the neighbors are not sorted by distance")
		//}
	}
	return ret
}

// this function is used when we have already m neighbors
func robustPruneWithOneExtra(vectors [][]float32, u int, neighbors []int, v int, m int, alpha float32) []int {

	// TODO: quality degrades??

	// first make sure that v is not already in the neighbors
	for i := 0; i < len(neighbors); i++ {
		if neighbors[i] == v {
			return neighbors
		}
	}

	dist_uv := L2Dist(vectors[u], vectors[v])

	// we compute the distance from u to all neighbors
	dist2u := make([]IdWithDist, len(neighbors))
	for i := 0; i < len(neighbors); i++ {
		dist2u[i] = IdWithDist{
			id:   neighbors[i],
			dist: L2Dist(vectors[u], vectors[neighbors[i]]),
		}

		if i > 0 && dist2u[i].dist < dist2u[i-1].dist {
			fmt.Println("Error in verifying: the neighbors are not sorted by distance to u")
		}
	}

	// now we check that if all the neighbors that are closer to u
	// will cause v to be pruned

	accept := make([]IdWithDist, 0)

	for i := 0; i < len(dist2u); i++ {
		if dist2u[i].dist >= dist_uv {
			break
		}

		accept = append(accept, dist2u[i])
		// in this case we don't need to add v
		if L2Dist(vectors[dist2u[i].id], vectors[v])*alpha < dist_uv {
			return neighbors
		}
	}

	alreadyAdded := len(accept)
	if alreadyAdded == m {
		// in this case we don't need to add v
		return neighbors
	}
	// so now we need to add v
	accept = append(accept, IdWithDist{id: v, dist: dist_uv})

	for i := alreadyAdded; i < len(dist2u) && len(accept) < m; i++ {
		if L2Dist(vectors[dist2u[i].id], vectors[v])*alpha < dist2u[i].dist {
			// in this case we don't need to add remain[i]
			continue
		}
		accept = append(accept, dist2u[i])
	}

	ret := make([]int, len(accept))
	for i := 0; i < len(accept); i++ {
		ret[i] = accept[i].id
		if i > 0 && accept[i].dist < accept[i-1].dist {
			fmt.Println("Error in adding one: the neighbors are not sorted by distance")
			fmt.Println("Trying to add ", v, " to ", u, " with neighbors ", neighbors)
			x, y := accept[i].id, accept[i-1].id
			fmt.Println("The distance between ", x, " and ", u, " is ", L2Dist(vectors[x], vectors[u]))
			fmt.Println("The distance between ", y, " and ", u, " is ", L2Dist(vectors[y], vectors[u]))
			panic("Error in adding one: the neighbors are not sorted by distance")
		}
	}

	return ret
}

func CreateGraphBasedOnNGT(vectors [][]float32, ngtFile string, m int) [][]int {

	n := len(vectors)
	dim := len(vectors[0])

	start := time.Now()

	var ngt *gongt.NGT
	if _, err := os.Stat(ngtFile); os.IsNotExist(err) {

		start := time.Now()
		ngt = gongt.New(ngtFile).SetObjectType(gongt.Float).SetDimension(dim).Open()

		for _, v := range vectors {
			// convert the vector to float64
			tmp := make([]float64, len(v))
			for i := 0; i < len(v); i++ {
				tmp[i] = float64(v[i])
			}
			ngt.Insert(tmp)
		}

		if err := ngt.CreateAndSaveIndex(runtime.NumCPU()); err != nil {
			fmt.Println("Error in creating ngt index: ", err)
		}

		end := time.Now()
		fmt.Printf("NGT index created, time = %v\n", end.Sub(start))

		// test the ngt index

		hit := 0
		for i := 0; i < 1000; i++ {
			tmp := make([]float64, len(vectors[i]))
			for j := 0; j < len(vectors[i]); j++ {
				tmp[j] = float64(vectors[i][j])
			}
			res, err := ngt.Search(tmp, 10, gongt.DefaultEpsilon)
			if err != nil {
				fmt.Println("Error in searching: ", err)
			}
			// verify that the search has found the vertex itself
			for j := 0; j < len(res); j++ {
				if int(res[j].ID-1) == i {
					hit += 1
					break
				}
			}
		}

		fmt.Print("Hit rate for NGT: ", float32(hit)/float32(1000), "\n")
	} else {
		ngt = gongt.New(ngtFile).Open()
	}
	defer ngt.Close()

	alpha := float32(1.2) // the alpha parameter in the robust prune function

	graph := make([][]int, n)

	//maxThread := runtime.NumCPU() - 1
	maxThread := 16
	fmt.Print("Number of threads: ", maxThread, "\n")

	// we now use multithread to build the graph
	// each thread will process n/maxThread vertices

	var wg sync.WaitGroup
	wg.Add(maxThread)

	perThreadVertices := (n + maxThread - 1) / maxThread

	for t := 0; t < maxThread; t++ {
		start := t * perThreadVertices
		end := min((t+1)*perThreadVertices, n)

		go func(start, end int) {
			for u := start; u < end; u++ {
				//  convert the vectors to float64
				tmp := make([]float64, len(vectors[u]))
				for j := 0; j < len(vectors[u]); j++ {
					tmp[j] = float64(vectors[u][j])
				}
				candidatesList, err := ngt.Search(tmp, int(float32(m)*1.5), gongt.DefaultEpsilon)
				if err != nil {
					fmt.Println("Error in searching: ", err)
				}

				// we cast the candidates to int
				candidates := make([]int, 0)
				for j := 0; j < len(candidatesList); j++ {
					v := int(candidatesList[j].ID) - 1 // very important to -1
					if v != u && v < n {
						candidates = append(candidates, v)
					}
				}

				// also, there may already some connected vertices
				//candidates = append(candidates, graph[u]...)

				// we prune the neighbors to m
				candidates = robustPrune(vectors, u, candidates, m, alpha)
				graph[u] = candidates
			}

			wg.Done()
		}(start, end)
	}

	wg.Wait()

	fmt.Printf("First pass done\n")

	// we now add the bi-directional edges
	biGraph := make([][]int, n)
	for u := 0; u < n; u++ {
		biGraph[u] = make([]int, 0)
	}
	for u := 0; u < n; u++ {
		for _, v := range graph[u] {
			biGraph[u] = append(biGraph[u], v)
			biGraph[v] = append(biGraph[v], u)
		}
	}

	// do a count of inbound degree
	inbounds := make([]int, n)
	for i := 0; i < n; i++ {
		inbounds[i] = len(biGraph[i])
	}

	// now we enumerate all edges (u -> v), and sample the edge with prob. (1.5*m)/inbounds[v]
	// again we use multithread to do this

	var wg2 sync.WaitGroup
	wg2.Add(maxThread)

	for t := 0; t < maxThread; t++ {
		start := t * perThreadVertices
		end := min((t+1)*perThreadVertices, n)

		go func(start, end int) {
			r := rand.New(rand.NewSource(int64(start)))
			for u := start; u < end; u++ {
				connection := make([]int, 0)
				for _, v := range biGraph[u] {
					prob := math.Min(float64(1.5*float64(m))/float64(inbounds[v]), 1.0)
					if r.Float64() < prob {
						connection = append(connection, v)
					}
				}

				if len(connection) > m {
					connection = robustPrune(vectors, u, connection, m, alpha)
				}

				// we fill the connection by random neighbors
				for len(connection) < m {
					// we add a random vertex to the outbounds
					// make sure it's not i and not already in the outbounds
					v := r.Intn(n)
					if v == u {
						continue
					}
					ok := true
					for _, vv := range connection {
						if vv == v {
							ok = false
							break
						}
					}
					if ok {
						connection = append(connection, v)
					}
				}

				graph[u] = connection
			}

			wg2.Done()
		}(start, end)
	}

	wg2.Wait()

	inbounds = make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < len(graph[i]); j++ {
			inbounds[graph[i][j]]++
		}
	}

	// now we check the min and the max inbounds
	minInbound := n
	maxInbound := 0
	for i := 0; i < n; i++ {
		if inbounds[i] < minInbound {
			minInbound = inbounds[i]
		}
		if inbounds[i] > maxInbound {
			maxInbound = inbounds[i]
		}
	}

	fmt.Printf("Min inbound: %d, Max inbound: %d\n", minInbound, maxInbound)

	end := time.Now()
	fmt.Println("Graph built, time = ", end.Sub(start))

	return graph
}

func CreateGraphBasedOnHNSW(vectors [][]float32, hnsw *hnswgo.HNSW, m int) [][]int {

	start := time.Now()

	n := len(vectors)
	alpha := float32(1.2) // the alpha parameter in the robust prune function

	// we enumerate all vertices in a random order
	perm := rand.Perm(n)

	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]int, 0)
	}

	for i := 0; i < n; i++ {

		if i%10000 == 0 {
			fmt.Printf("Processing %d-th vertex\n", i)
		}

		//if i%1000 == 0 {
		//	fmt.Printf("Processing %d-th vertex\n", i)
		//}
		u := perm[i]

		// we first find the m nearest neighbors of v
		candidatesList, _ := hnsw.SearchKNN(vectors[u], 2*m)
		// we cast the candidates to int
		candidates := make([]int, 0)
		for j := 0; j < len(candidatesList); j++ {
			v := int(candidatesList[j])
			if v != u && v < n {
				candidates = append(candidates, v)
			}
		}

		// also, there may already some connected vertices
		//candidates = append(candidates, graph[u]...)

		// we prune the neighbors to m
		candidates = robustPrune(vectors, u, candidates, m, alpha)
		graph[u] = append(graph[u], candidates...)
		//graph[u] = candidates

		// we intend to add u to the outbound of all its neighbors
		for j := 0; j < len(candidates); j++ {
			v := candidates[j]
			graph[v] = append(graph[v], u)
			//graph[v] = robustPrune(vectors, v, graph[v], m, alpha)
		}
	}

	// do a count of inbounds, simultaneously remove duplicates
	inbounds := make([]int, n)
	for i := 0; i < n; i++ {
		// remove duplicates
		seen := make(map[int]bool)
		j := 0
		for _, v := range graph[i] {
			if _, ok := seen[v]; !ok {
				seen[v] = true
				graph[i][j] = v
				j++
			}
		}
		graph[i] = graph[i][:j]

		for j := 0; j < len(graph[i]); j++ {
			inbounds[graph[i][j]]++
		}
	}

	// now we enumerate all edges (u -> v), and sample the edge with prob. (1.5*m)/inbounds[v]
	for i := 0; i < n; i++ {
		u := i
		keep := make([]int, 0)
		for j := 0; j < len(graph[i]); j++ {
			v := graph[i][j]
			prob := math.Min(float64(1.5*float64(m))/float64(inbounds[v]), 1.0)
			if rand.Float64() < prob {
				keep = append(keep, v)
			}
		}

		if len(keep) > m {
			keep = robustPrune(vectors, u, keep, m, alpha)
		}

		graph[u] = keep
	}

	// now we make sure that all vertices have exactly m outbounds
	// otherwise we add enough outbounds to make it m
	for i := 0; i < n; i++ {
		for len(graph[i]) < m {
			// we add a random vertex to the outbounds
			// make sure it's not i and not already in the outbounds
			v := rand.Intn(n)
			if v == i {
				continue
			}
			ok := true
			for j := 0; j < len(graph[i]); j++ {
				if graph[i][j] == v {
					ok = false
				}
			}
			if ok {
				graph[i] = append(graph[i], v)
			}
		}
	}

	inbounds = make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < len(graph[i]); j++ {
			inbounds[graph[i][j]]++
		}
	}

	// now we check the min and the max inbounds
	minInbound := n
	maxInbound := 0
	for i := 0; i < n; i++ {
		if inbounds[i] < minInbound {
			minInbound = inbounds[i]
		}
		if inbounds[i] > maxInbound {
			maxInbound = inbounds[i]
		}
	}

	fmt.Printf("Min inbound: %d, Max inbound: %d\n", minInbound, maxInbound)

	end := time.Now()
	fmt.Println("Graph built, time = ", end.Sub(start))

	return graph
}

/*

// the old one
func CreateGraphBasedOnHNSW(vectors [][]float32, hnsw *hnswgo.HNSW, m int) [][]int {

	start := time.Now()

	n := len(vectors)
	alpha := float32(1.2) // the alpha parameter in the robust prune function

	// we enumerate all vertices in a random order
	perm := rand.Perm(n)

	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]int, 0)
	}

	for i := 0; i < n; i++ {

		if i%10000 == 0 {
			fmt.Printf("Processing %d-th vertex\n", i)
		}

		//if i%1000 == 0 {
		//	fmt.Printf("Processing %d-th vertex\n", i)
		//}
		u := perm[i]

		// we first find the m nearest neighbors of v
		candidatesList, _ := hnsw.SearchKNN(vectors[u], 2*m)
		// we cast the candidates to int
		candidates := make([]int, 0)
		for j := 0; j < len(candidatesList); j++ {
			v := int(candidatesList[j])
			if v != u && v < n {
				candidates = append(candidates, v)
			}
		}

		// also, there may already some connected vertices
		//candidates = append(candidates, graph[u]...)

		// we prune the neighbors to m
		candidates = robustPrune(vectors, u, candidates, m, alpha)
		graph[u] = append(graph[u], candidates...)
		//graph[u] = candidates

		for j := 0; j < len(candidates); j++ {
			graph[candidates[j]] = append(graph[candidates[j]], u)
		}


		// we intend to add u to the outbound of all its neighbors
		for j := 0; j < len(candidates); j++ {
			v := candidates[j]
			graph[v] = robustPruneWithOneExtra(vectors, v, graph[v], u, m, alpha)
		}
	}

	// now we make sure that all vertices have exactly m outbounds
	// otherwise we add enough outbounds to make it m
	for i := 0; i < n; i++ {
		for len(graph[i]) < m {
			// we add a random vertex to the outbounds
			// make sure it's not i and not already in the outbounds
			v := rand.Intn(n)
			if v == i {
				continue
			}
			ok := true
			for j := 0; j < len(graph[i]); j++ {
				if graph[i][j] == v {
					ok = false
				}
			}
			if ok {
				graph[i] = append(graph[i], v)
			}
		}
	}

	inbounds := make([]int, n)
	for i := 0; i < n; i++ {
		for j := 0; j < len(graph[i]); j++ {
			inbounds[graph[i][j]]++
		}
	}

	// now we check the min and the max inbounds
	minInbound := n
	maxInbound := 0
	for i := 0; i < n; i++ {
		if inbounds[i] < minInbound {
			minInbound = inbounds[i]
		}
		if inbounds[i] > maxInbound {
			maxInbound = inbounds[i]
		}
	}

	fmt.Printf("Min inbound: %d, Max inbound: %d\n", minInbound, maxInbound)

	end := time.Now()
	fmt.Println("Graph built, time = ", end.Sub(start))

	return graph
}
*/

func EvaluateGraphQuality(vectors [][]float32, graph [][]int) {
	n := len(vectors)
	dim := len(vectors[0])
	m := len(graph[0])

	g := BasicGraphInfo{
		N:       n,
		Dim:     dim,
		M:       m,
		Graph:   graph,
		Vectors: vectors,
	}

	frontend := GraphANNFrontend{
		Graph: &g,
	}

	frontend.Preprocess()

	// we evaluate the quality of the graph by
	// do search query for random vertices in the graph
	// and report the average steps to reach the target

	numQueries := 100
	hit := 0
	avgSteps := 0.0

	for i := 0; i < numQueries; i++ {
		target := rand.Intn(n)
		//fmt.Println("Query ", i, " target: ", target)
		knn, steps := frontend.SearchKNN(vectors[target], 20, 20, 2, false)
		if knn[0] == target {
			hit++
			avgSteps += float64(steps[0])
		}

		//fmt.Printf("Query %d: steps = %d\n", i, steps)
	}

	avgSteps /= float64(hit)
	fmt.Print("Hit rate: ", float64(hit)/float64(numQueries), " Average steps: ", avgSteps, "\n")
}

// compute the recall for a batch of queries @ k

func ComputeRecall(gnd [][]int, response [][]int, k int) float32 {

	numQueries := len(response)
	if len(gnd) < numQueries {
		fmt.Println("The number of queries in the response is larger than the number of queries in the ground truth")
	}

	recall := float32(0)

	for i := 0; i < numQueries; i++ {
		if len(response[i]) < k {
			fmt.Println("The number of neighbors in the response is less than k")
		}
		hit := 0
		for j := 0; j < k; j++ {
			// first check if this is a repeated answer. If so, we ignore it
			rept := false
			for l := 0; l < j; l++ {
				if response[i][j] == response[i][l] {
					rept = true
					break
				}
			}
			if rept {
				continue
			}

			// now we check if this is a hit
			for l := 0; l < k; l++ {
				if response[i][j] == gnd[i][l] {
					hit++
					break
				}
			}
		}
		//  we are just computing recall@k by using the top10 ground truths as the relevant set
		recall += float32(hit) / float32(k)
	}

	recall /= float32(numQueries)

	return recall
}
