package graphann

import (
	"container/heap"
	"fmt"
	"math"
	"math/rand"
	"sort"
)

// define a struct that represents a vertex in a graph
type Vertex struct {
	Id        int
	Neighbors []int
	Vector    []float32
}

// define an interface that provides GeVertexInfo and GetStartVertex methods

type GetGraphInfo interface {
	Preprocess()
	GetMetadata() (int, int, int)          // n, dim, m
	GetVertexInfo([]int) ([]Vertex, error) // given a list of vertex ids, return the corresponding vertices
	GetStartVertex() ([]Vertex, error)     // return the start vertices (could be more than one)
}

// define a basic graph info struct that implements the GetGraphInfo interface

type BasicGraphInfo struct {
	N       int
	Dim     int
	M       int
	Graph   [][]int
	Vectors [][]float32
}

func (g *BasicGraphInfo) Preprocess() {}

func (g *BasicGraphInfo) GetMetadata() (int, int, int) {
	return g.N, g.Dim, g.M
}

func (g *BasicGraphInfo) GetVertexInfo(ids []int) ([]Vertex, error) {
	vertices := make([]Vertex, len(ids))
	for i, id := range ids {
		vertices[i] = Vertex{Id: id, Neighbors: g.Graph[id], Vector: g.Vectors[id]}
	}
	return vertices, nil
}

func (g *BasicGraphInfo) GetStartVertex() ([]Vertex, error) {
	n, _, _ := g.GetMetadata()

	targetNum := int(math.Sqrt(float64(n)))
	batch := make([]int, targetNum)
	for i := 0; i < targetNum; i++ {
		batch[i] = i
	}

	v, err := g.GetVertexInfo(batch)
	if err != nil {
		return nil, err
	}
	return v, nil
}

// define a struct that represents a search frontend

type GraphANNFrontend struct {
	Graph         GetGraphInfo
	StartVertices []Vertex
}

func (f *GraphANNFrontend) Preprocess() {
	f.Graph.Preprocess()
	v, err := f.Graph.GetStartVertex()
	if err != nil {
		panic(err)
	}
	f.StartVertices = v
}

func (f *GraphANNFrontend) GetMetadata() (int, int, int) {
	return f.Graph.GetMetadata()
}

type VertexWithDist struct {
	dist   float32
	vertex Vertex
}

type exploreQueue []*VertexWithDist

func (pq exploreQueue) Len() int { return len(pq) }
func (pq exploreQueue) Less(i, j int) bool {
	return pq[i].dist < pq[j].dist
}
func (pq exploreQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}
func (pq *exploreQueue) Push(x interface{}) {
	item := x.(*VertexWithDist)
	*pq = append(*pq, item)
}
func (pq *exploreQueue) Pop() interface{} {
	old := *pq
	n := len(old)
	item := old[n-1]
	*pq = old[0 : n-1]
	return item
}

// return the found k nearest neighbors and the step to reach them
func (g GraphANNFrontend) SearchKNN(queryVector []float32, k int, maxStep int, parallel int, benchmarking bool) ([]int, []int) {
	n, _, m := g.GetMetadata()

	reachStep := map[int]int{}
	knownVertices := map[int]Vertex{}
	// define a priority queue
	// the priority queue is a min heap, ranked by the distance to the query vector
	// each element is a tuple (*potential distance, vertex index)
	// we first push the first parallel * m vertices into the heap
	toBeExploredVertices := make(exploreQueue, 0)
	heap.Init(&toBeExploredVertices)

	//fmt.Println("Start vertices: ", g.startVertices)

	// we first find the top parallel vertices from fastStartVertices by their distance to the query vector
	if !benchmarking {
		fastStartQueue := make(exploreQueue, 0)
		for _, v := range g.StartVertices {
			dist := L2Dist(v.Vector, queryVector)
			fastStartQueue.Push(&VertexWithDist{dist: dist, vertex: v})
		}
		sort.Sort(fastStartQueue)
		for i := 0; len(toBeExploredVertices) < parallel && i < len(fastStartQueue); i++ {
			v := fastStartQueue[i]
			id := v.vertex.Id
			if _, ok := knownVertices[id]; ok {
				// we have already known this vertex
				continue
			}
			knownVertices[id] = v.vertex
			heap.Push(&toBeExploredVertices, v)
			reachStep[id] = 0
			//toBeExploredItems = append(toBeExploredItems, v)
		}
	}

	for step := 0; step < maxStep; step++ {

		// each time we issue parallel batches, each exploring one vertex's neighbors
		batchQ := make([]int, 0, m)
		for rept := 0; rept < parallel; rept++ {
			if len(toBeExploredVertices) == 0 || benchmarking {
				// in this case we simply make random queries
				for i := 0; i < m; i++ {
					batchQ = append(batchQ, rand.Intn(n))
				}
			} else {
				item := heap.Pop(&toBeExploredVertices).(*VertexWithDist)
				v := item.vertex.Id
				//log.Print("Exploring vertex ", v, " at step ", step, " with distance ", item.dist)
				// copy the neighbors of v to the batchQ
				batchQ = append(batchQ, knownVertices[v].Neighbors...)
			}

			//log.Printf("Exploring %d vertices at step %d", len(batchQ), step)
			//log.Printf("The first 5 vertices are %v", batchQ[:5])

		}

		//fmt.Println("Querying vertices, batch = ", batchQ[:5])
		queryResults, err := g.Graph.GetVertexInfo(batchQ)
		//fmt.Println("Querying vertices done")

		if err != nil {
			fmt.Printf("Error when querying vertices: %v\n", err)
			panic(err)
		}

		if benchmarking {
			// if we are just benchmarking, we don't care about the return
			continue
		}

		for _, v := range queryResults {
			if _, ok := knownVertices[v.Id]; ok {
				// we have already known this vertex
				continue
			}
			// if the neighbor list is all zeroes, we skip this vertex
			ok := false
			for _, neighbor := range v.Neighbors {
				if neighbor != 0 {
					ok = true
					break
				}
			}
			if ok {
				knownVertices[v.Id] = v
				reachStep[v.Id] = step
				// calculate the distance to the query vector
				dist := L2Dist(v.Vector, queryVector)
				heap.Push(&toBeExploredVertices, &VertexWithDist{dist: dist, vertex: v})
			}
		}
	}

	// extract all known vertices and sort them by distance by ascending order
	allKnownVertices := make([]VertexWithDist, 0, len(knownVertices))
	for _, v := range knownVertices {
		allKnownVertices = append(allKnownVertices,
			VertexWithDist{
				dist:   L2Dist(v.Vector, queryVector),
				vertex: v,
			})
	}
	sort.Slice(allKnownVertices, func(i, j int) bool {
		return allKnownVertices[i].dist < allKnownVertices[j].dist
	})
	ret := make([]int, k)
	stepRet := make([]int, k)
	for i := 0; i < k; i++ {
		if i >= len(allKnownVertices) {
			ret[i] = -1
			stepRet[i] = -1
		} else {
			ret[i] = allKnownVertices[i].vertex.Id
			stepRet[i] = reachStep[ret[i]]
		}
	}
	return ret, stepRet
}

func (g *GraphANNFrontend) SearchKNNBatch(queryVectors [][]float32, k int, maxStep int, parallel int, benchmarking bool) ([][]int, [][]int) {
	ret := make([][]int, len(queryVectors))
	stepRet := make([][]int, len(queryVectors))
	for i, queryVector := range queryVectors {
		r, s := g.SearchKNN(queryVector, k, maxStep, parallel, benchmarking)
		ret[i] = r
		stepRet[i] = s
	}
	return ret, stepRet
}
