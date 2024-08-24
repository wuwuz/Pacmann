// TODO: add non-private mode

package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"example.com/private-search/graphann"
	"example.com/private-search/pianopir"
)

// TODO: check if the variables are used correctly

var syntheticTest bool
var nonPrivateMode bool
var vectors [][]float32
var graph [][]int
var queries [][]float32
var n int
var dim int
var m int
var k int
var q int

var PIR *pianopir.SimpleBatchPianoPIR

//var skipPrep bool

// embeddings file name

//const graphFile = "graph100.txt"

func genRandomMatrix(n int, dim int) [][]float32 {
	ret := make([][]float32, n)

	for i := 0; i < n; i++ {
		ret[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			ret[i][j] = rand.Float32()
		}
	}
	return ret
}

func genRandomGraph(n int, m int) [][]int {
	ret := make([][]int, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]int, m)
		for j := 0; j < m; j++ {
			k := rand.Intn(n)
			for k == i {
				// no self loop
				k = rand.Intn(n)
			}
			ret[i][j] = k
		}
	}

	return ret
}

func main() {
	numVectors := flag.Int("n", 100000, "number of vectors")
	dimVectors := flag.Int("d", 128, "dimension of the vectors")
	neighborNum := flag.Int("m", 32, "number of neighbors")
	outputNum := flag.Int("k", 100, "top K output")
	queryNum := flag.Int("q", 100, "number of queries")
	inputFile := flag.String("input", "", "input file name")
	graphFile := flag.String("graph", "", "graph file name")
	queryFile := flag.String("query", "", "file name")
	outputFile := flag.String("output", "", "output file name")
	gndFile := flag.String("gnd", "", "ground truth file name")
	reportFile := flag.String("report", "", "report file name")
	stepN := flag.Int("step", 15, "searching max depth")
	parallelN := flag.Int("parallel", 2, "how many parallel vertices are accessed in the same round")
	benchmarking := flag.Bool("benchmark", false, "benchmarking mode")
	rtt := flag.Int("rtt", 0, "round trip time in milliseconds")
	nonPrivate := flag.Bool("nonprivate", false, "non-private mode")

	flag.Parse()

	n = *numVectors
	dim = *dimVectors
	m = *neighborNum
	k = *outputNum
	q = *queryNum
	nonPrivateMode = *nonPrivate
	workingDir := filepath.Dir(*inputFile)
	fmt.Println("Working directory: ", workingDir)
	dataName := filepath.Base(*inputFile)
	dataName = strings.TrimSuffix(dataName, filepath.Ext(dataName))
	fmt.Println("Data name: ", dataName)
	dataset := dataName + fmt.Sprintf("_%d_%d_%d", n, dim, m)
	fmt.Println("Dataset name: ", dataset)

	// step 1: load vector

	if *inputFile == "" {
		log.Printf("No input file specified. If you want to use synthetic data, use -input synthetic instead.")
		return
	}

	if *inputFile == "synthetic" {
		syntheticTest = true
		vectors = genRandomMatrix(n, dim)
		log.Printf("Generated synthetic data with n=%d, dim=%d\n", n, dim)
	} else {
		// it means we need to read the file
		log.Print("Loading vectors from file: ", *inputFile)
		var err error
		vectors, err = graphann.LoadFloat32Matrix(*inputFile, n, dim)
		if err != nil {
			log.Fatalf("Error reading the input file: %v", err)
		}
	}

	// step 2: load graph. If not exists, generate the graph

	graph = make([][]int, n)
	graphFileName := *graphFile
	if syntheticTest {
		graph = genRandomGraph(n, m)
		log.Print("Generated synthetic graph...")
	} else {
		if *graphFile == "" {
			// we will use the default name
			graphFileName = filepath.Join(workingDir, dataset+"_graph.npy")
		}

		if _, err := os.Stat(graphFileName); os.IsNotExist(err) {
			// in this case we need to generate the graph
			log.Printf("Graph file %s does not exist. Generating the graph...\n", graphFileName)
			start := time.Now()
			graph = graphann.BuildGraph(n, dim, m, vectors, workingDir, dataset)
			end := time.Now()
			graphann.SaveGraphToFile(graphFileName, graph)
			log.Printf("Graph generation time: %v\n", end.Sub(start))

			// we write the graph generation time to an auxiliary file

			auxFileName := filepath.Join(workingDir, dataset+"_graph_aux.txt")
			auxFile, _ := os.Create(auxFileName)
			fmt.Fprintf(auxFile, "Dataset: %s\n", dataset)
			fmt.Fprintf(auxFile, "Graph generation time: %v\n", end.Sub(start))
		} else {
			log.Printf("Loading graph from file %s\n", graphFileName)
			graph, err = graphann.LoadIntMatrixFromFile(graphFileName, n, m)
			if err != nil {
				log.Fatalf("Error reading the graph file: %v", err)
			}
		}
	}

	// step 3: load queries

	queries = make([][]float32, q)
	if syntheticTest {
		queries = genRandomMatrix(q, dim)
		log.Print("Generated synthetic queries...")
	} else {
		if *queryFile == "" {
			log.Fatalf("No query file specified. Please specify the query file.")
		}
		log.Print("Loading queries from file: ", *queryFile)
		var err error
		queries, err = graphann.LoadFloat32Matrix(*queryFile, q, dim)
		if err != nil {
			log.Fatalf("Error reading the query file: %v", err)
		}
	}

	// step 4: build PIR instace

	queryEngine := PIRGraphInfo{
		N:              n,
		Dim:            dim,
		M:              m,
		graph:          graph,
		vectors:        vectors,
		skipPrep:       *benchmarking, // if benchmarking, we will skip PIR prep
		NonPrivateMode: nonPrivateMode,

		// the following will be set during prep
		DBEntryByteNum: 0,
		DBTotalSize:    0,
		rawDB:          nil,
		PIR:            nil,
	}

	frontend := graphann.GraphANNFrontend{
		Graph: &queryEngine,
	}

	start := time.Now()
	frontend.Preprocess()
	end := time.Now()
	prepTime := end.Sub(start)
	log.Println("Preprocessing time: ", prepTime)

	windowSize := queryEngine.PIR.SupportBatchNum / (uint64(*stepN) * uint64(*parallelN))
	//expectedMaintainenceTime := prepTime.Seconds() / float64(windowSize)

	// we now make queries

	start = time.Now()
	answers := make([][]int, q)

	maintainenceTime := time.Duration(0)
	for i := 0; i < q; i++ {
		if i%100 == 0 {
			log.Printf("Processing query %d\n", i)
		}
		answers[i], _ = frontend.SearchKNN(queries[i], k, *stepN, *parallelN, *benchmarking)

		if queryEngine.PIR.FinishedBatchNum+uint64(*stepN)*uint64(*parallelN)+10 >= queryEngine.PIR.SupportBatchNum {
			// in this case we need to re-run the preprocessing
			start := time.Now()
			queryEngine.PIR.Preprocessing()
			end := time.Now()
			maintainenceTime += end.Sub(start)
		}
	}
	end = time.Now()
	searchTime := end.Sub(start) - maintainenceTime
	avgTime := searchTime.Seconds() / float64(q)
	avgMaintainenceTime := maintainenceTime.Seconds() / float64(q)
	log.Println("Total Online time: ", searchTime)
	log.Println("Average search time: ", avgTime, " seconds per query")
	log.Println("Average maintainence time: ", avgMaintainenceTime, " seconds per query")

	// some stats
	log.Println("Total query number: ", queryEngine.totalQueryNum)
	log.Println("Successful query number: ", queryEngine.succQueryNum)
	log.Println("Success rate: ", float32(queryEngine.succQueryNum)/float32(queryEngine.totalQueryNum))

	if *outputFile == "" {
		// we use the default output file name
		*outputFile = filepath.Join(workingDir, dataset+"_output.txt")
	}

	// write the answers to the output file
	log.Println("Writing answers to the output file: ", *outputFile)

	file, err := os.Create(*outputFile)
	if err != nil {
		log.Printf("Error creating the output file: %v", err)
	} else {
		_ = graphann.SaveIntMatrixToFile(*outputFile, answers)
	}
	file.Close()

	// finally we evaluate the recall
	recall := float32(-1.0) // if -1, it means we don't have ground truth
	if *gndFile != "" {
		log.Println("Evaluating recall...")
		gnd, err := graphann.LoadIntMatrixFromFile(*gndFile, q, k)
		if err != nil {
			log.Fatalf("Error reading the ground truth file: %v", err)
		}
		recall = graphann.ComputeRecall(gnd, answers, k)
		log.Println("Recall: ", recall)
	}

	// we finally write the report

	if *reportFile == "" {
		// use a default report file name
		reportFileName := filepath.Join(workingDir, dataset+"_report.txt")
		reportFile = &reportFileName
		log.Printf("Using the default report file name: %s\n", *reportFile)
	}

	if *reportFile != "" {

		log.Printf("Writing the report to the file: %s\n", *reportFile)

		file, err := os.OpenFile(*reportFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			log.Printf("Error creating the report file: %v", err)
			return
		}

		instance := queryEngine.PIR
		config := instance.Config()
		DBSize := config.DBSize * config.DBEntryByteNum // in bytes
		PrepTime := instance.PreprocessingTime()
		MainTimePerQ := PrepTime / float64(instance.SupportBatchNum) * float64(*stepN) * float64(*parallelN)
		Storage := instance.LocalStorageSize()
		OnlineComm := instance.CommCostPerBatchOnline()
		OfflineComm := instance.CommCostPerBatchOffline()

		fmt.Fprintf(file, "-------------------------\n")
		fmt.Fprintf(file, "Private ANN Benchmarking w/ Go Frontend\n")
		fmt.Fprintf(file, "Settings:\n")
		fmt.Fprintf(file, "** Vector Num: %d\n", n)
		fmt.Fprintf(file, "** DB Size (MB): %f\n", float64(DBSize)/1024.0/1024.0)
		fmt.Fprintf(file, "** Top K: %d\n", k)
		fmt.Fprintf(file, "** Rounds: %d\n", *stepN)
		fmt.Fprintf(file, "** Parallel Exploration: %d\n", *parallelN)
		fmt.Fprintf(file, "** RTT (ms): %d\n", *rtt)
		fmt.Fprintf(file, "** Window Size: %d\n", windowSize)
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "Preprocessing Cost:\n")
		fmt.Fprintf(file, "** Storage (MB): %f\n", float64(Storage)/1024.0/1024.0)
		fmt.Fprintf(file, "** Preparation Time (s): %f\n", PrepTime)
		fmt.Fprintf(file, "** Offline Communication Cost Per Q (KB, amt.): %f\n", float64(OfflineComm)*float64(*stepN)*float64(*parallelN)/1024.0)
		fmt.Fprintf(file, "** Amortized Maintainence Time Per Q (s): %f\n", MainTimePerQ)
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "Online Cost:\n")
		fmt.Fprintf(file, "** Average Computation Time Per Query (s): %f\n", avgTime)
		fmt.Fprintf(file, "** Average Total Time Per Q (s): %f\n", avgTime+float64(*rtt)/1000.0*float64(*stepN))
		//fmt.Fprintf(file, "** Average Maintainence Time Per Q (s): %f\n", avgMaintainenceTime)
		fmt.Fprintf(file, "** Online Communication Per Q (KB): %f\n", float64(OnlineComm)*float64(*stepN)*float64(*parallelN)/1024.0)
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "Quality:\n")
		fmt.Fprintf(file, "** Recall: %f\n", recall)
		fmt.Fprintf(file, "-----------------------\n")

	}
}

// define a basic graph info struct that implements the GetGraphInfo interface

type PIRGraphInfo struct {
	N       int
	Dim     int
	M       int
	graph   [][]int
	vectors [][]float32

	skipPrep       bool
	NonPrivateMode bool
	DBEntryByteNum uint64 // per entry bytes
	DBTotalSize    uint64 // in bytes
	rawDB          []uint64
	PIR            *pianopir.SimpleBatchPianoPIR

	// some stats
	totalQueryNum int
	succQueryNum  int
}

func (g *PIRGraphInfo) Preprocess() {
	// now we set up the PIR

	// first step, we need to convert the matrix and graph into a rawDB

	N := g.N
	Dim := g.Dim
	M := g.M
	DBEntryByteNum := uint64(Dim*4 + M*4)

	fmt.Println("DBEntryByteNum: ", DBEntryByteNum)
	fmt.Println("DB Entry Number: ", N)
	fmt.Println("The raw DB has size (GB): ", float64(N)*float64(DBEntryByteNum)/1024.0/1024.0/1024.0)

	rawDB := make([]uint64, N*int(DBEntryByteNum)/8)

	for i := 0; i < N; i++ {
		// we first convert the matrix row to a byte slice
		vector := g.vectors[i]
		vectorBytes := make([]byte, Dim*4)
		for j := 0; j < Dim; j++ {
			binary.LittleEndian.PutUint32(vectorBytes[j*4:], math.Float32bits(vector[j]))
		}

		// we also convert the graph row to a byte slice
		neighbors := g.graph[i]
		neighborsBytes := make([]byte, M*4)
		for j := 0; j < M; j++ {
			binary.LittleEndian.PutUint32(neighborsBytes[j*4:], uint32(neighbors[j]))
		}

		// then we concatenate the two byte slices
		entryBytes := append(vectorBytes, neighborsBytes...)

		// then we convert the byte slice to a uint64 slice
		entry := make([]uint64, DBEntryByteNum/8)
		for j := uint64(0); j < DBEntryByteNum/8; j++ {
			entry[j] = binary.LittleEndian.Uint64(entryBytes[j*8:])
		}

		// we then copy the entry to the rawDB
		copy(rawDB[i*int(DBEntryByteNum)/8:], entry)
	}

	g.rawDB = rawDB
	fmt.Println("DB size: ", len(rawDB))
	g.DBEntryByteNum = DBEntryByteNum
	g.DBTotalSize = uint64(N) * DBEntryByteNum

	// now we set up the PIR
	g.PIR = pianopir.NewSimpleBatchPianoPIR(uint64(g.N), g.DBEntryByteNum, uint64(len(g.graph[0])), g.rawDB, 8)

	if g.skipPrep {
		g.PIR.DummyPreprocessing()
	} else {
		g.PIR.Preprocessing()
	}
}

func (g *PIRGraphInfo) GetMetadata() (int, int, int) {
	return g.N, g.Dim, g.M
}

func Entry2VectorAndNeighbors(dim int, m int, entry []uint64) ([]float32, []int) {
	// we first convert the entry to a byte slice
	entryBytes := make([]byte, len(entry)*8)
	for i := 0; i < len(entry); i++ {
		binary.LittleEndian.PutUint64(entryBytes[i*8:], entry[i])
	}

	// for the first vectorSize*4 bytes, we convert it to a float32 slice
	vector := make([]float32, dim)
	for i := 0; i < dim; i++ {
		vector[i] = math.Float32frombits(binary.LittleEndian.Uint32(entryBytes[i*4:]))
	}

	// for the next numNeighbors*4 bytes, we convert it to a uint32 slice
	neighbors := make([]int, m)
	for i := 0; i < m; i++ {
		tmp := binary.LittleEndian.Uint32(entryBytes[(dim+i)*4:])
		neighbors[i] = int(tmp)
	}

	return vector, neighbors
}

func (g *PIRGraphInfo) GetVertexInfo(vertexIds []int) ([]graphann.Vertex, error) {

	g.totalQueryNum += len(vertexIds)

	if g.NonPrivateMode {
		vertices := make([]graphann.Vertex, len(vertexIds))
		for i := 0; i < len(vertexIds); i++ {
			vertices[i] = graphann.Vertex{
				Id:        vertexIds[i],
				Vector:    g.vectors[vertexIds[i]],
				Neighbors: g.graph[vertexIds[i]],
			}
		}
		return vertices, nil
	}

	// convert the vertexIds to uint64
	indices := make([]uint64, len(vertexIds))
	for i := 0; i < len(vertexIds); i++ {
		indices[i] = uint64(vertexIds[i])
	}

	responses, err := g.PIR.Query(indices)
	if err != nil {
		return nil, err
	}

	/*
		responses := make([][]uint64, 0)
		for i := 0; i < len(indices); i += g.M {
			// we make a batch query
			response, err := g.PIR.Query(indices[i : i+g.M])
			if err != nil {
				return nil, err
			}
			responses = append(responses, response...)
		}
	*/

	vertices := make([]graphann.Vertex, len(vertexIds))
	for i, response := range responses {
		vector, neighbors := Entry2VectorAndNeighbors(g.Dim, g.M, response)
		vertices[i] = graphann.Vertex{
			Id:        vertexIds[i],
			Vector:    vector,
			Neighbors: neighbors,
		}

		// if the neighbors are not the same as the ground truth, we will record it as a failed query
		correctQ := true
		for j := 0; j < len(neighbors); j++ {
			if neighbors[j] != g.graph[vertexIds[i]][j] {
				if neighbors[j] != 0 {
					fmt.Print("Error: ", neighbors[j], " vs ", g.graph[vertexIds[i]][j])
				}
				correctQ = false
				break
			}
		}
		if correctQ {
			g.succQueryNum++
		}
	}

	return vertices, nil
}

func (g *PIRGraphInfo) GetStartVertex() ([]graphann.Vertex, error) {
	n, _, _ := g.GetMetadata()

	added := make(map[int]bool)
	// we randomly select sqrt(n) vertices as the starting vertices
	targetNum := int(math.Sqrt(float64(n)))
	batch := make([]int, targetNum)
	ret := make([]graphann.Vertex, targetNum)
	for i := 0; i < targetNum; i++ {
		x := rand.Intn(n)
		for added[x] {
			x = rand.Intn(n)
		}
		added[x] = true
		batch[i] = x
		ret[i] = graphann.Vertex{
			Id:        x,
			Vector:    g.vectors[x],
			Neighbors: g.graph[x],
		}
	}

	return ret, nil
}
