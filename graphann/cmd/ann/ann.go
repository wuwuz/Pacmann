package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"example.com/graphann"
)

func main() {
	// Parameters
	// "-n": number of vectors
	// "-d": dimension of the vectors
	// "-m": number of neighbors
	// "-q": number of queries
	// "-k": top K output
	// "-input": input file name, default to synthetic
	// "-output": output file name, default to null
	// "--parallel": how many parallel vertices are accessed in the same round, default to 1

	numVectors := flag.Int("n", 100000, "number of vectors")
	dimVectors := flag.Int("d", 128, "dimension of the vectors")
	neighborNum := flag.Int("m", 32, "number of neighbors")
	outputNum := flag.Int("k", 100, "top K output")
	queryNum := flag.Int("q", 100, "number of queries")
	inputFile := flag.String("input", "", "input file name")
	queryFile := flag.String("query", "", "file name")
	outputFile := flag.String("output", "", "output file name")
	gndFile := flag.String("gnd", "", "ground truth file name")
	//reportFile := flag.String("report", "", "report file name")
	stepN := flag.Int("step", 15, "searching max depth")
	parallelN := flag.Int("parallel", 2, "how many parallel vertices are accessed in the same round")

	flag.Parse()

	n := *numVectors
	d := *dimVectors
	m := *neighborNum
	k := *outputNum
	q := *queryNum

	// default: we will store the index graph in the same directory as the input file

	if *inputFile == "" {
		fmt.Println("Please specify the input file")
		return
	}

	var vectors [][]float32
	if _, err := os.Stat(*inputFile); err == nil {
		fmt.Println("Loading vectors from file")
		vectors, err = graphann.LoadFloat32Matrix(*inputFile, n, d)
		if err != nil {
			fmt.Println("Error loading vectors from file: ", err)
			return
		}
	} else {
		fmt.Printf("Error: Loading files %e", err)
		return
	}

	workingDir := filepath.Dir(*inputFile)
	fmt.Println("Working directory: ", workingDir)
	dataName := filepath.Base(*inputFile)
	dataName = strings.TrimSuffix(dataName, filepath.Ext(dataName))
	fmt.Println("Data name: ", dataName)
	dataset := dataName + fmt.Sprintf("_%d_%d_%d", n, d, m)
	fmt.Println("Dataset name: ", dataset)
	graphFile := workingDir + "/" + dataset + "_graph.npy"
	fmt.Println("Graph file: ", graphFile)

	var graph [][]int
	if _, err := os.Stat(graphFile); err == nil {
		fmt.Println("Graph file already exists, skipping building graph")
		graph, err = graphann.LoadIntMatrixFromFile(graphFile, n, m)
		if err != nil {
			fmt.Printf("Error loading graph from file: %v\n", err)
		}
	} else {
		fmt.Println("Building graph")
		start := time.Now()
		graph = graphann.BuildGraph(n, d, m, vectors, workingDir, dataset)
		end := time.Now()
		graphann.SaveGraphToFile(graphFile, graph)
		fmt.Println("Graph built and saved to file. Time = ", end.Sub(start))
	}

	fmt.Println("Graph size: ", len(graph), len(graph[0]))

	// loading the query vectors
	var queryVectors [][]float32
	if *queryFile == "" {
		fmt.Println("No query file specified. Skipping queries.")
		return
	}

	if _, err := os.Stat(*queryFile); err == nil {
		fmt.Println("Loading query vectors from file")
		queryVectors, err = graphann.LoadFloat32Matrix(*queryFile, q, d)
		if err != nil {
			fmt.Println("Error loading query vectors from file: ", err)
			return
		}
	} else {
		fmt.Printf("Error: Loading files %e", err)
		return
	}

	// now we run the query

	output := *outputFile
	if output == "" {
		output = workingDir + "/" + dataset + "_output.txt"
		fmt.Println("No output file specified. New name: ", output)
	}

	frontend := graphann.GraphANNFrontend{
		Graph: &graphann.BasicGraphInfo{
			N:       n,
			Dim:     d,
			M:       m,
			Graph:   graph,
			Vectors: vectors,
		},
	}

	frontend.Preprocess()

	start := time.Now()
	answer, _ := frontend.SearchKNNBatch(queryVectors, k, *stepN, *parallelN, false)
	end := time.Now()

	fmt.Println("Search time: ", end.Sub(start))
	fmt.Println("Average search time: ", end.Sub(start)/time.Duration(q))

	// output the result to output

	outputF, err := os.Create(output)
	if err != nil {
		fmt.Println("Error creating output file: ", err)
		return
	}

	for i := 0; i < q; i++ {
		for j := 0; j < k; j++ {
			outputF.WriteString(fmt.Sprintf("%d ", answer[i][j]))
		}
		outputF.WriteString("\n")
	}
	fmt.Println("Output written to file: ", output)

	if *gndFile != "" {
		gnd, err := graphann.LoadGraphFromFile(*gndFile, q, k) // it's just the same as loading the graph
		if err != nil {
			fmt.Println("Error loading ground truth file: ", err)
			return
		}

		recall := graphann.ComputeRecall(gnd, answer, k)
		fmt.Println("Recall: ", recall)
	}
}
