// TODO: add non-private mode

package main

import (
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"example.com/private-search/graphann"
	"github.com/yahoojapan/gongt"
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
	ngtFile := flag.String("ngt", "", "ngt file name")
	queryFile := flag.String("query", "", "file name")
	outputFile := flag.String("output", "", "output file name")
	gndFile := flag.String("gnd", "", "ground truth file name")
	reportFile := flag.String("report", "", "report file name")

	flag.Parse()

	n = *numVectors
	dim = *dimVectors
	m = *neighborNum
	k = *outputNum
	q = *queryNum
	workingDir := filepath.Dir(*inputFile)
	fmt.Println("Working directory: ", workingDir)
	dataName := filepath.Base(*inputFile)
	dataName = strings.TrimSuffix(dataName, filepath.Ext(dataName))
	fmt.Println("Data name: ", dataName)
	dataset := dataName + fmt.Sprintf("_%d_%d_%d", n, dim, m)
	fmt.Println("Dataset name: ", dataset)

	factor := 20.0

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

	// step 2: load ngt. If not exists, generate the ngt file

	ngtFileName := *ngtFile
	if ngtFileName == "" {
		// we will use the default name
		ngtFileName = filepath.Join(workingDir, dataset+".ngt")
	}

	log.Printf("NGT file name: %s\n", ngtFileName)

	var ngt *gongt.NGT
	if _, err := os.Stat(ngtFileName); os.IsNotExist(err) {

		// in this case we need to build the ngt file
		log.Printf("Building the NGT index...")

		start := time.Now()
		ngt = gongt.New(ngtFileName).SetObjectType(gongt.Float).SetDimension(dim).Open()

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

	} else {
		ngt = gongt.New(ngtFileName).Open()
	}
	defer ngt.Close()

	// test the ngt index

	hit := 0
	reptN := 100
	for i := 0; i < reptN; i++ {
		tmp := make([]float64, len(vectors[i]))
		for j := 0; j < len(vectors[i]); j++ {
			tmp[j] = float64(vectors[i][j])
		}
		res, err := ngt.Search(tmp, 10, gongt.DefaultEpsilon*factor)
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

	fmt.Print("Hit rate for NGT: ", float32(hit)/float32(reptN), "\n")

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

	// we now make queries

	start := time.Now()
	answers := make([][]int, q)

	for i := 0; i < q; i++ {
		if i%100 == 0 {
			log.Printf("Processing query %d\n", i)
		}

		// convert the query to float64
		tmp := make([]float64, len(queries[i]))
		for j := 0; j < len(queries[i]); j++ {
			tmp[j] = float64(queries[i][j])
		}

		res, err := ngt.Search(tmp, k, gongt.DefaultEpsilon*factor)
		if err != nil {
			fmt.Println("Error in searching: ", err)
		}
		// verify that the search has found the vertex itself
		answers[i] = make([]int, k)
		for j := 0; j < len(res) && j < k; j++ {
			answers[i][j] = int(res[j].ID - 1)
		}
	}
	end := time.Now()
	searchTime := end.Sub(start)
	avgTime := searchTime.Seconds() / float64(q)
	log.Println("Total Online time: ", searchTime)
	log.Println("Average search time: ", avgTime, " seconds per query")

	if *outputFile == "" {
		// we use the default output file name
		*outputFile = filepath.Join(workingDir, dataset+"_output_ngt.txt")
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
		reportFileName := filepath.Join(workingDir, dataset+"_report_ngt.txt")
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

		fmt.Fprintf(file, "-------------------------\n")
		fmt.Fprintf(file, "NGT Stats\n")
		fmt.Fprintf(file, "Settings:\n")
		fmt.Fprintf(file, "** Vector Num: %d\n", n)
		fmt.Fprintf(file, "** Top K: %d\n", k)
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "** Average Computation Time Per Query (s): %f\n", avgTime)
		//fmt.Fprintf(file, "** Average Maintainence Time Per Q (s): %f\n", avgMaintainenceTime)
		fmt.Fprintf(file, "\n")
		fmt.Fprintf(file, "Quality:\n")
		fmt.Fprintf(file, "** Recall: %f\n", recall)
		fmt.Fprintf(file, "-----------------------\n")

	}
}
