package graphann

import (
	"fmt"
	"os"
	"testing"
)

func TestVectorLoad(t *testing.T) {

	// first test the LoadBvecsFile function
	vectors, err := LoadBvecsFile("../SIFT-dataset/bigann_base.bvecs", 10, 128)

	// print the first vector
	fmt.Println(vectors[1])

	if err != nil {
		t.Errorf("Error loading bvecs file: %v", err)
	}

	if len(vectors) != 10 {
		t.Errorf("Expected 10 vectors, got %d", len(vectors))
	}

	if len(vectors[0]) != 128 {
		t.Errorf("Expected 128 dimensions, got %d", len(vectors[0]))
	}

	if vectors[0][3] != 1.0 {
		t.Errorf("Expected 1, got %f", vectors[0][3])
	}

	if vectors[1][0] != 65.0 {
		t.Errorf("Expected 65, got %f", vectors[1][0])
	}

	// now test the LoadTxtFile function

	// we need to create a test file
	testFile := "test.txt"
	n := 4
	d := 5
	matrix := make([][]float32, n)
	for i := 0; i < n; i++ {
		matrix[i] = make([]float32, d)
		for j := 0; j < d; j++ {
			matrix[i][j] = float32(i*d + j)
		}
	}

	file, err := os.Create(testFile)
	if err != nil {
		t.Errorf("Error creating test file: %v", err)
	}

	for _, row := range matrix {
		for _, val := range row {
			fmt.Fprintf(file, "%f ", val)
		}
		fmt.Fprintln(file)
	}

	file.Close()

	vectors, err = LoadFloat32Matrix(testFile, n, d)
	if err != nil {
		t.Errorf("Error loading txt file: %v", err)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			if vectors[i][j] != matrix[i][j] {
				t.Errorf("Expected %f, got %f", matrix[i][j], vectors[i][j])
			}
		}
	}

	// delete the test file
	//err = os.Remove(testFile)
	//if err != nil {
	//t.Errorf("Error deleting test file: %v", err)
	//}
}

func TestGraphLoad(t *testing.T) {
	extensions := []string{".npy", ".txt"}

	n := 2
	m := 4

	graph := make([][]int, n)
	for i := 0; i < n; i++ {
		graph[i] = make([]int, m)
		for j := 0; j < m; j++ {
			graph[i][j] = i*m + j + 1
		}
	}

	for _, ext := range extensions {

		filename := "test_graph" + ext
		err := SaveGraphToFile(filename, graph)
		if err != nil {
			t.Errorf("Error saving graph: %v", err)
		}

		loadedGraph, err := LoadGraphFromFile(filename, n, m)
		if err != nil {
			t.Errorf("Error loading graph: %v", err)
		}

		for i, row := range loadedGraph {
			for j, val := range row {
				if val != graph[i][j] {
					t.Errorf("Expected %d, got %d", graph[i][j], val)
				}
			}
		}

		// delete the test file
		//err = os.Remove(filename)
		//if err != nil {
		//	t.Errorf("Error deleting test file: %v", err)
		//}
	}
}

func TestIvecsLoad(t *testing.T) {

	gnd, err := LoadIvecsFile("/home/mingxunz/Private-Search/SIFT-dataset/gnd/idx_1M.ivecs", 10, 100) // the last dim is not important

	if err != nil {
		t.Errorf("Error loading ivecs file: %v", err)
	}

	if len(gnd) != 10 {
		t.Errorf("Expected 10 vectors, got %d", len(gnd))
	}

	fmt.Println("Shape of gnd", len(gnd), len(gnd[0]))
	fmt.Println(gnd[0][0:5])
}
