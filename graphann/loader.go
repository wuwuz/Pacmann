package graphann

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/kshard/fvecs"
	"github.com/kshedden/gonpy"
)

func LoadBvecsFile(filename string, n int, dim int) ([][]float32, error) {
	// Reading vectors
	r, err := os.Open(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	d := fvecs.NewDecoder[byte](r)

	// we allocate n*dim memory space for the vectors
	mem_space := make([]float32, n*dim)
	ret := make([][]float32, 0, n)
	for i := 0; i < n; i++ {
		// we set the pointer to the right position
		ret = append(ret, mem_space[i*dim:(i+1)*dim])
	}

	for i := 0; i < n; i++ {
		v, err := d.Read()
		if err == io.EOF {
			fmt.Print("Unexpected EOF\n")
			break
		}

		if err != nil {
			fmt.Printf("Error reading vector %d\n", i)
			break
		}

		// create a float32 slice from the byte slice
		vf := make([]float32, len(v))
		for i, b := range v {
			vf[i] = float32(b)
			//vf[i] = vf[i] / 255.0
		}

		// now we copy the vector to the memory space
		copy(ret[i][:], vf)
	}

	return ret, nil
}

func LoadFloat32MatrixFromBvecs(filename string, n int, dim int) ([][]float32, error) {
	return LoadBvecsFile(filename, n, dim)
}

func LoadFvecsFile(filename string, n int, dim int) ([][]float32, error) {
	// Reading vectors
	r, err := os.Open(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	d := fvecs.NewDecoder[float32](r)

	ret := make([][]float32, 0)

	for i := 0; i < n; i++ {
		v, err := d.Read()
		if err != nil {
			break
		}
		ret = append(ret, v)
	}

	return ret, nil
}

func LoadFloat32MatrixFromFvecs(filename string, n int, dim int) ([][]float32, error) {
	return LoadFvecsFile(filename, n, dim)
}

func LoadIvecsFile(filename string, n int, dim int) ([][]int, error) {
	// Reading vectors
	r, err := os.Open(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	d := fvecs.NewDecoder[uint32](r)

	ret := make([][]int, n)

	for i := 0; i < n; i++ {
		v, err := d.Read()
		if err != nil {
			fmt.Printf("Error reading vector %d, %e\n", i, err)
			panic(err)
		}
		ret[i] = make([]int, len(v))
		for j, val := range v {
			ret[i][j] = int(val)
		}
	}

	return ret, nil
}

func LoadIntMatrixFromIvecs(filename string, n int, dim int) ([][]int, error) {
	return LoadIvecsFile(filename, n, dim)
}

func LoadTxtFileFloat32(filename string, n int, dim int) ([][]float32, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	matrix := make([][]float32, n)
	for i := range matrix {
		matrix[i] = make([]float32, dim)
	}

	scanner := bufio.NewScanner(file)
	for i := 0; i < n && scanner.Scan(); i++ {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) != dim {
			return nil, fmt.Errorf("line %d has %d fields, expected %d", i+1, len(fields), dim)
		}

		for j := 0; j < dim; j++ {
			value, err := strconv.ParseFloat(fields[j], 32)
			if err != nil {
				return nil, fmt.Errorf("failed to parse field %d on line %d: %v", j+1, i+1, err)
			}

			matrix[i][j] = float32(value)
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return matrix, nil
}

func LoadFloat32MatrixFromTxt(filename string, n int, dim int) ([][]float32, error) {
	return LoadTxtFileFloat32(filename, n, dim)
}

func LoadFloat32MatrixFromNpy(filename string, n int, dim int) ([][]float32, error) {
	r, err := gonpy.NewFileReader(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	shape := r.Shape

	// check the shape
	if len(shape) != 2 || shape[0] < n || shape[1] != dim {
		fmt.Printf("Invalid shape: %v\n", shape)
		fmt.Printf("Expected shape: (%d, %d)\n", n, dim)
		return nil, fmt.Errorf("invalid shape: %v", shape)
	}

	data, err := r.GetFloat64()
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	// we now convert the data to a 2D slice
	ret := make([][]float32, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]float32, dim)
		for j := 0; j < dim; j++ {
			ret[i][j] = float32(data[i*dim+j])
		}
	}

	return ret, nil
}

func LoadFloat32Matrix(filename string, n int, dim int) ([][]float32, error) {
	// first we check the file extension
	ext := filepath.Ext(filename)

	// write a if case for each extension
	switch ext {
	case ".bvecs":
		return LoadFloat32MatrixFromBvecs(filename, n, dim)
	case ".fvecs":
		return LoadFloat32MatrixFromFvecs(filename, n, dim)
	case ".txt":
		return LoadFloat32MatrixFromTxt(filename, n, dim)
	case ".npy":
		return LoadFloat32MatrixFromNpy(filename, n, dim)
	default:
		fmt.Printf("Unknown file extension: %s\n", ext)
		return nil, fmt.Errorf("unknown file extension: %s", ext)
	}
}

func LoadGraphFromNpyFile(filename string, n int, m int) ([][]int, error) {
	r, err := gonpy.NewFileReader(filename)
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	shape := r.Shape

	// check the shape
	if len(shape) != 2 || shape[0] < n || shape[1] != m {
		fmt.Printf("Invalid shape: %v\n", shape)
		return nil, fmt.Errorf("invalid shape: %v", shape)
	}

	data, err := r.GetInt32()
	if err != nil {
		fmt.Println(err)
		return nil, err
	}

	// we now convert the data to a 2D slice
	ret := make([][]int, n)
	for i := 0; i < n; i++ {
		ret[i] = make([]int, m)
		for j := 0; j < m; j++ {
			ret[i][j] = int(data[i*m+j])
		}
	}

	return ret, nil
}

func LoadGraphFromTxtFile(filename string, n int, m int) ([][]int, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	graph := make([][]int, n)
	for i := range graph {
		graph[i] = make([]int, m)
	}

	scanner := bufio.NewScanner(file)
	for i := 0; i < n && scanner.Scan(); i++ {
		line := scanner.Text()
		fields := strings.Fields(line)
		if len(fields) != m {
			return nil, fmt.Errorf("line %d has %d fields, expected %d", i+1, len(fields), m)
		}

		for j := 0; j < m; j++ {
			value, err := strconv.Atoi(fields[j])
			if err != nil {
				return nil, fmt.Errorf("failed to parse field %d on line %d: %v", j+1, i+1, err)
			}

			graph[i][j] = value
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	return graph, nil
}

func LoadGraphFromFile(filename string, n int, m int) ([][]int, error) {
	ext := filepath.Ext(filename)
	switch ext {
	case ".npy":
		return LoadGraphFromNpyFile(filename, n, m)
	case ".txt":
		return LoadGraphFromTxtFile(filename, n, m)
	case ".ivecs":
		return LoadIvecsFile(filename, n, m)
	default:
		fmt.Printf("Unknown file extension: %s\n", ext)
		return nil, fmt.Errorf("unknown file extension: %s", ext)
	}
}

func LoadIntMatrixFromFile(filename string, n int, m int) ([][]int, error) {
	return LoadGraphFromFile(filename, n, m) // same thing
}

func SaveGraphToNpyFile(filename string, graph [][]int) error {
	n := len(graph)
	m := len(graph[0])

	data := make([]int32, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			data[i*m+j] = int32(graph[i][j])
		}
	}

	w, err := gonpy.NewFileWriter(filename)
	if err != nil {
		fmt.Println(err)
		return err
	}

	w.Shape = []int{n, m}
	err = w.WriteInt32(data)
	return err
}

func SaveGraphToTxtFile(filename string, graph [][]int) error {
	w, err := os.Create(filename)
	if err != nil {
		fmt.Println(err)
		return err
	}
	defer w.Close()

	n := len(graph)
	m := len(graph[0])

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			fmt.Fprintf(w, "%d ", graph[i][j])
		}
		fmt.Fprintln(w)
	}

	return nil
}

func SaveGraphToFile(filename string, graph [][]int) error {
	ext := filepath.Ext(filename)
	switch ext {
	case ".npy":
		return SaveGraphToNpyFile(filename, graph)
	case ".txt":
		return SaveGraphToTxtFile(filename, graph)
	default:
		fmt.Printf("Unknown file extension: %s\n", ext)
		return fmt.Errorf("unknown file extension: %s", ext)
	}
}

func SaveIntMatrixToFile(filename string, matrix [][]int) error {
	return SaveGraphToFile(filename, matrix) // same thing
}
