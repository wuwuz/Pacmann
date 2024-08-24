package pianopir

import (
	"math/rand"
	"testing"
	"time"
)

func TestPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(18750)
	DBEntrySize := uint64(4)
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewPianoPIR(DBSize, DBEntrySize*8, rawDB, 40)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("PIR config: %v", config)
	t.Logf("hint num: %v", PIR.client.primaryHintNum)
	t.Logf("max query num: %v", PIR.client.MaxQueryNum)

	maxQueryNum := PIR.client.MaxQueryNum

	PIR.Preprocessing()

	// make 1000 random queries
	for i := 0; i < int(maxQueryNum); i++ {
		idx := rand.Uint64() % DBSize
		query, err := PIR.Query(idx, true)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", idx, err)
		}

		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}

		if i == 0 {
			t.Logf("response = %v", query)
		}

		// just output a message to show the progress
		//t.Logf("PIR.Query(%v) passed", idx)
	}
}

func TestBatchPIRBasic(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(1000000)
	DBEntrySize := uint64(16)
	BatchSize := uint64(32)

	// a seed that's depending on the current time
	//seed := time.Now().UnixNano()
	//rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = uint64(i) //rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 20)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v", config)

	PIR.Preprocessing()

	// make a single batch query
	// for each partition, make PartitionQueryNum queries
	batchQuery := make([]uint64, 0, BatchSize)

	for i := uint64(0); i < config.PartitionNum; i++ {
		start := i * config.PartitionSize
		end := min((i+1)*config.PartitionSize, DBSize)

		for j := uint64(0); j < QueryPerPartition-1; j++ {
			offset := rand.Uint64() % (end - start)
			// append the query to the batch query
			batchQuery = append(batchQuery, start+offset)
		}
	}

	// now make a batch query
	// they should be all correct

	responses, err := PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := 0; i < len(batchQuery); i++ {
		idx := batchQuery[i]
		query := responses[i]
		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}

	// we make a batch query with each partition having 4 queries

	batchQuery = make([]uint64, 4*config.PartitionNum)
	for i := 0; i < int(config.PartitionNum); i++ {
		start := i * int(config.PartitionSize)
		end := min((i+1)*int(config.PartitionSize), int(DBSize))

		for j := 0; j < 4; j++ {
			offset := int(rand.Uint64() % uint64(end-start))
			// append the query to the batch query
			batchQuery[i*4+j] = uint64(start + offset)
		}
	}

	responses, err = PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := 0; i < len(batchQuery); i++ {
		idx := batchQuery[i]
		query := responses[i]
		for j := uint64(0); j < DBEntrySize; j++ {
			if query[j] != rawDB[idx*DBEntrySize+j] {
				t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
			}
		}
	}

	//t.Logf("Batch PIR.Query(%v) passed", batchQuery)

	// now make another batch query
	// it only has queries in the first partition
	// so only the first PartitionQueryNum queries should be correct

	querySet := make(map[uint64]bool)
	batchQuery = make([]uint64, 0, BatchSize)
	for i := uint64(0); i < BatchSize; i++ {
		idx := rand.Uint64() % config.PartitionSize
		if _, ok := querySet[idx]; ok {
			// resample the index
			i--
			continue
		}
		querySet[idx] = true
		batchQuery = append(batchQuery, idx)
	}

	// now make a batch query
	// only the first PartitionQueryNum queries should be correct
	// the rest should be all zeros

	//fmt.Println("batchQuery: ", batchQuery)

	responses, err = PIR.Query(batchQuery)

	if err != nil {
		t.Errorf("PIR.Query(%v) failed: %v", batchQuery, err)
	}

	for i := uint64(0); i < BatchSize; i++ {
		idx := batchQuery[i]
		query := responses[i]

		if i < QueryPerPartition {
			// check if the first PartitionQueryNum queries are correct
			for j := uint64(0); j < DBEntrySize; j++ {
				if query[j] != rawDB[idx*DBEntrySize+j] {
					t.Errorf("query[%v] = %v; want %v", idx, query[j], rawDB[idx*DBEntrySize+j])
				}
			}
		} else {
			// otherwise check if they are all zeros
			for j := uint64(0); j < DBEntrySize; j++ {
				if query[j] != 0 {
					t.Errorf("query[%v] = %v; want 0", idx, query[j])
				}
			}
		}
	}
}

func TestBatchPIRPerf(t *testing.T) {
	// Arrange
	// Set up any necessary data or arguments

	DBSize := uint64(3201821)
	//DBSize := uint64(100000000)
	//DBSize := uint64(300000)
	DBEntrySize := uint64(112)
	BatchSize := uint64(32)

	// a seed that's depending on the current time
	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))

	rawDB := make([]uint64, DBEntrySize*DBSize)
	for i := uint64(0); i < DBSize; i++ {
		for j := uint64(0); j < DBEntrySize; j++ {
			rawDB[i*DBEntrySize+j] = rng.Uint64()
		}
	}

	PIR := NewSimpleBatchPianoPIR(DBSize, DBEntrySize*8, BatchSize, rawDB, 8)

	// print the config of the PIR
	config := PIR.Config()
	t.Logf("Batch PIR config: %v\n", config)
	t.Logf("Batch PIR storage %v MB\n", PIR.LocalStorageSize()/1024/1024)
	t.Logf("Batch PIR max query num%v\n", PIR.subPIR[0].client.MaxQueryNum)
	t.Logf("Sub PIR config: %v\n", PIR.subPIR[0].Config())
	t.Logf("Sub PIR primary hint num: %v\n", PIR.subPIR[0].client.primaryHintNum)
	t.Logf("Sub PIR strorae %v MB\n", PIR.subPIR[0].LocalStorageSize()/1024/1024)
	PIR.subPIR[0].client.PrintStorageBreakdown()

	start := time.Now()
	PIR.Preprocessing()
	end := time.Now()
	t.Logf("Preprocessing time = %v\n", end.Sub(start))

	// now we make 1000 random batchQuery

	queryNum := 300

	start = time.Now()
	for i := 0; i < queryNum; i++ {
		batch := make([]uint64, 0, BatchSize)
		for j := 0; j < int(BatchSize); j++ {
			batch = append(batch, rng.Uint64()%DBSize)
		}
		response, err := PIR.Query(batch)
		if err != nil {
			t.Errorf("PIR.Query(%v) failed: %v", batch, err)
		}
		//we check the first response, either it's all zeros, or it's correct
		for j := uint64(0); j < DBEntrySize; j++ {
			if response[0][j] != 0 && response[0][j] != rawDB[batch[0]*DBEntrySize+j] {
				t.Errorf("response[0][%v] = %v; want %v", j, response[0][j], rawDB[batch[0]*DBEntrySize+j])
			}
		}
	}
	end = time.Now()
	t.Logf("Total query time = %v\n", end.Sub(start))
	t.Logf("Average query time per batch = %v\n", end.Sub(start)/time.Duration(queryNum))
	avgBatchTime := end.Sub(start) / time.Duration(queryNum)

	rtt := time.Duration(50) * time.Millisecond
	parallel := 2
	step := 15

	// now we estimate the average ann latency by (avgBatchTime * parallel + rtt) * step
	annLatency := (avgBatchTime*time.Duration(parallel) + rtt) * time.Duration(step)
	t.Logf("Estimated ANN latency = %v\n", annLatency)
}

func TestXORPerf(t *testing.T) {

	p := make([]uint64, 8)
	q := make([]uint64, 8)
	for i := 0; i < 8; i++ {
		p[i] = 12312312
		q[i] = 12312
	}
	xorSlices(p, q, 8)
	for i := 0; i < 8; i++ {
		if p[i] != 12312312^12312 {
			t.Errorf("p[%v] = %v; want %v", i, p[i], 12312312^12312)
		}
	}

	n := 1000000
	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] = 12312312
			b[i*l+j] = 12312
		}
	}

	// naive xor

	start := time.Now()
	for i := 0; i < n; i++ {
		for j := 0; j < l; j++ {
			a[i*l+j] ^= b[i*l+j]
		}
	}
	end := time.Now()
	t.Logf("Naive XOR time = %v\n", end.Sub(start))

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()
	xorSlices(a, b, l*n)
	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))

	// verify the result
	for i := 0; i < l*n; i++ {
		if a[i] != 12312312^12312 {
			t.Errorf("a[%v] = %v; want %v", i, a[i], 12312312^12312)
		}
	}
}

func TestAESPerf(t *testing.T) {

	seed := time.Now().UnixNano()
	rng := rand.New(rand.NewSource(seed))
	masterKey := RandKey(rng)
	longKey := GetLongKey((*PrfKey128)(&masterKey))

	n := 1000000
	tag := make([]uint64, n)
	results := make([]uint64, n)

	for i := 0; i < n; i++ {
		tag[i] = rng.Uint64()
		results[i] = 0
	}

	start := time.Now()
	for i := 0; i < n; i++ {
		results[i] = PRFEvalWithLongKeyAndTag(longKey, tag[i], uint64(i))
	}
	end := time.Now()
	t.Logf("PRFEvalWithLongKeyAndTag time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))

	l := 112
	a := make([]uint64, l*n)
	b := make([]uint64, l*n)

	for i := 0; i < l*n; i++ {
		a[i] = 12312312
		b[i] = 12312
	}

	// use XorSlice
	start = time.Now()

	for i := 0; i < n; i++ {
		xorSlices(a[i*l:(i+1)*l], b[i*l:(i+1)*l], l)
	}

	end = time.Now()
	t.Logf("XorSlices time = %v\n", end.Sub(start))
	t.Logf("average time = %v ns", end.Sub(start).Nanoseconds()/int64(n))
}
