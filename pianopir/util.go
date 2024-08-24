package pianopir

import (
	//"crypto/sha256"
	"hash/fnv"
	"math"

	// "fmt"
	//////////////////"io"
	//"crypto/aes"
	//"crypto/cipher"

	"encoding/binary"
	rand "math/rand"
)

type PrfNonce [12]byte
type PrfKey128 [16]byte
type block [16]byte

//type DBEntry [DBEntryLength]uint64

type PrfKey PrfKey128

func RandKey128(rng *rand.Rand) PrfKey128 {
	var key [16]byte
	//rand.Read(key[:])
	binary.LittleEndian.PutUint64(key[0:8], rng.Uint64())
	binary.LittleEndian.PutUint64(key[8:16], rng.Uint64())
	return key
}

func RandKey(rng *rand.Rand) PrfKey {
	return PrfKey(RandKey128(rng))
}

func PRFEval(key *PrfKey, x uint64) uint64 {
	return PRFEval4((*PrfKey128)(key), x)
}

/*
func DBEntryXor(dst *DBEntry, src *DBEntry) {
	for i := 0; i < DBEntryLength; i++ {
		(*dst)[i] ^= (*src)[i]
	}
}

func DBEntryXorFromRaw(dst *DBEntry, src []uint64) {
	for i := 0; i < DBEntryLength; i++ {
		(*dst)[i] ^= src[i]
	}
}

func EntryIsEqual(a *DBEntry, b *DBEntry) bool {
	for i := 0; i < DBEntryLength; i++ {
		if (*a)[i] != (*b)[i] {
			return false
		}
	}
	return true
}

func RandDBEntry(rng *rand.Rand) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = rng.Uint64()
	}
	return entry
}

func GenDBEntry(key uint64, id uint64) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = DefaultHash((key ^ id) + uint64(i))
	}
	return entry
}

func ZeroEntry() DBEntry {
	ret := DBEntry{}
	for i := 0; i < DBEntryLength; i++ {
		ret[i] = 0
	}
	return ret
}

func DBEntryFromSlice(s []uint64) DBEntry {
	var entry DBEntry
	for i := 0; i < DBEntryLength; i++ {
		entry[i] = s[i]
	}
	return entry
}
*/

// return ChunkSize, SetSize
func GenParams(DBSize uint32) (uint32, uint32) {
	targetChunkSize := uint32(2 * math.Sqrt(float64(DBSize)))
	ChunkSize := uint32(1)
	for ChunkSize < targetChunkSize {
		ChunkSize *= 2
	}
	SetSize := uint32(math.Ceil(float64(DBSize) / float64(ChunkSize)))
	// round up to the next mulitple of 4
	SetSize = (SetSize + 3) / 4 * 4
	return ChunkSize, SetSize
}

type AesPrf struct {
	// block cipher.Block
	enc []uint32
}

func xor16(dst, a, b *byte)
func encryptAes128(xk *uint32, dst, src *byte)
func aes128MMO(xk *uint32, dst, src *byte)
func expandKeyAsm(key *byte, enc *uint32)

func NewCipher(key uint64) (*AesPrf, error) {
	k := make([]byte, 16)
	binary.LittleEndian.PutUint64(k, key)
	// n := 11*4
	c := AesPrf{make([]uint32, 4)}
	expandKeyAsm(&k[0], &c.enc[0])
	// fmt.Println("NEW CIPHER")
	// fmt.Println(k)
	// fmt.Println(c.enc)
	return &c, nil
}

func (c *AesPrf) Encrypt(dst, src []byte) {
	encryptAes128(&c.enc[0], &dst[0], &src[0])
}

func DefaultHash(key uint64) uint64 {
	hash := fnv.New64a()
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, key)
	hash.Write(b)
	return hash.Sum64()
}

func nonSafePRFEval(key uint64, x uint64) uint64 {
	return DefaultHash(key ^ x)
}

func PRFEval4(key *PrfKey128, x uint64) uint64 {
	var longKey = make([]uint32, 11*4)
	expandKeyAsm(&key[0], &longKey[0])
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)
	binary.LittleEndian.PutUint64(src, x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	return binary.LittleEndian.Uint64(dsc)
}

func PRFEvalWithLongKeyAndTag(longKey []uint32, tag uint64, x uint64) uint64 {
	var src = make([]byte, 16)
	var dsc = make([]byte, 16)

	// the tag has to be less than 2^29
	binary.LittleEndian.PutUint64(src, (tag<<35)+x)
	aes128MMO(&longKey[0], &dsc[0], &src[0])
	return binary.LittleEndian.Uint64(dsc)
}

func GetLongKey(key *PrfKey128) []uint32 {
	var longKey = make([]uint32, 11*4)
	expandKeyAsm(&key[0], &longKey[0])
	return longKey
}

func xorSlices(dst, src []uint64, n int)
