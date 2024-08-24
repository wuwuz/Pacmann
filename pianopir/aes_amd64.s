// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!appengine,!gccgo

// func xor16(dst, a, b *byte)
TEXT ·xor16(SB),4,$0
	MOVQ dst+0(FP), AX
	MOVQ a+8(FP), BX
	MOVQ b+16(FP), CX
	MOVUPS 0(BX), X0
	MOVUPS 0(CX), X1
	PXOR X1, X0
	MOVUPS X0, 0(AX)
	RET

// func encryptAes128(xk *uint32, dst, src *byte)
TEXT ·encryptAes128(SB),4,$0
	MOVQ xk+0(FP), AX
	MOVQ dst+8(FP), DX
	MOVQ src+16(FP), BX
	MOVUPS 0(AX), X1
	MOVUPS 0(BX), X0
	ADDQ $16, AX
	PXOR X1, X0
	MOVUPS 0(AX), X1
	AESENC X1, X0
	MOVUPS 16(AX), X1
	AESENC X1, X0
	MOVUPS 32(AX), X1
	AESENC X1, X0
	MOVUPS 48(AX), X1
	AESENC X1, X0
	MOVUPS 64(AX), X1
	AESENC X1, X0
	MOVUPS 80(AX), X1
	AESENC X1, X0
	MOVUPS 96(AX), X1
	AESENC X1, X0
	MOVUPS 112(AX), X1
	AESENC X1, X0
	MOVUPS 128(AX), X1
	AESENC X1, X0
	MOVUPS 144(AX), X1
	AESENCLAST X1, X0
	MOVUPS X0, 0(DX)
	RET

// func aes128MMO(xk *uint32, dst, src *byte)
TEXT ·aes128MMO(SB),4,$0
	MOVQ xk+0(FP), AX
	MOVQ dst+8(FP), DX
	MOVQ src+16(FP), BX
	MOVUPS 0(AX), X1
	MOVUPS 0(BX), X0
	ADDQ $16, AX
	PXOR X1, X0
	MOVUPS 0(AX), X1
	AESENC X1, X0
	MOVUPS 16(AX), X1
	AESENC X1, X0
	MOVUPS 32(AX), X1
	AESENC X1, X0
	MOVUPS 48(AX), X1
	AESENC X1, X0
	MOVUPS 64(AX), X1
	AESENC X1, X0
	MOVUPS 80(AX), X1
	AESENC X1, X0
	MOVUPS 96(AX), X1
	AESENC X1, X0
	MOVUPS 112(AX), X1
	AESENC X1, X0
	MOVUPS 128(AX), X1
	AESENC X1, X0
	MOVUPS 144(AX), X1
	AESENCLAST X1, X0
	MOVUPS 0(BX), X1
	PXOR X1, X0
	MOVUPS X0, 0(DX)
	RET


// func expandKeyAsm(key *byte, enc *uint32) {
// Note that round keys are stored in uint128 format, not uint32
TEXT ·expandKeyAsm(SB),4,$0
	MOVQ key+0(FP), AX
	MOVQ enc+8(FP), BX
	MOVUPS (AX), X0
	// enc
	MOVUPS X0, (BX)
	ADDQ $16, BX
	PXOR X4, X4 // _expand_key_* expect X4 to be zero
	AESKEYGENASSIST $0x01, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x02, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x04, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x08, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x10, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x20, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x40, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x80, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x1b, X0, X1
	CALL _expand_key_128<>(SB)
	AESKEYGENASSIST $0x36, X0, X1
	CALL _expand_key_128<>(SB)
	RET

TEXT _expand_key_128<>(SB),4,$0
	PSHUFD $0xff, X1, X1
	SHUFPS $0x10, X0, X4
	PXOR X4, X0
	SHUFPS $0x8c, X0, X4
	PXOR X4, X0
	PXOR X1, X0
	MOVUPS X0, (BX)
	ADDQ $16, BX
	RET

// Ensure your build environment and CPU support AVX before using this.
// This is a conceptual example to illustrate 256-bit XOR operations using AVX.

// func xorSlices(dst, src []uint64, len int)

TEXT ·xorSlices(SB), $0-56
    MOVQ dst+0(FP), SI         // Load pointer to dst slice
    MOVQ src+24(FP), DI        // Load pointer to src slice
    MOVQ n+32(FP), CX          // Load number of elements to process into CX

    // Calculate the number of 256-bit chunks (4 uint64 elements per chunk)
    SHRQ $2, CX                // Divide CX by 4 because we process 4 elements per iteration

loop:
    TESTQ CX, CX               // Test if the loop counter is zero
    JZ    done                 // If zero, we're done

    VMOVDQU (SI), Y0         // Load 256 bits from dst into YMM0
    VMOVDQU (DI), Y1         // Load 256 bits from src into YMM1
    VPXOR Y1, Y0, Y0     // Perform XOR operation between YMM1 and YMM0, result in YMM0
    VMOVDQU Y0, (SI)         // Store result back to dst from YMM0

    ADDQ $32, SI               // Advance pointers by 256 bits (32 bytes)
    ADDQ $32, DI
    DECQ CX                    // Decrement loop counter
    JNZ  loop                  // Continue if not done

done:
    VZEROUPPER                 // Clear upper part of YMM registers to avoid AVX-SSE transition penalty
    RET                        // Return
