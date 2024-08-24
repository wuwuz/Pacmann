#include "textflag.h"

// Function signature: func L2DistanceSIMD(a, b *float32, n int) float32
TEXT ·L2DistanceSIMD(SB), NOSPLIT, $0-32
    MOVQ a+0(FP), AX              // Load address of a into AX
    MOVQ b+8(FP), BX              // Load address of b into BX
    MOVQ n+16(FP), CX             // Load length n into CX (number of floats, not chunks)

    VXORPS Y0, Y0, Y0             // Initialize Y0 to zero for accumulating sums

    XORQ DX, DX                   // Initialize loop counter DX to 0
    LOOP:
        VMOVUPS (AX)(DX*4), Y1    // Load 8 floats from a into Y1
        VMOVUPS (BX)(DX*4), Y2    // Load 8 floats from b into Y2

        VSUBPS Y2, Y1, Y1         // Y1 = Y1 - Y2
        VMULPS Y1, Y1, Y1         // Y1 = Y1 * Y1 (square differences)

        VADDPS Y1, Y0, Y0         // Y0 += Y1 (accumulate sums)

        ADDQ $8, DX               // Increment loop counter by 8 (processing 8 floats at a time)
        CMPQ DX, CX               // Compare loop counter with length
        JL LOOP                   // If counter < length, continue loop

    // Extract upper 128 bits to X1
    VEXTRACTF128 $1, Y0, X1

    // Add upper 128 bits to lower 128 bits
    VHADDPS X1, X0, X0 // VADDPS or VHADDPS?

    // Horizontal add of the 4 floats in X0
    VHADDPS X0, X0, X0
    VHADDPS X0, X0, X0

    MOVSS X0, ret+24(FP)          // Move the result to return value slot
    RET

// func InnerProduct(a *uint32, b *uint32, n int) uint32
TEXT ·InnerProduct(SB), NOSPLIT, $0-24
    MOVQ a+0(FP), SI         // Load pointer to a
    MOVQ b+8(FP), DI         // Load pointer to b
    MOVQ n+16(FP), CX        // Load length of vectors
    VPXORD Z0, Z0, Z0        // Clear Z0 for accumulation

    // Main loop using SIMD with ZMM registers
loop:
    VMOVDQU32 (SI), Z1       // Load 16 uint32 from a
    VMOVDQU32 (DI), Z2       // Load 16 uint32 from b
    VPMULLD Z1, Z2, Z3       // Multiply packed unsigned doubleword integers
    VPADDD Z3, Z0, Z0        // Add results to accumulator

    ADDQ $64, SI             // Advance a pointer by 64 bytes (16 uint32)
    ADDQ $64, DI             // Advance b pointer by 64 bytes (16 uint32)
    SUBQ $16, CX             // Decrement counter by 16
    JNZ loop                 // Continue if not zero

    // Sum up the elements in the SIMD accumulator
    VEXTRACTI32X8 $1, Z0, Y1 // Extract upper 256 bits
    VPADDD Y1, Y0, Y0        // Add upper 256 bits to lower 256 bits
    VEXTRACTI32X4 $1, Y0, X1 // Extract upper 128 bits
    PADDD X1, X0             // Add upper 128 bits to lower 128 bits
    PSHUFD $0xEE, X0, X1     // Shuffle to sum 64-bit halves
    PADDD X1, X0
    PSHUFD $0xE5, X0, X1     // Shuffle to sum 32-bit halves
    PADDD X1, X0

    MOVSS X0, ret+24(FP)     // Store result (32-bit) directly from XMM register
    RET
