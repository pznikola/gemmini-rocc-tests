// See LICENSE for license details.

#ifndef COUNTER_H_
#define COUNTER_H_

#define DISABLE 0

// Highest incremental (event) counter code. Codes above this are external
// counters (counter_configure subtracts this offset and sets the external bit).
// Bumped (44 -> 50) to make room for the MX scale-load + serialization diagnostic
// events; the external-counter macros below are defined relative to it, so they shift
// automatically and keep mapping to the same CounterExternal indices.
#define INCREMENTAL_COUNTERS 50

// All existing Gemmini performance counters

#define MAIN_LD_CYCLES 1
#define MAIN_ST_CYCLES 2
#define MAIN_EX_CYCLES 3
#define MAIN_LD_ST_CYCLES 4
#define MAIN_LD_EX_CYCLES 5
#define MAIN_ST_EX_CYCLES 6
#define MAIN_LD_ST_EX_CYCLES 7

#define LOAD_DMA_WAIT_CYCLE 8
#define LOAD_ACTIVE_CYCLE 9
#define LOAD_SCRATCHPAD_WAIT_CYCLE 10

#define STORE_DMA_WAIT_CYCLE 11
#define STORE_ACTIVE_CYCLE 12
#define STORE_POOLING_CYCLE 13
#define STORE_SCRATCHPAD_WAIT_CYCLE 14

#define DMA_TLB_MISS_CYCLE 15
#define DMA_TLB_HIT_REQ 16
#define DMA_TLB_TOTAL_REQ 17

#define RDMA_ACTIVE_CYCLE 18
#define RDMA_TLB_WAIT_CYCLES 19
#define RDMA_TL_WAIT_CYCLES 20

#define WDMA_ACTIVE_CYCLE 21
#define WDMA_TLB_WAIT_CYCLES 22
#define WDMA_TL_WAIT_CYCLES 23

#define EXE_ACTIVE_CYCLE 24
#define EXE_FLUSH_CYCLE 25
#define EXE_CONTROL_Q_BLOCK_CYCLE 26
#define EXE_PRELOAD_HAZ_CYCLE 27
#define EXE_OVERLAP_HAZ_CYCLE 28

#define SCRATCHPAD_A_WAIT_CYCLE 29
#define SCRATCHPAD_B_WAIT_CYCLE 30
#define SCRATCHPAD_D_WAIT_CYCLE 31

#define ACC_A_WAIT_CYCLE 32
#define ACC_B_WAIT_CYCLE 33
#define ACC_D_WAIT_CYCLE 34

#define A_GARBAGE_CYCLES 35
#define B_GARBAGE_CYCLES 36
#define D_GARBAGE_CYCLES 37

#define IM2COL_MEM_CYCLES 38
#define IM2COL_ACTIVE_CYCLES 39
#define IM2COL_TRANSPOSER_WAIT_CYCLE 40

#define RESERVATION_STATION_FULL_CYCLES 41
#define RESERVATION_STATION_ACTIVE_CYCLES 42

#define LOOP_MATMUL_ACTIVE_CYCLES 43
#define TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES 44

// MX scale-load diagnostics (must match CounterFile.scala)
#define MX_SCALE_DMA_ACTIVE_CYCLE 45
#define MX_SCALE_DMA_SOLO_CYCLE 46

// MX serialization diagnostics (must match CounterFile.scala)
#define MX_DBG_WAIT_CMD_CYCLE 47
#define MX_DBG_ENQ_NOT_READY_CYCLE 48
#define MX_DBG_REQ_STALL_CYCLE 49
#define MX_DBG_DRAINING_CYCLE 50

#define RESERVATION_STATION_LD_COUNT (INCREMENTAL_COUNTERS + 1)
#define RESERVATION_STATION_ST_COUNT (INCREMENTAL_COUNTERS + 2)
#define RESERVATION_STATION_EX_COUNT (INCREMENTAL_COUNTERS + 3)

#define RDMA_BYTES_REC (INCREMENTAL_COUNTERS + 4)
#define WDMA_BYTES_SENT (INCREMENTAL_COUNTERS + 5)

#define RDMA_TOTAL_LATENCY (INCREMENTAL_COUNTERS + 6)
#define WDMA_TOTAL_LATENCY (INCREMENTAL_COUNTERS + 7)

#endif
