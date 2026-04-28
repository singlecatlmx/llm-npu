#pragma once

#include <hexagon_types.h>
#include <stddef.h>

#define HMX_INLINE_ALWAYS inline __attribute__((unused, always_inline))

#define TILE              32
#define TILE_ELMS         (TILE * TILE)
#define TILE_BYTES        (TILE_ELMS * 2)    /* 2048 bytes per 32x32 f16 tile */
#define N_ITERS           1000

static inline __attribute__((always_inline)) void hmx_clear_acc(void) {
// #ifdef __hexagon__
    asm volatile("mxclracc.hf" ::: "memory");
// #endif
}

static inline __attribute__((always_inline)) void hmx_set_scales(const void *scales) {
// #ifdef __hexagon__
    asm volatile("bias = mxmem2(%0)" :: "r"(scales) : "memory");
// #endif
}

static inline __attribute__((always_inline)) void hmx_load_tiles(
    const void *act_tiles, const void *wt_tiles, uint32_t n_tiles)
{
// #ifdef __hexagon__
    uint32_t limit = n_tiles * TILE_BYTES - 1;
    asm volatile(
        "{ activation.hf = mxmem(%0, %1):deep\n"
        "  weight.hf = mxmem(%2, %3) }\n"
        :: "r"(act_tiles), "r"(limit), "r"(wt_tiles), "r"(limit)
        : "memory");
// #endif
}


static inline __attribute__((always_inline)) void hmx_store_acc(_Float16 *out) {
// #ifdef __hexagon__
    asm volatile(
        "cvt.hf = acc(%0)\n"
        "mxmem(%1, %2) = cvt\n"
        :: "r"(2), "r"(out), "r"(0)
        : "memory");
// #endif
}

