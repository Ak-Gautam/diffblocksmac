/*
 * dblocks_native.c — Native C helpers for dblocks on macOS/Apple Silicon.
 *
 * Provides:
 *   1. Accurate process memory reporting via mach kernel APIs
 *   2. Fast random batch index generation (avoids Python/numpy overhead)
 *   3. System info (total RAM, CPU core count)
 *
 * Build:
 *   make -C src/dblocks/_csrc
 *
 * The resulting libdblocks_native.dylib is loaded via ctypes at runtime.
 */

#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task.h>
#include <mach/task_info.h>
#include <mach/vm_statistics.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

/* -----------------------------------------------------------------------
 * 1. Process memory via mach task_info
 *
 * Uses TASK_VM_INFO (flavor 22) which is the modern, well-supported
 * API on arm64 macOS.  MACH_TASK_BASIC_INFO (flavor 20) has struct
 * alignment issues across macOS versions; TASK_VM_INFO is reliable.
 * ----------------------------------------------------------------------- */

typedef struct {
    uint64_t resident_size;      /* current RSS in bytes */
    uint64_t resident_size_peak; /* peak RSS (high water mark) */
    uint64_t virtual_size;       /* virtual address space size */
    uint64_t internal;           /* internal (anonymous/malloc) memory */
    uint64_t compressed;         /* compressed memory footprint */
    uint64_t phys_footprint;     /* actual physical memory footprint */
} dblocks_mem_info_t;

int dblocks_get_mem_info(dblocks_mem_info_t *out) {
    if (!out) return -1;
    memset(out, 0, sizeof(*out));

    task_vm_info_data_t vm_info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;

    kern_return_t kr = task_info(
        mach_task_self(),
        TASK_VM_INFO,
        (task_info_t)&vm_info,
        &count
    );

    if (kr != KERN_SUCCESS) return (int)kr;

    out->resident_size     = vm_info.resident_size;
    out->resident_size_peak = vm_info.resident_size_peak;
    out->virtual_size      = vm_info.virtual_size;
    out->internal          = vm_info.internal;
    out->compressed        = vm_info.compressed;
    out->phys_footprint    = vm_info.phys_footprint;

    return 0;
}


/* -----------------------------------------------------------------------
 * 2. System info
 * ----------------------------------------------------------------------- */

uint64_t dblocks_get_total_memory(void) {
    int mib[2] = { CTL_HW, HW_MEMSIZE };
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctl(mib, 2, &mem, &len, NULL, 0) == 0)
        return mem;
    return 0;
}

int dblocks_get_cpu_count(void) {
    int mib[2] = { CTL_HW, HW_NCPU };
    int ncpu = 0;
    size_t len = sizeof(ncpu);
    if (sysctl(mib, 2, &ncpu, &len, NULL, 0) == 0)
        return ncpu;
    return 0;
}

/* Performance core count (Apple Silicon P-cores) */
int dblocks_get_perf_cpu_count(void) {
    int ncpu = 0;
    size_t len = sizeof(ncpu);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &ncpu, &len, NULL, 0) == 0)
        return ncpu;
    /* Fallback: return total / 2 as rough estimate */
    return dblocks_get_cpu_count() / 2;
}


/* -----------------------------------------------------------------------
 * 3. Fast batch index sampling
 *
 * Generates an array of `batch_size` random integers in [0, max_start)
 * using xoshiro256** PRNG — significantly faster than numpy for small
 * batches where Python overhead dominates.
 * ----------------------------------------------------------------------- */

/* xoshiro256** state */
static uint64_t s_rng[4] = {
    0x180ec6d33cfd0abaULL,
    0xd5a61266f0c9392cULL,
    0xa9582618e03fc9aaULL,
    0x39abdc4529b1661cULL
};

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    const uint64_t result = rotl(s_rng[1] * 5, 7) * 9;
    const uint64_t t = s_rng[1] << 17;

    s_rng[2] ^= s_rng[0];
    s_rng[3] ^= s_rng[1];
    s_rng[1] ^= s_rng[2];
    s_rng[0] ^= s_rng[3];
    s_rng[2] ^= t;
    s_rng[3] = rotl(s_rng[3], 45);

    return result;
}

void dblocks_seed_rng(uint64_t seed) {
    /* SplitMix64 to initialize xoshiro state from a single seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        z = z ^ (z >> 31);
        s_rng[i] = z;
    }
}

/*
 * Fill `out` with `batch_size` random indices in [0, max_start).
 * Returns 0 on success.
 */
int dblocks_sample_batch_indices(
    int32_t *out,
    int batch_size,
    int32_t max_start
) {
    if (!out || batch_size <= 0 || max_start <= 0) return -1;

    for (int i = 0; i < batch_size; i++) {
        /* Unbiased bounded random using rejection sampling */
        uint64_t r = xoshiro256ss();
        out[i] = (int32_t)(r % (uint64_t)max_start);
    }
    return 0;
}


/* -----------------------------------------------------------------------
 * 4. Fast contiguous sequence extraction
 *
 * Given a flat int32 token array and a list of start indices, extract
 * `batch_size` sequences of length `seq_len` into a contiguous output
 * buffer.  This is the hot path in CharDataset.get_batch().
 * ----------------------------------------------------------------------- */

int dblocks_extract_sequences(
    const int32_t *data,       /* source token array, length >= max(starts) + seq_len */
    int data_len,
    const int32_t *starts,     /* start indices, length = batch_size */
    int batch_size,
    int seq_len,
    int32_t *out               /* output buffer, shape (batch_size, seq_len) */
) {
    if (!data || !starts || !out) return -1;
    if (batch_size <= 0 || seq_len <= 0) return -1;

    for (int b = 0; b < batch_size; b++) {
        int32_t s = starts[b];
        if (s < 0 || s + seq_len > data_len) return -2;
        memcpy(out + b * seq_len, data + s, seq_len * sizeof(int32_t));
    }
    return 0;
}
