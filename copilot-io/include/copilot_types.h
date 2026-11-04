#ifndef COPILOT_TYPES_H
#define COPILOT_TYPES_H

#include <cstdint>

// Shared CPU/GPU notification layout. Keep this independent of CUDA runtime,
// GDRCopy and polling threads so device headers can include complete types.
struct copilot_notify_entry {
    volatile uint32_t ready;
    uint32_t pos;
    uint32_t loc;
    uint32_t cq_head;
    uint32_t status; // NVMe status without phase; survives reuse of the CQ slot.
};

#define COPILOT_NUM_CIDS 65536
#define COPILOT_NOTIFY_BYTES (COPILOT_NUM_CIDS * sizeof(copilot_notify_entry))

#define COPILOT_CQ_MAX_DEPTH 4096

struct copilot_cq_table_entry {
    uint16_t cid;
    uint16_t sq_head;
};

struct copilot_cq_table {
    volatile uint32_t ready[COPILOT_CQ_MAX_DEPTH];
    copilot_cq_table_entry entries[COPILOT_CQ_MAX_DEPTH];
    volatile uint32_t drain_lock;
    volatile uint32_t gpu_head;
};

#endif /* COPILOT_TYPES_H */
