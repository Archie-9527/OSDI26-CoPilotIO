#pragma once

// CPU publishes a CQ-ordered table independently of which GPU thread consumes
// each CID notification. Reclaim all published completions, including those
// owned by other threads; waiting for per-CID dequeue order can deadlock a batch.
inline __device__ void batch_cq_doorbell(nvm_queue_t* cq, uint32_t head) {
    uint32_t db = head & cq->qs_minus_1;
    __threadfence_system();
    asm volatile("st.mmio.relaxed.sys.global.u32 [%0], %1;"
                 :: "l"(cq->db), "r"(db) : "memory");
}

inline __device__ void drain_batch_cq(QueuePair* qp, volatile copilot_cq_table* table) {
    auto cq = &qp->cq;
    auto sq = &qp->sq;
    // Lock and counters live in device memory, even when the table is in host RAM.
    while (cq->head_lock.fetch_or(LOCKED, cuda::std::memory_order_acquire) == LOCKED)
        __nanosleep(64);
    uint32_t head = cq->head.load(cuda::std::memory_order_relaxed);
    uint32_t count = 0;
    uint16_t last_sq_head = 0;
    while (table->ready[head & cq->qs_minus_1]) {
        uint32_t pos = head & cq->qs_minus_1;
        __threadfence_system();
        last_sq_head = table->entries[pos].sq_head;
        table->ready[pos] = 0;
        ++head;
        ++count;
    }
    if (count) {
        uint32_t old_sq_head = sq->head.load(cuda::std::memory_order_relaxed);
        uint32_t move = (uint32_t(last_sq_head) - old_sq_head) & sq->qs_minus_1;
        for (uint32_t i = 0; i < move; ++i)
            sq->tickets[(old_sq_head + i) & sq->qs_minus_1].val.fetch_add(1, cuda::std::memory_order_relaxed);
        sq->head.fetch_add(move, cuda::std::memory_order_release);
        cq->tail.fetch_add(count, cuda::std::memory_order_relaxed);
        cq->head.store(head, cuda::std::memory_order_release);
        // Clear table flags before allowing the CPU/controller to reuse slots.
        batch_cq_doorbell(cq, head);
    }
    cq->head_lock.store(UNLOCKED, cuda::std::memory_order_release);
}
