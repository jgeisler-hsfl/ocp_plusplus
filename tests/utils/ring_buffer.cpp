// Unit tests for ocp::RingBuffer (include/ocp/utils/ring_buffer.hpp):
// fixed-capacity FIFO (drop-oldest) semantics, size/empty/full tracking,
// oldest->newest ordering across a wrap-around, reset, and mutable access
// through next()/at(i)/newest(). The stats tables (STATS_RING_PLAN.md) are
// the motivating consumer, so the wrap-around / stamping behaviour is the
// focus.

#include <cstdio>
#include <string>
#include <vector>

#include "ocp/utils/ring_buffer.hpp"

namespace
{

using ocp::RingBuffer;

int failures = 0;

void check(bool cond, const std::string& msg)
{
    if (!cond)
    {
        std::fprintf(stderr, "FAIL: %s\n", msg.c_str());
        ++failures;
    }
}

// One row mirrors a statistics row: a payload value plus a stamp that the
// producer writes into the slot handed back by next().
struct Row
{
    int value = 0;
    int stamp = -1;
};

void test_initial_state()
{
    RingBuffer<Row, 4> rb;
    check(rb.empty(), "fresh buffer empty");
    check(!rb.full(), "fresh buffer not full");
    check(rb.size() == 0, "fresh buffer size 0");
}

void test_fill_up_to_capacity()
{
    RingBuffer<int, 3> rb;
    for (int v = 1; v <= 3; ++v)
    {
        rb.next() = v;
    }
    check(rb.full(), "buffer full at capacity");
    check(rb.size() == 3, "size == capacity once full");
    check(rb.at(0) == 1 && rb.at(1) == 2 && rb.at(2) == 3,
          "oldest->newest order before wrap");
    check(rb.newest() == 3, "newest is the last write");
}

void test_drop_oldest_on_overflow()
{
    RingBuffer<int, 3> rb;
    for (int v = 1; v <= 5; ++v)
    {
        rb.next() = v;
    }
    // Writes 1..5 into a 3-slot ring keep 3,4,5 (1 and 2 evicted).
    check(rb.full(), "still full after overflow");
    check(rb.size() == 3, "size stays at capacity after overflow");
    check(rb.at(0) == 3 && rb.at(1) == 4 && rb.at(2) == 5,
          "drop-oldest keeps the three most recent values");
    check(rb.newest() == 5, "newest is the most recent write");
}

void test_ordering_across_wrap()
{
    // Enough writes that the head index wraps more than once.
    RingBuffer<int, 4> rb;
    for (int v = 1; v <= 9; ++v)
    {
        rb.next() = v;
    }
    // 1..9 into 4 slots keeps 6,7,8,9.
    check(rb.at(0) == 6 && rb.at(1) == 7 && rb.at(2) == 8 && rb.at(3) == 9,
          "oldest->newest order after a full wrap-around");
    check(rb.newest() == 9, "newest after wrap");
}

void test_stamping_into_next_slot()
{
    // Mirrors HpipmStatistics::next_row(): the producer writes a stamp into
    // the fresh slot, so each stored row carries its own sequence number.
    RingBuffer<Row, 3> rb;
    int seq = 0;
    for (int i = 0; i < 7; ++i)
    {
        Row& slot = rb.next();
        ++seq;
        slot.value = i;
        slot.stamp = seq;
    }
    // Oldest->newest should be writes 4,5,6 carrying stamps 5,6,7.
    check(rb.at(0).value == 4 && rb.at(0).stamp == 5,
          "oldest row keeps its own stamp");
    check(rb.at(1).value == 5 && rb.at(1).stamp == 6, "middle row stamp");
    check(rb.at(2).value == 6 && rb.at(2).stamp == 7, "newest row stamp");
}

void test_mutable_at_and_newest()
{
    RingBuffer<int, 3> rb;
    for (int v = 1; v <= 3; ++v)
    {
        rb.next() = v;
    }
    // Mutate through the non-const at() / newest() overloads.
    rb.at(0) = 10;
    rb.newest() = 30;
    check(rb.at(0) == 10, "non-const at() mutates the stored entry");
    check(rb.newest() == 30, "non-const newest() mutates the stored entry");
}

void test_reset()
{
    RingBuffer<int, 3> rb;
    for (int v = 1; v <= 5; ++v)
    {
        rb.next() = v;
    }
    check(rb.full(), "full before reset");
    rb.reset();
    check(rb.empty(), "empty after reset");
    check(rb.size() == 0, "size 0 after reset");
    // A fresh fill after reset must start clean (no stale entries).
    rb.next() = 100;
    check(rb.size() == 1 && rb.at(0) == 100, "first write after reset");
}

void test_single_slot_capacity()
{
    RingBuffer<int, 1> rb;
    rb.next() = 1;
    check(rb.full() && rb.size() == 1, "1-slot buffer fills on first write");
    rb.next() = 2;
    check(rb.at(0) == 2 && rb.newest() == 2,
          "1-slot buffer overwrites in place");
}

void test_capacity_constant()
{
    check(RingBuffer<Row, 8>::kCapacity == 8, "kCapacity is the extent");
}

}  // namespace

int main()
{
    test_initial_state();
    test_fill_up_to_capacity();
    test_drop_oldest_on_overflow();
    test_ordering_across_wrap();
    test_stamping_into_next_slot();
    test_mutable_at_and_newest();
    test_reset();
    test_single_slot_capacity();
    test_capacity_constant();

    if (failures == 0)
    {
        std::printf("All ring_buffer checks passed.\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed.\n", failures);
    return 1;
}
