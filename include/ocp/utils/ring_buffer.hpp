// include/ocp/utils/ring_buffer.hpp
//
// Fixed-capacity FIFO ring buffer with a compile-time extent (no heap
// allocation, no per-write zeroing). Once full, the next write evicts the
// oldest entry. Used by the per-iteration statistics tables to keep the
// most recent K records of a possibly longer iteration history
// (docs/plans/finished/STATS_RING_PLAN.md).

#pragma once

#include <array>
#include <cstddef>
#include <type_traits>

namespace ocp
{

/// Fixed-capacity FIFO (ring) buffer.
///
/// Stores at most `N` entries of `T`. When the buffer is full, the next
/// `next()` call overwrites the oldest entry (drop-oldest / FIFO policy).
/// Not thread-safe.
template <class T, std::size_t N>
class RingBuffer
{
    static_assert(N >= 1,
                  "ocp::RingBuffer: the capacity must be at least 1");
    static_assert(!std::is_same_v<T, void>,
                  "ocp::RingBuffer: T must be a complete type");
public:
    using value_type = T;
    static constexpr std::size_t kCapacity = N;

    /// Drop all entries (the buffer becomes empty). Existing elements are
    /// left as-is; only the bookkeeping is reset.
    void reset()
    {
        head_ = 0;
        size_ = 0;
    }

    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == N; }

    /// Number of valid entries: `min(total writes, N)`.
    std::size_t size() const { return size_; }

    /// The slot the next write goes to. When the buffer is full, the
    /// oldest entry is evicted first. The returned element is *not*
    /// zero-initialized; the caller fills it.
    T& next()
    {
        const std::size_t idx = head_;
        if (size_ < N)
        {
            ++size_;
        }
        head_ = (head_ + 1) % N;
        return data_[idx];
    }

    /// The i-th valid entry, oldest (i = 0) to newest
    /// (i = size() - 1). Precondition: `0 <= i < size()`.
    const T& at(std::size_t i) const
    {
        return data_[(head_ + N - size_ + i) % N];
    }
    T& at(std::size_t i)
    {
        return data_[(head_ + N - size_ + i) % N];
    }

    /// The most recent entry. Precondition: !empty().
    const T& newest() const { return at(size_ - 1); }
    T& newest() { return at(size_ - 1); }

private:
    std::array<T, N> data_{};
    std::size_t head_ = 0;  // next write slot (oldest entry when full)
    std::size_t size_ = 0;  // valid entries
};

}  // namespace ocp
