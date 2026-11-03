// CUDA implementation of the bdcudatag detector pipeline.
//
// Stage overview (all on the GPU, one stream):
//   1. per-8x8-tile min/max threshold                 (k_tile_threshold)
//   2. bit-packed binarization, 1 = white             (k_binarize)
//   3. run-length extraction per row, one block/row   (k_find_runs)
//   4. union-find merge of runs (8-connected)         (k_merge_runs)
//   5. component area / bounding box statistics       (k_region_stats)
//   6. tag-like component filtering                   (k_collect_candidates)
//   7. contour trace, quad fit, subpixel corner refine(k_fit_quads)
//   8. four-rotation sampling + Hamming decode        (k_decode_quads)
//   9. de-duplicate and sort detections               (k_finalize)
//
// The host path is captured as a CUDA graph at creation, so a frame costs one
// graph launch rather than nine kernel launches plus two memcpys.

#include "pipeline.hpp"
#include "tag_families.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace bdcudatag::detail {
namespace {

constexpr int kTile = 8;
constexpr int kMinTileContrast = 12;
constexpr int kFindRunsThreads = 256;
constexpr int kMergeWarps = 4;
constexpr int kMergeThreads = kMergeWarps * 32;
constexpr int kMinContourCapacity = 2048;
constexpr int kMaxContourCapacity = 16384;
// Contours up to this length are copied to shared memory before polygon
// simplification, which reads the same points many times.
constexpr int kSharedContourPoints = 8192;

struct PipelineArgs {
    const uint8_t* gray = nullptr;
    int width = 0;
    int height = 0;
    int stride = 0;
    int tile_w = 0;
    int tile_h = 0;
    int words = 0;
    int row_cap = 0;
    int total_slots = 0;
    int min_cluster_pixels = 0;
    int min_tag_pixels = 0;
    int max_candidates = 0;
    int contour_cap = 0;
    int bits_per_side = 0;
    int dim = 0;
    int max_hamming = 0;
    int code_count = 0;
    int refine_edges = 0;
    int max_detections = 0;
    float min_decision_margin = 0.0f;
    float sharpening = 0.0f;
    uint8_t* tile_thresh = nullptr;
    uint8_t* tile_active = nullptr;
    uint8_t* row_active = nullptr;
    int* row_start = nullptr;
    int* row_end = nullptr;
    uint32_t* bits = nullptr;
    uint32_t* row_count = nullptr;
    uint32_t* runs = nullptr;
    uint32_t* parent = nullptr;
    uint32_t* area = nullptr;
    uint32_t* bbox = nullptr;
    uint32_t* candidates = nullptr;
    uint32_t* counts = nullptr;  // [0] candidates, [1] detections, [2] runs, [3] quads
    uint32_t* contour = nullptr;
    float* quads = nullptr;
    bd_detection* detections = nullptr;
    uint64_t* codes = nullptr;
};

// Tests bit `local` (0..33) of a 32-bit word pair covering a run of pixels.
__device__ __forceinline__ bool black_bit(uint32_t lo, uint32_t hi, int local) {
    const uint32_t word = local < 32 ? lo : hi;
    return ((word >> (local & 31)) & 1u) == 0u;
}

// ---------------------------------------------------------------------------
// 1. Tile threshold
// ---------------------------------------------------------------------------
__global__ void k_tile_threshold(const uint8_t* __restrict__ gray, int width, int height,
                                 int stride, uint8_t* __restrict__ thresh,
                                 uint8_t* __restrict__ active, int tile_w, int tile_h) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tile_count = tile_w * tile_h;
    if (idx >= tile_count) return;

    const int tx = idx % tile_w;
    const int ty = idx / tile_w;
    const int x0 = tx * kTile;
    const int y0 = ty * kTile;
    const int x1 = min(x0 + kTile, width);
    const int y1 = min(y0 + kTile, height);

    int mn = 255;
    int mx = 0;
    for (int y = y0; y < y1; ++y) {
        const uint8_t* row = gray + y * stride;
        for (int x = x0; x < x1; ++x) {
            const int v = row[x];
            mn = min(mn, v);
            mx = max(mx, v);
        }
    }

    // Flat tiles fall back to a global threshold. Using the local midpoint
    // there would classify a uniform white tile as black (or vice versa)
    // depending on rounding, which breaks the tag border.
    int t = 127;
    if (mx - mn >= kMinTileContrast) t = (mn + mx + 1) >> 1;
    thresh[idx] = static_cast<uint8_t>(t);
    active[idx] = (mx - mn >= kMinTileContrast) ? 1u : 0u;
}

// Coarse ROI: a tile row with any contrast is searched only between the
// leftmost and rightmost contrasting tile (plus a halo). Every row that
// crosses a tag also crosses the tag's left and right border, so the whole
// tag is always inside its row's band; the flat background is skipped.
__global__ void k_build_row_roi(const uint8_t* __restrict__ active, int tile_w, int tile_h,
                                uint8_t* __restrict__ row_active, int* __restrict__ row_start,
                                int* __restrict__ row_end, uint32_t* __restrict__ counts) {
    __shared__ int partial_top[256];
    __shared__ int partial_bottom[256];
    int local_top = tile_h;
    int local_bottom = -1;
    const int total = tile_w * tile_h;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (active[i]) {
            const int ty = i / tile_w;
            local_top = min(local_top, ty);
            local_bottom = max(local_bottom, ty);
        }
    }
    partial_top[threadIdx.x] = local_top;
    partial_bottom[threadIdx.x] = local_bottom;
    __syncthreads();
    if (threadIdx.x == 0) {
        int top = tile_h;
        int bottom = -1;
        for (int i = 0; i < blockDim.x; ++i) {
            top = min(top, partial_top[i]);
            bottom = max(bottom, partial_bottom[i]);
        }
        partial_top[0] = top;
        partial_bottom[0] = bottom;
    }
    __syncthreads();
    const int s_top = partial_top[0];
    const int s_bottom = partial_bottom[0];

    // Everything between the first and last contrasting tile row is searched.
    // Closing the vertical gaps keeps a tag whole even when a tile row inside
    // its black border happens to be flat (for example a tile-aligned edge).
    for (int ty = threadIdx.x; ty < tile_h; ty += blockDim.x) {
        if (s_bottom < 0 || ty < s_top - 1 || ty > s_bottom + 1) {
            row_active[ty] = 0;
            row_start[ty] = 0;
            row_end[ty] = -1;
            continue;
        }
        row_active[ty] = 1;
        atomicAdd(&counts[4], 1u);

        const uint8_t* row = active + ty * tile_w;
        int mn = tile_w;
        int mx = -1;
        for (int x = 0; x < tile_w; ++x) {
            if (row[x]) {
                mn = min(mn, x);
                mx = max(mx, x);
            }
        }
        if (mx < 0) {
            row_start[ty] = 0;
            row_end[ty] = tile_w - 1;
        } else {
            row_start[ty] = max(0, mn - 4);
            row_end[ty] = min(tile_w - 1, mx + 4);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Binarize into a 1-bit-per-pixel white mask.
// ---------------------------------------------------------------------------
__global__ void k_binarize(const uint8_t* __restrict__ gray, int width, int height,
                           int stride, const uint8_t* __restrict__ thresh,
                           const uint8_t* __restrict__ row_active,
                           const int* __restrict__ row_start,
                           const int* __restrict__ row_end, int tile_w,
                           uint32_t* __restrict__ bits, int words) {
    const int warp_in_block = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wx = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    const int y = blockIdx.y;
    if (wx >= words || y >= height) return;

    const int ty = y >> 3;
    const bool row_on = row_active[ty] != 0;
    const int tile0 = wx << 2;
    const bool overlap = row_on && tile0 <= row_end[ty] && tile0 + 3 >= row_start[ty];
    if (!overlap) {
        if (lane == 0) bits[y * words + wx] = 0xffffffffu;
        return;
    }

    const int x = (wx << 5) + lane;
    bool white = true;
    if (x < width) {
        const int t = thresh[ty * tile_w + (x >> 3)];
        white = gray[y * stride + x] > t;
    }
    const uint32_t mask = __ballot_sync(0xffffffffu, white);
    if (lane == 0) bits[y * words + wx] = mask;
}

// ---------------------------------------------------------------------------
// Union-find helpers. Labels only ever move to smaller run-slot indices, so
// the lock-free atomicMin hooking is deadlock free and path halving is safe
// even with concurrent readers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t uf_find(uint32_t* parent, uint32_t x) {
    for (;;) {
        const uint32_t p = parent[x];
        if (p == x) return x;
        const uint32_t gp = parent[p];
        parent[x] = gp;
        x = gp;
    }
}

__device__ __forceinline__ void uf_union(uint32_t* parent, uint32_t a, uint32_t b) {
    a = uf_find(parent, a);
    b = uf_find(parent, b);
    while (a != b) {
        if (a > b) {
            const uint32_t t = a;
            a = b;
            b = t;
        }
        const uint32_t old = atomicMin(&parent[b], a);
        if (old == a) break;
        b = uf_find(parent, old);
    }
}

// ---------------------------------------------------------------------------
// 3. Runs of black pixels per row.  A run may be split at a 32-pixel chunk
// boundary; k_merge_runs rejoins adjacent fragments in the same row.
// ---------------------------------------------------------------------------
__global__ void k_find_runs(const uint32_t* __restrict__ bits, int width, int words,
                            int row_cap, uint32_t* __restrict__ row_count,
                            uint32_t* __restrict__ runs, uint32_t* __restrict__ parent,
                            uint32_t* __restrict__ area, uint32_t* __restrict__ bbox,
                            uint32_t* __restrict__ counts,
                            const uint8_t* __restrict__ row_active) {
    const int y = blockIdx.x;
    if (!row_active[y >> 3]) {
        if (threadIdx.x == 0) row_count[y] = 0;
        return;
    }

    extern __shared__ uint32_t smem[];
    uint32_t* chunk_counts = smem;
    uint32_t* chunk_offsets = smem + words;

    const int warps = blockDim.x >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const uint32_t* row_bits = bits + y * words;

    for (int c = warp; c < words; c += warps) {
        const uint32_t mask = row_bits[c];
        const uint32_t prev = (c == 0) ? 1u : ((row_bits[c - 1] >> 31) & 1u);
        // Also start a fragment when the previous word ends black: runs may
        // span 32-pixel chunks and k_merge_runs rejoins the fragments.
        const uint32_t starts = ~mask & ((mask << 1) | (c == 0 ? 1u : (1u - prev)));
        if (lane == 0) chunk_counts[c] = __popc(starts);
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        uint32_t off = 0;
        for (int c = 0; c < words; ++c) {
            const uint32_t n = chunk_counts[c];
            chunk_offsets[c] = off;
            off += n;
        }
        const uint32_t total = min(off, static_cast<uint32_t>(row_cap));
        row_count[y] = total;
        atomicAdd(&counts[2], total);
    }
    __syncthreads();

    for (int c = warp; c < words; c += warps) {
        const uint32_t mask = row_bits[c];
        const uint32_t prev = (c == 0) ? 1u : ((row_bits[c - 1] >> 31) & 1u);
        // Also start a fragment when the previous word ends black: runs may
        // span 32-pixel chunks and k_merge_runs rejoins the fragments.
        const uint32_t starts = ~mask & ((mask << 1) | (c == 0 ? 1u : (1u - prev)));
        if ((starts & (1u << lane)) == 0u) continue;

        const int s = lane;
        const int rank = __popc(starts & ((1u << s) - 1u));
        const int x0 = (c << 5) + s;
        if (x0 >= width) continue;
        const int last = min(31, width - 1 - (c << 5));
        const int next_white = __ffs(mask >> s);
        int e = next_white ? (s + next_white - 2) : last;
        e = min(e, last);
        // x0 is global; e is chunk-local. Store both in image coordinates.
        const int x1 = (c << 5) + e;

        const int index = static_cast<int>(chunk_offsets[c]) + rank;
        if (index >= row_cap) continue;
        const uint32_t id = static_cast<uint32_t>(y * row_cap + index);
        runs[id] = (static_cast<uint32_t>(x0) & 0xffffu) |
                   ((static_cast<uint32_t>(x1) & 0xffffu) << 16);
        parent[id] = id;
        area[id] = 0;
        bbox[id * 4 + 0] = 0xffffffffu;
        bbox[id * 4 + 1] = 0xffffffffu;
        bbox[id * 4 + 2] = 0;
        bbox[id * 4 + 3] = 0;
    }
}

// ---------------------------------------------------------------------------
// 4. Merge fragments: same-row neighbours and overlapping runs in row y-1.
// ---------------------------------------------------------------------------
__global__ void k_merge_runs(const uint32_t* __restrict__ runs,
                             const uint32_t* __restrict__ row_count,
                             uint32_t* __restrict__ parent, int row_cap, int height,
                             const uint8_t* __restrict__ row_active) {
    const int y = (blockIdx.x * kMergeWarps) + (threadIdx.x >> 5);
    if (y <= 0 || y >= height) return;
    if (!row_active[y >> 3]) return;
    const int lane = threadIdx.x & 31;

    const int n = static_cast<int>(row_count[y]);
    const int prev_n = static_cast<int>(row_count[y - 1]);
    const uint32_t* cur_runs = runs + static_cast<size_t>(y) * row_cap;
    const uint32_t* prev_runs = runs + static_cast<size_t>(y - 1) * row_cap;
    const uint32_t cur_base = static_cast<uint32_t>(y * row_cap);
    const uint32_t prev_base = static_cast<uint32_t>((y - 1) * row_cap);

    for (int i = lane; i < n; i += 32) {
        const uint32_t cur = cur_runs[i];
        const int x0 = static_cast<int>(cur & 0xffffu);
        const int x1 = static_cast<int>(cur >> 16);

        if (i > 0) {
            const uint32_t p = cur_runs[i - 1];
            const int px1 = static_cast<int>(p >> 16);
            if (px1 + 1 >= x0) uf_union(parent, cur_base + i, cur_base + i - 1);
        }
        if (prev_n == 0) continue;

        // First run in the previous row with x1 >= x0 - 1.
        int lo = 0;
        int hi = prev_n - 1;
        int first = prev_n;
        while (lo <= hi) {
            const int mid = (lo + hi) >> 1;
            if (static_cast<int>(prev_runs[mid] >> 16) >= x0 - 1) {
                first = mid;
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }
        for (int k = first; k < prev_n; ++k) {
            const uint32_t pr = prev_runs[k];
            const int px0 = static_cast<int>(pr & 0xffffu);
            if (px0 > x1 + 1) break;
            uf_union(parent, cur_base + i, prev_base + k);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Component stats
// ---------------------------------------------------------------------------
__global__ void k_region_stats(const uint32_t* __restrict__ runs,
                               const uint32_t* __restrict__ row_count,
                               uint32_t* __restrict__ parent, uint32_t* __restrict__ area,
                               uint32_t* __restrict__ bbox, int row_cap, int height,
                               const uint8_t* __restrict__ row_active) {
    const int y = (blockIdx.x * kMergeWarps) + (threadIdx.x >> 5);
    if (y >= height || !row_active[y >> 3]) return;
    const int lane = threadIdx.x & 31;
    const int n = static_cast<int>(row_count[y]);
    const uint32_t* row_runs = runs + static_cast<size_t>(y) * row_cap;
    const uint32_t base = static_cast<uint32_t>(y * row_cap);

    for (int i = lane; i < n; i += 32) {
        const uint32_t idx = base + static_cast<uint32_t>(i);
        const uint32_t root = uf_find(parent, idx);
        const uint32_t run = row_runs[i];
        const int x0 = static_cast<int>(run & 0xffffu);
        const int x1 = static_cast<int>(run >> 16);
        atomicAdd(&area[root], static_cast<uint32_t>(x1 - x0 + 1));
        atomicMin(&bbox[root * 4 + 0], static_cast<uint32_t>(x0));
        atomicMin(&bbox[root * 4 + 1], static_cast<uint32_t>(y));
        atomicMax(&bbox[root * 4 + 2], static_cast<uint32_t>(x1));
        atomicMax(&bbox[root * 4 + 3], static_cast<uint32_t>(y));
    }
}

// ---------------------------------------------------------------------------
// 6. Filter components that could be the black border square of a tag.
// ---------------------------------------------------------------------------
__global__ void k_collect_candidates(const uint32_t* __restrict__ row_count,
                                     const uint32_t* __restrict__ parent,
                                     const uint32_t* __restrict__ area,
                                     const uint32_t* __restrict__ bbox, int row_cap,
                                     int height, int width, int min_area, int min_tag_px,
                                     int max_candidates, uint32_t* __restrict__ candidates,
                                     uint32_t* __restrict__ counts,
                                     const uint8_t* __restrict__ row_active) {
    const int y = (blockIdx.x * kMergeWarps) + (threadIdx.x >> 5);
    if (y >= height || !row_active[y >> 3]) return;
    const int lane = threadIdx.x & 31;
    const int n = static_cast<int>(row_count[y]);
    const uint32_t base = static_cast<uint32_t>(y * row_cap);

    for (int i = lane; i < n; i += 32) {
        const uint32_t idx = base + static_cast<uint32_t>(i);
        if (parent[idx] != idx) continue;

        const int a = static_cast<int>(area[idx]);
        if (a < min_area) continue;

        const int x0 = static_cast<int>(bbox[idx * 4 + 0]);
        const int y0 = static_cast<int>(bbox[idx * 4 + 1]);
        const int x1 = static_cast<int>(bbox[idx * 4 + 2]);
        const int y1 = static_cast<int>(bbox[idx * 4 + 3]);
        const int bw = x1 - x0 + 1;
        const int bh = y1 - y0 + 1;
        if (bw < min_tag_px || bh < min_tag_px) continue;
        if (x0 <= 0 || y0 <= 0 || x1 >= width - 1 || y1 >= height - 1) continue;
        if (bw > 6 * bh || bh > 6 * bw) continue;

        const float fill = static_cast<float>(a) / static_cast<float>(bw * bh);
        if (fill < 0.18f || fill > 0.95f) continue;

        const uint32_t slot = atomicAdd(&counts[0], 1u);
        if (slot < static_cast<uint32_t>(max_candidates)) candidates[slot] = idx;
    }
}

// ---------------------------------------------------------------------------
// 7. Contour tracing, polygon simplification and subpixel corner refinement.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t pack_point(int x, int y) {
    return (static_cast<uint32_t>(x) & 0xffffu) | (static_cast<uint32_t>(y) << 16);
}

__device__ int find_region_start(const uint32_t* runs, const uint32_t* row_count,
                                 uint32_t* parent, int row_cap, uint32_t root, int bx0,
                                 int by0, int bx1, int width, int* out_y) {
    const int n = static_cast<int>(row_count[by0]);
    const uint32_t* row_runs = runs + static_cast<size_t>(by0) * row_cap;
    int lo = 0;
    int hi = n - 1;
    int first = n;
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (static_cast<int>(row_runs[mid] >> 16) >= bx0) {
            first = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }
    const uint32_t base = static_cast<uint32_t>(by0 * row_cap);
    for (int k = first; k < n; ++k) {
        const uint32_t run = row_runs[k];
        const int x0 = static_cast<int>(run & 0xffffu);
        const int x1 = static_cast<int>(run >> 16);
        if (x0 > bx1) break;
        if (x1 < bx0) continue;
        if (uf_find(parent, base + k) == root) {
            *out_y = by0;
            return max(x0, bx0);
        }
    }
    return -1;
}

// Moore-neighbour trace of the outer boundary, clockwise in image coordinates.
// Returns the number of points, or -1 if the trace was truncated.
//
// A step only needs the 3x3 neighbourhood, so the three rows around the current
// pixel are loaded as word pairs once and every neighbour is a register test.
// The straightforward version issued up to eight guarded dependent loads per
// step, which cost about a thousand cycles per contour point.
__device__ int trace_contour(const uint32_t* bits, int words, int width, int height, int sx,
                             int sy, uint32_t* out, int capacity) {
    const int dx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    const int dy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    int px = sx;
    int py = sy;
    int bdir = 4;  // west neighbour is background at the topmost-leftmost pixel
    int n = 0;
    for (int step = 0; step < capacity; ++step) {
        out[n++] = pack_point(px, py);
        if (px < 1 || py < 1 || px >= width - 1 || py >= height - 1) return 0;

        const int left_word = (px - 1) >> 5;
        const int right_word = (px + 1) >> 5;
        const int lx = px - (left_word << 5);
        const uint32_t* top = bits + static_cast<size_t>(py - 1) * words;
        const uint32_t* mid = bits + static_cast<size_t>(py) * words;
        const uint32_t* bot = bits + static_cast<size_t>(py + 1) * words;
        uint32_t tl, tr, ml, mr, bl, br;
        if (left_word == right_word) {
            // The 3x3 neighbourhood fits in one word per row almost always.
            tl = tr = top[left_word];
            ml = mr = mid[left_word];
            bl = br = bot[left_word];
        } else {
            tl = top[left_word];
            tr = top[right_word];
            ml = mid[left_word];
            mr = mid[right_word];
            bl = bot[left_word];
            br = bot[right_word];
        }

        const uint32_t black_mask =
            (static_cast<uint32_t>(black_bit(ml, mr, lx + 1)) << 0) |
            (static_cast<uint32_t>(black_bit(bl, br, lx + 1)) << 1) |
            (static_cast<uint32_t>(black_bit(bl, br, lx)) << 2) |
            (static_cast<uint32_t>(black_bit(bl, br, lx - 1)) << 3) |
            (static_cast<uint32_t>(black_bit(ml, mr, lx - 1)) << 4) |
            (static_cast<uint32_t>(black_bit(tl, tr, lx - 1)) << 5) |
            (static_cast<uint32_t>(black_bit(tl, tr, lx)) << 6) |
            (static_cast<uint32_t>(black_bit(tl, tr, lx + 1)) << 7);

        // Scan clockwise from the direction after the backtrack: rotate the
        // 8-bit mask so bit 0 is that direction, then take the first set bit.
        const int start = (bdir + 1) & 7;
        const uint32_t scan =
            ((black_mask >> start) | (black_mask << (8 - start))) & 0xffu;
        const int first = __ffs(scan);
        if (first == 0) return 0;
        const int dir = (bdir + first) & 7;
        bdir = (dir + 4) & 7;
        px += dx[dir];
        py += dy[dir];
        if (px == sx && py == sy) return n;
    }
    return -1;
}

__device__ __forceinline__ void contour_point(const uint32_t* contour, int n, int index,
                                              float* x, float* y) {
    const int m = (index == n) ? 0 : index;
    const uint32_t p = contour[m];
    *x = static_cast<float>(p & 0xffffu);
    *y = static_cast<float>(p >> 16);
}

__device__ float point_segment_distance(float px, float py, float ax, float ay, float bx,
                                        float by) {
    const float vx = bx - ax;
    const float vy = by - ay;
    const float wx = px - ax;
    const float wy = py - ay;
    const float len2 = vx * vx + vy * vy;
    float t = 0.0f;
    if (len2 > 1e-6f) t = fminf(fmaxf((wx * vx + wy * vy) / len2, 0.0f), 1.0f);
    const float ex = wx - t * vx;
    const float ey = wy - t * vy;
    return sqrtf(ex * ex + ey * ey);
}

__device__ bool simplify_to_quad(const uint32_t* contour, int n, float tolerance,
                                 float quad[4][2]) {
    if (n < 8) return false;

    float x0, y0;
    contour_point(contour, n, 0, &x0, &y0);
    int farthest = 1;
    float best_d2 = -1.0f;
    for (int m = 1; m < n; ++m) {
        float x, y;
        contour_point(contour, n, m, &x, &y);
        const float dx = x - x0;
        const float dy = y - y0;
        const float d2 = dx * dx + dy * dy;
        if (d2 > best_d2) {
            best_d2 = d2;
            farthest = m;
        }
    }
    if (best_d2 < 16.0f) return false;

    constexpr int kMaxVertices = 128;
    int poly[kMaxVertices];
    int np = 0;
    poly[np++] = 0;

    int stack_i[kMaxVertices];
    int stack_j[kMaxVertices];
    for (int seg = 0; seg < 2; ++seg) {
        int sp = 0;
        stack_i[sp] = (seg == 0) ? 0 : farthest;
        stack_j[sp] = (seg == 0) ? farthest : n;
        ++sp;

        while (sp > 0) {
            --sp;
            const int i = stack_i[sp];
            const int j = stack_j[sp];
            if (j <= i) continue;

            float ax, ay, bx, by;
            contour_point(contour, n, i, &ax, &ay);
            contour_point(contour, n, j, &bx, &by);
            const float dx = bx - ax;
            const float dy = by - ay;
            const float len2 = dx * dx + dy * dy;
            // One reciprocal square root per segment instead of one per point;
            // this loop is the hot spot of the whole fit.
            const float inv_len = len2 > 1e-6f ? rsqrtf(len2) : 0.0f;

            int k = -1;
            float dmax = 0.0f;
            for (int m = i + 1; m < j; ++m) {
                float px, py;
                contour_point(contour, n, m, &px, &py);
                float d;
                if (inv_len == 0.0f) {
                    const float ex = px - ax;
                    const float ey = py - ay;
                    d = sqrtf(ex * ex + ey * ey);
                } else {
                    d = fabsf((px - ax) * dy - (py - ay) * dx) * inv_len;
                }
                if (d > dmax) {
                    dmax = d;
                    k = m;
                }
            }

            if (k < 0 || dmax <= tolerance) {
                if (j < n) {
                    if (np >= kMaxVertices) return false;
                    poly[np++] = j;
                }
            } else {
                if (sp + 2 > kMaxVertices) return false;
                stack_i[sp] = k;
                stack_j[sp] = j;
                ++sp;
                stack_i[sp] = i;
                stack_j[sp] = k;
                ++sp;
            }
        }
    }

    float vx[kMaxVertices];
    float vy[kMaxVertices];
    int count = 0;
    for (int i = 0; i < np; ++i) {
        float x, y;
        contour_point(contour, n, poly[i], &x, &y);
        vx[count] = x;
        vy[count] = y;
        ++count;
    }
    if (count < 4) return false;

    // Collapse down to four vertices, always removing the least significant
    // bump. This keeps the four dominant corners of a perspective quad.
    while (count > 4) {
        int best = -1;
        float best_area = FLT_MAX;
        for (int i = 0; i < count; ++i) {
            const int prev = (i + count - 1) % count;
            const int next = (i + 1) % count;
            const float ax = vx[i] - vx[prev];
            const float ay = vy[i] - vy[prev];
            const float bx = vx[next] - vx[i];
            const float by = vy[next] - vy[i];
            const float area = fabsf(ax * by - ay * bx) * 0.5f;
            if (area < best_area) {
                best_area = area;
                best = i;
            }
        }
        if (best < 0) return false;
        for (int i = best; i + 1 < count; ++i) {
            vx[i] = vx[i + 1];
            vy[i] = vy[i + 1];
        }
        --count;
    }

    float area2 = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        area2 += vx[i] * vy[j] - vx[j] * vy[i];
    }
    if (fabsf(area2) < 24.0f) return false;
    if (area2 < 0.0f) {
        for (int i = 0; i < 2; ++i) {
            const float tx = vx[i + 1];
            const float ty = vy[i + 1];
            vx[i + 1] = vx[3 - i];
            vy[i + 1] = vy[3 - i];
            vx[3 - i] = tx;
            vy[3 - i] = ty;
        }
    }

    // Convexity and sane corner angles.
    float min_side = FLT_MAX;
    float max_side = 0.0f;
    float perimeter = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        const float ex = vx[j] - vx[i];
        const float ey = vy[j] - vy[i];
        const float len = sqrtf(ex * ex + ey * ey);
        if (len < 4.0f) return false;
        min_side = fminf(min_side, len);
        max_side = fmaxf(max_side, len);
        perimeter += len;

        const int p = (i + 3) & 3;
        const float ax = vx[i] - vx[p];
        const float ay = vy[i] - vy[p];
        const float bx = vx[j] - vx[i];
        const float by = vy[j] - vy[i];
        if (ax * by - ay * bx <= 0.0f) return false;  // not convex/consistent
        const float dot = ax * bx + ay * by;
        const float n1 = sqrtf(ax * ax + ay * ay);
        const float n2 = sqrtf(bx * bx + by * by);
        if (n1 < 1e-3f || n2 < 1e-3f) return false;
        if (fabsf(dot / (n1 * n2)) > 0.97f) return false;
    }
    if (max_side > 12.0f * min_side) return false;

    // Every contour point must lie close to one of the four quad edges;
    // otherwise this is a blob, a circle, or an L-shape, not a tag.
    for (int m = 0; m < n; ++m) {
        float px, py;
        contour_point(contour, n, m, &px, &py);
        float best = FLT_MAX;
        for (int i = 0; i < 4; ++i) {
            const int j = (i + 1) & 3;
            best = fminf(best, point_segment_distance(px, py, vx[i], vy[i], vx[j], vy[j]));
        }
        if (best > fmaxf(3.0f, 0.02f * perimeter)) return false;
    }

    for (int i = 0; i < 4; ++i) {
        quad[i][0] = vx[i];
        quad[i][1] = vy[i];
    }
    return true;
}

__device__ float sample_bilinear(const uint8_t* image, int width, int height, int stride,
                                 float x, float y) {
    if (!(x >= 0.0f && y >= 0.0f && x < static_cast<float>(width - 1) &&
          y < static_cast<float>(height - 1))) {
        return -1.0f;
    }
    const int ix = static_cast<int>(x);
    const int iy = static_cast<int>(y);
    const float fx = x - static_cast<float>(ix);
    const float fy = y - static_cast<float>(iy);
    const uint8_t* p = image + iy * stride + ix;
    const float a = p[0];
    const float b = p[1];
    const float c = p[stride];
    const float d = p[stride + 1];
    return (a * (1.0f - fx) + b * fx) * (1.0f - fy) + (c * (1.0f - fx) + d * fx) * fy;
}

__device__ bool intersect_lines(const float li[4], const float lj[4], float* x, float* y) {
    const float det = li[2] * lj[3] - li[3] * lj[2];
    if (fabsf(det) < 0.2f) return false;
    const float di = li[2] * li[0] + li[3] * li[1];
    const float dj = lj[2] * lj[0] + lj[3] * lj[1];
    *x = (di * lj[3] - li[3] * dj) / det;
    *y = (li[2] * dj - di * lj[2]) / det;
    return true;
}

// Fits one edge of the quad to gradient samples.  The four edges are
// independent, so k_fit_quads runs this one edge per thread.
__device__ bool fit_edge_line(const float quad[4][2], int edge, const uint8_t* image,
                              int width, int height, int stride, int dim, float line[4]) {
    const int b = (edge + 1) & 3;
    const float ax = quad[edge][0];
    const float ay = quad[edge][1];
    const float bx = quad[b][0];
    const float by = quad[b][1];
    const float ex = bx - ax;
    const float ey = by - ay;
    const float len = sqrtf(ex * ex + ey * ey);
    if (len < 8.0f) return false;

    const float nx = -ey / len;
    const float ny = ex / len;
    const float cell = len / static_cast<float>(dim);
    const float range = fminf(fmaxf(0.35f * cell, 0.5f), 2.5f);
    const int samples = min(max(static_cast<int>(len / 8.0f), 6), 20);

    float mx = 0.0f;
    float my = 0.0f;
    float mxx = 0.0f;
    float mxy = 0.0f;
    float myy = 0.0f;
    int used = 0;        for (int s = 1; s <= samples; ++s) {
            const float alpha = static_cast<float>(s) / static_cast<float>(samples + 1);
            const float px = ax + alpha * ex;
            const float py = ay + alpha * ey;
            float best_mag = 0.0f;
            float best_t = 0.0f;
            for (int step = -4; step <= 4; ++step) {
                const float t = range * static_cast<float>(step) * 0.25f;
                const float x = px + nx * t;
                const float y = py + ny * t;
                const float xp = sample_bilinear(image, width, height, stride, x + 0.7f, y);
                const float xm = sample_bilinear(image, width, height, stride, x - 0.7f, y);
                const float yp = sample_bilinear(image, width, height, stride, x, y + 0.7f);
                const float ym = sample_bilinear(image, width, height, stride, x, y - 0.7f);
                if (xp < 0.0f || xm < 0.0f || yp < 0.0f || ym < 0.0f) continue;
                const float gx = xp - xm;
                const float gy = yp - ym;
                const float mag = sqrtf(gx * gx + gy * gy);
                if (mag > best_mag) {
                    best_mag = mag;
                    best_t = t;
                }
            }
            if (best_mag < 6.0f) continue;
        const float x = px + nx * best_t;
        const float y = py + ny * best_t;
        mx += x;
        my += y;
        mxx += x * x;
        mxy += x * y;
        myy += y * y;
        ++used;
    }
    if (used < 4) return false;

    const float inv = 1.0f / static_cast<float>(used);
    const float cx = mx * inv;
    const float cy = my * inv;
    const float cxx = mxx * inv - cx * cx;
    const float cxy = mxy * inv - cx * cy;
    const float cyy = myy * inv - cy * cy;
    const float theta = 0.5f * atan2f(2.0f * cxy, cxx - cyy);
    line[0] = cx;
    line[1] = cy;
    line[2] = -sinf(theta);
    line[3] = cosf(theta);
    return true;
}

// Intersects the fitted edge lines and keeps the result when every corner
// stays close to the traced quad and the quad remains convex.
__device__ void apply_refined_lines(float quad[4][2], const float lines[4][4],
                                    const int have[4]) {
    int line_count = 0;
    for (int i = 0; i < 4; ++i) line_count += have[i] ? 1 : 0;
    if (line_count < 3) return;

    float refined[4][2];
    for (int i = 0; i < 4; ++i) {
        refined[i][0] = quad[i][0];
        refined[i][1] = quad[i][1];
    }
    float extent = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        extent = fmaxf(extent, hypotf(quad[j][0] - quad[i][0], quad[j][1] - quad[i][1]));
    }
    const float max_shift = fminf(fmaxf(3.0f, 0.04f * extent), 10.0f);

    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        if (!have[i] || !have[j]) continue;
        float x, y;
        if (!intersect_lines(lines[i], lines[j], &x, &y)) continue;
        const int corner = j;
        const float dx = x - quad[corner][0];
        const float dy = y - quad[corner][1];
        if (dx * dx + dy * dy > max_shift * max_shift) continue;
        refined[corner][0] = x;
        refined[corner][1] = y;
    }

    // Reject refinements that produce a non-convex or degenerate quad.
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        if (hypotf(refined[j][0] - refined[i][0], refined[j][1] - refined[i][1]) < 3.0f) {
            return;
        }
    }
    float area2 = 0.0f;
    for (int i = 0; i < 4; ++i) {
        const int j = (i + 1) & 3;
        area2 += refined[i][0] * refined[j][1] - refined[j][0] * refined[i][1];
    }
    if (area2 <= 0.0f) return;
    for (int i = 0; i < 4; ++i) {
        quad[i][0] = refined[i][0];
        quad[i][1] = refined[i][1];
    }
}

__global__ void k_fit_quads(const uint32_t* __restrict__ bits,
                            const uint32_t* __restrict__ runs,
                            const uint32_t* __restrict__ row_count,
                            uint32_t* __restrict__ parent,
                            const uint32_t* __restrict__ bbox,
                            const uint32_t* __restrict__ candidates,
                            uint32_t* __restrict__ counts,
                            const uint8_t* __restrict__ gray, int width, int height, int stride,
                            int words, int row_cap, int max_candidates, int contour_cap,
                            int dim, int refine_edges,
                            uint32_t* __restrict__ contour, float* __restrict__ quads) {
    __shared__ float traced_quad[4][2];
    __shared__ float edge_lines[4][4];
    __shared__ int edge_have[4];
    __shared__ int have_quad;
    __shared__ int contour_n;
    __shared__ uint32_t shared_contour[kSharedContourPoints];

    // One block per candidate: candidates are few (tens), and a single block
    // per candidate spreads them across SMs instead of serialising them on the
    // two SMs that a single 128-thread block would use.
    const int ci = blockIdx.x;
    if (ci >= max_candidates) return;

    float* out = quads + ci * 8;
    if (threadIdx.x == 0) {
        const float nan_value = __int_as_float(0x7fc00000);
        for (int i = 0; i < 8; ++i) out[i] = nan_value;
    }
    if (ci >= static_cast<int>(counts[0])) return;

    const uint32_t root = candidates[ci];
    const int x0 = static_cast<int>(bbox[root * 4 + 0]);
    const int y0 = static_cast<int>(bbox[root * 4 + 1]);
    const int x1 = static_cast<int>(bbox[root * 4 + 2]);
    const int y1 = static_cast<int>(bbox[root * 4 + 3]);

    // Union-find is only read now; keep it on one thread instead of letting the
    // whole block race through parent path-halving before the trace starts.
    if (threadIdx.x == 0) {
        int sy = 0;
        const int sx = find_region_start(runs, row_count, parent, row_cap, root, x0, y0, x1,
                                         width, &sy);
        int n = 0;
        if (sx >= 0) {
            uint32_t* points = contour + static_cast<size_t>(ci) * contour_cap;
            n = trace_contour(bits, words, width, height, sx, sy, points, contour_cap);
        }
        contour_n = n;
    }
    __syncthreads();

    uint32_t* points = contour + static_cast<size_t>(ci) * contour_cap;
    const int n = contour_n;
    const bool cached = n >= 8 && n <= kSharedContourPoints;
    // Every second contour point is enough to locate the four corners, and it
    // halves the polygon-simplification work on long contours.
    const bool decimate = cached && n > 256;
    const int kept = decimate ? (n + 1) / 2 : n;
    if (cached) {
        for (int i = threadIdx.x; i < kept; i += blockDim.x) {
            shared_contour[i] = points[decimate ? i * 2 : i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        bool ok = n >= 8;
        if (ok) {
            const float tolerance =
                fmaxf(1.5f, 0.015f * static_cast<float>(max(x1 - x0, y1 - y0)));
            float quad[4][2];
            ok = simplify_to_quad(cached ? shared_contour : points, cached ? kept : n, tolerance,
                                  quad);
            if (ok) {
                // The trace lives on black pixel centres, half a pixel inside
                // the true edge. Move corners outward before refinement.
                const float cx = 0.25f * (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]);
                const float cy = 0.25f * (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]);
                for (int i = 0; i < 4; ++i) {
                    const float dx = quad[i][0] - cx;
                    const float dy = quad[i][1] - cy;
                    const float len = sqrtf(dx * dx + dy * dy);
                    if (len > 1e-3f) {
                        quad[i][0] += dx / len * 0.65f;
                        quad[i][1] += dy / len * 0.65f;
                    }
                }
                for (int i = 0; i < 4; ++i) {
                    traced_quad[i][0] = quad[i][0];
                    traced_quad[i][1] = quad[i][1];
                }
            }
        }
        have_quad = ok ? 1 : 0;
        for (int i = 0; i < 4; ++i) edge_have[i] = 0;
    }
    __syncthreads();
    if (!have_quad) return;

    if (refine_edges && threadIdx.x < 4) {
        edge_have[threadIdx.x] =
            fit_edge_line(traced_quad, threadIdx.x, gray, width, height, stride, dim,
                          edge_lines[threadIdx.x])
                ? 1
                : 0;
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    float quad[4][2];
    for (int i = 0; i < 4; ++i) {
        quad[i][0] = traced_quad[i][0];
        quad[i][1] = traced_quad[i][1];
    }
    if (refine_edges) apply_refined_lines(quad, edge_lines, edge_have);

    for (int i = 0; i < 4; ++i) {
        out[i * 2 + 0] = quad[i][0];
        out[i * 2 + 1] = quad[i][1];
    }
    atomicAdd(&counts[3], 1u);
}

// ---------------------------------------------------------------------------
// 8. Homography sampling and decoding
// ---------------------------------------------------------------------------
// Direct square-to-quad projective mapping.  This is the same homography an
// 8x8 solve produces, but it needs no eliminated system and stays in single
// precision, which matters because consumer GPUs run FP64 at 1/32 rate.
__device__ bool homography_from_unit_quad(const float quad[4][2], float h[9]) {
    const float x0 = quad[0][0], y0 = quad[0][1];
    const float x1 = quad[1][0], y1 = quad[1][1];
    const float x2 = quad[2][0], y2 = quad[2][1];
    const float x3 = quad[3][0], y3 = quad[3][1];

    const float dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    const float dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    const float det = dx1 * dy2 - dx2 * dy1;
    if (fabsf(det) < 1e-9f) return false;
    const float g = (dx3 * dy2 - dx2 * dy3) / det;
    const float k = (dx1 * dy3 - dx3 * dy1) / det;

    h[0] = x1 - x0 + g * x1;
    h[1] = x3 - x0 + k * x3;
    h[2] = x0;
    h[3] = y1 - y0 + g * y1;
    h[4] = y3 - y0 + k * y3;
    h[5] = y0;
    h[6] = g;
    h[7] = k;
    h[8] = 1.0f;
    return true;
}

__device__ int match_code(const uint64_t* codes, int count, uint64_t code, int max_hamming,
                          int* out_hamming) {
    int best = max_hamming + 1;
    int best_id = -1;
    for (int i = 0; i < count; ++i) {
        const int d = __popcll(codes[i] ^ code);
        if (d < best) {
            best = d;
            best_id = i;
            if (d == 0) break;
        }
    }
    if (best_id < 0) return -1;
    *out_hamming = best;
    return best_id;
}

__device__ bool decode_quad(const uint8_t* gray, int width, int height, int stride,
                            int image_origin_x, int image_origin_y,
                            const float base_quad[4][2], int dim, int rotation,
                            const uint64_t* codes, int code_count, int max_hamming,
                            float sharpening, float min_margin, int* out_id, int* out_hamming,
                            float* out_margin, float out_quad[4][2]) {
    int best_id = -1;
    int best_hamming = 0;
    float best_margin = 0.0f;
    float best_corners[4][2];
    float best_score = -1.0f;

    constexpr int kMaxCells = 64;
    float raw[kMaxCells];
    float diff[kMaxCells];
    float sharp[kMaxCells];

    const int rot = rotation;
    {
        float corners[4][2];
        for (int k = 0; k < 4; ++k) {
            const int src = (rot + k) & 3;
            corners[k][0] = base_quad[src][0];
            corners[k][1] = base_quad[src][1];
        }

        float h[9];
        if (!homography_from_unit_quad(corners, h)) return false;

        bool valid = true;
        for (int cy = 0; cy < dim && valid; ++cy) {
            for (int cx = 0; cx < dim; ++cx) {
                const float u = (cx + 0.5f) / static_cast<float>(dim);
                const float v = (cy + 0.5f) / static_cast<float>(dim);
                const float den = h[6] * u + h[7] * v + 1.0f;
                if (fabsf(den) < 1e-9f) {
                    valid = false;
                    break;
                }
                const float x = (h[0] * u + h[1] * v + h[2]) / den;
                const float y = (h[3] * u + h[4] * v + h[5]) / den;
                const float value = sample_bilinear(
                    gray, width, height, stride, x - static_cast<float>(image_origin_x),
                    y - static_cast<float>(image_origin_y));
                if (value < 0.0f) {
                    valid = false;
                    break;
                }
                raw[cy * dim + cx] = value;
            }
        }
        if (!valid) return false;

        float mn = 255.0f;
        float mx = 0.0f;
        for (int i = 0; i < dim * dim; ++i) {
            mn = fminf(mn, raw[i]);
            mx = fmaxf(mx, raw[i]);
        }
        if (mx - mn < 10.0f) return false;
        const float thresh = 0.5f * (mn + mx);

        bool border_ok = true;
        for (int cy = 0; cy < dim && border_ok; ++cy) {
            for (int cx = 0; cx < dim; ++cx) {
                if (cx == 0 || cy == 0 || cx == dim - 1 || cy == dim - 1) {
                    if (raw[cy * dim + cx] >= thresh) {
                        border_ok = false;
                        break;
                    }
                }
            }
        }
        if (!border_ok) return false;

        for (int i = 0; i < dim * dim; ++i) diff[i] = raw[i] - thresh;
        for (int i = 0; i < dim * dim; ++i) sharp[i] = diff[i];
        const int lo = 1;
        const int hi = dim - 2;
        for (int cy = lo; cy <= hi; ++cy) {
            for (int cx = lo; cx <= hi; ++cx) {
                const int i = cy * dim + cx;
                const float c = diff[i];
                const float lap = 4.0f * c - diff[i - dim] - diff[i + dim] - diff[i - 1] -
                                  diff[i + 1];
                sharp[i] = c + sharpening * lap;
            }
        }

        uint64_t code = 0;
        float white_sum = 0.0f;
        float black_sum = 0.0f;
        int white_count = 0;
        int black_count = 0;
        for (int cy = lo; cy <= hi; ++cy) {
            for (int cx = lo; cx <= hi; ++cx) {
                const float d = sharp[cy * dim + cx];
                code = (code << 1) | (d > 0.0f ? 1ull : 0ull);
                if (d > 0.0f) {
                    white_sum += d;
                    ++white_count;
                } else {
                    black_sum -= d;
                    ++black_count;
                }
            }
        }

        int hamming = 0;
        const int id = match_code(codes, code_count, code, max_hamming, &hamming);
        if (id < 0) return false;

        const float white_avg = white_sum / static_cast<float>(max(white_count, 1));
        const float black_avg = black_sum / static_cast<float>(max(black_count, 1));
        const float margin = fminf(white_avg, black_avg);
        if (margin < min_margin) return false;

        const float score = static_cast<float>(4 - hamming) * 1000.0f + margin;
        if (score > best_score) {
            best_score = score;
            best_id = id;
            best_hamming = hamming;
            best_margin = margin;
            for (int k = 0; k < 4; ++k) {
                best_corners[k][0] = corners[k][0];
                best_corners[k][1] = corners[k][1];
            }
        }
    }

    if (best_id < 0) return false;
    *out_id = best_id;
    *out_hamming = best_hamming;
    *out_margin = best_margin;
    for (int k = 0; k < 4; ++k) {
        out_quad[k][0] = best_corners[k][0];
        out_quad[k][1] = best_corners[k][1];
    }
    return true;
}

__global__ void k_decode_quads(const uint8_t* __restrict__ gray, int width, int height,
                               int stride, const float* __restrict__ quads,
                               uint32_t* __restrict__ counts, int max_candidates, int dim,
                               const uint64_t* __restrict__ codes, int code_count,
                               int max_hamming, float sharpening, float min_margin,
                               bd_detection* __restrict__ detections, int max_detections) {
    const int ci = blockIdx.x;
    if (ci >= max_candidates) return;
    if (ci >= static_cast<int>(counts[0])) return;

    const float* q = quads + ci * 8;
    if (q[0] != q[0]) return;  // NaN sentinel

    float quad[4][2];
    for (int i = 0; i < 4; ++i) {
        quad[i][0] = q[i * 2 + 0];
        quad[i][1] = q[i * 2 + 1];
    }

    // The four rotations are independent: one thread decodes each, then the
    // block picks the same winner the serial loop used to (highest score,
    // first rotation on ties).
    struct RotationResult {
        int id;
        int hamming;
        float margin;
        float corners[4][2];
    };
    __shared__ RotationResult results[4];
    if (threadIdx.x < 4) {
        results[threadIdx.x].id = -1;
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        int id = 0;
        int hamming = 0;
        float margin = 0.0f;
        float corners[4][2];
        if (decode_quad(gray, width, height, stride, 0, 0, quad, dim, threadIdx.x, codes,
                        code_count, max_hamming, sharpening, min_margin, &id, &hamming,
                        &margin, corners)) {
            RotationResult& result = results[threadIdx.x];
            result.id = id;
            result.hamming = hamming;
            result.margin = margin;
            for (int k = 0; k < 4; ++k) {
                result.corners[k][0] = corners[k][0];
                result.corners[k][1] = corners[k][1];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x != 0) return;

    int best = -1;
    float best_score = -1.0f;
    for (int r = 0; r < 4; ++r) {
        if (results[r].id < 0) continue;
        const float score = static_cast<float>(4 - results[r].hamming) * 1000.0f +
                            results[r].margin;
        if (score > best_score) {
            best_score = score;
            best = r;
        }
    }
    if (best < 0) return;

    const uint32_t slot = atomicAdd(&counts[1], 1u);
    if (slot >= static_cast<uint32_t>(max_detections)) return;
    bd_detection d;
    d.id = results[best].id;
    d.hamming = results[best].hamming;
    d.decision_margin = results[best].margin;
    for (int k = 0; k < 4; ++k) {
        d.corners[k * 2 + 0] = results[best].corners[k][0];
        d.corners[k * 2 + 1] = results[best].corners[k][1];
    }
    d.reserved = 0;
    detections[slot] = d;
}

// ---------------------------------------------------------------------------
// 9. Deduplicate overlapping detections of the same id, then sort by id.
// ---------------------------------------------------------------------------
__global__ void k_finalize(bd_detection* __restrict__ detections, uint32_t* __restrict__ counts,
                           int max_detections) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    bd_detection* out = reinterpret_cast<bd_detection*>(smem_raw);
    if (threadIdx.x != 0) return;

    const int n = min(static_cast<int>(counts[1]), max_detections);
    int kept = 0;
    for (int i = 0; i < n; ++i) {
        const bd_detection d = detections[i];
        float side = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const int j = (k + 1) & 3;
            side += hypotf(d.corners[j * 2] - d.corners[k * 2],
                           d.corners[j * 2 + 1] - d.corners[k * 2 + 1]);
            cx += d.corners[k * 2];
            cy += d.corners[k * 2 + 1];
        }
        side *= 0.25f;
        cx *= 0.25f;
        cy *= 0.25f;

        int duplicate = -1;
        for (int k = 0; k < kept; ++k) {
            if (out[k].id != d.id) continue;
            float ocx = 0.0f;
            float ocy = 0.0f;
            for (int c = 0; c < 4; ++c) {
                ocx += out[k].corners[c * 2];
                ocy += out[k].corners[c * 2 + 1];
            }
            ocx *= 0.25f;
            ocy *= 0.25f;
            const float limit = 0.35f * side;
            if (fabsf(ocx - cx) < limit && fabsf(ocy - cy) < limit) {
                duplicate = k;
                break;
            }
        }
        if (duplicate >= 0) {
            if (d.decision_margin > out[duplicate].decision_margin) out[duplicate] = d;
        } else if (kept < max_detections) {
            out[kept++] = d;
        }
    }

    for (int i = 1; i < kept; ++i) {
        const bd_detection value = out[i];
        int j = i - 1;
        while (j >= 0 && out[j].id > value.id) {
            out[j + 1] = out[j];
            --j;
        }
        out[j + 1] = value;
    }
    for (int i = 0; i < kept; ++i) detections[i] = out[i];
    counts[1] = static_cast<uint32_t>(kept);
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
void launch_pipeline(const PipelineArgs& a, cudaStream_t stream) {
    // BD_PROFILE_TEMP: per-stage GPU timings, only for local optimization work.
    const bool profile = std::getenv("BD_PROFILE") != nullptr;
    cudaEvent_t p0 = nullptr;
    cudaEvent_t p1 = nullptr;
    if (profile) {
        cudaEventCreate(&p0);
        cudaEventCreate(&p1);
    }
#define BD_STAGE(label, ...)                                         \
    do {                                                             \
        if (profile) cudaEventRecord(p0, stream);                    \
        __VA_ARGS__;                                                 \
        if (profile) {                                               \
            cudaEventRecord(p1, stream);                             \
            cudaEventSynchronize(p1);                                \
            float ms = 0.0f;                                         \
            cudaEventElapsedTime(&ms, p0, p1);                       \
            std::printf("  %-16s %.3f ms\n", label, ms);             \
        }                                                            \
    } while (0)

    BD_STAGE("memset", cudaMemsetAsync(a.counts, 0, 5 * sizeof(uint32_t), stream));

    const int tile_count = a.tile_w * a.tile_h;
    const int tile_threads = 128;
    BD_STAGE("tile_threshold",
             k_tile_threshold<<<(tile_count + tile_threads - 1) / tile_threads, tile_threads,
                                0, stream>>>(a.gray, a.width, a.height, a.stride,
                                             a.tile_thresh, a.tile_active, a.tile_w,
                                             a.tile_h));
    BD_STAGE("row_roi", k_build_row_roi<<<1, 256, 0, stream>>>(
                            a.tile_active, a.tile_w, a.tile_h, a.row_active, a.row_start,
                            a.row_end, a.counts));

    const dim3 bin_grid((a.words + 7) / 8, a.height);
    BD_STAGE("binarize", k_binarize<<<bin_grid, 256, 0, stream>>>(
                             a.gray, a.width, a.height, a.stride, a.tile_thresh,
                             a.row_active, a.row_start, a.row_end, a.tile_w, a.bits,
                             a.words));

    const size_t smem = static_cast<size_t>(a.words) * 2 * sizeof(uint32_t);
    BD_STAGE("find_runs", k_find_runs<<<a.height, kFindRunsThreads, smem, stream>>>(
                              a.bits, a.width, a.words, a.row_cap, a.row_count, a.runs,
                              a.parent, a.area, a.bbox, a.counts, a.row_active));

    BD_STAGE("merge_runs", k_merge_runs<<<(a.height + kMergeWarps - 1) / kMergeWarps,
                                          kMergeThreads, 0, stream>>>(
                               a.runs, a.row_count, a.parent, a.row_cap, a.height,
                               a.row_active));

    const int row_grid = (a.height + kMergeWarps - 1) / kMergeWarps;
    BD_STAGE("region_stats", k_region_stats<<<row_grid, kMergeThreads, 0, stream>>>(
                                 a.runs, a.row_count, a.parent, a.area, a.bbox, a.row_cap,
                                 a.height, a.row_active));

    BD_STAGE("collect", k_collect_candidates<<<row_grid, kMergeThreads, 0, stream>>>(
                            a.row_count, a.parent, a.area, a.bbox, a.row_cap, a.height,
                            a.width, a.min_cluster_pixels, a.min_tag_pixels,
                            a.max_candidates, a.candidates, a.counts, a.row_active));

    BD_STAGE("fit_quads", k_fit_quads<<<a.max_candidates, 32, 0, stream>>>(
                              a.bits, a.runs, a.row_count, a.parent, a.bbox, a.candidates,
                              a.counts, a.gray, a.width, a.height, a.stride, a.words,
                              a.row_cap, a.max_candidates, a.contour_cap, a.dim,
                              a.refine_edges, a.contour, a.quads));

    BD_STAGE("decode", k_decode_quads<<<a.max_candidates, 32, 0, stream>>>(
                           a.gray, a.width, a.height, a.stride, a.quads, a.counts,
                           a.max_candidates, a.dim, a.codes, a.code_count, a.max_hamming,
                           a.sharpening, a.min_decision_margin, a.detections,
                           a.max_detections));

    const size_t final_smem = static_cast<size_t>(a.max_detections) * sizeof(bd_detection);
    BD_STAGE("finalize", k_finalize<<<1, 1, final_smem, stream>>>(a.detections, a.counts,
                                                                  a.max_detections));

    if (profile) {
        cudaEventDestroy(p0);
        cudaEventDestroy(p1);
    }
#undef BD_STAGE
}

// ---------------------------------------------------------------------------
// Host-side device detector
// ---------------------------------------------------------------------------
std::string cuda_error_string(const char* what, cudaError_t status) {
    std::ostringstream ss;
    ss << what << ": " << cudaGetErrorString(status);
    return ss.str();
}

bool device_alloc(void** ptr, size_t bytes, std::string* error) {
    const cudaError_t status = cudaMalloc(ptr, bytes == 0 ? 1 : bytes);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaMalloc", status);
        return false;
    }
    return true;
}

bool host_alloc(void** ptr, size_t bytes, std::string* error) {
    const cudaError_t status = cudaHostAlloc(ptr, bytes == 0 ? 1 : bytes, cudaHostAllocPortable);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaHostAlloc", status);
        return false;
    }
    return true;
}

// Homography from the tag's canonical square (coordinates in [-1,1], x right,
// y down) to image pixels, derived from the four refined corners TL, TR, BR,
// BL.  Normalised so H[8] == 1, matching AprilTag's tag-frame convention.
void fill_homography(const float corners[8], float H[9]) {
    const double x0 = corners[0], y0 = corners[1];
    const double x1 = corners[2], y1 = corners[3];
    const double x2 = corners[4], y2 = corners[5];
    const double x3 = corners[6], y3 = corners[7];

    // Unit-square (0,0)..(1,1) to quad, then remapped to [-1,1].
    const double dx1 = x1 - x2, dx2 = x3 - x2;
    const double dx3 = x0 - x1 + x2 - x3;
    const double dy1 = y1 - y2, dy2 = y3 - y2;
    const double dy3 = y0 - y1 + y2 - y3;

    double g = 0.0, h = 0.0;
    const double det = dx1 * dy2 - dx2 * dy1;
    if (std::fabs(det) > 1e-12) {
        g = (dx3 * dy2 - dx2 * dy3) / det;
        h = (dx1 * dy3 - dx3 * dy1) / det;
    }

    const double a = x1 - x0 + g * x1;
    const double b = x3 - x0 + h * x3;
    const double d = y1 - y0 + g * y1;
    const double e = y3 - y0 + h * y3;
    const double w = g + h + 2.0;
    const double inv = (std::fabs(w) > 1e-12) ? 1.0 / w : 0.0;

    H[0] = static_cast<float>(a * inv);
    H[1] = static_cast<float>(b * inv);
    H[2] = static_cast<float>((a + b + 2.0 * x0) * inv);
    H[3] = static_cast<float>(d * inv);
    H[4] = static_cast<float>(e * inv);
    H[5] = static_cast<float>((d + e + 2.0 * y0) * inv);
    H[6] = static_cast<float>(g * inv);
    H[7] = static_cast<float>(h * inv);
    H[8] = static_cast<float>(w * inv);
}

int next_power_of_two(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

}  // namespace

struct GpuDetector::Impl {
    GpuConfig cfg;
    const TagFamilyData* family = nullptr;
    PipelineArgs args;
    int max_detections = 0;
    int max_candidates = 0;
    int hamming = 0;
    bool graph_ready = false;

    uint8_t* d_gray = nullptr;
    uint8_t* d_tile_thresh = nullptr;
    uint8_t* d_tile_active = nullptr;
    uint8_t* d_row_active = nullptr;
    int* d_row_start = nullptr;
    int* d_row_end = nullptr;
    uint32_t* d_bits = nullptr;
    uint32_t* d_row_count = nullptr;
    uint32_t* d_runs = nullptr;
    uint32_t* d_parent = nullptr;
    uint32_t* d_area = nullptr;
    uint32_t* d_bbox = nullptr;
    uint32_t* d_candidates = nullptr;
    uint32_t* d_counts = nullptr;
    uint32_t* d_contour = nullptr;
    float* d_quads = nullptr;
    bd_detection* d_detections = nullptr;
    uint64_t* d_codes = nullptr;

    uint8_t* h_input = nullptr;
    uint32_t* h_counts = nullptr;
    bd_detection* h_detections = nullptr;

    cudaStream_t stream = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;

    uint64_t frames = 0;
    float last_gpu_ms = 0.0f;
    float average_gpu_ms = 0.0f;
    float last_total_ms = 0.0f;
    int last_candidates = 0;
    int last_runs = 0;
    int last_detections = 0;
    int last_quads = 0;
    int last_roi_rows = 0;

    // One detector owns one stream, one set of buffers and one CUDA graph, so
    // calls are serialized.  This makes a single instance safe to share.
    mutable std::mutex mutex;

    ~Impl() { release(); }

    void release() {
        if (stream) cudaStreamSynchronize(stream);
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (ev_start) cudaEventDestroy(ev_start);
        if (ev_stop) cudaEventDestroy(ev_stop);
        if (stream) cudaStreamDestroy(stream);

#define BD_FREE_DEV(p)       \
    do {                     \
        if (p) cudaFree(p);  \
        p = nullptr;         \
    } while (0)
        BD_FREE_DEV(d_gray);
        BD_FREE_DEV(d_tile_thresh);
        BD_FREE_DEV(d_tile_active);
        BD_FREE_DEV(d_row_active);
        BD_FREE_DEV(d_row_start);
        BD_FREE_DEV(d_row_end);
        BD_FREE_DEV(d_bits);
        BD_FREE_DEV(d_row_count);
        BD_FREE_DEV(d_runs);
        BD_FREE_DEV(d_parent);
        BD_FREE_DEV(d_area);
        BD_FREE_DEV(d_bbox);
        BD_FREE_DEV(d_candidates);
        BD_FREE_DEV(d_counts);
        BD_FREE_DEV(d_contour);
        BD_FREE_DEV(d_quads);
        BD_FREE_DEV(d_detections);
        BD_FREE_DEV(d_codes);
#undef BD_FREE_DEV

        if (h_input) cudaFreeHost(h_input);
        if (h_counts) cudaFreeHost(h_counts);
        if (h_detections) cudaFreeHost(h_detections);
        h_input = nullptr;
        h_counts = nullptr;
        h_detections = nullptr;

        graph_exec = nullptr;
        stream = nullptr;
        ev_start = nullptr;
        ev_stop = nullptr;
    }

    bool initialize(std::string* error);
    bool capture_graph(std::string* error);
    int finish(const std::string& where, bd_detection* out, int out_capacity, std::string* error,
               float gpu_ms, float total_ms);
};

bool GpuDetector::Impl::initialize(std::string* error) {
    family = kFamilies[cfg.family];
    max_detections = std::min(std::max(cfg.max_detections, 1), 256);
    hamming = cfg.max_hamming >= 0
                  ? std::min(cfg.max_hamming, 3)
                  : (cfg.family == 0 ? 2 : 1);

    const int width = cfg.width;
    const int height = cfg.height;
    max_candidates = cfg.max_candidates > 0
                         ? std::min(std::max(cfg.max_candidates, 16), 1024)
                         : std::min(std::max(next_power_of_two((width * height) / 4096), 64), 1024);

    args.width = width;
    args.height = height;
    args.stride = width;
    args.tile_w = (width + kTile - 1) / kTile;
    args.tile_h = (height + kTile - 1) / kTile;
    args.words = (width + 31) / 32;
    args.row_cap = width / 2 + 2;
    args.total_slots = height * args.row_cap;
    args.min_cluster_pixels = std::max(cfg.min_cluster_pixels, 1);
    args.min_tag_pixels = std::max(cfg.min_tag_pixels, 4);
    args.max_candidates = max_candidates;
    args.contour_cap = std::min(std::max(2 * (width + height), kMinContourCapacity),
                                kMaxContourCapacity);
    args.bits_per_side = family->bits_per_side;
    args.dim = family->bits_per_side + 2 * family->border_cells;
    args.max_hamming = hamming;
    args.code_count = family->code_count;
    args.refine_edges = cfg.refine_edges ? 1 : 0;
    args.max_detections = max_detections;
    args.min_decision_margin = cfg.min_decision_margin;
    args.sharpening = cfg.decode_sharpening;

    const size_t pixels = static_cast<size_t>(width) * height;
    const size_t slots = static_cast<size_t>(args.total_slots);

    bool ok = true;
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_gray), pixels, error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_tile_thresh),
                            static_cast<size_t>(args.tile_w) * args.tile_h, error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_tile_active),
                            static_cast<size_t>(args.tile_w) * args.tile_h, error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_row_active),
                            static_cast<size_t>(args.tile_h), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_row_start),
                            static_cast<size_t>(args.tile_h) * sizeof(int), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_row_end),
                            static_cast<size_t>(args.tile_h) * sizeof(int), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_bits),
                            static_cast<size_t>(args.words) * height * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_row_count),
                            static_cast<size_t>(height) * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_runs), slots * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_parent), slots * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_area), slots * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_bbox), slots * 4 * sizeof(uint32_t),
                            error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_candidates),
                            static_cast<size_t>(max_candidates) * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_counts), 5 * sizeof(uint32_t), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_contour),
                            static_cast<size_t>(max_candidates) * args.contour_cap *
                                sizeof(uint32_t),
                            error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_quads),
                            static_cast<size_t>(max_candidates) * 8 * sizeof(float), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_detections),
                            static_cast<size_t>(max_detections) * sizeof(bd_detection), error);
    ok = ok && device_alloc(reinterpret_cast<void**>(&d_codes),
                            static_cast<size_t>(family->code_count) * sizeof(uint64_t), error);
    ok = ok && host_alloc(reinterpret_cast<void**>(&h_input), pixels, error);
    ok = ok && host_alloc(reinterpret_cast<void**>(&h_counts), 5 * sizeof(uint32_t), error);
    ok = ok && host_alloc(reinterpret_cast<void**>(&h_detections),
                          static_cast<size_t>(max_detections) * sizeof(bd_detection), error);
    if (!ok) return false;

    cudaError_t status = cudaMemcpy(d_codes, family->codes,
                                    static_cast<size_t>(family->code_count) * sizeof(uint64_t),
                                    cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaMemcpy(tag codes)", status);
        return false;
    }

    status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaStreamCreate", status);
        return false;
    }
    status = cudaEventCreate(&ev_start);
    if (status == cudaSuccess) status = cudaEventCreate(&ev_stop);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaEventCreate", status);
        return false;
    }

    cudaMemset(d_counts, 0, 5 * sizeof(uint32_t));

    args.gray = d_gray;
    args.tile_thresh = d_tile_thresh;
    args.tile_active = d_tile_active;
    args.row_active = d_row_active;
    args.row_start = d_row_start;
    args.row_end = d_row_end;
    args.bits = d_bits;
    args.row_count = d_row_count;
    args.runs = d_runs;
    args.parent = d_parent;
    args.area = d_area;
    args.bbox = d_bbox;
    args.candidates = d_candidates;
    args.counts = d_counts;
    args.contour = d_contour;
    args.quads = d_quads;
    args.detections = d_detections;
    args.codes = d_codes;

    if (cfg.use_cuda_graph) {
        std::string graph_error;
        graph_ready = capture_graph(&graph_error);
        // Graph capture is an optimization, not a requirement; on failure the
        // detector falls back to direct launches.
    }
    return true;
}

bool GpuDetector::Impl::capture_graph(std::string* error) {
    const cudaError_t begin = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (begin != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaStreamBeginCapture", begin);
        return false;
    }

    cudaMemcpyAsync(d_gray, h_input, static_cast<size_t>(cfg.width) * cfg.height,
                    cudaMemcpyHostToDevice, stream);
    launch_pipeline(args, stream);
    cudaMemcpyAsync(h_counts, d_counts, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_detections, d_detections,
                    static_cast<size_t>(max_detections) * sizeof(bd_detection),
                    cudaMemcpyDeviceToHost, stream);

    cudaGraph_t captured = nullptr;
    cudaError_t status = cudaStreamEndCapture(stream, &captured);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaStreamEndCapture", status);
        cudaGetLastError();
        cudaStreamDestroy(stream);
        stream = nullptr;
        if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) {
            return false;
        }
        return false;
    }

    cudaGraphExec_t exec = nullptr;
    status = cudaGraphInstantiate(&exec, captured, 0);
    if (status != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaGraphInstantiate", status);
        cudaGraphDestroy(captured);
        return false;
    }
    graph_exec = exec;
    cudaGraphDestroy(captured);
    return true;
}

int GpuDetector::Impl::finish(const std::string& where, bd_detection* out, int out_capacity,
                              std::string* error, float gpu_ms, float total_ms) {
    (void)where;
    (void)error;
    const int found = std::min(static_cast<int>(h_counts[1]), max_detections);
    const int written = std::min(found, std::max(out_capacity, 0));
    if (written > 0 && out) {
        std::memcpy(out, h_detections, static_cast<size_t>(written) * sizeof(bd_detection));
        for (int i = 0; i < written; ++i) {
            fill_homography(out[i].corners, out[i].homography);
        }
    }

    last_candidates = std::min(static_cast<int>(h_counts[0]), max_candidates);
    last_runs = static_cast<int>(h_counts[2]);
    last_detections = found;
    last_quads = static_cast<int>(h_counts[3]);
    last_roi_rows = std::min(static_cast<int>(h_counts[4]) * kTile, cfg.height);
    last_gpu_ms = gpu_ms;
    last_total_ms = total_ms;
    if (frames == 0) {
        average_gpu_ms = gpu_ms;
    } else {
        average_gpu_ms = 0.9f * average_gpu_ms + 0.1f * gpu_ms;
    }
    ++frames;
    return found;
}

GpuDetector::GpuDetector() : impl_(new Impl()) {}

GpuDetector::~GpuDetector() { delete impl_; }

GpuDetector* GpuDetector::create(const GpuConfig& config, std::string* error) {
    GpuDetector* detector = new (std::nothrow) GpuDetector();
    if (!detector) {
        if (error) *error = "out of memory allocating detector";
        return nullptr;
    }
    detector->impl_->cfg = config;
    if (!detector->impl_->initialize(error)) {
        delete detector;
        return nullptr;
    }
    return detector;
}

void GpuDetector::destroy(GpuDetector* detector) { delete detector; }

int GpuDetector::max_detections() const { return impl_->max_detections; }

int GpuDetector::detect_host(const uint8_t* gray, int stride, bd_detection* out,
                             int out_capacity, std::string* error) {
    Impl& im = *impl_;
    const std::lock_guard<std::mutex> lock(im.mutex);
    if (!gray) {
        if (error) *error = "null image pointer";
        return -1;
    }
    if (stride < im.cfg.width) {
        if (error) *error = "stride is smaller than the image width";
        return -1;
    }
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        if (error) *error = "invalid output buffer";
        return -1;
    }

    const auto start = std::chrono::steady_clock::now();
    if (stride == im.cfg.width) {
        std::memcpy(im.h_input, gray, static_cast<size_t>(im.cfg.width) * im.cfg.height);
    } else {
        for (int y = 0; y < im.cfg.height; ++y) {
            std::memcpy(im.h_input + static_cast<size_t>(y) * im.cfg.width,
                        gray + static_cast<size_t>(y) * stride, im.cfg.width);
        }
    }

    cudaEventRecord(im.ev_start, im.stream);
    if (im.graph_ready && im.graph_exec) {
        const cudaError_t status = cudaGraphLaunch(im.graph_exec, im.stream);
        if (status != cudaSuccess) {
            if (error) *error = cuda_error_string("cudaGraphLaunch", status);
            return -1;
        }
    } else {
        cudaMemcpyAsync(im.d_gray, im.h_input,
                        static_cast<size_t>(im.cfg.width) * im.cfg.height,
                        cudaMemcpyHostToDevice, im.stream);
        launch_pipeline(im.args, im.stream);
        cudaMemcpyAsync(im.h_counts, im.d_counts, 5 * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, im.stream);
        cudaMemcpyAsync(im.h_detections, im.d_detections,
                        static_cast<size_t>(im.max_detections) * sizeof(bd_detection),
                        cudaMemcpyDeviceToHost, im.stream);
    }
    cudaEventRecord(im.ev_stop, im.stream);

    const cudaError_t sync = cudaEventSynchronize(im.ev_stop);
    if (sync != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaEventSynchronize", sync);
        return -1;
    }
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, im.ev_start, im.ev_stop);

    const auto end = std::chrono::steady_clock::now();
    const float total_ms =
        std::chrono::duration<float, std::milli>(end - start).count();
    return im.finish("detect", out, out_capacity, error, gpu_ms, total_ms);
}

int GpuDetector::detect_device(const uint8_t* gray_device, int stride, bd_detection* out,
                               int out_capacity, std::string* error) {
    Impl& im = *impl_;
    const std::lock_guard<std::mutex> lock(im.mutex);
    if (!gray_device) {
        if (error) *error = "null device image pointer";
        return -1;
    }
    if (stride < im.cfg.width) {
        if (error) *error = "stride is smaller than the image width";
        return -1;
    }
    if (out_capacity < 0 || (out_capacity > 0 && !out)) {
        if (error) *error = "invalid output buffer";
        return -1;
    }

    const auto start = std::chrono::steady_clock::now();
    PipelineArgs a = im.args;
    a.gray = gray_device;
    a.stride = stride;

    cudaEventRecord(im.ev_start, im.stream);
    launch_pipeline(a, im.stream);
    cudaMemcpyAsync(im.h_counts, im.d_counts, 5 * sizeof(uint32_t), cudaMemcpyDeviceToHost,
                    im.stream);
    cudaMemcpyAsync(im.h_detections, im.d_detections,
                    static_cast<size_t>(im.max_detections) * sizeof(bd_detection),
                    cudaMemcpyDeviceToHost, im.stream);
    cudaEventRecord(im.ev_stop, im.stream);

    const cudaError_t sync = cudaEventSynchronize(im.ev_stop);
    if (sync != cudaSuccess) {
        if (error) *error = cuda_error_string("cudaEventSynchronize", sync);
        return -1;
    }
    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, im.ev_start, im.ev_stop);

    const auto end = std::chrono::steady_clock::now();
    const float total_ms = std::chrono::duration<float, std::milli>(end - start).count();
    return im.finish("detectDevice", out, out_capacity, error, gpu_ms, total_ms);
}

void GpuDetector::get_stats(bd_stats* out) const {
    if (!out) return;
    const Impl& im = *impl_;
    const std::lock_guard<std::mutex> lock(im.mutex);
    out->frames = im.frames;
    out->last_gpu_ms = im.last_gpu_ms;
    out->average_gpu_ms = im.average_gpu_ms;
    out->last_total_ms = im.last_total_ms;
    out->runs = im.last_runs;
    out->candidates = im.last_candidates;
    out->detections = im.last_detections;
    out->quads = im.last_quads;
    out->roi_rows = im.last_roi_rows;
}

}  // namespace bdcudatag::detail
