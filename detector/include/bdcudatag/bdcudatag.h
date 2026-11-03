/*
 * bdcudatag - CUDA AprilTag-family detector.
 *
 * C ABI used by the Java FFM wrapper and by plain C consumers.  A detector
 * instance handles exactly one tag family and one image resolution.
 */
#ifndef BDCUDATAG_BDCUDATAG_H
#define BDCUDATAG_BDCUDATAG_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#if defined(BDCUDATAG_BUILDING)
#define BD_API __declspec(dllexport)
#elif defined(BDCUDATAG_STATIC)
#define BD_API
#else
#define BD_API __declspec(dllimport)
#endif
#else
#define BD_API __attribute__((visibility("default")))
#endif

#define BD_API_VERSION 2u
#define BD_MAX_WIDTH 8192
#define BD_MAX_HEIGHT 16384

typedef struct bd_detector bd_detector;

typedef enum bd_family {
    BD_FAMILY_TAG36H11 = 0,
    BD_FAMILY_TAG16H5 = 1
} bd_family;

typedef enum bd_status {
    BD_OK = 0,
    BD_ERROR_INVALID_ARGUMENT = -1,
    BD_ERROR_CUDA = -2,
    BD_ERROR_OUT_OF_MEMORY = -3,
    BD_ERROR_NOT_SUPPORTED = -4,
    BD_ERROR_UNKNOWN = -5
} bd_status;

/*
 * Detector configuration.  Always initialise with bd_config_init(), or set
 * struct_size and all fields.  max_hamming = -1 selects the family default
 * (2 for tag36h11, 1 for tag16h5); max_candidates = 0 selects an automatic
 * value based on the image size; max_detections is capped at 256.
 */
typedef struct bd_config {
    uint32_t struct_size;
    int32_t max_hamming;
    int32_t min_cluster_pixels;
    int32_t min_tag_pixels;
    int32_t max_detections;
    int32_t max_candidates;
    int32_t refine_edges;
    int32_t use_cuda_graph;
    float min_decision_margin;
    float decode_sharpening;
} bd_config;

/*
 * One detected tag.  Corners are (x0,y0) .. (x3,y3) in image pixel
 * coordinates, ordered starting at the tag's top-left corner and proceeding
 * top-right, bottom-right, bottom-left in the tag's canonical orientation.
 * This matches AprilTag's detection corner ordering, but no pose is computed.
 *
 * homography is a row-major 3x3 matrix mapping the tag's canonical square to
 * image pixels, using AprilTag's tag frame: (x, y) in [-1, 1], x right, y
 * down, with the first corner at (-1, -1).
 *
 *     s * (px, py, 1)^T = homography * (x, y, 1)^T
 *
 * It is normalised so homography[8] == 1 and is derived from the refined
 * corners; no camera intrinsics or pose are involved.
 */
typedef struct bd_detection {
    int32_t id;
    int32_t hamming;
    float decision_margin;
    float corners[8];
    int32_t reserved;
    float homography[9];
} bd_detection;

typedef struct bd_stats {
    uint64_t frames;
    float last_gpu_ms;
    float average_gpu_ms;
    float last_total_ms;
    int32_t runs;
    int32_t candidates;
    int32_t detections;
    int32_t quads;
    int32_t roi_rows;
} bd_stats;

BD_API uint32_t bd_api_version(void);
BD_API const char* bd_version_string(void);

BD_API int32_t bd_family_count(void);
BD_API const char* bd_family_name(int32_t family);
BD_API int32_t bd_family_bits_per_side(int32_t family);
BD_API int32_t bd_family_min_hamming(int32_t family);
BD_API int32_t bd_family_default_max_hamming(int32_t family);

BD_API void bd_config_init(bd_config* config);

/*
 * Creates a detector.  Returns NULL on failure and, when status_out is not
 * NULL, writes a bd_status value.  bd_last_error() describes the failure.
 * The detector allocates all device memory up front; detect calls do not
 * allocate.
 */
BD_API bd_detector* bd_detector_create(int32_t family, int32_t width, int32_t height,
                                       const bd_config* config, int32_t* status_out);
BD_API void bd_detector_destroy(bd_detector* detector);

/*
 * Detects tags in an 8-bit grayscale image with the given row stride in
 * bytes (>= width).  Returns the number of detections written, or a negative
 * bd_status on error.  If more tags are found than out_capacity, only
 * out_capacity are written and the full count is returned.
 */
BD_API int32_t bd_detector_detect_host(bd_detector* detector, const uint8_t* gray,
                                       int32_t stride, bd_detection* out,
                                       int32_t out_capacity);

/*
 * Same, but reads a CUDA device pointer directly (no host->device copy).  The
 * pointer must be accessible to the detector's CUDA context.
 */
BD_API int32_t bd_detector_detect_device(bd_detector* detector, const uint8_t* gray_device,
                                         int32_t stride, bd_detection* out,
                                         int32_t out_capacity);

BD_API void bd_detector_get_stats(bd_detector* detector, bd_stats* out);

/* Thread-local message for the most recent failure on this thread. */
BD_API const char* bd_last_error(void);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
static_assert(sizeof(bd_config) == 40, "bd_config layout changed");
static_assert(sizeof(bd_detection) == 84, "bd_detection layout changed");
static_assert(sizeof(bd_stats) == 40, "bd_stats layout changed");
#endif

#endif /* BDCUDATAG_BDCUDATAG_H */
