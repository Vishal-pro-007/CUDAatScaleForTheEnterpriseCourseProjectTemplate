import math
import time
from numba import cuda

@cuda.jit
def gpu_sobel_kernel(d_img, d_out, width, height):
    x, y = cuda.grid(2)
    if 1 <= x < width - 1 and 1 <= y < height - 1:
        gx = (-1 * d_img[y - 1, x - 1] + 1 * d_img[y - 1, x + 1] - 2 * d_img[y, x - 1] + 2 * d_img[y, x + 1] - 1 * d_img[y + 1, x - 1] + 1 * d_img[y + 1, x + 1])
        gy = (-1 * d_img[y - 1, x - 1] - 2 * d_img[y - 1, x] - 1 * d_img[y - 1, x + 1] + 1 * d_img[y + 1, x - 1] + 2 * d_img[y + 1, x] + 1 * d_img[y + 1, x + 1])
        mag = math.sqrt(gx * gx + gy * gy)
        d_out[y, x] = min(255, int(mag))

def run_gpu_sobel(img):
    height, width = img.shape
    d_img = cuda.to_device(img)
    d_out = cuda.device_array_like(img)
    threads_per_block = (16, 16)
    blocks_per_grid_x = math.ceil(width / threads_per_block[0])
    blocks_per_grid_y = math.ceil(height / threads_per_block[1])
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    cuda.synchronize()
    start = time.time()
    gpu_sobel_kernel[blocks_per_grid, threads_per_block](d_img, d_out, width, height)
    cuda.synchronize()
    end = time.time()
    return d_out.copy_to_host(), end - start
