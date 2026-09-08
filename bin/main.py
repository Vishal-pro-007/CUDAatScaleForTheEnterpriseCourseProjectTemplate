import os
import sys
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from cpu_sobel import run_cpu_sobel
from gpu_sobel import run_gpu_sobel
from utils import plot_visualizations, plot_performance

def main():
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "sample.png"))
    img_np = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_np is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    cpu_result, cpu_time = run_cpu_sobel(img_np)
    gpu_result, gpu_time = run_gpu_sobel(img_np)
    plot_visualizations(img_np, cpu_result, gpu_result)
    plot_performance(cpu_time, gpu_time)

if __name__ == "__main__":
    main()
