import matplotlib.pyplot as plt

def plot_visualizations(img_np, cpu_result, gpu_result):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap="gray")
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(cpu_result, cmap="gray")
    plt.title("CPU Sobel Edge Detection")
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(gpu_result, cmap="gray")
    plt.title("GPU CUDA Sobel Edge Detection")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def plot_performance(cpu_time, gpu_time):
    labels = ["CPU", "GPU"]
    times = [cpu_time, gpu_time]
    plt.bar(labels, times)
    plt.ylabel("Execution Time in Seconds")
    plt.title("CPU vs GPU Performance")
    plt.show()
    print("Final Performance Summary")
    print("-------------------------")
    print("CPU Time:", cpu_time, "seconds")
    print("GPU Time:", gpu_time, "seconds")
    print("Speedup:", cpu_time / gpu_time)
