import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, ttk, Canvas, Toplevel
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Low-Pass Filter Algorithms ----

# Ideal Low-Pass Filter (Color)
def ideal_low_pass_filter_color(img, radius=50):
    """Applies an ideal low-pass filter in the frequency domain and preserves color."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]

    F = np.fft.fft2(y_channel)
    Fshift = np.fft.fftshift(F)
    rows, cols = y_channel.shape
    center_x, center_y = cols // 2, rows // 2

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    low_pass_mask = distance <= radius

    Fshift_filtered = Fshift * low_pass_mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    filtered_y_channel = np.abs(np.fft.ifft2(F_ishift))

    img_yuv[:, :, 0] = filtered_y_channel.astype(np.uint8)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# Gaussian Low-Pass Filter (Color)
def gaussian_low_pass_filter_color(img, cutoff=50):
    """Applies a Gaussian low-pass filter in the frequency domain and preserves color."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:, :, 0]

    F = np.fft.fft2(y_channel)
    Fshift = np.fft.fftshift(F)
    rows, cols = y_channel.shape
    center_x, center_y = cols // 2, rows // 2

    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    gaussian_mask = np.exp(- (distance ** 2) / (2 * (cutoff ** 2)))

    Fshift_filtered = Fshift * gaussian_mask
    F_ishift = np.fft.ifftshift(Fshift_filtered)
    filtered_y_channel = np.abs(np.fft.ifft2(F_ishift))

    img_yuv[:, :, 0] = filtered_y_channel.astype(np.uint8)
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

# ---- Edge Detection Algorithms ----

# Sobel Operator
def sobel_operator(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(sobel.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Canny Edge Detector
def canny_edge_detector(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_img, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

# Kirsch Operator
def kirsch_operator(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]]),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
    ]
    edge_images = [cv2.filter2D(gray_img, -1, k) for k in kirsch_kernels]
    kirsch = np.max(edge_images, axis=0)
    return cv2.cvtColor(kirsch.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# ---- Additional Algorithms ----

# Median Filter
def median_filter(img):
    return cv2.medianBlur(img, 5)

# Laplacian Filter
def laplacian_filter(img):
    kernel = np.array([[0, 1, 0], 
                       [1, -4, 1], 
                       [0, 1, 0]], np.float32)
    return cv2.filter2D(img, -1, kernel)

# Histogram Stretching (Grayscale)
def histogram_stretch_gray(img):
    in_min, in_max = 80 / 255.0, 200 / 255.0
    img_normalized = img / 255.0
    stretched_img = np.clip((img_normalized - in_min) / (in_max - in_min), 0, 1)
    return (stretched_img * 255).astype(np.uint8)

# Histogram Stretching (RGB Channels)
def histogram_stretch_rgb(img):
    def contrast_stretch(channel, in_min, in_max):
        channel_normalized = channel / 255.0
        stretched_channel = np.clip((channel_normalized - in_min) / (in_max - in_min), 0, 1)
        return (stretched_channel * 255).astype(np.uint8)

    img[:, :, 0] = contrast_stretch(img[:, :, 0], 30/255, 180/255)  # Red
    img[:, :, 1] = contrast_stretch(img[:, :, 1], 40/255, 160/255)  # Green
    img[:, :, 2] = contrast_stretch(img[:, :, 2], 20/255, 150/255)  # Blue
    return img

# ---- GUI Setup ----

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.geometry("1200x800")

        # Variables
        self.img = None
        self.processed_img = None
        self.algorithms = {
            "Ideal Low-Pass Filter (Color)": self.apply_ideal_low_pass_filter_color,
            "Gaussian Low-Pass Filter (Color)": self.apply_gaussian_low_pass_filter_color,
            "Sobel Operator": self.apply_sobel_operator,
            "Canny Edge Detector": self.apply_canny_edge_detector,
            "Kirsch Operator": self.apply_kirsch_operator,
            "Laplacian Filter": self.apply_laplacian_filter,
            "Median Filter": self.apply_median_filter,
            "Histogram Stretch (Grayscale)": self.apply_histogram_stretch_gray,
            "Histogram Stretch (RGB)": self.apply_histogram_stretch_rgb,
        }

        # Widgets
        self.label = Label(root, text="Choose an image and apply a filter!", font=("Helvetica", 14))
        self.label.pack(pady=10)

        self.canvas_frame = Canvas(root, width=1200, height=400, bg="white")
        self.canvas_frame.pack()

        self.original_canvas = Canvas(self.canvas_frame, width=600, height=400, bg="gray")
        self.original_canvas.place(x=0, y=0)

        self.processed_canvas = Canvas(self.canvas_frame, width=600, height=400, bg="gray")
        self.processed_canvas.place(x=600, y=0)

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image, font=("Helvetica", 12))
        self.upload_button.pack(pady=5)

        self.algorithm_selector = ttk.Combobox(root, values=list(self.algorithms.keys()), state="readonly", font=("Helvetica", 12))
        self.algorithm_selector.set("Select Filter")
        self.algorithm_selector.pack(pady=5)

        self.apply_button = Button(root, text="Apply Filter", command=self.apply_filter, font=("Helvetica", 12))
        self.apply_button.pack(pady=10)

        self.histogram_button = Button(root, text="Show Histograms", command=self.show_histograms, font=("Helvetica", 12))
        self.histogram_button.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp;*.gif")]
        )
        if file_path:
            self.img = cv2.imread(file_path)
            self.display_original_image(self.img)

    def display_original_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((600, 400)))
        self.original_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.original_canvas.image = img_tk

    def display_processed_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil.resize((600, 400)))
        self.processed_canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.processed_canvas.image = img_tk

    def apply_filter(self):
        if self.img is None:
            self.label.config(text="Please upload an image first!", fg="red")
            return

        filter_name = self.algorithm_selector.get()
        if filter_name not in self.algorithms:
            self.label.config(text="Please select a valid filter!", fg="red")
            return

        self.label.config(text="Processing...", fg="black")
        self.algorithms[filter_name]()
        self.display_processed_image(self.processed_img)
        self.label.config(text=f"Applied: {filter_name}", fg="green")

    def apply_ideal_low_pass_filter_color(self):
        self.processed_img = ideal_low_pass_filter_color(self.img)

    def apply_gaussian_low_pass_filter_color(self):
        self.processed_img = gaussian_low_pass_filter_color(self.img)

    def apply_sobel_operator(self):
        self.processed_img = sobel_operator(self.img)

    def apply_canny_edge_detector(self):
        self.processed_img = canny_edge_detector(self.img)

    def apply_kirsch_operator(self):
        self.processed_img = kirsch_operator(self.img)

    def apply_laplacian_filter(self):
        self.processed_img = laplacian_filter(self.img)

    def apply_median_filter(self):
        self.processed_img = median_filter(self.img)

    def apply_histogram_stretch_gray(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.processed_img = cv2.cvtColor(histogram_stretch_gray(gray_img), cv2.COLOR_GRAY2BGR)

    def apply_histogram_stretch_rgb(self):
        self.processed_img = histogram_stretch_rgb(self.img.copy())

    def show_histograms(self):
        if self.img is None or self.processed_img is None:
            self.label.config(text="Please upload an image and apply an algorithm first!", fg="red")
            return

        if not hasattr(self, 'hist_window') or not self.hist_window.winfo_exists():
            self.hist_window = Toplevel(self.root)
            self.hist_window.title("Histograms")
            self.hist_window.geometry("1000x600")

            self.hist_canvas = Canvas(self.hist_window, width=900, height=500)
            self.hist_canvas.pack()

            fig, axs = plt.subplots(1, 2, figsize=(10, 5))

            axs[0].hist(self.img.ravel(), bins=256, color='gray')
            axs[0].set_title("Original Image Histogram")
            axs[0].set_xlim(0, 255)

            axs[1].hist(self.processed_img.ravel(), bins=256, color='gray')
            axs[1].set_title("Processed Image Histogram")
            axs[1].set_xlim(0, 255)

            canvas = FigureCanvasTkAgg(fig, master=self.hist_canvas)
            canvas.draw()
            canvas.get_tk_widget().pack()

root = Tk()
app = ImageProcessingApp(root)
root.mainloop()
