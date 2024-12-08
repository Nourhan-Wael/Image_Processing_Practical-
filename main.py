import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog, Button, Label, ttk, Canvas, Toplevel
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---- Algorithm Implementations ----

# Algorithm 1: Laplacian Filter
def laplacian_filter(img):
    kernel = np.array([[0, 1, 0], 
                       [1, -4, 1], 
                       [0, 1, 0]], np.float32)
    return cv2.filter2D(img, -1, kernel)

# Algorithm 2: Histogram Stretching (Grayscale)
def histogram_stretch_gray(img):
    in_min, in_max = 80 / 255.0, 200 / 255.0
    img_normalized = img / 255.0
    stretched_img = np.clip((img_normalized - in_min) / (in_max - in_min), 0, 1)
    return (stretched_img * 255).astype(np.uint8)

# Algorithm 3: Histogram Stretching (RGB Channels)
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
            "Laplacian Filter": self.apply_laplacian_filter,
            "Histogram Stretch (Grayscale)": self.apply_histogram_stretch_gray,
            "Histogram Stretch (RGB)": self.apply_histogram_stretch_rgb,
        }

        # Widgets
        self.label = Label(root, text="Choose an image and apply an algorithm!", font=("Helvetica", 14))
        self.label.pack(pady=10)

        # Create canvases for original and processed images
        self.canvas_frame = Canvas(root, width=1200, height=400, bg="white")
        self.canvas_frame.pack()

        self.original_canvas = Canvas(self.canvas_frame, width=600, height=400, bg="gray")
        self.original_canvas.place(x=0, y=0)

        self.processed_canvas = Canvas(self.canvas_frame, width=600, height=400, bg="gray")
        self.processed_canvas.place(x=600, y=0)

        # Buttons and dropdown
        self.upload_button = Button(root, text="Upload Image", command=self.upload_image, font=("Helvetica", 12))
        self.upload_button.pack(pady=5)

        self.algorithm_selector = ttk.Combobox(root, values=list(self.algorithms.keys()), state="readonly", font=("Helvetica", 12))
        self.algorithm_selector.set("Select Algorithm")
        self.algorithm_selector.pack(pady=5)

        self.apply_button = Button(root, text="Apply Algorithm", command=self.apply_algorithm, font=("Helvetica", 12))
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

    def apply_algorithm(self):
        if self.img is None:
            self.label.config(text="Please upload an image first!", fg="red")
            return

        algorithm_name = self.algorithm_selector.get()
        if algorithm_name not in self.algorithms:
            self.label.config(text="Please select a valid algorithm!", fg="red")
            return

        self.label.config(text="Processing...", fg="black")
        self.algorithms[algorithm_name]()
        self.display_processed_image(self.processed_img)
        self.label.config(text=f"Applied: {algorithm_name}", fg="green")

    def apply_laplacian_filter(self):
        self.processed_img = laplacian_filter(self.img)

    def apply_histogram_stretch_gray(self):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.processed_img = cv2.cvtColor(histogram_stretch_gray(gray_img), cv2.COLOR_GRAY2BGR)

    def apply_histogram_stretch_rgb(self):
        self.processed_img = histogram_stretch_rgb(self.img.copy())

    def show_histograms(self):
        if self.img is None or self.processed_img is None:
            self.label.config(text="Please upload an image and apply an algorithm first!", fg="red")
            return

        # Create a new window for histograms if it doesn't exist
        if not hasattr(self, 'hist_window') or not self.hist_window.winfo_exists():
            self.hist_window = Toplevel(self.root)
            self.hist_window.title("Histograms")
            self.hist_window.geometry("1000x600")

            # Create a canvas for embedding the matplotlib plot
            self.hist_canvas = Canvas(self.hist_window, width=900, height=550, bg="white")
            # self.hist_canvas.pack(pady=10)

        # Clear any previous plots from the canvas
        self.hist_canvas.delete("all")

        # Create a new matplotlib figure with specific size
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1)

        # Plot histogram of the original image
        plt.subplot(1, 2, 1)
        if len(self.img.shape) == 2:  # Grayscale
            axes[0].hist(self.img.ravel(), bins=256, range=[0, 256], color='gray')
            axes[0].set_title("Histogram of Original Image")
            axes[0].set_xlabel('Pixel Value')
            axes[0].set_ylabel('Frequency')
        else:  # RGB
            colors = ('r', 'g', 'b')
            for i, color in enumerate(colors):
                axes[0].hist(self.img[:, :, i].ravel(), bins=256, range=[0, 256], color=color, alpha=0.5)
            axes[0].set_title("Histogram of Original Image")
            axes[0].set_xlabel('Pixel Value')
            axes[0].set_ylabel('Frequency')

        # Plot histogram of the processed image
        plt.subplot(1, 2, 2)
        if len(self.processed_img.shape) == 2:  # Grayscale
            axes[1].hist(self.processed_img.ravel(), bins=256, range=[0, 256], color='gray')
            axes[1].set_title("Histogram of Processed Image")
            axes[1].set_xlabel('Pixel Value')
            axes[1].set_ylabel('Frequency')
        else:  # RGB
            for i, color in enumerate(colors):
                axes[1].hist(self.processed_img[:, :, i].ravel(), bins=256, range=[0, 256], color=color, alpha=0.5)
            axes[1].set_title("Histogram of Processed Image")
            axes[1].set_xlabel('Pixel Value')
            axes[1].set_ylabel('Frequency')

        # Adjust layout for a cleaner look
        plt.tight_layout()

        # Embed the matplotlib figure in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.hist_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack()
        plt.close(fig)  # Close the matplotlib figure to prevent displaying in a separate window


if __name__ == "__main__":
    root = Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
