# Image_classifier


## Installation and Setup

#### Prerequisites
- Python 3.8+
- `pip` (Python package manager)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Gabriele-tomai00/Image_classifier.git
2. Navigate to the project directory:
    ```bash
    cd Image_classifier
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
4. Edit the folder path (optional)
Keep in mind that you may have to edit the paths in which the training and test images are present (It is for Colab, but you have the images in this repo, in `/images` folder)
    ```bash
    drive.mount('/content/drive/', force_remount=True)
    # Upload image training
    folder_training = '/content/drive/MyDrive/CV_progetto2/train'
    images_training, train_labels  = load_images_labels(folder_training)

    # Loading test images
    folder_test = '/content/drive/MyDrive/CV_progetto2/test'
    images_test, test_labels  = load_images_labels(folder_test)
### Final note
Please keep in mind that we have worked on Colab so to avoid having to edit some paths and packages, we suggest viewing the work from Colab.
