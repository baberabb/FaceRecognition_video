import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np



class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(input_dim, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, output_dim)
    )

  def forward(self, x):
    '''Forward pass'''
    #batch_size = x.shape[0]
    #x = x.view(batch_size, -1)
    return self.layers(x)


class ConvNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*22*22, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 7)
        # Define proportion or neurons to dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*22*22)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DatasetHog(Dataset):
    #adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """Images dataset for pytorch."""

    def __init__(self, descriptors, labels, transform=None):
        """
        Args:
            labels : Array of labels.
            descriptors (string): Array of HOG descriptors.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.labels = pd.read_csv(
                                  #label_path,
                                  #delimiter=' ',
                                  #header=None,
                                  #names=['filename', 'label']
                                #)
        #self.root_dir = root_dir
        self.labels = (labels - 1) #convert labels to 0-6
        self.descriptors = descriptors
        self.transform = transform
        #self.image_path = sorted(os.listdir(root_dir))
        #self.transform = transforms.Compose(
                                #[transforms.Normalize(),
                                #transforms.ToTensor()]
                                #)
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
                                #self.labels.iloc[idx, 0])
        #image = io.imread(img_name)
        descriptors = self.descriptors[idx]
        #label = self.labels.iloc[idx, 1]
        label = self.labels[idx]
        

        if self.transform:
            image = self.transform(descriptors)
        
        return descriptors, label



class SiftBoVW(object):
    """
    Class used for training and testing BoWV algorithm on SVM.
    """
    def __init__(self,
                 X_train,
                 y_train,
                 key_points=None,
                 k_mul=10,
                 batches=4,
                 log=False):
        """"
        Args:
        X_train: An array containing training images
        y_train: An array of labels
        key_points: key points for SIFT
        k_mul (integer): k multiplier for number of unique classes
        batches: Minibatch size for kmeans
        log: logging to wandb.io

        """
        self.X_train = X_train
        self.y_train = y_train
        self.key_points = key_points
        self.k_mul = k_mul
        self.batches = batches

        self.sift = cv2.SIFT_create()
        if log:
            run = wandb.init(reinit=True)
        # Create empty lists for feature descriptors and labels
        descriptors_list = []
        self.y_train_list = []

        for i in trange(len(X_train), desc='Extracting SIFT descriptors', leave=False):
            # Identify key points and extract descriptors with SIFT
            img = img_as_ubyte(color.rgb2gray(X_train[i]))
            kp, descriptors = self.sift.detectAndCompute(img, self.key_points)

            # Append list of descriptors and label to respective lists
            if descriptors is not None:
                descriptors_list.append(descriptors)
                self.y_train_list.append(self.y_train[i])

        # Convert to array for easier handling
        des_array = np.vstack(descriptors_list)

        # kmeans

        self.k = len(np.unique(self.y_train)) * int(k_mul)
        print(f'Number of descriptors: {des_array.shape[0]}.')
        batch_size = des_array.shape[0] // int(batches)
        print(f'K-Means batch size: {batch_size}.')
        self.kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=batch_size).fit(des_array)


        histogram_list = []
        idx_list = []

        for descriptors in tqdm(descriptors_list, desc='Predicting Clusters', leave=False):
            histogram = np.zeros(self.k)

            idx = self.kmeans.predict(descriptors)
            if log:
                wandb.sklearn.plot_elbow_curve(self.kmeans, des_array)
                wandb.sklearn.plot_silhouette(self.kmeans,
                                              des_array,)

                #wandb.sklearn.plot_summary_metrics(idx,
                                                   #des_array,
                                                   #self.y_train_list,
                                                   #model_name='KMeans')
            idx_list.append(idx)
            for j in idx:
                histogram[j] = histogram[j] + (1 / len(descriptors))
            histogram_list.append(histogram)

        self.histogram_array = np.vstack(histogram_list)

    def get(self):
        """
        Return a tuple of a list of image descriptors and respective labels for training.
        """
        return self.histogram_array, self.y_train_list

    def test(self, X_test, y_test):
        """
        Return a tuple of a list of image descriptors and respective labels for test.
        """
        hist_test_list = []

        for i in trange(len(X_test), desc='Extracting clusters for Test', leave=False):
            img = img_as_ubyte(color.rgb2gray(X_test[i]))
            kp, des = self.sift.detectAndCompute(img, None)

            if des is not None:
                hist = np.zeros(self.k)

                idx = self.kmeans.predict(des)

                for j in idx:
                    hist[j] = hist[j] + (1 / len(des))

                # hist = scale.transform(hist.reshape(1, -1))
                hist_test_list.append(hist)

            else:
                hist_test_list.append(None)

        # Remove potential cases of images with no descriptors
            idx_not_empty = [i for i, x in enumerate(hist_test_list) if x is not None]
            hist_test_list = [hist_test_list[i] for i in idx_not_empty]
            labels = [y_test[i] for i in idx_not_empty]
            hist_test_array = np.vstack(hist_test_list)

        return hist_test_array, labels

class DatasetTorch(Dataset):
    #adapted from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """Images dataset for pytorch."""

    def __init__(self, root_dir, label_path, transform=None):
        """
        Args:
            csv_file (string): Path to the text file with *exact image filenames* and  respective labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        #self.labels = pd.read_csv(
                                  #label_path,
                                  #delimiter=' ',
                                  #header=None,
                                  #names=['filename', 'label']
                                #)
        #self.root_dir = root_dir
        self.labels = label_path - 1
        self.root_dir = root_dir
        self.transform = transform
        #self.image_path = sorted(os.listdir(root_dir))
        #self.transform = transforms.Compose(
                                #[transforms.Normalize(),
                                #transforms.ToTensor()]
                                #)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()

        #img_name = os.path.join(self.root_dir,
                                #self.labels.iloc[idx, 0])
        #image = io.imread(img_name)
        image = self.root_dir[idx]
        #label = self.labels.iloc[idx, 1]
        label = self.labels[idx]
        

        if self.transform:
            image = self.transform(image)
        
        return image, label