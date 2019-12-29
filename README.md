Part 1: Omniglot dataset

Run matchingnetwork.py to execute code and see results of 5-way 1-shot learning on Omniglot dataset.

Other files:
data.npy - contains Omniglot data.
datanway.py - extracts samples from the dataset and creates traning set and validation set.
matchnn.py - contains cosine similarity + softmax function.

Part 2: 5 Celebrity Faces dataset

Step 1: Run createfacesdataset.py to create a compressed folder containing the dataset.
Step 2: Run createfaceembeddings.py to create a compressed folder containing face embeddings.
Step 3: Run matchingnetworksforfaces.py to perform one shot learning and see results.

Other files:
facenet_keras.h5 - pre-trained FaceNet model.
5-celebrity-faces-dataset - folder containing 5 Celebrity Faces data
5-celebrity-faces-dataset.npz - compressed folder containing 5 Celebrity Faces data
5-celebrity-faces-embeddings.npz - compressed folder containing embeddings of all faces in the dataset

*********************************************************************************************************

Acknowledgements

Matching Networks code available at https://github.com/cnichkawde/MatchingNetwork
FaceNet pre-trained model available at https://github.com/nyoki-mtl/keras-facenet

Files taken from other sources:
datanway.py (unmodified)
matchnn.py (modified)
matchingnetwork.py (modified)

Files created as part of implementation:
matchingnetworksforfaces.py
createfacesdataset.py
createfaceembeddings.py

Face embeddings extraction assisted by https://machinelearningmastery.com/

