import numpy as np
from scipy.stats import wasserstein_distance

# calculate the Euclidean Distance between feature vectors
# formula: d(A, B) = square root of sum((ai-bi)^2), i = 1, 2, ..., n
def get_Euclidean_Distance(vector_1, vector_2):
    # transform input feature vectors into numpy.ndarray if they are not numpy.ndarray type
    if (type(vector_1) != 'numpy.ndarray' and type(vector_2) != 'numpy.ndarray'):
        vector_1 = np.asarray(vector_1)
        vector_2 = np.asarray(vector_2)  
    Euclidean_distance = np.sqrt(np.square(vector_1 - vector_2).sum())

    return Euclidean_distance


def get_Cosine_Distance(vector_1, vector_2):
    # transform input feature vectors into numpy.ndarray if they are not numpy.ndarray type
    if (type(vector_1) != 'numpy.ndarray' and type(vector_2) != 'numpy.ndarray'):
        vector_1 = np.asarray(vector_1)
        vector_2 = np.asarray(vector_2) 

    Cosine_distance = (float(np.dot(vector_1, vector_2)) / 
               (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))

    # Sometimes using value ranges in [0, 1] to indicate the distance, so I normalize it here in case we need this
    normalized_dist = (1 - Cosine_distance)/ 2.0

    return Cosine_distance

def get_Earth_Mover_Distance(vector_1, vector_2):

    EMD = wasserstein_distance(vector1, vector2)
    return EMD

if __name__ == "__main__":
    # test Euclidean Distance
    list1 = [1,1,1]
    list2 = [2,3,4]
    vector1 = np.asarray(list1)
    vector2 = np.asarray(list2)
    euclidean_dist = get_Euclidean_Distance(vector1, vector2)
    print('Euclidean distance: %.5f' %euclidean_dist)
    print("get_Euclidean_Distance test finished!")


    cosine_dist = get_Cosine_Distance(vector1, vector2)
    print('Cosine distance: %.5f' %cosine_dist)
    print("get_Cosine_Distance test finished!")
    
    EMD = get_Earth_Mover_Distance(vector1, vector2)
    print("Earth Mover distance: %.5f" %EMD)
    print("get_Earth_Mover_Distance test finished!")
