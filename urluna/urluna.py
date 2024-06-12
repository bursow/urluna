

import random
import math

class Flex:
    """
    A class for performing various flexible array operations.
    """

    @classmethod
    def fill_array(cls, value, num_rows, num_cols):
        """
        Generates a 2D array filled with the specified value.

        :param value: Value to fill the array with.
        :param num_rows: Number of rows in the array.
        :param num_cols: Number of columns in the array.
        :return: 2D array filled with the value.
        """
        if not isinstance(value, int) or not isinstance(num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("All parameters must be integers")
        if num_rows < 0 or num_cols < 0:
            raise ValueError("Number of rows and columns must be non-negative")
        return [[value] * num_cols for _ in range(num_rows)]
    @classmethod
    def sorted_random_array(cls, min_value, max_value, num_rows, num_cols):
        """
        Generates a 2D array filled with random values sorted in ascending order.

        :param min_value: Minimum value for random generation.
        :param max_value: Maximum value for random generation.
        :param num_rows: Number of rows in the array.
        :param num_cols: Number of columns in the array.
        :return: 2D array filled with random values sorted in ascending order.
        """
        if not isinstance(min_value, int) or not isinstance(max_value, int) or not isinstance(num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("All parameters must be integers")
        if min_value > max_value:
            raise ValueError("Minimum value must be less than or equal to Maximum value")
        random_list = [random.randint(min_value, max_value) for _ in range(num_rows * num_cols)]
        random_list.sort()
        return [random_list[i * num_cols: (i + 1) * num_cols] for i in range(num_rows)]

    @classmethod
    def random_array(cls, min_value, max_value, num_rows, num_cols):
        """
        Generates a 2D array filled with random values.

        :param min_value: Minimum value for random generation.
        :param max_value: Maximum value for random generation.
        :param num_rows: Number of rows in the array.
        :param num_cols: Number of columns in the array.
        :return: 2D array filled with random values.
        """
        if not isinstance(min_value, int) or not isinstance(max_value, int) or not isinstance(num_rows, int) or not isinstance(num_cols, int):
            raise TypeError("All parameters must be integers")
        if min_value > max_value:
            raise ValueError("Minimum value must be less than or equal to Maximum value")
        return [[random.randint(min_value, max_value) for _ in range(num_cols)] for _ in range(num_rows)]

    @classmethod
    def range_array(cls, start, stop=None, step=1):
        """
        Generates an array containing evenly spaced values within a given range.

        :param start: Start of the range.
        :param stop: End of the range.
        :param step: Step between each value.
        :return: Array containing evenly spaced values within the given range.
        """
        if stop is None:
            start, stop = 0, start

        if not isinstance(start, int) or not isinstance(stop, int) or not isinstance(step, int):
            raise TypeError("Start, stop, and step must be integers")

        if step == 0:
            raise ValueError("Step must not be zero")

        result = list(range(start, stop, step))
        return [result]

    @classmethod
    def zeros_array(cls, num_rows, num_cols):
        """
        Generates a 2D array filled with zeros.

        :param num_rows: Number of rows in the array.
        :param num_cols: Number of columns in the array.
        :return: 2D array filled with zeros.
        """
        return cls.fill_array(0, num_rows, num_cols)

    @classmethod
    def ones_array(cls, num_rows, num_cols):
        """
        Generates a 2D array filled with ones.

        :param num_rows: Number of rows in the array.
        :param num_cols: Number of columns in the array.
        :return: 2D array filled with ones.
        """
        return cls.fill_array(1, num_rows, num_cols)
    
    @classmethod
    def identity_matrix(cls, size):
        """
        Generates an identity matrix of given size.

        :param size: Size of the identity matrix.
        :return: Identity matrix.
        """
        if not isinstance(size, int) or size < 0:
            raise TypeError("Size must be a non-negative integer")
        return [[1 if i == j else 0 for j in range(size)] for i in range(size)]

    @classmethod
    def diagonal_matrix(cls, diagonal):
        """
        Generates a diagonal matrix with the given diagonal elements.

        :param diagonal: List of diagonal elements.
        :return: Diagonal matrix.
        """
        if not isinstance(diagonal, list):
            raise TypeError("Diagonal must be a list of numbers")
        size = len(diagonal)
        return [[diagonal[i] if i == j else 0 for j in range(size)] for i in range(size)]



class Operator:
    """
    A class for performing basic array operations.
    """

    @classmethod
    def add_arrays(cls, array1, array2):
        """
        Adds two arrays element-wise.

        :param array1: First array.
        :param array2: Second array.
        :return: Resultant array after element-wise addition.
        """
        if not isinstance(array1, list) or not isinstance(array2, list):
            raise TypeError("Both parameters must be lists")
        if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
            raise ValueError("Both arrays must have the same dimensions")

        return [[array1[i][j] + array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]

    @classmethod
    def subtract_arrays(cls, array1, array2):
        """
        Subtracts two arrays element-wise.

        :param array1: First array.
        :param array2: Second array.
        :return: Resultant array after element-wise subtraction.
        """
        if not isinstance(array1, list) or not isinstance(array2, list):
            raise TypeError("Both parameters must be lists")
        if len(array1) != len(array2) or len(array1[0]) != len(array2[0]):
            raise ValueError("Both arrays must have the same dimensions")

        return [[array1[i][j] - array2[i][j] for j in range(len(array1[0]))] for i in range(len(array1))]

    @classmethod
    def transpose(cls, matrix):
        """
        Transposes the given matrix.

        :param matrix: Matrix to be transposed.
        :return: Transposed matrix.
        """
        if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            raise TypeError("Matrix must be a list of lists")
        return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

    @classmethod
    def multiply_matrices(cls, matrix1, matrix2):
        """
        Multiplies two matrices.

        :param matrix1: First matrix.
        :param matrix2: Second matrix.
        :return: Resultant matrix after multiplication.
        """
        if not isinstance(matrix1, list) or not isinstance(matrix2, list):
            raise TypeError("Both parameters must be lists")
        if not all(isinstance(row, list) for row in matrix1) or not all(isinstance(row, list) for row in matrix2):
            raise TypeError("Both parameters must be lists of lists")
        if len(matrix1[0]) != len(matrix2):
            raise ValueError("Number of columns in the first matrix must equal the number of rows in the second matrix")

        result = [[0 for _ in range(len(matrix2[0]))] for _ in range(len(matrix1))]
        for i in range(len(matrix1)):
            for j in range(len(matrix2[0])):
                for k in range(len(matrix2)):
                    result[i][j] += matrix1[i][k] * matrix2[k][j]
        return result

    @classmethod
    def determinant(cls, matrix):
        """
        Calculates the determinant of the given square matrix.

        :param matrix: Square matrix to calculate the determinant.
        :return: Determinant of the matrix.
        """
        if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            raise TypeError("Matrix must be a list of lists")
        n = len(matrix)
        if n != len(matrix[0]):
            raise ValueError("Matrix must be square")
        if n == 1:
            return matrix[0][0]
        if n == 2:
            return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        det = 0
        for i in range(n):
            det += ((-1) ** i) * matrix[0][i] * cls.determinant([row[:i] + row[i + 1:] for row in matrix[1:]])
        return det

    @classmethod
    def inverse_matrix(cls, matrix):
        """
        Calculates the inverse of a matrix.

        :param matrix: Input matrix.
        :return: Inverse of the input matrix.
        """
        if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            raise TypeError("Matrix must be a list of lists")
        n = len(matrix)
        if n != len(matrix[0]):
            raise ValueError("Matrix must be square")
        identity = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        for i in range(n):
            row = matrix[i]
            if row[i] == 0:
                for j in range(i + 1, n):
                    if matrix[j][i] != 0:
                        matrix[i], matrix[j] = matrix[j], matrix[i]
                        identity[i], identity[j] = identity[j], identity[i]
                        break
            if matrix[i][i] == 0:
                raise ValueError("Matrix is singular")
            scalar = 1 / matrix[i][i]
            matrix[i] = [elem * scalar for elem in matrix[i]]
            identity[i] = [elem * scalar for elem in identity[i]]
            for j in range(n):
                if i != j:
                    scalar = matrix[j][i]
                    matrix[j] = [elem1 - elem2 * scalar for elem1, elem2 in zip(matrix[j], matrix[i])]
                    identity[j] = [elem1 - elem2 * scalar for elem1, elem2 in zip(identity[j], identity[i])]
        return identity
    
    @classmethod
    def hadamard_product(cls, matrix1, matrix2):
        """
        Computes the Hadamard product (element-wise multiplication) of two matrices.

        :param matrix1: First matrix.
        :param matrix2: Second matrix.
        :return: Resultant matrix after element-wise multiplication.
        """
        if not isinstance(matrix1, list) or not isinstance(matrix2, list):
            raise TypeError("Both parameters must be lists")
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            raise ValueError("Both matrices must have the same dimensions")
        
        return [[matrix1[i][j] * matrix2[i][j] for j in range(len(matrix1[0]))] for i in range(len(matrix1))]

    @classmethod
    def scalar_multiply(cls, matrix, scalar):
        """
        Multiplies a matrix by a scalar value.

        :param matrix: Input matrix.
        :param scalar: Scalar value.
        :return: Resultant matrix after scalar multiplication.
        """
        if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
            raise TypeError("Matrix must be a list of lists")
        if not isinstance(scalar, (int, float)):
            raise TypeError("Scalar must be a number")

        return [[scalar * element for element in row] for row in matrix]




class MachineLearning:
    """
    A class for machine learning algorithms.
    """

    @staticmethod
    def logistic_regression(X_train, y_train, X_test):
        """
        Logistic regression classifier.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :return: Predicted labels.
        """
        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        def predict(weights, sample):
            z = sum(weight * feature for weight, feature in zip(weights, sample))
            return sigmoid(z)

        def train(X, y, learning_rate=0.01, num_epochs=100):
            num_features = len(X[0])
            weights = [0] * num_features
            for _ in range(num_epochs):
                for features, label in zip(X, y):
                    prediction = predict(weights, features)
                    error = label - prediction
                    for i in range(num_features):
                        weights[i] += learning_rate * error * features[i]
            return weights

        weights = train(X_train, y_train)
        return [1 if predict(weights, sample) >= 0.5 else 0 for sample in X_test]

    @staticmethod
    def k_nearest_neighbors(X_train, y_train, X_test, k=3):
        """
        k-nearest neighbors classifier.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :param k: Number of neighbors.
        :return: Predicted labels.
        """
        def euclidean_distance(point1, point2):
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

        def most_common_label(labels):
            return max(set(labels), key=labels.count)

        def predict(X_train, y_train, sample, k):
            distances = [(euclidean_distance(sample, train_sample), label) for train_sample, label in zip(X_train, y_train)]
            sorted_distances = sorted(distances, key=lambda x: x[0])
            k_nearest_labels = [label for _, label in sorted_distances[:k]]
            return most_common_label(k_nearest_labels)

        return [predict(X_train, y_train, test_sample, k) for test_sample in X_test]

    @staticmethod
    def naive_bayes_classifier(X_train, y_train, X_test):
        """
        Naive Bayes classifier.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :return: Predicted labels.
        """
        def calculate_class_probabilities(X_train, y_train):
            class_probabilities = {}
            total_samples = len(y_train)
            for class_label in set(y_train):
                class_samples = [X_train[i] for i in range(total_samples) if y_train[i] == class_label]
                class_probabilities[class_label] = len(class_samples) / total_samples
            return class_probabilities

        def calculate_feature_statistics(X_train, y_train):
            feature_statistics = {}
            total_samples = len(y_train)
            num_features = len(X_train[0])
            for class_label in set(y_train):
                class_samples = [X_train[i] for i in range(total_samples) if y_train[i] == class_label]
                class_feature_values = list(zip(*class_samples))
                class_feature_stats = [(sum(feature) / len(feature), math.sqrt(sum((x - (sum(feature) / len(feature))) ** 2 for x in feature) / len(feature))) for feature in class_feature_values]
                feature_statistics[class_label] = class_feature_stats
            return feature_statistics

        def calculate_probability(x, mean, stdev):
            exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
            return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

        class_probabilities = calculate_class_probabilities(X_train, y_train)
        feature_statistics = calculate_feature_statistics(X_train, y_train)
        predicted_labels = []
        for sample in X_test:
            probabilities = {}
            for class_label, class_prob in class_probabilities.items():
                probabilities[class_label] = class_prob
                for i in range(len(sample)):
                    mean, stdev = feature_statistics[class_label][i]
                    probabilities[class_label] *= calculate_probability(sample[i], mean, stdev)
            predicted_label = max(probabilities, key=probabilities.get)
            predicted_labels.append(predicted_label)
        return predicted_labels

    @staticmethod
    def linear_regression(X_train, y_train, X_test):
        """
        Linear regression.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :return: Predicted values.
        """
        def predict(weights, sample):
            return sum(weight * feature for weight, feature in zip(weights, sample))

        def train(X, y):
            num_features = len(X[0])
            weights = [0] * num_features
            transpose_X = Operator.transpose(X)
            X_transpose_X = Operator.multiply_matrices(transpose_X, X)
            inverse_X_transpose_X = Operator.inverse_matrix(X_transpose_X)
            X_transpose_y = Operator.multiply_matrices(transpose_X, [[label] for label in y])
            weights_matrix = Operator.multiply_matrices(inverse_X_transpose_X, X_transpose_y)
            weights = [weight[0] for weight in weights_matrix]
            return weights

        weights = train(X_train, y_train)
        return [predict(weights, sample) for sample in X_test]

    @staticmethod
    def decision_tree(X_train, y_train, X_test, max_depth=None, min_samples_split=2):
        """
        Decision tree classifier with additional parameters for max depth and min samples split.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :param max_depth: Maximum depth of the tree.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :return: Predicted labels.
        """
        class Node:
            def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
                self.feature_index = feature_index
                self.threshold = threshold
                self.value = value
                self.left = left
                self.right = right

        def gini_impurity(labels):
            total_samples = len(labels)
            if total_samples == 0:
                return 0
            probabilities = [labels.count(label) / total_samples for label in set(labels)]
            return 1 - sum(probability ** 2 for probability in probabilities)

        def find_best_split(X, y):
            best_gini = float('inf')
            best_feature_index = None
            best_threshold = None
            for feature_index in range(len(X[0])):
                for sample in X:
                    for threshold in set(sample):
                        left_X, right_X, left_y, right_y = [], [], [], []
                        for sample, label in zip(X, y):
                            if sample[feature_index] < threshold:
                                left_X.append(sample)
                                left_y.append(label)
                            else:
                                right_X.append(sample)
                                right_y.append(label)
                        gini = (len(left_y) / len(y)) * gini_impurity(left_y) + (len(right_y) / len(y)) * gini_impurity(right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature_index = feature_index
                            best_threshold = threshold
            return best_feature_index, best_threshold

        def build_tree(X, y, depth=0):
            if len(set(y)) == 1:
                return Node(value=y[0])
            if max_depth is not None and depth >= max_depth:
                return Node(value=max(set(y), key=y.count))
            if len(y) < min_samples_split:
                return Node(value=max(set(y), key=y.count))
            
            best_feature_index, best_threshold = find_best_split(X, y)
            if best_feature_index is None:
                return Node(value=max(set(y), key=y.count))
            
            left_X, right_X, left_y, right_y = [], [], [], []
            for sample, label in zip(X, y):
                if sample[best_feature_index] < best_threshold:
                    left_X.append(sample)
                    left_y.append(label)
                else:
                    right_X.append(sample)
                    right_y.append(label)
            
            left = build_tree(left_X, left_y, depth + 1)
            right = build_tree(right_X, right_y, depth + 1)
            return Node(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)

        def predict(node, sample):
            if node.value is not None:
                return node.value
            if sample[node.feature_index] < node.threshold:
                return predict(node.left, sample)
            else:
                return predict(node.right, sample)

        root = build_tree(X_train, y_train)
        return [predict(root, sample) for sample in X_test]

    @staticmethod
    def random_forest(X_train, y_train, X_test, n_trees=100, max_depth=None, min_samples_split=2, max_features=None):
        """
        Random forest classifier.

        :param X_train: Training data.
        :param y_train: Training labels.
        :param X_test: Test data.
        :param n_trees: Number of trees in the random forest.
        :param max_depth: Maximum depth of the trees.
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param max_features: Number of features to consider when looking for the best split. If None, all features will be considered.
        :return: Predicted labels.
        """
        def bootstrap_sample(X, y):
            n_samples = len(X)
            indices = [random.randint(0, n_samples - 1) for _ in range(n_samples)]
            return [X[i] for i in indices], [y[i] for i in indices]

        def most_common_label(labels):
            return max(set(labels), key=labels.count)

        def build_tree(X, y, depth=0):
            if len(set(y)) == 1 or depth == max_depth:
                return most_common_label(y)

            n_features = len(X[0])
            if max_features and max_features <= n_features:
                feature_indices = random.sample(range(n_features), max_features)
                X = [[sample[i] for i in feature_indices] for sample in X]

            feature_index, threshold = find_best_split(X, y)

            if feature_index is None:
                return most_common_label(y)

            left_X, right_X, left_y, right_y = [], [], [], []
            for sample, label in zip(X, y):
                if sample[feature_index] < threshold:
                    left_X.append(sample)
                    left_y.append(label)
                else:
                    right_X.append(sample)
                    right_y.append(label)

            left_subtree = build_tree(left_X, left_y, depth + 1)
            right_subtree = build_tree(right_X, right_y, depth + 1)

            return {'feature_index': feature_index, 'threshold': threshold,
                    'left': left_subtree, 'right': right_subtree}

        def find_best_split(X, y):
            best_gini = float('inf')
            best_feature_index = None
            best_threshold = None
            for feature_index in range(len(X[0])):
                for sample in X:
                    for threshold in set(sample):
                        left_X, right_X, left_y, right_y = [], [], [], []
                        for sample, label in zip(X, y):
                            if sample[feature_index] < threshold:
                                left_X.append(sample)
                                left_y.append(label)
                            else:
                                right_X.append(sample)
                                right_y.append(label)
                        gini = (len(left_y) / len(y)) * gini_impurity(left_y) + (
                                    len(right_y) / len(y)) * gini_impurity(right_y)
                        if gini < best_gini:
                            best_gini = gini
                            best_feature_index = feature_index
                            best_threshold = threshold
            return best_feature_index, best_threshold

        def gini_impurity(labels):
            total_samples = len(labels)
            if total_samples == 0:
                return 0
            probabilities = [labels.count(label) / total_samples for label in set(labels)]
            return 1 - sum(probability ** 2 for probability in probabilities)

        def predict(tree, sample):
            if isinstance(tree, dict):
                if sample[tree['feature_index']] < tree['threshold']:
                    return predict(tree['left'], sample)
                else:
                    return predict(tree['right'], sample)
            else:
                return tree

        forest = []
        for _ in range(n_trees):
            X_sample, y_sample = bootstrap_sample(X_train, y_train)
            tree = build_tree(X_sample, y_sample)
            forest.append(tree)

        predictions = []
        for sample in X_test:
            tree_predictions = [predict(tree, sample) for tree in forest]
            predictions.append(most_common_label(tree_predictions))

        return predictions
    

    
    

class Preprocessing:
    """
    A class for data preprocessing.
    """

    @staticmethod
    def standardize_data(data):
        """
        Standardizes the input data.

        :param data: Input data.
        :return: Standardized data.
        """
        means = [sum(feature) / len(feature) for feature in zip(*data)]
        std_devs = [math.sqrt(sum((x - mean) ** 2 for x in feature) / len(feature)) for mean, feature in zip(means, zip(*data))]
        return [[(x - mean) / std_dev for x, mean, std_dev in zip(sample, means, std_devs)] for sample in data]

    @staticmethod
    def clean_data(data):
        """
        Cleans the input data by removing missing or inconsistent values.

        :param data: Input data.
        :return: Cleaned data.
        """
        cleaned_data = []
        for row in data:
            cleaned_row = [value for value in row if value is not None]
            if len(cleaned_row) == len(row):
                cleaned_data.append(cleaned_row)
        return cleaned_data

    @staticmethod
    def transform_data(data, transformation):
        """
        Transforms the input data using the specified transformation.

        :param data: Input data.
        :param transformation: Transformation function.
        :return: Transformed data.
        """
        return [transformation(row) for row in data]

    @staticmethod
    def prepare_data(data):
        """
        Prepares the input data for machine learning algorithms.

        :param data: Input data.
        :return: Prepared data.
        """
        return Preprocessing.standardize_data(Preprocessing.clean_data(data))