import pandas as pd
import numpy as np
from json import dumps, JSONEncoder
from numpy import array
from operator import add, eq, ge
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

# Supporting Override for Converting Numpy Types into Python Values
class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

# Revised tree_classifier.py snippet for binary classification probabilities

class TreeClassifier:

    def __init__(self, source, encoder=None, X=None, y=None):
        self.source = source
        self.encoder = encoder
        self.class_distributions = {}  # Store class counts per leaf for probabilities
        if not X is None and not y is None:
            self.__initialize_training_distributions__(X, y)  # Use distributions instead of loss

    # def __initialize_training_distributions__(self, X, y):
    #     """
    #     Computes class distributions (proportions) for each leaf to estimate π̂_0(x) and π̂_1(x).
    #     """
    #     # Initialize class counts for each leaf
    #     for node in self.__all_leaves__():
    #         node["class_counts"] = {0: 0, 1: 0}  # Binary labels 0 and 1
    #         node["sample_count"] = 0

    #     (n, m) = X.shape
    #     for i in range(n):
    #         node = self.__find_leaf__(X.values[i, :])
    #         label = y.values[i] if y.ndim == 1 else y.values[i, -1]
    #         node["class_counts"][label] += 1
    #         node["sample_count"] += 1

    #     # Compute probabilities for each leaf
    #     for node in self.__all_leaves__():
    #         total = node["sample_count"]
    #         if total > 0:
    #             node["probabilities"] = {
    #                 0: node["class_counts"][0] / total,
    #                 1: node["class_counts"][1] / total
    #             }
    #             node["prediction"] = 1 if node["probabilities"][1] > node["probabilities"][0] else 0
    #         else:
    #             node["probabilities"] = {0: 0.5, 1: 0.5}  # Default uniform if no samples

    def __initialize_training_distributions__(self, X, y):
        """
        Computes class distributions (proportions) for each leaf to estimate π̂_0(x) and π̂_1(x).
        """
        # Initialize or update class counts for each leaf
        for node in self.__all_leaves__():
            if "class_counts" not in node:
                node["class_counts"] = {0: 0, 1: 0}
            if "sample_count" not in node:
                node["sample_count"] = 0

        (n, m) = X.shape
        for i in range(n):
            node = self.__find_leaf__(X.values[i, :])
            label = y.values[i] if y.ndim == 1 else y.values[i, -1]
            node["class_counts"][label] += 1
            node["sample_count"] += 1

        # Compute probabilities for each leaf
        for node in self.__all_leaves__():
            total = node["sample_count"]
            if total > 0:
                node["probabilities"] = {
                    0: node["class_counts"][0] / total,
                    1: node["class_counts"][1] / total
                }
                if "prediction" not in node or node["prediction"] is None:
                    node["prediction"] = 1 if node["probabilities"][1] > node["probabilities"][0] else 0
            else:
                node["probabilities"] = {0: 0.5, 1: 0.5}  # Default uniform if no samples
                if "prediction" not in node or node["prediction"] is None:
                    node["prediction"] = 0  # Default prediction for empty leaf



    def classify(self, sample):
        """
        Returns the predicted label and class probabilities.
        """
        node = self.__find_leaf__(sample)
        return node["prediction"], node["probabilities"]

    def predict(self, X):
        """
        Predict labels using class probabilities.
        """
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)
        
        predictions = []
        (n, m) = X.shape
        for i in range(n):
            prediction, _ = self.classify(X.values[i, :])
            predictions.append(prediction)
        return np.array(predictions)

    def confidence(self, X):
        """
        Returns probabilities for the predicted class (for backward compatibility, but probabilities are now primary).
        """
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        confidences = []
        (n, m) = X.shape
        for i in range(n):
            _, probs = self.classify(X.values[i, :])
            pred, _ = self.classify(X.values[i, :])  # Get predicted label
            confidences.append(probs[pred])  # Return probability of the predicted class
        return np.array(confidences)

    def probabilities(self, X):
        """
        Returns full class probabilities (π̂_0(x), π̂_1(x)) for each sample.
        """
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:, :]), columns=self.encoder.headers)

        probs_list = []
        (n, m) = X.shape
        for i in range(n):
            _, probs = self.classify(X.values[i, :])
            probs_list.append([probs[0], probs[1]])  # [π̂_0(x), π̂_1(x)]
        return np.array(probs_list)

# class TreeClassifier:
#     """
#     Unified representation of a tree classifier in Python

#     This class accepts a dictionary representation of a tree classifier and decodes it into an interactive object

#     Additional support for encoding/decoding layer can be layers if the feature-space of the model differs from the feature space of the original data
#     """
#     def __init__(self, source, encoder=None, X=None, y=None):
#         self.source = source # The classifier stored in a recursive dictionary structure
#         self.encoder = encoder # Optional encoder / decoder unit to run before / after prediction
#         if not X is None and not y is None: # Original training features and labels to fill in missing training loss values
#             self.__initialize_training_loss__(X, y)

    
#     def __initialize_training_loss__(self, X, y):
#         """
#         Compares every prediction y_hat against the labels y, then incorporates the misprediction into the stored loss values
#         and computes confidence as the proportion of samples in each leaf that match the predicted label.
#         This is used when parsing models from an algorithm that doesn't provide the training loss in the output
#         """
#         # Initialize loss and counts for each leaf
#         for node in self.__all_leaves__():
#             node["loss"] = 0.0
#             node["sample_count"] = 0  # Total samples in the leaf
#             node["correct_count"] = 0  # Samples matching the prediction

#         (n, m) = X.shape
#         for i in range(n):
#             node = self.__find_leaf__(X.values[i,:])
#             label = y.values[i] if y.ndim == 1 else y.values[i, -1]
#             weight = 1 / n
#             node["sample_count"] += 1
#             if node["prediction"] == label:
#                 node["correct_count"] += 1
#             else:
#                 node["loss"] += weight
        
#         # Compute confidence for each leaf
#         for node in self.__all_leaves__():
#             if node["sample_count"] > 0:
#                 node["confidence"] = node["correct_count"] / node["sample_count"]
#             else:
#                 node["confidence"] = 1.0  # Default to 1 if no samples (edge case)
#         return

    # def __find_leaf__(self, sample):
    #     """
    #     Returns
    #     ---
    #     the leaf by which this sample would be classified        
    #     """
    #     nodes = [self.source]
    #     while len(nodes) > 0:
    #         node = nodes.pop()
    #         if "prediction" in node:
    #             return node
    #         else:
    #             value = sample[node["feature"]]
    #             if value == 1:
    #                 nodes.append(node["true"])
    #             else:
    #                 nodes.append(node["false"])

    # def __all_leaves__(self):
    #     """
    #     Returns
    #     ---
    #     list : a list of all leaves in this model
    #     """
    #     nodes = [self.source]
    #     leaf_list = []
    #     while len(nodes) > 0:
    #         node = nodes.pop()
    #         if "prediction" in node:
    #             leaf_list.append(node)
    #         else:
    #             nodes.append(node["true"])
    #             nodes.append(node["false"])
    #     return leaf_list

    def __find_leaf__(self, sample):
        """
        Returns the leaf by which this sample would be classified.
        """
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return node
            else:
                value = sample[node["feature"]]
                if value == 1:
                    nodes.append(node["true"])
                else:
                    nodes.append(node["false"])

    def __all_leaves__(self):
        """
        Returns a list of all leaves in this model.
        """
        nodes = [self.source]
        leaf_list = []
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaf_list.append(node)
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaf_list
        
    def loss(self):
        """
        Returns
        ---
        real number : values between [0,1]
            the training loss of this model
        """
        return sum( node["loss"] for node in self.__all_leaves__() )

    # def classify(self, sample):
    #     """
    #     Parameters
    #     ---
    #     sample : array-like, shape = [m_features]
    #         a 1-by-m row representing each feature of a single sample

    #     Returns
    #     ---
    #     tuple : (prediction, confidence) where prediction is the predicted label and confidence is the proportion of training samples in the leaf matching the prediction
    #     """
    #     node = self.__find_leaf__(sample)
    #     return node["prediction"], node.get("confidence", 1.0)  # Default to 1.0 if confidence not computed

    # def predict(self, X):
    #     """
    #     Requires
    #     ---
    #     the set of features used should be pre-encoding if an encoder is used

    #     Parameters
    #     ---
    #     X : matrix-like, shape = [n_samples by m_features]
    #         a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

    #     Returns
    #     ---
    #     array-like, shape = [n_samples by 1] : a column where each element is the prediction associated with each row
    #     """
    #     if not self.encoder is None: # Perform an encoding if an encoding unit is specified
    #         X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)
        
    #     predictions = []
    #     (n, m) = X.shape
    #     for i in range(n):
    #         prediction, _ = self.classify(X.values[i,:])
    #         predictions.append(prediction)
    #     return array(predictions)

    # def confidence(self, X):
    #     """
    #     Requires
    #     ---
    #     the set of features used should be pre-encoding if an encoder is used

    #     Parameters
    #     ---
    #     X : matrix-like, shape = [n_samples by m_features]
    #         a matrix where each row is a sample to be predicted and each column is a feature to be used for prediction

    #     Returns
    #     ---
    #     array-like, shape = [n_samples by 1] : a column where each element is the confidence of each prediction
    #     """
    #     if not self.encoder is None:
    #         X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)

    #     confidences = []
    #     (n, m) = X.shape
    #     for i in range(n):
    #         _, confidence = self.classify(X.values[i,:])
    #         confidences.append(confidence)
    #     return array(confidences)
    
    def error(self, X, y, weight=None):
        """
        Parameters
        --- 
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification

        Returns
        ---
        real number : the inaccuracy produced by applying this model over the given dataset, with optionals for weighted inaccuracy
        """
        return 1 - self.score(X, y, weight=weight)

    
    def score(self, X, y, weight=None):
        """
        Parameters
        --- 
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification

        Returns
        ---
        real number : the accuracy produced by applying this model over the given dataset, with optionals for weighted accuracy
        """
        y_hat = self.predict(X)
        if weight == "balanced":
            return balanced_accuracy_score(y, y_hat)
        else:
            return accuracy_score(y, y_hat, normalize=True, sample_weight=weight)
    
    def confusion(self, X, y, weight=None):
        """
        Parameters
        --- 
        X : matrix-like, shape = [n_samples by m_features]
            an n-by-m matrix of sample and their features
        y : array-like, shape = [n_samples by 1]
            an n-by-1 column of labels associated with each sample
        weight : real number
            an n-by-1 column of weights to apply to each sample's misclassification

        Returns
        ---
        matrix-like, shape = [k_classes by k_classes] : the confusion matrix of all classes present in the dataset
        """
        return confusion_matrix(y, self.predict(X), sample_weight=weight)

    def __len__(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        return self.leaves()

    def leaves(self):
        """
        Returns
        ---
        natural number : The number of terminal nodes present in this tree
        """
        leaves_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaves_counter += 1
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaves_counter
    
    def nodes(self):
        """
        Returns
        ---
        natural number : The number of nodes present in this tree
        """
        nodes_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                nodes_counter += 1
            else:
                nodes_counter += 1
                nodes.append(node["true"])
                nodes.append(node["false"])
        return nodes_counter


    def features(self):
        """
        Returns
        ---
        set : A set of strings each describing the features used by this model
        """
        feature_set = set()
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                continue
            else:
                feature_set.add(node["feature"])
                nodes.append(node["true"])
                nodes.append(node["false"])
        return feature_set 

    def encoded_features(self):
        """
        Returns
        ---
        natural number : The number of encoded features used by the supplied encoder to represent the data set
        """
        return len(self.encoder.headers) if not self.encoder is None else None

    def maximum_depth(self, node=None):
        """
        Returns
        ---
        natural number : the length of the longest decision path in this tree. A single-node tree will return 1.
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            return 1
        else:
            return 1 + max(self.maximum_depth(node["true"]), self.maximum_depth(node["false"]))

    def __str__(self):
        """
        Returns
        ---
        string : pseudocode representing the logic of this classifier
        """
        cases = []
        for group in self.__groups__():
            predicates = []
            for name in sorted(group["rules"].keys()):
                domain = group["rules"][name]
                if domain["type"] == "Categorical":
                    if len(domain["positive"]) > 0:
                        predicates.append("{} = {}".format(name, list(domain["positive"])[0]))
                    elif len(domain["negative"]) > 0:
                        if len(domain["negative"]) > 1:
                            predicates.append("{} not in {{ {} }}".format(name, ", ".join([ str(v) for v in domain["negative"] ])) )
                        else:
                            predicates.append("{} != {}".format(name, str(list(domain["negative"])[0])))
                    else:
                        raise "Invalid Rule"
                elif domain["type"] == "Numerical":
                    predicate = name
                    if domain["min"] != -float("INF"):
                        predicate = "{} <= ".format(domain["min"]) + predicate
                    if domain["max"] != float("INF"):
                        predicate = predicate + " < {}".format(domain["max"])
                    predicates.append(predicate)
            
            if len(predicates) == 0:
                condition = "if true then:"
            else:
                condition = "if {} then:".format(" and ".join(predicates))
            outcomes = []
            outcomes.append("    predicted {}: {}".format(group["name"], group["prediction"]))
            outcomes.append("    confidence: {:.3f}".format(group.get("confidence", 1.0)))
            result = "\n".join(outcomes)
            cases.append("{}\n{}".format(condition, result))
        return "\n\nelse ".join(cases)
    
    def __repr__(self):
        """
        Returns
        ---
        dictionary : The recursive dictionary used to represent the model
        """
        return dumps(self.source, indent=2, cls=NumpyEncoder)

    def latex(self, node=None):
        """
        Note
        ---
        This method doesn't work well for label headers that contain underscores due to underscore being a reserved character in LaTeX

        Returns
        ---
        string : A LaTeX string representing the model
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            return "[ ${}$ [ ${}$ ] ]".format(name, node["prediction"])
        else:
            if "name" in node:
                if "=" in node["name"]:
                    name = "{}".format(node["name"])
                else:
                    name = "{} {} {}".format(node["name"], node["relation"], node["reference"])
            else:
                name = "feature_{} {} {}".format(node["feature"], node["relation"], node["reference"])
            return "[ ${}$ {} {} ]".format(name, self.latex(node["true"]), self.latex(node["false"])).replace("==", " \eq ").replace(">=", " \ge ").replace("<=", " \le ")

    def json(self):
        """
        Returns
        ---
        string : A JSON string representing the model
        """
        return dumps(self.source, cls=NumpyEncoder)


    def __groups__(self, node=None):
        """
        Parameters
        --- 
        node : node within the tree from which to start
        Returns
        ---
        list : Object representation of each leaf for conversion to a case in an if-then-else statement
        """
        if node is None:
            node = self.source
        if "prediction" in node:
            node["rules"] = {}
            groups = [node]
            return groups
        else:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            reference = node["reference"]
            groups = []
            for condition_result in ["true", "false"]:
                subtree = node[condition_result]
                for group in self.__groups__(subtree):

                    # For each group, add the corresponding rule
                    rules = group["rules"]
                    if not name in rules:
                        rules[name] = {}
                    rule = rules[name]
                    if node["relation"] == "==":
                        rule["type"] = "Categorical"
                        if "positive" not in rule:
                            rule["positive"] = set()
                        if "negative" not in rule:
                            rule["negative"] = set()
                        if condition_result == "true":
                            rule["positive"].add(reference)
                        elif condition_result == "false":
                            rule["negative"].add(reference)
                        else:
                            raise "OptimalSparseDecisionTree: Malformatted source {}".format(node)
                    elif node["relation"] == ">=":
                        rule["type"] = "Numerical"
                        if "max" not in rule:
                            rule["max"] = float("INF")
                        if "min" not in rule:
                            rule["min"] = -float("INF")
                        if condition_result == "true":
                            rule["min"] = max(reference, rule["min"])
                        elif condition_result == "false":
                            rule["max"] = min(reference, rule["max"])
                        else:
                            raise "OptimalSparseDecisionTree: Malformatted source {}".format(node)
                    else:
                        raise "Unsupported relational operator {}".format(node["relation"])
                    
                    # Add the modified group to the group list
                    groups.append(group)
            return groups