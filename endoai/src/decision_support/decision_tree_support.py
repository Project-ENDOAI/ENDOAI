"""
Decision Tree Support Module

This script provides utilities for decision tree-based support systems.
"""

from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd

def train_decision_tree(X, y, max_depth=None):
    """
    Train a decision tree classifier.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target labels.
        max_depth (int, optional): Maximum depth of the tree.

    Returns:
        DecisionTreeClassifier: Trained decision tree model.
    """
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y)
    return model

def display_decision_tree(model, feature_names):
    """
    Display the decision tree rules.

    Args:
        model (DecisionTreeClassifier): Trained decision tree model.
        feature_names (list): List of feature names.
    """
    tree_rules = export_text(model, feature_names=feature_names)
    print("Decision Tree Rules:")
    print(tree_rules)

def main():
    """
    Main function to demonstrate decision tree support.
    """
    # Example data
    data = pd.DataFrame({
        "feature1": [1, 2, 3, 4],
        "feature2": [10, 20, 30, 40],
        "target": [0, 1, 0, 1]
    })
    X = data[["feature1", "feature2"]]
    y = data["target"]

    model = train_decision_tree(X, y, max_depth=3)
    display_decision_tree(model, feature_names=X.columns.tolist())

if __name__ == "__main__":
    main()
