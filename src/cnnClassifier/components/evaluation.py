class Evaluator:
    def __init__(self, model, test_dataset):
        self.model = model
        self.test_dataset = test_dataset

    def evaluate(self):
        # Evaluate the model
        results = self.model.evaluate(self.test_dataset)
        return results
