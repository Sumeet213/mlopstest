import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast


class DataIngestor:
    def __init__(self, config):
        self.source_file = config.source_file
        self.intents_dict = config.intents_dict
        self.threshold = config.threshold


    def load_and_process_data(self):
        # Load your dataset
        df = pd.read_csv(self.source_file)
        df = df[:10000]

        # Add new columns for each intent in your DataFrame
        def count_keywords(description, keywords):
            if isinstance(description, str):
                return sum([word in description.lower().split() for word in keywords])
            else:
                return 0

        for intent, intent_dict in self.intents_dict.items():
            df[intent] = df['text'].apply(
                lambda x: count_keywords(x, intent_dict))

        # Convert counts to binary (0 or 1) based on the threshold
        for intent in self.intents_dict.keys():
            df[intent] = df[intent].apply(
                lambda x: 1 if x >= self.threshold else 0)

        # Save to a new CSV file
        df.to_csv('new_dataset1.csv', index=False)

        # Load the BERT tokenizer
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Encode the text data
        input_encodings = tokenizer(
            df['text'].tolist(), truncation=True, padding=True, max_length=512)

        # Create binary labels for each intent
        intents = list(self.intents_dict.keys())
        labels = df[intents].values

        # Split the data into training and validation sets
        train_inputs, val_inputs, train_labels, val_labels = train_test_split(
            input_encodings['input_ids'], labels, test_size=0.2)

        return train_inputs, val_inputs, train_labels, val_labels
