import re

from keras.preprocessing.sequence import pad_sequences

from load_glove_embeddings import load_glove_embeddings
from model import MAX_SEQUENCE_LENGTH, get_model
from utills import load_cui_dataset

DATASET_DIR = "data/cui/processed/"

train, dev, test = load_cui_dataset(DATASET_DIR, MAX_SEQUENCE_LENGTH)

EMBEDDING_DIM = 50
print("Loading word embedding")
word2index, embedding_matrix = load_glove_embeddings(
    fp="glove/glove.6B.50d.txt", embedding_dim=EMBEDDING_DIM
)


class CoherenceModel(object):
    def __init__(self, weight_file):

        print("Stating to load model config")
        self.model = get_model()
        print("Model config loaded")

        print("Starting to load model weights")
        self.model.load_weights(weight_file)
        print("Model weight loaded")

    @staticmethod
    def _process_line(line):
        splits = line.split(" ")
        res = []
        for element in splits:
            res.extend([x for x in re.split("(\W+)", element) if (len(x) != 0)])

        full_num = []
        for word in res:
            try:
                full_num.append(word2index[word])
            except Exception:
                pass
        return pad_sequences(
            [full_num],
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
            value=0,
        )

    def predict(self, first_sentence, second_sentence, third_sentence):
        x = [
            CoherenceModel._process_line(first_sentence),
            CoherenceModel._process_line(second_sentence),
            CoherenceModel._process_line(third_sentence),
        ]

        print("Starting to predict")
        value = self.model.predict(x)
        return value


#
# if __name__ == "__main__":
#     obj = CoherenceModel(weight_file="trained_models/model.05-0.652.weights.h5")
#
#     coherent_paragraph = [
#         "I start my day with a cup of coffee.",
#         "After that, I read the news to stay informed.",
#         "Once I've finished reading, I go for a short run in the park.",
#         "The fresh air and exercise help me feel energized.",
#         "After my run, I take a quick shower.",
#         "I then prepare a healthy breakfast of oatmeal and fruit.",
#         "By 8 AM, I am ready to start working on my projects.",
#         "I find that starting my day early helps me be more productive.",
#     ]
#
#     non_coherent_paragraph = [
#         "The library is closed on Sundays.",
#         "A mathematician won the Nobel Prize for his research.",
#         "The stock market experienced a significant drop yesterday.",
#         "I bought a new pair of running shoes last week.",
#         "The concert was canceled due to the rain.",
#         "We decided to have dinner at the new Italian restaurant.",
#         "My friend moved to a new apartment downtown.",
#         "The coffee machine in the office broke yesterday.",
#     ]
#
#     def predict_and_average(paragraph):
#         scores = []
#         for i in range(len(paragraph) - 2):
#             prob = obj.predict(paragraph[i], paragraph[i + 1], paragraph[i + 2])
#             scores.append(prob[0])
#         average_score = sum(scores) / len(scores)
#         return average_score
#
#     coherent_avg_score = predict_and_average(coherent_paragraph)
#     non_coherent_avg_score = predict_and_average(non_coherent_paragraph)
#
#     print("Average Coherent Paragraph Score:", coherent_avg_score)
#     print("Average Non-Coherent Paragraph Score:", non_coherent_avg_score)

if __name__ == "__main__":
    obj = CoherenceModel(weight_file="trained_models/model.05-0.679.weights.h5")

    coherent_prob = obj.predict(
        "I start my day with a cup of coffee.",
        "After that, I read the news to stay informed.",
        "Finally, I head to work feeling prepared for the day.",
    )
    less_coherent_prob = obj.predict(
        "The dog barked loudly.",
        "She went to the market to buy vegetables.",
        "It started raining heavily in the afternoon.",
    )
    random_prob = obj.predict(
        "The museum exhibits ancient artifacts from Egypt.",
        "THe cricketer enjoys playing the guitar in his free time.",
        "The stock market experienced a significant drop yesterday.",
    )
    print("Coherent sentences prediction:", coherent_prob[0])
    print("Less coherent sentences prediction:", less_coherent_prob[0])
    print("Random sentences prediction:", random_prob[0])
