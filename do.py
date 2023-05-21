import tensorflow as tf
import tensorflow_text
import requests
import pandas as pd


def LoadAndPredict(review):
    BERT = './bert_model'
    model = tf.saved_model.load(BERT)

    prediction = model([review])
    data = prediction.numpy()[0][0]

    evaluation = True
    if data >= 0:
        evaluation = True
    else:
        evaluation = False
    return evaluation


csv_file = './Review.csv'
data = pd.read_csv(csv_file, header=None, sep=',', names=[
                   'id', 'movieId', 'authorId', 'content', 'a', 'b', 'c', 'd', 'e', 'f'])

for _, row in data[::-1].iterrows():
    id = row['id']
    if int(id) >= 44076:
        continue
    movie_id = row['movieId']
    review = row['content']
    result = LoadAndPredict(review)

    print(f"ReviewId: {id}")
    print(f"MovieId: {movie_id}")
    print(f"Prediction: {result}")

    url = 'http://localhost:8080/api/reviews/ai?code=bummmjin'
    data = {
        'result': result,
        'id': id,
        'movie_id': movie_id
    }
    res = requests.post(url, json=data)
    print(res)
    print()
