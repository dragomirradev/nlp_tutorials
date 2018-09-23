import tensorflow as tf
import tensorflow_hub as hub

with tf.Graph().as_default():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    embed = hub.Module(module_url)
    embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        print(sess.run(embeddings))
