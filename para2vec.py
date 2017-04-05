import collections
from collections import namedtuple
import math
import string
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


exclude = set(string.punctuation)
LabelDoc = namedtuple('LabelDoc', 'words tags')
filename = 'alldata-shuf.txt'
def read_data(filename):
    all_docs = []
    count = 0
    with open(filename) as f:

        for line in iter(f):
            word_list = line.split()
            if len(word_list) < 3:
                continue
            tag = ['SEN_' + str(count)]
            count += 1
            sen = ''.join(ch for ch in line if ch not in exclude)
            all_docs.append(LabelDoc(sen.split(), tag))
            if count > 10001:
                break;

    return all_docs


all_docs = read_data(filename)
print('We have %s docs.' %len(all_docs))


# Build the dictionary and replace rare words with UNK token
def build_dataset(input_data, min_cut_freq):
    words = []
    for i in input_data:
        for j in i.words:
            words.append(j)

    # unknown token is used to denote words that are not in the dictionary
    count_org = [['UNK', -1]]
    # returns set of tuples (word, count) with all words
    count_org.extend(collections.Counter(words).most_common())
    count = [['UNK', -1]]
    for word, c in count_org:
        word_tuple = [word, c]
        if word == 'UNK':
            count[0][1] = c
            continue
        if c > min_cut_freq:
            count.append(word_tuple)
    dictionary = dict()
    # set word count for all the words to the current number of keys in the dictionary
    # in other words values act as indices for each word
    # first word is 'UNK' representing unknown words we encounter
    for word, _ in count:
        dictionary[word] = len(dictionary)
    # this contains the words replaced by assigned indices
    data = []
    unk_count = 0
    for tup in input_data:
        word_data = []
        for word in tup.words:
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0
                unk_count += 1
            word_data.append(index)
        data.append(LabelDoc(word_data, tup.tags))
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary

min_cut_freq = 3
data, count, dictionary, reverse_dictionary = build_dataset(all_docs, min_cut_freq)

vocabulary_size = len(reverse_dictionary)
paragraph_size = len(all_docs)
print("paragraph size is: %s" %paragraph_size)

word_index = 0
sentence_index = 0


def generate_pv_batch(batch_size, skip_window):
    # skip window is the amount of words we're looking at from left side of a given word(Like Mikolov)
    # create a single batch
    global word_index
    global sentence_index

    span = skip_window + 1
    batch = np.ndarray(shape=(batch_size, span-1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    parag_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # e.g if skip_window = 2 then span = 3
    # span is the length of the whole frame we are considering for a single word (left + word)
    # skip_window is the length of left side

    # queue which add and pop at the end
    # specified a maxlen, when new items added, a corresponding number of items are discarded from
    # the opposite end.
    buffer = collections.deque(maxlen=span)

    # get words starting from index 0 to span
    for _ in range(span):
        buffer.append((data[sentence_index].words[word_index]))
        sent_len = len(data[sentence_index].words)
        if sent_len - 1 == word_index: # reaching the end of a sentence
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:
            word_index += 1

    #
    for i in range(batch_size):
        batch_temp = np.ndarray(shape=(span-1), dtype=np.int32)
        for j in range(span-1):
            batch_temp[j] = buffer[j]
        batch[i] = batch_temp
        labels[i, 0] = buffer[skip_window]
        parag_labels[i, 0] = sentence_index
        buffer.append(data[sentence_index].words[word_index])
        sent_len = len(data[sentence_index].words)
        if sent_len - 1 == word_index:
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:
            word_index += 1
    return batch, labels, parag_labels




num_steps = 10001

if __name__ == '__main__':
    batch_size = 128
    embedding_size = 128 # Dimension of the embedding vector
    skip_window = 2 # How many words to consider left and right

    # we pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100 # Only pick dev samples in the head of the distribution.
    # pick 16 samples from 100
    valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
    valid_examples = np.append(valid_examples, random.sample(range(1000, 1000+valid_window), valid_size//2))
    num_sampled = 64 # Number of negative examples to sample.

    graph = tf.Graph()
    with graph.as_default():
        # input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size, skip_window])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        # paragraph vector placeholder
        train_parag_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

        with tf.device('/cpu:0'):
            # Variables.
            # embedding, vector for each word in the vocabulary
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
            )
            para_embeddings = tf.Variable(
                tf.random_uniform([paragraph_size, embedding_size], -1.0, 1.0))
            embed_word = tf.nn.embedding_lookup(embeddings, train_inputs)
            print("embed word shape is:%s"%embed_word.shape)
            embed_para = tf.nn.embedding_lookup(para_embeddings, train_parag_labels)
            print("embed para shape is:%s"%embed_para.shape)
            embed = tf.concat([embed_word, embed_para], 1)
            print("embed shape is:%s"%embed.shape)
            reduced_embed = tf.div(tf.reduce_sum(embed, 1), skip_window+1)
            print("reduced embed shape is:%s"%reduced_embed.shape)

            softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

            # Model
            # Lookup embeddings for inputs
            # this might efficiently find the embeddings for given ids (trained dataset)
            # manually doing this might not be efficient given there are 50000 entries in embeddings
            # embeds = None
            # for i in range(2*skip_window):
            #     embedding_i = tf.nn.embedding_lookup(embeddings, train_dataset[:,i])
            #     print('embedding %d shape: %s' % (i, embedding_i.get_shape().as_list()))
            #     emb_x, emb_y = embedding_i.get_shape().as_list()
            #     if embeds is None:
            #         embeds = tf.reshape(embedding_i, [emb_x, emb_y, 1])
            #     else:
            #         embeds = tf.concat([embeds, tf.reshape(embedding_i, [emb_x, emb_y, 1])], 2)
            #
            # assert embeds.get_shape().as_list()[2] == 2*skip_window
            # print('Concat embedding size: %s' %embeds.get_shape().as_list())
            # avg_embed = tf.reduce_mean(embeds, 2, keep_dims=False)
            # print("Avg embedding size: %s" %avg_embed.get_shape().as_list())

            # compute the softmax loss, using a sample of the negative labels each time.
            # inputs are embeddings of the train words
            # with this loss we optimize weights, biases, embeddings

            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                                             biases=softmax_biases,
                                                             labels=train_labels,
                                                             inputs=reduced_embed,
                                                             num_sampled=num_sampled,
                                                             num_classes=vocabulary_size))


            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities
            # that contribute to the tensor it is passed.
            # Adagrad is required because there are too many things to optimize
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

            # compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            normalized_embeddings = embeddings / norm
            para_norm = tf.sqrt(tf.reduce_sum(tf.square(para_embeddings), 1, keep_dims=True))
            normalized_para_embeddings = para_embeddings / para_norm
            valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')

        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels, batch_para_labels = generate_pv_batch(
                                       batch_size,  skip_window)
            # print some items to check
            # for batch in batch_inputs:
            #     print("printing batch: ", batch)
            # print("the shape of batch_inputs: ", batch_inputs.shape)
            # print("the shape of batch_labels: ", batch_labels.shape)
            # print("the shape of batch_para_labels: ", batch_para_labels.shape)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, train_parag_labels: batch_para_labels}

            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            # if step % 10000 == 0:
            #     sim = similarity.eval()
            #     for i in range(valid_size):
            #         valid_word = reverse_dictionary[valid_examples[i]]
            #         top_k = 8
            #         nearest = (-sim[i, :]).argsort()[1:top_k+1]
            #         log_str = "Nearest to %s:" % valid_word
            #         for k in range(top_k):
            #             close_word = reverse_dictionary[nearest[k]]
            #             log_str = "%s %s," % (log_str, close_word)
            #         print(log_str)
            final_embeddings = normalized_embeddings.eval()
            final_para_embeddings = normalized_para_embeddings.eval()

# Testing final embedding
            if step % 10000 == 0:
                input_dictionary = dict([(v,k) for (k, v) in reverse_dictionary.items()])
                test_word_idx_a = np.random.randint(0, len(input_dictionary) - 1)
                a = final_embeddings[test_word_idx_a, :]

                similarity = final_embeddings.dot(a)
                top_k = 9
                nearest = (-similarity).argsort()[0:top_k]

                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    print(close_word)

                doc_id = 1234

                para_embeddings = final_para_embeddings[doc_id, :]
                print (doc_id)
                similarity_para = final_para_embeddings.dot(para_embeddings)
                nearest_para = (-similarity_para).argsort()[0:top_k]
                for k in range(top_k):
                    close_sentence = all_docs[nearest_para[k]]
                    print(close_sentence)

        #
        # def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        #     assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        #     plt.figure(figsize=(18, 18))
        #     for i, labels in enumerate(labels):
        #         x, y = low_dim_embs[i, :]
        #         plt.scatter(x, y)
        #         plt.annotate(labels,
        #                      xy=(x, y),
        #                      xytext=(5, 2),
        #                      textcoords='offset points',
        #                      ha='right',
        #                      va='bottom')
        #     plt.savefig(filename)
        #
        #
        # tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        # plot_only = 100
        # low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
        # labels = [reverse_dictionary[i] for i in range(plot_only)]
        # plot_with_labels(low_dim_embs, labels)
