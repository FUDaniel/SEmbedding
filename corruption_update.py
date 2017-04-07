import collections
from collections import namedtuple
import math
import argparse
import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

word_index = 0
sentence_index = 0


def main(_):
    # read data
    all_docs = read_data(FLAGS.datafile)
    print('We have %s docs.' % len(all_docs))
    # build data
    data, count, dictionary, reverse_dictionary = build_dataset(all_docs, FLAGS.min_cut_freq)
    vocabulary_size = len(reverse_dictionary)
    print("vocabulary size is: %s" % vocabulary_size)
    paragraph_size = len(all_docs)
    print("paragraph size is: %s" % paragraph_size)

    # for valid
    valid_examples = np.array(dictionary['man'])
    valid_examples = np.append(valid_examples, dictionary['german'])
    valid_examples = np.append(valid_examples, dictionary['two'])

    # Train
    with tf.Graph().as_default():
        # input data
        train_inputs = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.skip_window])
        train_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        # paragraph vector placeholder
        # train_parag_labels = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, 1])
        # doc2vecc placeholder
        train_sentence_cor = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.max_sentence_length])
        train_sen_len_T = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 1])

        # build a graph
        # Variables.
        # embedding, vector for each word in the vocabulary
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, FLAGS.embedding_size], -0.5, 0.5), name='word_embedding'
        )
        embeddings = tf.concat([embeddings, tf.zeros([1, FLAGS.embedding_size])], 0, name='word_embedding')
        print("embedding shape is:%s" % embeddings.get_shape())
        # Format
        config = projector.ProjectorConfig()
        # add embedding
        embed2visual = config.embeddings.add()
        embed2visual.tensor_name = embeddings.name
        # Link this tensor to its metadata file 9e.g. labels)
        embed2visual.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')


        # para_embeddings = tf.Variable(
        #     tf.random_uniform([paragraph_size, FLAGS.embedding_size], -0.5, 0.5))

        embed_word = tf.nn.embedding_lookup(embeddings, train_inputs)
        print("embed word shape is:%s" % embed_word.shape)
        # embed_para = tf.nn.embedding_lookup(para_embeddings, train_parag_labels)
        # print("embed para shape is:%s"%embed_para.shape)

        # sentence corruption embeddings
        embed_cor = tf.nn.embedding_lookup(embeddings, train_sentence_cor)
        print("embed cor shape is: %s" % embed_cor.shape)
        embed_cor_scale = tf.div(10.0, train_sen_len_T)
        # print(embed_cor_scale)
        print("embed_cor_scale shape is: %s" % embed_cor_scale.shape)
        embed_c = tf.multiply(tf.reduce_sum(embed_cor, 1, keep_dims=True), embed_cor_scale)
        print("embed_c shape is: %s" % embed_c.shape)
        corrpution = tf.reduce_sum(embed_c, 1, keep_dims=True)
        print("reduce embed_c %s" % corrpution.shape)
        embed = tf.concat([embed_word, corrpution], 1)
        print("embed shape is:%s" % embed.shape)
        reduced_embed = tf.div(tf.reduce_sum(embed, 1), FLAGS.skip_window + 1)
        print("reduced embed shape is:%s" % reduced_embed.shape)

        # summary
        #
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, FLAGS.embedding_size],
                                                          stddev=1.0 / math.sqrt(FLAGS.embedding_size)))
            variable_summaries(nce_weights)
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
            variable_summaries(nce_biases)

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                             biases=nce_biases,
                                             labels=train_labels,
                                             inputs=reduced_embed,
                                             num_sampled=FLAGS.num_sampled,
                                             num_classes=vocabulary_size))
        # summary
        tf.summary.scalar('nce_loss', loss)

        # Adagrad is required because there are too many things to optimize
        optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

        # compute the similarity between minibatch examples and all embeddings.
        # We use the cosine distance
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        # para_norm = tf.sqrt(tf.reduce_sum(tf.square(para_embeddings), 1, keep_dims=True))
        # normalized_para_embeddings = para_embeddings / para_norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
        similarity_w = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

        #
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            print('Initialized')

            merged = tf.summary.merge_all()
            # summaries_writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)
            # use the same log_dir to stored checkpoint
            summaries_writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

            # write a projector_config.pbtxt in the log_dir
            projector.visualize_embeddings(summaries_writer, config)

            average_loss = 0
            # saver
            saver = tf.train.Saver()

            for step in range(FLAGS.epochs):
                batch_inputs, batch_cor, batch_labels, sen_len_T = generate_cor_batch(
                    data, vocabulary_size)

                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels,
                             train_sentence_cor: batch_cor, train_sen_len_T: sen_len_T}

                summary, _, loss_val = session.run([merged, optimizer, loss], feed_dict=feed_dict)
                # write summaries
                summaries_writer.add_summary(summary, step)

                average_loss += loss_val

                if step % 2000 == 0 and step > 0:
                    average_loss /= 2000
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0
                    # save check point
                    saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'), step)

                if step % 10000 == 0 and step > 0:
                    sim = similarity_w.eval()
                    for i in range(len(valid_examples)):
                        valid_word = reverse_dictionary[valid_examples[i]]
                        top_k = 8
                        nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                        log_str = "Nearest to %s:" % valid_word
                        for k in range(top_k):
                            close_word = reverse_dictionary[nearest[k]]
                            log_str = "%s %s," % (log_str, close_word)
                        print(log_str)

                final_embeddings = normalized_embeddings.eval()
            #
            # get final embeddings
            final_embeddings = normalized_embeddings.eval()
            generate_corruption_sentence(final_embeddings, dictionary)


def read_data(filename):
    LabelDoc = namedtuple('LabelDoc', 'words tags')
    all_docs = []
    count = 0
    with open(filename) as f:
        for line in iter(f):
            tag = ['SEN_' + str(count)]
            count += 1
            all_docs.append(LabelDoc(line.split(), tag))
            # if count > 10001:
            #     break
    # print("max is: %s and min is: %s and avg is: %s"%(max(leng), min(leng), float(sum(leng))/len(leng)))
    return all_docs


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
    with open('log/metadata.tsv', 'w') as f:
        for word, freq in count:
            dictionary[word] = len(dictionary)
            f.write("%s\t%s\n" % (word, freq))
        f.close()
    # this contains the words replaced by assigned indices
    data = []
    unk_count = 0
    LabelDoc = namedtuple('LabelDoc', 'words tags')
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


def generate_cor_batch(data, vocabulary_size):
    # skip window is the amount of words we're looking at from left side of a given word(Like Mikolov)
    # create a single batch
    global word_index
    global sentence_index
    skip_window = FLAGS.skip_window
    batch_size = FLAGS.batch_size
    max_sentence_length = FLAGS.max_sentence_length
    rp_sample = FLAGS.rp_sample

    span = skip_window + 1
    batch = np.ndarray(shape=(batch_size, span - 1), dtype=np.int32)
    batch_cor = np.ndarray(shape=(batch_size, max_sentence_length), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # parag_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    sentence_length_T = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # skip_window is the length of left side

    # queue which add and pop at the end
    # specified a maxlen, when new items added, a corresponding number of items are discarded from
    # the opposite end.
    buffer = collections.deque(maxlen=span)

    # get words starting from index 0 to span
    for _ in range(span):
        buffer.append((data[sentence_index].words[word_index]))
        sent_len = len(data[sentence_index].words)
        if sent_len - 1 == word_index:  # reaching the end of a sentence
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:
            word_index += 1

    #
    for i in range(batch_size):
        batch_temp = np.ndarray(shape=(span - 1), dtype=np.int32)
        for j in range(span - 1):
            batch_temp[j] = buffer[j]
        batch[i] = batch_temp
        labels[i, 0] = buffer[skip_window]
        # parag_labels[i, 0] = sentence_index
        buffer.append(data[sentence_index].words[word_index])
        sent_len = len(data[sentence_index].words)
        # sample sentence
        sentence_length_T[i, 0] = sent_len
        batch_tmp_cor = np.ndarray(shape=(max_sentence_length), dtype=np.int32)
        already_sampled = 0
        for wd in data[sentence_index].words:
            if np.random.random() > rp_sample:
                continue
            batch_tmp_cor[already_sampled] = wd
            already_sampled += 1
            if already_sampled >= max_sentence_length:
                break
        if already_sampled <= max_sentence_length:
            while max_sentence_length - already_sampled > 0:
                batch_tmp_cor[already_sampled] = vocabulary_size
                already_sampled += 1
        batch_cor[i] = batch_tmp_cor

        if sent_len - 1 == word_index:
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
        else:
            word_index += 1

    # return batch, batch_cor, labels, parag_labels, sentence_length_T
    return batch, batch_cor, labels, sentence_length_T


def generate_corruption_sentence(embeddings, dictionary):
    with open('alldata.txt') as fp, open('doc2vector.txt', 'w')as fpw:
        for index, line in enumerate(fp):
            words_idx = []
            word_list = line.split()
            sen_len = len(word_list)
            if sen_len <= FLAGS.max_sentence_length:
                for w in word_list:
                    if w in dictionary:
                        words_idx.append(dictionary[w])
            else:
                for i in range(FLAGS.max_sentence_length):
                    if word_list[i] in dictionary:
                        words_idx.append(dictionary[word_list[i]])
            # words_np_idx = np.array(words_idx)
            final_embeddings_np = embeddings

            sentence = final_embeddings_np[words_idx]
            corr_embed = np.sum(sentence, axis=0)
            w = 1.0 / sen_len
            cor_embedding = w * corr_embed
            norm_cor = np.sqrt(np.sum(np.square(cor_embedding)))
            cor_norm_embedding = cor_embedding / norm_cor

            if index % 1000 == 0:
                print("index: %s" % index)
            fpw.write(np.array_str(cor_norm_embedding, 10000)[1:-1])
            fpw.write('\n')
        fpw.close()
        fp.close()


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_sum(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--embedding_size', type=int, default=100, help='embedding size')
    parser.add_argument('--skip_window', type=int, default=10, help='how many words to consider left')
    parser.add_argument('--epochs', type=int, default=100001, help='number of steps')
    parser.add_argument('--datafile', type=str, default='alldata-shuf.txt', help='train data file path')
    parser.add_argument('--min_cut_freq', type=int, default=9, help='min cut frequent words')
    parser.add_argument('--max_sentence_length', type=int, default=100, help='limit sentence length to max')
    parser.add_argument('--rp_sample', type=float, default=0.1, help='the rate of corruption sample 1-q')
    parser.add_argument('--num_sampled', type=int, default=5, help='negative sampling, noise-contrastive estimation')
    parser.add_argument('--log_dir', type=str, default='log/', help='summary log dir')

    FLAGS, unparsed = parser.parse_known_args()
    print('flags', FLAGS)

    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
