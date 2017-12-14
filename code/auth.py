import os, re, json, time
from glob import glob
from collections import defaultdict
from operator import itemgetter
from copy import deepcopy
from random import shuffle
from math import ceil
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import pylab as pl


def sorting(d, rev=False):
    return sorted(d.iteritems(), key=itemgetter(0), reverse=rev)

#============================#

exceptions = set('vb vbd vbg vbn vbp vbz jj jjr jjs nn nns nnp nnps cd'.upper().split())
# not counting exact nouns, verbs and adjectives - they do not determine style
# only count how many of them.

def feat_form_dict(para):

    d = defaultdict(lambda: 0)
    l = WordNetLemmatizer()
    f = l.lemmatize
    for word, tag in para:
        feature = (f(word.lower()) if tag not in exceptions else '') + '-' + tag
        d[feature] += 1
    return dict(d)


def norma(l, ln=2):
    norm = sum([i**ln for i in l]) ** (1/ln)
    i2 = [i/norm for i in l]
    return i2


#============#

class clf_data(object):
    """
    by default splits data into 30% test set and 70% training set
    """
    def __init__(self, vec, split=.3):
        self.vec = deepcopy(vec)
        #self.vec = vec
        shuffle(self.vec)
        self.n = len(vec)
        labels, vectors = zip(*self.vec)
        if split:
            cut = int(ceil(self.n * split))
            self.train_data = vectors[:-cut]
            self.test_data  = vectors[-cut:]
            self.train_labels = labels[:-cut]
            self.test_labels  = labels[-cut:]
        else:
            self.train_data = vectors
            self.test_data  = []
            self.train_labels = labels
            self.test_labels  = []


class my_filedata_builder(object):
    def __init__(self, filename, mastertags):
        self.filename = filename
        self.master_tags = mastertags
        self.tag_word_vecs = {}
        self.vec_list = []
        if os.path.exists(self.filename):
            pos_words = []
            fname = os.path.basename(self.filename)
            print ' ' * 4 + 'Opening file', fname
            with open(self.filename) as f:
                f_data = f.read().decode("utf-8").lower()
            paras = re.split("[\n\r]*", f_data)
            for para in paras:
                pos_words1 = [nltk.pos_tag(re.findall(r"\b\w+\b", para))]
                pos_words.extend(pos_words1)
            self.pos_data = pos_words
            self.tag_word_vecs = []
            print ' Processing tag vectors'
            for para in self.pos_data:
                if not para:
                    continue
                para_dict = self.master_tags.copy()
                author_pos_dict = feat_form_dict(para)
                for apd in author_pos_dict:
                    if apd not in para_dict:
                        continue
                    para_dict[apd] = author_pos_dict[apd]
                data_vector = [v for (k, v) in sorting(para_dict)]
                data_vector = norma(data_vector, 2)
                self.tag_word_vecs.append(data_vector)
            for vec in self.tag_word_vecs:
                self.vec_list.append((0, vec))


class my_data_buider(object):
    def __init__(self, foldername):
        self.init_folder = foldername
        self.authors = [d for d in os.listdir(self.init_folder)
                     if os.path.isdir(os.path.join(self.init_folder, d))]
        self.pos_data = {}
        self.master_tags = {}
        self.tag_word_vecs = {}
        self.vec_list = []
        self.author_ind = {}
        for auth in self.authors:
            print "Pos for auth", auth
            self.extract_data_for_author(self.init_folder + "/" + auth, auth)

    def extract_data_for_author(self, dir, author):
        pos_fname = os.path.join(self.init_folder, author + '_pos1.txt')
        self.author_ind[author] = len(self.author_ind) + 1
        files = sorted(glob(os.path.join(dir, "book*.txt")))
        print 'Extracting data from files...'
        print files
        if os.path.exists(pos_fname):
            with open(pos_fname) as f:
                pos_words = json.loads(f.read())
                self.pos_data[author] = pos_words
        else:
            pos_words = []
            for full_fname in files:
                fname = os.path.basename(full_fname)
                print ' ' * 4 + 'Opening file', fname
                with open(full_fname) as f:
                    f_data = f.read().decode("utf-8").lower()
                paras = re.split("[\n\r]*", f_data)
                for para in paras:
                    pos_words1 = [nltk.pos_tag(re.findall(r"\b\w+\b", para))]
                    pos_words.extend(pos_words1)
            with open(pos_fname, 'w') as f:
                self.pos_data[author] = pos_words
                json.dump(pos_words, f)

    def pos_vectorize(self):
        if not self.pos_data:
            print 'No tagged data to vectorize'
            return
        #Build master tag dict
        master_tags = dict()
        for author in self.pos_data:
            print ' Adding info from author "%s" into master tag dict' % author
            for book in self.pos_data[author]:
                master_tags.update(feat_form_dict(book))
        self.master_tags = {key:0 for key in master_tags}
        st = time.time()
        for author in self.pos_data:
            self.tag_word_vecs[author] = []
            print ' Processing tag vectors for author "%s"' % author

            for book in self.pos_data[author]:
                if book == []:
                    continue
                para_dict = self.master_tags.copy()
                author_pos_dict = feat_form_dict(book)
                para_dict.update(author_pos_dict)
                data_vector = [v for (k, v) in sorting(para_dict)]
                data_vector = norma(data_vector, 2)
                self.tag_word_vecs[author].append(data_vector)

        print 'Labeling vectors'
        for auth in self.tag_word_vecs:
            print 'Adding vec for', auth
            for vec in self.tag_word_vecs[auth]:
                self.vec_list.append((self.author_ind[auth], vec))
        tt = time.time() - st
        print 'Completed in %2.5f seconds' % tt


def final (vec_list, iters=10):
    Prec, Rec, fscores = [], [], []
    global Y, y_, cm
    Y = []  # y-labels
    y_ = []  # predicted labels
    clf = LinearSVC()
    for _ in range(iters):
        dat = clf_data(vec_list)
        clf = clf.fit(dat.train_data, dat.train_labels)
        pos = dat.test_labels
        guess_pos = clf.predict(dat.test_data)
        tpos = [1 if (dat.test_labels[i] == guess_pos[i]) else 0 for i in range(len(guess_pos))]
        prec  = sum(tpos) / (len(pos) + 1e-10)

        Prec.append(prec)
        print 'Precision: %2.2f' % (prec)
    print 'Overall Precision: %2.4f' % (sum(Prec) / len(Prec))


def batch_test(vec_list, iters=1, target=1):
    Prec, Rec, fscores = [], [], []
    global Y, y_, cm
    Y = []  # y-labels
    y_ = []  # predicted labels

    clf = LinearSVC()
    for _ in range(iters):
        dat = clf_data(vec_list)
        clf = clf.fit(dat.train_data, dat.train_labels)
        pred = clf.predict(dat.test_data)
        pos = [1 if dat.test_labels[i] == target else 0 for i in range(len(pred))]
        #print pos
        guess_pos = [1 if pred[i]==target else 0 for i in range(len(pred))]
        #print guess_pos
        tpos = [1 if (dat.test_labels[i] == target and pred[i] == target) else 0 for i in range(len(pred))]
        #print tpos
        prec, rec = sum(tpos) / (sum(guess_pos) + 1e-10), sum(tpos) / (sum(pos) + 1e-10)
        f1 = 2 * prec * rec / (prec + rec + 1e-10)
        Y.extend(dat.test_labels)
        y_.extend(pred)
        fscores.append(f1)
        Prec.append(prec)
        Rec.append(rec)
        print 'Precision: %2.2f, Recall: %2.2f, Fscore: %2.2f' % (prec, rec, f1)
    print 'Overall Fscore: %2.4f' % (sum(fscores) / len(fscores))
    print 'Overall Precision: %2.4f' % (sum(Prec) / len(Prec))
    print 'Overall Recall: %2.4f' % (sum(Rec) / len(Rec))
    cm = confusion_matrix(Y, y_)
    print cm
    print ''
    return Y, y_, cm


def subs_crossval(myclass, iters=10):
    Yauth = []
    yauth = []
    for author in myclass.author_ind:
        print '%s:' % author
        Y, y, _ = batch_test(myclass.vec_list, iters, myclass.author_ind[author])
        Yauth.extend(Y)
        yauth.extend(y)
    return Yauth, yauth, confusion_matrix(Yauth, yauth)


def subs_val_2(myclass, iters=10):
    final(myclass.vec_list, iters)


dir_path = os.path.dirname(os.path.realpath(__file__))
data_folder = dir_path + "/books"
mm = my_data_buider(data_folder)
mm.pos_vectorize()
Y, y_, cm = subs_crossval(mm, 1)


mm1 = my_filedata_builder(dir_path + "/books/unknown.txt", mm.master_tags)
clff = LinearSVC()
datt = clf_data(mm.vec_list)
clff = clff.fit(datt.train_data, datt.train_labels)
labels, vectors = zip(*mm1.vec_list)
predd = clff.predict(vectors)
overall = {}
for p in predd:
    if p in overall:
        overall[p] += 1
    else:
        overall[p] = 1
for auth_i in mm.author_ind:
    print "Probability of", auth_i, "being the " \
            "author:", ((overall[mm.author_ind[auth_i]] + 1e-10)/(len(predd) + 1e-10))

print predd
pl.matshow(cm)
pl.title('Confusion matrix')
pl.colorbar()
for i, cas in enumerate(cm):
    for j, c in enumerate(cas):
        if c>0:
            pl.text(j-.2, i+.2, c, fontsize=14)
pl.xlabel('Actual author')
pl.ylabel('Predicted author')
pl.savefig('confusion_matrix.pdf')
