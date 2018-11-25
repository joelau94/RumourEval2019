"""
This is outer preprocessing file

To run:

python prep_pipeline.py

Main function has parameter that can be changed:

feats ('text' or 'SemEval')

"""
from preprocessing_tweets import load_dataset
from preprocessing_reddit import load_data
from transform_feature_dict import transform_feature_dict
from extract_thread_features import extract_thread_features_incl_response
import help_prep_functions
import numpy as np
import os
# from keras.preprocessing.sequence import pad_sequences
#%%

DATA_PATH = 'text_data/'


def convert_label(label):
  if label == "true":
    return(0)
  elif label == "false":
    return(1)
  elif label == "unverified":
    return(2)
  else:
    print(label)


def prep_pipeline(dataset='RumEval2019', feature_set=['avgw2v']):

  # path = 'saved_data'+dataset
  folds = {}
  folds = load_dataset()
  reddit = load_data()

  folds['train'].extend(reddit['train'])
  folds['dev'].extend(reddit['dev'])
  folds['test'].extend(reddit['test'])

  # help_prep_functions.loadW2vModel()

#%%
  for fold in folds.keys():

    print(fold)
    feature_fold = []
    tweet_ids = []
    fold_stance_labels = []
    labels = []
    ids = []
    for conversation in folds[fold]:

      thread_feature_dict = extract_thread_features_incl_response(
          conversation)

      # thread_features_array, thread_stance_labels, branches = \
      #   transform_feature_dict(
      #                        thread_feature_dict, conversation,
      #                        feature_set=feature_set)

      thread_text, thread_stance_labels, branches = \
          transform_feature_dict(thread_feature_dict, conversation,
                                 feature_set=feature_set)

      fold_stance_labels.extend(thread_stance_labels)
      tweet_ids.extend(branches)
      feature_fold.extend(thread_features_array)
      for i in range(len(thread_features_array)):
        # labels.append(convert_label(conversation['veracity']))
        labels.append(conversation['veracity'])
        ids.append(conversation['id'])

    # result: thread_text (nested),
    #         fold_stance_labels (nested),
    #         labels (veracity)

    datafile = os.path.join(DATA_PATH, fold + '.txt')
    assert len(thread_text) == len(fold_stance_labels)
    assert len(thread_text) == len(labels)
    fdata = open(datafile, 'w')
    for thread, stance, veracity in zip(thread_text,
                                        fold_stance_labels,
                                        labels):
      orig_tweet = ' '.join(thread_text[0])
      fdata.write('{} ||| {} ||| {}\n'.format(orig_tweet, stance[0], veracity))
      for thr, sdqc in zip(thread[1:], stance[1:]):
        tweet = ' '.join(thr)
        fdata.write('{} ||| {}\n'.format(tweet, sdqc))
      fdata.write('\n')

#%
      # if feature_fold!=[]:

      # feature_fold = pad_sequences(feature_fold, maxlen=None,
      #                              dtype='float32',
      #                              padding='post',
      #                              truncating='post', value=0.)

      # fold_stance_labels = pad_sequences(fold_stance_labels, maxlen=None,
      #                                    dtype='float32',
      #                                    padding='post', truncating='post',
      #                                    value=0.)

      # labels = np.asarray(labels)
      # path_fold = os.path.join(path, fold)
      # if not os.path.exists(path_fold):
      #     os.makedirs(path_fold)

      # np.save(os.path.join(path_fold, 'train_array'), feature_fold)
      # np.save(os.path.join(path_fold, 'labels'), labels)
      # np.save(os.path.join(path_fold, 'fold_stance_labels'),
      #         fold_stance_labels)
      # np.save(os.path.join(path_fold, 'ids'), ids)
      # np.save(os.path.join(path_fold, 'tweet_ids'), tweet_ids)

#%%
# def main(data ='RumEval2019', feats = 'SemEvalfeatures'):

  # if feats == 'text':
  #   prep_pipeline(dataset='RumEval2019', feature_set=['avgw2v'])
  # elif feats == 'SemEvalfeatures':
  #   SemEvalfeatures = ['avgw2v', 'hasnegation', 'hasswearwords',
  #                      'capitalratio', 'hasperiod', 'hasqmark',
  #                      'hasemark', 'hasurl', 'haspic',
  #                      'charcount', 'wordcount', 'issource',
  #                      'Word2VecSimilarityWrtOther',
  #                      'Word2VecSimilarityWrtSource',
  #                      'Word2VecSimilarityWrtPrev']
  #   prep_pipeline(dataset='RumEval2019', feature_set=SemEvalfeatures)


def main():
  prep_pipeline()


if __name__ == '__main__':
  main()
