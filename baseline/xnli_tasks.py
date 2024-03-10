## The task does not belong to XTREME UP. Added here for easier comparision.
import seqio
from t5.evaluation import metrics
import tensorflow as tf
import functools

from xtreme_up.baseline import tasks_lib
from xtreme_up.evaluation import constants

XNLI_LANGS = [
    "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr",
    "ur", "vi", "zh"
]

DEFAULT_PREPROCESSORS = [
    seqio.preprocessors.tokenize,
    seqio.CacheDatasetPlaceholder(),
    seqio.preprocessors.append_eos_after_trim,
]

def process_mnli(dataset):
  """Convert MNLI dataset into a text2text format.

  This function will return examples of the form:
  {
     'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
     'targets': '<target>'
  }

  Args:
    dataset: tf.data.Dataset to process.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['hypothesis']]),
        'targets': tf.strings.as_string(x['label'])
    }

  return dataset.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def xnli_map_hypothesis_premise(dataset, target_language):
  """Generates XNLI dataset with the hypothesis restricted to a target language.

  The XNLI dataset (https://www.tensorflow.org/datasets/catalog/xnli) contains
  the hypothesis in a TranslationVariableLanguages feature. The hypothesis looks
  like:
    hypothesis: {
      'language': ['ar', 'bg', 'en', ...],
      'translation': ['t1', 't2', 't3', ...]
    }
  This function processes this hypothesis to return a dataset of the form:
    {
      'language': 'ar',
      'translation': 't1',
      'label': '1'
    }
  The label is also extracted along with the hypothesis.

  Args:
    dataset: tf.data.Dataset to process.
    target_language: string, the target language to restrict the hypothesis in
      the dataset to.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    languages = x['hypothesis']['language']
    translations = x['hypothesis']['translation']

    # Create a tensor of the same length as languages so that everything can be
    # unbatched into examples for each language later.
    label = tf.fill(tf.shape(languages), x['label'])
    premise = tf.fill(tf.shape(languages), x['premise'][target_language])

    return {
        'language': languages,
        'translation': translations,
        'label': label,
        'premise': premise
    }

  dataset = dataset.map(
      _process, num_parallel_calls=tf.data.experimental.AUTOTUNE).unbatch()
  dataset = dataset.filter(
      lambda x: tf.math.equal(x['language'], target_language))
  return dataset


def process_xnli(dataset, target_languages):
  """Processes the XNLI dataset into examples by language in a text2text format.

  The XNLI dataset contains examples of the form:
  {
        'hypothesis': {
            'language': ['lang1', 'lang2', 'lang3'],
            'translation': ['translation1', 'translation2', 'translation3'],
        },
        'label': 1,
        'premise': {
            'lang1': 'premise1',
            'lang2': 'premise2',
            'lang3': 'premise3'
        }
    }

  This function processes the XNLI dataset and returns examples of the form:
  {
    'inputs': 'xnli: premise: <premise> hypothesis: <hypothesis>',
    'targets': <target>
  }
  for each language in the list of input target languages.
  Args:
    dataset: tf.data.Dataset to process.
    target_languages: list of strings, the target languages.
  Returns:
    A preprocessed tf.data.Dataset with the format listed above.
  """
  def _process(x):
    return {
        'inputs': tf.strings.join(['xnli: premise: ', x['premise'],
                                   ' hypothesis: ', x['translation']]),
        'targets': tf.strings.as_string(x['label'])
    }

  output = []
  for language in target_languages:
    examples = xnli_map_hypothesis_premise(dataset, target_language=language)
    d = examples.map(_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    output.append(d)

  output_dataset = output[0]
  for lang_dataset in output[1:]:
    output_dataset = output_dataset.concatenate(lang_dataset)
  return output_dataset


def _add_xnli_tasks(model):
    seqio.TaskRegistry.add(
        f"{model}_xnli_train",
        source=seqio.TfdsDataSource(tfds_name="multi_nli:1.1.0", splits=["train"]),
        preprocessors=[
            process_mnli,
            *DEFAULT_PREPROCESSORS,
        ],
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[metrics.accuracy])
    for lang in XNLI_LANGS:
      seqio.TaskRegistry.add(
          f"{model}_xnli_dev_test.{lang}",
          source=seqio.TfdsDataSource(
              tfds_name="xnli:1.1.0", splits=["validation", "test"]),
          preprocessors=[
              functools.partial(
                  process_xnli, target_languages=[lang]),
              *DEFAULT_PREPROCESSORS,
          ],
          output_features=tasks_lib.get_output_features(model),
          metric_fns=[metrics.accuracy])
      if lang == "en":
        continue
      seqio.TaskRegistry.add(
          f"{model}_xnli_translate_train.{lang}",
          source=seqio.TfdsDataSource(
              tfds_name="xtreme_xnli:1.1.0", splits=["train"]),
          preprocessors=[
              functools.partial(
                  process_xnli, target_languages=[lang]),
              *DEFAULT_PREPROCESSORS,
          ],
          output_features=tasks_lib.get_output_features(model),
          metric_fns=[metrics.accuracy])
    seqio.TaskRegistry.add(
        f"{model}_xnli_dev_test.all_langs",
        source=seqio.TfdsDataSource(
            tfds_name="xnli:1.1.0", splits=["validation", "test"]),
        preprocessors=[
            functools.partial(
                process_xnli, target_languages=XNLI_LANGS),
            *DEFAULT_PREPROCESSORS,
        ],
        output_features=tasks_lib.get_output_features(model),
        metric_fns=[metrics.accuracy])

    xnli_zeroshot = ([f"{model}_xnli_train", f"{model}_xnli_dev_test.all_langs"] +
                     [f"{model}_xnli_dev_test.{lang}" for lang in XNLI_LANGS])
    seqio.MixtureRegistry.add(f"xtreme_up_xnli_zero_shot_{model}", xnli_zeroshot, default_rate=1.0)
    xnli_translate_train = xnli_zeroshot + [
        f"{model}_xnli_translate_train.{lang}"
        for lang in XNLI_LANGS
        if lang != "en"
    ]
    seqio.MixtureRegistry.add(
        f"xtreme_up_xnli_{model}", xnli_translate_train, default_rate=1.0)


_add_xnli_tasks('myt5')
_add_xnli_tasks('byt5')
