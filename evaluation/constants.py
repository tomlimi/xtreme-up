# coding=utf-8
# Copyright 2023 The Google Research authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Sequence


TASK_TO_LANGUAGES = {
    'asr': (
        'af_za',
        'am_et',
        'ar_eg',
        'as_in',
        'ast_es',
        'az_az',
        'be_by',
        'bn_in',
        'bs_ba',
        'ca_es',
        'ceb',
        'cmn_hans_cn',
        'cmn_hant_hk',
        'cs_cz',
        'cy_gb',
        'da_dk',
        'de_de',
        'el_gr',
        'en_us',
        'es_419',
        'et_ee',
        'fa_ir',
        'ff_sn',
        'fi_fi',
        'fil_ph',
        'fr_fr',
        'ga_ie',
        'gl_es',
        'gu_in',
        'ha_ng',
        'he_il',
        'hi_in',
        'hr_hr',
        'hu_hu',
        'hy_am',
        'id_id',
        'ig_ng',
        'is_is',
        'it_it',
        'ja_jp',
        'jv_id',
        'ka_ge',
        'kam_ke',
        'kea_cv',
        'kk_kz',
        'km_kh',
        'kn_in',
        'ko_kr',
        'ckb_iq',
        'ky_kg',
        'lb_lu',
        'lg_ug',
        'ln_cd',
        'lo_la',
        'lt_lt',
        'luo_ke',
        'lv_lv',
        'mi_nz',
        'mk_mk',
        'ml_in',
        'mn_mn',
        'mr_in',
        'ms_my',
        'mt_mt',
        'my_mm',
        'nb_no',
        'ne_np',
        'nl_nl',
        'nso_za',
        'ny_mw',
        'oc_fr',
        'om_et',
        'or_in',
        'pa_in',
        'pl_pl',
        'ps_af',
        'pt_br',
        'ro_ro',
        'rup_bg',
        'ru_ru',
        'sd_arab_in',
        'sk_sk',
        'sl_si',
        'sn_zw',
        'so_so',
        'sr_rs',
        'sv_se',
        'sw_ke',
        'ta_in',
        'te_in',
        'tg_tj',
        'th_th',
        'tr_tr',
        'uk_ua',
        'umb_ao',
        'ur_pk',
        'uz_uz',
        'vi_vn',
        'wo_sn',
        'xh_za',
        'yo_ng',
        'zu_za',
    ),
    'autocomplete': (
        'be',
        'lv',
        'is',
        'et',
        'el',
        'ug',
        'ga',
        'ro',
        'pcm',
        'he',
        'id',
        'bg',
        'sl',
        'uk',
        'sk',
        'gl',
        'eu',
        'ur',
        'hy',
        'gd',
        'en',
        'lt',
        'da',
    ),
    'translation': (
        'af',
        'am',
        'ar',
        'as',
        'az',
        'be',
        'bg',
        'bn',
        'bs',
        'ca',
        'ceb',
        'ckb',
        'cs',
        'cy',
        'da',
        'de',
        'el',
        'es',
        'et',
        'fa',
        'ff',
        'fi',
        'fil',
        'fr',
        'ga',
        'gl',
        'gu',
        'ha',
        'he',
        'hi',
        'hr',
        'hu',
        'hy',
        'id',
        'ig',
        'is',
        'it',
        'ja',
        'jv',
        'ka',
        'kk',
        'km',
        'kn',
        'ko',
        'ky',
        'lb',
        'lg',
        'ln',
        'lo',
        'lt',
        'lv',
        'mi',
        'mk',
        'ml',
        'mn',
        'mr',
        'ms',
        'mt',
        'my',
        'ne',
        'nl',
        'no',
        'nso',
        'ny',
        'om',
        'or',
        'pa',
        'pl',
        'ps',
        'pt',
        'ro',
        'ru',
        'sd',
        'sk',
        'sl',
        'sn',
        'so',
        'sr',
        'sv',
        'sw',
        'ta',
        'te',
        'tg',
        'th',
        'tr',
        'uk',
        'ur',
        'uz',
        'vi',
        'xh',
        'yo',
        'zh',
        'zu',
    ),
    'ner': (
        'am',
        'bm',
        'bbj',
        'ee',
        'ha',
        'ig',
        'rw',
        'lg',
        'luo',
        'mos',
        'ny',
        'pcm',
        'sn',
        'sw',
        'tn',
        'tw',
        'wo',
        'xh',
        'yo',
        'zu',
    ),
    'semantic_parsing': (
        # XTREME-UP languages.
        'am',
        'be',
        'bn',
        'bn_cs',
        'fi',
        'ha',
        'hi_cs',
        'hu',
        'ja',
        'pt_br',
        'ru',
        'sw',
        'ta',
        'ta_cs',
        'tr',
        'yo',
        'zu',
        'de_localized',
        # MTOP languages.
        'en',
        'de',
        'es',
        'fr',
        'hi',
        'th',
    ),
    'qa_in_lang': ('ar', 'bn', 'fi', 'id', 'ko', 'ru', 'sw', 'te', 'en'),
    'qa_cross_lang': (
        # Original XOR-TyDi QA languages:
        'ar',
        'bn',
        'fi',
        'ko',
        'ru',
        'te',
        # New Indic languages:
        'as',
        'bho',
        'brx',
        'gbm',
        'gom',
        'gu',
        'hi',
        'hne',
        'kn',
        'mai',
        'ml',
        'mni',
        'mr',
        'mwr',
        'or',
        'pa',
        'ps',
        'sa',
        'ta',
        'ur',
    ),
}

XTREME_UP_HIGH_RESOURCE_LANGS = frozenset([
    'en',  # English
    'ar',  # Arabic
    'ca',  # Catalan
    'zh',  # Chinese
    'hr',  # Croatian
    'cs',  # Czech
    'nl',  # Dutch
    'fi',  # Finnish
    'fr',  # French
    'de',  # German
    'hi',  # Hindi
    'hu',  # Hungarian
    'it',  # Italian
    'ja',  # Japanese
    'ko',  # Korean
    'fa',  # Farsi/Persian
    'pl',  # Polish
    'pt',  # Portuguese
    'ru',  # Russian
    'sr',  # Serbian
    'es',  # Spanish
    'sv',  # Swedish
    'tr',  # Turkish
    'vi',  # Vietnamese
    # From ASR:
    'cmn',
])

XTREME_UP_UNDER_REPRESENTED_LANGS = frozenset(
    [
        'af',
        'bn',
        'be',
        'bg',
        'bs',
        'my',
        'ceb',
        'da',
        'et',
        'gl',
        'ka',
        'el',
        'he',
        'id',
        'kk',
        'lv',
        'lt',
        'ms',
        'pcm',
        'ro',
        'sk',
        'sl',
        'tl',
        'ta',
        'th',
        'ug',
        'uk',
        'ur',
        'uz',
        'no',
        'or',
        # Languages new from Indic QA:
        'bho',
        'brx',
        'gbm',
        'gom',
        'hne',
        'mai',
        'mni',
        'mwr',
        'ps',
        'ta',
        # Added the following level 1 languages:
        'ff',
        'rw',
        # Added the following level 2 languages:
        'mr',
        'mt',
        # Added the following languages with unknown status:
        'fil',  # Filipino (note: Tagalog is level 2)
        # Added from ASR:
        'ku',
        'nb',
        'oci',
        'rup',
        'sk',
        # Added from Autocomplete:
        'eu',
        'sme',
        # Already registered:
        'am',
        'hy',
        'as',
        'ast',
        'az',
        'bm',
        'bem',
        'ber',
        'my',
        'ckb',
        'ee',
        'fon',
        'ful',
        'bbj',
        'gu',
        'ha',
        'is',
        'ig',
        'ga',
        'jv',
        'kea',
        'kam',
        'kn',
        'km',
        'nw',
        'ky',
        'lo',
        'ln',
        'lij',
        'olo',
        'lg',
        'luo',
        'lb',
        'mk',
        'ml',
        'mg',
        'mi',
        'mn',
        'mos',
        'nd',
        'ne',
        'nso',
        'se',
        'ny',
        'oc',
        'om',
        'ps',
        'pa',
        'sa',
        'gd',
        'tn',
        'sn',
        'si',
        'sd',
        'so',
        'ckb',
        'st',
        'ss',
        'sw',
        'tg',
        'te',
        'bo',
        'ts',
        'tw',
        'umb',
        'hsb',
        've',
        'cy',
        'wo',
        'xh',
        'yo',
        'zu',
    ],
)

TRANSLIT_LANGS_AND_SCRIPTS = {
    "am": [("Latn", "Ethi")],
    "bn": [("Latn", "Beng")],
    "gu": [("Latn", "Gujr")],
    "hi": [("Latn", "Deva")],
    "kn": [("Latn", "Knda")],
    "ml": [("Latn", "Mlym")],
    "mr": [("Latn", "Deva")],
    # Gurmukhi (Guru) and Shahmukhi (Arab) for Punjabi.
    "pa": [("Latn", "Guru"), ("Arab", "Guru"), ("Latn", "Arab")],
    "sd": [("Latn", "Arab")],
    "si": [("Latn", "Sinh")],
    "ta": [("Latn", "Taml")],
    "te": [("Latn", "Telu")],
    "ur": [("Latn", "Arab")],
}

SELECTED_LANGS = ['ta', 'te', 'el', 'hy', 'ru', 'kk', 'am', 'vi', 'ja', 'fr', 'sm', 'st', 'ko', 'de', 'mt', 'pl', 'sn', 'en']

def is_under_represented(lang: str) -> bool:
  """Checks if this language code (~2 letter) is under-represented."""
  if '_' in lang:
    # For languages with locales such as 'af_za', remove locale.
    lang = lang.split('_')[0]
  if lang in XTREME_UP_UNDER_REPRESENTED_LANGS:
    return True
  else:
    if lang not in XTREME_UP_HIGH_RESOURCE_LANGS:
      raise ValueError(f'Unrecognized language code: {lang}')
    return False

def is_selected(lang: str) -> bool:
  """Checks if this language code was selected for analysis."""
  if '_' in lang:
    # For languages with locales such as 'af_za', remove locale.
    lang = lang.split('_')[0]
  if lang in SELECTED_LANGS:
    return True
  else:
    return False



def get_languages(
    task: str,
    under_represented_only: bool = False,
    include_code_switching: bool = False,
) -> Sequence[str]:
  """Gets the languages for `task`."""
  if 'retrieval_' in task:
    # QA and retrieval have the same set of tasks.
    task = task.replace('retrieval_', 'qa_')
  if task == "transliteration":
    result = list(TRANSLIT_LANGS_AND_SCRIPTS.keys())
  else:
    if task not in TASK_TO_LANGUAGES:
      raise ValueError(f'Unrecognized task: "{task}"')
    result = TASK_TO_LANGUAGES[task]
  if not include_code_switching:
    result = [lang for lang in result if not lang.endswith('_cs')]
  if under_represented_only:
    result = [lang for lang in result if is_under_represented(lang)]
  return result
