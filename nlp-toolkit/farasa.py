"""
Pythonic and thread-safe wrapper around Farasa.

Farasa is developed at QCRI and can be found at http://qatsdemo.cloudapp.net/farasa/
Paper can be found at http://www.aclweb.org/anthology/N16-3003
"""
import logging
import os.path
from functools import partial
from collections import defaultdict
from operator import itemgetter
from typing import List, Tuple, Dict

from functional import seq
from fuzzywuzzy import process
from py4j.java_gateway import GatewayParameters, JavaGateway, launch_gateway

import utils
from utils import preprorcess_text

LOGGER = utils.setup_logger('farasa', logging.INFO)

FILE_PATH = os.path.dirname(__file__)
FARASA_JARS = [
    os.path.join(FILE_PATH, 'Farasa/NER/NER.jar'),
    os.path.join(FILE_PATH, 'Farasa/POS/POS.jar'),
]

CACHE_SIZE = 100

if not seq(FARASA_JARS).map(os.path.isfile).all():
    raise FileNotFoundError(
        'could not locate Farasa .jar files, %s are required' % FARASA_JARS
    )

CLASS_PATH = ':'.join(FARASA_JARS)


class Farasa:
    """
    Pythonic wrapper around Farasa.

    Supports Farasa Segmenter, POS and NER taggers.
    """
    SEGMENT_TYPES = ['S', 'E',
                     'V', 'NOUN', 'PRON', 'ADJ', 'NUM',
                     'CONJ', 'PART', 'NSUFF', 'CASE', 'FOREIGN',
                     'DET', 'PREP', 'ABBREV', 'PUNC']

    NER_TOKEN_TYPES = ['B-LOC', 'B-ORG', 'B-PERS',
                       'I-LOC', 'I-ORG', 'I-PERS']

    def __init__(self) -> None:
        """Initialize Farasa."""
        self.gateway = self.__launch_java_gateway()

        base = self.gateway.jvm.com.qcri.farasa

        self.segmenter = base.segmenter.Farasa()
        self.pos_tagger = base.pos.FarasaPOSTagger(self.segmenter)
        self.ner = base.ner.ArabicNER(self.segmenter, self.pos_tagger)

    @preprorcess_text(remove_punct=False)
    def tag_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tag part of speech.

        :param text: text to process.

        :returns: List of (token, token_type) pairs.
        """
        result = []

        segments = self.segment(text)
        for segment in self.pos_tagger.tagLine(segments).clitics:
            result.append(
                (segment.surface, segment.guessPOS)
            )

        return result

    def filter_pos(self, text: str, keep: List[str]) -> str:
        """
        Filter parts of speech

        :param text: text to process.
        :param keep: list of parts of speech to keep.

        :returns: filtered text.
        """
        pos = self.tag_pos(text)
        get_match = partial(process.extractOne, choices=text.split())

        return ' '.join(seq(pos)
                        .filter(lambda x: x[1] in keep and '+' not in x[1])
                        .map(itemgetter(0))
                        .map(lambda word: get_match(word)[0])
                        .to_list()
                        )

    @preprorcess_text(remove_punct=False)
    def segment(self, text: str) -> List[str]:
        """
        Segment piece of text.

        :param text: text to process.

        :returns: Unaltered Farasa segmenter output.
        """
        return self.segmenter.segmentLine(text)

    @preprorcess_text(remove_punct=False)
    def get_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Get named entities.

        :param text: text to process.

        :returns: List of (token, token_type) pairs.
        """
        tokens = (seq(self.ner.tagLine(text))
                  .map(lambda token: token.split('/'))
                  .filter(lambda token: token[1] in self.NER_TOKEN_TYPES)
                  )

        result: Dict[Tuple[int, str], List[str]] = defaultdict(list)
        entities: List[Tuple[str, str]] = []

        index = -1
        # Farasa returns named entities in IOB Style (Inside, Outside and Begninning).
        # Related Entities are grouped together.
        for token, info in tokens:
            position, token_type = info.split('-')

            if position == 'B':
                index += 1

            result[(index, token_type)].append(token)

        # Return NE as a name and type pairs, i.e. ('Egypt', 'LOC').
        for key in sorted(result.keys(), key=lambda value: value[0]):
            entities.append(
                (' '.join(result[key]), key[1])
            )

        # Keep distinct NE ONLY.
        return seq(entities).distinct().to_list()

    @staticmethod
    def __launch_java_gateway() -> JavaGateway:
        """Launch java gateway."""
        port = launch_gateway(classpath=CLASS_PATH, die_on_exit=True)
        params = GatewayParameters(
            port=port, auto_convert=True, auto_field=True, eager_load=True
        )

        return JavaGateway(gateway_parameters=params)


f = Farasa()

text2 = '''
جماهير مصرية تشجع المنتخب القومي لكرة القدم.
تقع مهام إدارة الرياضة في مصر بشتى مجالاتها على عاتق وزارة الدولة للرياضة [366]، وتعدّ كرة القدم هي أكثر الألعاب شعبية 
في مصر. تأسس الاتحاد المصري لكرة القدم عام 1921م وانضم إلى الاتحاد الدولي لكرة القدم في 1923م لتكون بذلك 
أول دولة إفريقية وعربية تنضم للفيفا،[367] كما تعدّ أيضاً من المؤسسين للاتحاد الأفريقي لكرة القدم عام 1957م،
 يمثل مصر دوليًا "منتخب مصر لكرة القدم" والذي يعد أول فريق أفريقي يلعب في كأس العالم عام 1934،[368][369]
 وهو صاحب أكثر عدد مرات فوز ببطولة كأس الأمم الأفريقية لكرة القدم فقد نالها سبعة مرات آخرها عام (2010) بأنجولا
، وفي 2010 وصل المنتخب المصري في تصنيف الفيفا إلى المركز التاسع عالميًا.[370].

الناديان الأكثر شعبية هما النادي الأهلي المصري ونادي الزمالك، والنادي الأهلي المصري هو أكثر الفرق فوزًا بالبطولات
 الأفريقية للأندية أبطال الدوري برصيد ثمان بطولات يليه نادي الزمالك برصيد خمس بطولات بالإضافة إلى بطولة وحيدة
 للنادي الإسماعيلي، كما فاز الأهلي بالمركز الثالث في كأس العالم للأندية باليابان والتي تأهل لها خمسة مرات.[371]

كما أن هناك حضور مصري عالمي في العديد من الرياضات 
الأخرى ذات شعبية أقل من كرة القدم، ككرة اليد، والإسكواش ورفع الأثقال. حيث احتلت مصر المرتبة الأولي
 عالميا في الإسكواش بلاعبيها العالميين عمرو شبانة ورامي عاشور، كما تعدّ مصر هي منشأ لعبة كرة السرعة.[372]

تُشارك مصر كذلك في الألعاب الأولمبية بأنواعها، ويرجع التاريخ الأوليمبي إلى عام 1910 عندما انضمت مصر 
إلى اللجنة الأولمبية الدولية لتصبح الدولة رقم 14 في اللجنة. وتشارك مصر بشكل دائم في دورة الألعاب 
الأولمبية الصيفية ودورة ألعاب البحر المتوسط ودورة الألعاب العربية ودورة الألعاب الأفريقية، يعد المصارع 
المصري كرم جابر أشهر اللاعبين الأوليمبيين في الفترة الأخيرة حيث أحرز الميدالية الذهبية في أولمبياد 
أثينا 2004 والفضية في أوليمبياد لندن 2012، تحتفل مصر في 3 مارس من كل عام بعيد الرياضة المصرية.[373
بالقاهرة.
بدأ البث التلفزيوني في مصر عام 1960 عندما تم تأسيس "التلفزيون العربي" والذي تم بثه في مصر وسوريا أثناء الوحدة
، بدأ البث التلفزيوني بقناة واحدة وكان البث بمعدل 6 ساعات يوميا، وفي عام 1961 تم إطلاق القناة الثانية
 المصرية وفي عام 1962 بدأ إرسال ثالث قناة بالتليفزيون المصري، وبدأ بث أول قناة فضائية مصرية عام 1990
، ويضم قطاع الفضائيات الآن "المصرية 1" وقناة النيل الدولية التي تبث برامجها بالإنجليزية والفرنسية
 والعبرية، وفي فترة التسعينات تم إطلاق القنوات الإقليمية بدأ من القناة الرابعة حتى القناة الثامنة.[355]

في 1998 دخلت مصر عصر البث الفضائي مع إطلاق القمر المصري نايل سات 101 وبدأ
 البث التجريبي لقنوات النيل المتخصصة في 31 مايو 1998 والبث الفعلي أكتوبر من ذات العام. وتبث إرسالها على
 أقمار النايل سات وانتلسات واسيا سات وبنما سات وعددها 12 قناة منها: قناة النيل وقنوات للدراما والأسرة
 والطفل وللرياضة وللثقافة وللمنوعات وللتعليم والبحث العلمي، وفي عام 2000 تم إطلاق القمر نايل سات 102.

وفي عام 2000 تم 
إنشاء مدينة الإنتاج الإعلامي بمدينة السادس من أكتوبر، ويوجد بها وحدة التحكم الرئيسية للقمرين الصناعيين نايل 
سات 1و2، وبها العديد من استوديوهات الإنتاج التلفزيوني والسينمائي ومناطق التصوير المفتوحة.[356]

السينما[عدل]
Crystal Clear app kdict.png مقالة مفصلة: السينما المصرية

يوسف وهبي وأمينة رزق في فيلم أولاد الذوات الذي يعد أول فيلم مصري ناطق عام 1932.
بدأت علاقة مصر بالسينما مع بدء صناعة السينما في العالم، فقد قُدم أول عرض سينمائي في مصر بالإسكندرية في يناير عام 
1896 (منذ 122 سنة) وتبعه عرض في القاهرة في نفس الشهر، وذلك بعد أيام من أول عرض سينمائي في العالم الذي كان
 في باريس في ديسمبر عام 1895. ومن أشهر الأفلام الصامتة ليلى قبلة في الصحراء وزينب، وقد عُرض أول فيلم
 مصري ناطق عام 1932 (منذ 86 سنة) وهو فيلم أولاد الذوات من بطولة يوسف وهبي وأمينة رزق، وفي عام 1935 
تأسس ستديو مصر والذي كان بمثابة قاعدة لبداية نهضة سينمائية حقيقية في مصر.[357][358]

بعد قيام ثورة يوليو 1952 ومن بعدها إعلان الجمهورية أصبحت السينما المصرية أكثر ازدهاراً، وبدأ على شكل واسع 
انتشار الفيلم المصري في الدول العربية؛ كذلك وصل إلى دول غير عربية كأثيوبيا والهند وباكستان واليونان
 والولايات المتحدة والعديد من دول أوروبا، وأصبحت السينما صناعة قومية في البلاد.[359]

ازداد عدد دور العرض السينمائي مع ظهور الأفلام الناطقة، ووصل إلى 395 داراً عام 1958. بدأ هذا العدد في الانخفاض
 بعد إنشاء التلفزيون عام 1960 وإنشاء القطاع العام في السينما عام 1962 ووصل إلى 297 داراً عام 1965،
 ثم إلى 141 عام 1995 بسبب تداول الأفلام عبر أجهزة الفيديو على الرغم من رواج صناعة السينما في هذه
 الفترة. وبفضل قوانين وإجراءات شجعت الاستثمار في إنشاء دور العرض الخاصة، عادت تزداد من جديد خاصةً في
 المراكز التجارية حتى وصل عددها إلى 200 عام 2001، وإلى 400 عام 2009.[360] وعلى مدى أكثر من مائة عام 
قدمت السينما المصرية أكثر من أربعة آلاف فيلمًا، ويعدّ من أبرز الفنانين المصريين الذين حققوا شهرة واسعة
 عالمياً الفنان عمر الشريف الذي رشح للأوسكار وفاز بثلاثة جوائز جولدن جلوب.[361][362]

'''

tokens2 = f.tag_pos(text2)
filtered = f.filter_pos(text2, ['V'])
entities2 = f.get_named_entities(text2)

# print(entities2)
# print(f.segment(text2))
print(filtered)
