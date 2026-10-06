"""Generate the synthetic training/validation set.

Each message is built from a hand-written template (per intent and dialect) plus
a tone phrase and an optional urgency marker. Labels follow fixed rules so they
are consistent (see "Labeling rules" in the README):

- sentiment: negative if the template is a complaint or the tone is annoyed/angry;
  positive if the tone is thankful; otherwise neutral.
- urgency: high if there's an urgency marker or the tone is angry; medium if the
  sentiment is negative; otherwise low.

The test set (data/test.jsonl) is hand-written separately and never generated here.

Usage: python data/generate.py
"""

import json
import random
from pathlib import Path

SEED = 42
N_TRAIN, N_VAL = 900, 100
OUT = Path(__file__).parent

ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# (template, base_sentiment) per intent and dialect. "neg" = complaint by nature.
# Slots: {order}, {product}, {days}
TEMPLATES = {
    "delivery_delay": {
        "en": [
            ("My {product} still hasn't arrived and it's been {days} days. {order}", "neg"),
            ("Where is my package? Tracking hasn't updated in {days} days, {order}", "neg"),
            ("Hi, can you check on {order}? It was supposed to be delivered yesterday.", "neu"),
            ("I ordered a {product} {days} days ago and nothing yet", "neg"),
            ("When will {order} be delivered? The estimated date already passed.", "neu"),
            ("The courier never showed up for {order}", "neg"),
        ],
        "eg": [
            ("الأوردر بتاعي لسه موصلش بقاله {days} يوم، {order}", "neg"),
            ("فين ال{product} اللي طلبته؟ {order}", "neu"),
            ("المندوب مجاش خالص النهارده، {order}", "neg"),
            ("ممكن تشوفولي {order} وصل لفين؟", "neu"),
        ],
        "gulf": [
            ("طلبي للحين ما وصل صار له {days} أيام، {order}", "neg"),
            ("ممكن تتأكدون وين وصل {order}؟", "neu"),
        ],
        "msa": [
            ("لم يصلني الطلب حتى الآن رغم مرور {days} أيام، {order}", "neg"),
            ("أود الاستفسار عن موعد توصيل {order}", "neu"),
        ],
        "mixed": [
            ("el order lessa mawsalsh ba2alo {days} ayam, {order}", "neg"),
            ("فين الـ order بتاعي؟ الـ tracking مش بيتحرك، {order}", "neg"),
            ("emta el {product} hayewsal? {order}", "neu"),
        ],
    },
    "refund_request": {
        "en": [
            ("I want a refund for {order}", "neu"),
            ("Please refund my money for the {product}, {order}", "neu"),
            ("I returned the {product} {days} days ago and still no refund. {order}", "neg"),
            ("How long does a refund take? I sent back {order} last week.", "neu"),
            ("Give me my money back, the {product} is nothing like the photos", "neg"),
        ],
        "eg": [
            ("عايز فلوسي ترجع لو سمحت، {order}", "neu"),
            ("رجعت ال{product} من {days} يوم ولسه الفلوس مرجعتش، {order}", "neg"),
            ("ازاي أعمل refund لـ {order}؟", "neu"),
        ],
        "gulf": [
            ("أبي أسترجع فلوسي، {order}", "neu"),
            ("رجعت المنتج وللحين ما رجعت الفلوس، {order}", "neg"),
        ],
        "msa": [
            ("أرغب في استرداد المبلغ الخاص بـ {order}", "neu"),
            ("لم يتم استرداد المبلغ رغم إرجاع المنتج منذ {days} أيام، {order}", "neg"),
        ],
        "mixed": [
            ("3ayez refund lel {order} law sama7t", "neu"),
            ("raga3t el {product} w lessa el folos marga3etsh, {order}", "neg"),
        ],
    },
    "return_exchange": {
        "en": [
            ("Can I exchange the {product} for a different size? {order}", "neu"),
            ("I'd like to return the {product}, it doesn't fit. {order}", "neu"),
            ("How do I return an item? {order}", "neu"),
            ("Wrong color was sent, I need to exchange it. {order}", "neg"),
        ],
        "eg": [
            ("ينفع أبدل ال{product} بمقاس تاني؟ {order}", "neu"),
            ("عايز أرجع ال{product} مش عاجبني، {order}", "neu"),
            ("بعتولي لون غلط وعايز أبدله، {order}", "neg"),
        ],
        "gulf": [
            ("أبي أبدل ال{product} بمقاس ثاني، {order}", "neu"),
            ("كيف أرجع المنتج؟ {order}", "neu"),
        ],
        "msa": [
            ("أرغب في استبدال ال{product} بمقاس آخر، {order}", "neu"),
        ],
        "mixed": [
            ("momken a3mel exchange lel {product}? el size kbeer, {order}", "neu"),
            ("عايز أعمل return لل{product}، {order}", "neu"),
        ],
    },
    "cancel_order": {
        "en": [
            ("Please cancel {order}", "neu"),
            ("I need to cancel my order before it ships. {order}", "neu"),
            ("Cancel {order}, I found it cheaper somewhere else", "neu"),
            ("I placed the order by mistake, can you cancel it? {order}", "neu"),
        ],
        "eg": [
            ("لو سمحت الغوا {order}", "neu"),
            ("عايز ألغي الأوردر قبل ما يتشحن، {order}", "neu"),
        ],
        "gulf": [
            ("أبي ألغي {order} لو سمحتوا", "neu"),
        ],
        "msa": [
            ("أرجو إلغاء {order}", "neu"),
            ("أود إلغاء الطلب قبل شحنه، {order}", "neu"),
        ],
        "mixed": [
            ("3ayez a-cancel el order, {order}", "neu"),
            ("لو سمحت cancel لـ {order}", "neu"),
        ],
    },
    "payment_issue": {
        "en": [
            ("I was charged twice for {order}", "neg"),
            ("My card payment keeps failing at checkout", "neg"),
            ("The money was taken from my account but the order shows unpaid. {order}", "neg"),
            ("Do you accept cash on delivery?", "neu"),
            ("Why was I charged extra fees on {order}?", "neg"),
        ],
        "eg": [
            ("اتخصم مني الفلوس مرتين، {order}", "neg"),
            ("الفيزا مش راضية تدفع خالص", "neg"),
            ("الفلوس اتسحبت بس الأوردر مكتوب مش مدفوع، {order}", "neg"),
        ],
        "gulf": [
            ("انخصم المبلغ مرتين، {order}", "neg"),
            ("الدفع بالبطاقة ما يمشي", "neg"),
        ],
        "msa": [
            ("تم خصم المبلغ مرتين من بطاقتي، {order}", "neg"),
            ("هل يتوفر الدفع عند الاستلام؟", "neu"),
        ],
        "mixed": [
            ("el visa msh rady tedfa3", "neg"),
            ("اتخصم مني double charge على {order}", "neg"),
        ],
    },
    "damaged_product": {
        "en": [
            ("The {product} arrived broken. {order}", "neg"),
            ("My {product} stopped working after {days} days. {order}", "neg"),
            ("The box was crushed and the {product} inside is damaged, {order}", "neg"),
            ("Received my {product} with a cracked screen, {order}", "neg"),
        ],
        "eg": [
            ("ال{product} وصل مكسور، {order}", "neg"),
            ("ال{product} باظ بعد {days} يوم بس، {order}", "neg"),
        ],
        "gulf": [
            ("ال{product} وصلني خربان، {order}", "neg"),
        ],
        "msa": [
            ("وصل ال{product} تالفاً، {order}", "neg"),
        ],
        "mixed": [
            ("el {product} wesel maksoor, {order}", "neg"),
            ("ال{product} فيه defect من أول يوم، {order}", "neg"),
        ],
    },
    "account_issue": {
        "en": [
            ("I can't log in to my account", "neg"),
            ("How do I change the email on my account?", "neu"),
            ("I never received the password reset email", "neg"),
            ("Please delete my account", "neu"),
        ],
        "eg": [
            ("مش عارف أعمل تسجيل دخول على حسابي", "neg"),
            ("ازاي أغير رقم الموبايل في الحساب؟", "neu"),
        ],
        "gulf": [
            ("ما أقدر أدخل حسابي", "neg"),
        ],
        "msa": [
            ("كيف يمكنني تغيير كلمة المرور؟", "neu"),
        ],
        "mixed": [
            ("msh 3aref a3mel login 3ala el account", "neg"),
            ("الـ password reset email موصلنيش", "neg"),
        ],
    },
    "general_question": {
        "en": [
            ("Do you ship to Alexandria?", "neu"),
            ("Is the {product} available in black?", "neu"),
            ("What are your working hours?", "neu"),
            ("Do you have any discounts this week?", "neu"),
            ("Does the {product} come with a warranty?", "neu"),
        ],
        "eg": [
            ("بتوصلوا اسكندرية؟", "neu"),
            ("ال{product} متوفر باللون الأسود؟", "neu"),
            ("في عروض الأسبوع ده؟", "neu"),
        ],
        "gulf": [
            ("توصلون الرياض؟", "neu"),
            ("ال{product} عليه ضمان؟", "neu"),
        ],
        "msa": [
            ("ما هي مواعيد العمل لديكم؟", "neu"),
            ("هل يتوفر ال{product} بألوان أخرى؟", "neu"),
        ],
        "mixed": [
            ("fe offers el osbo3 da?", "neu"),
            ("el {product} 3aleh warranty?", "neu"),
            ("هل فيه delivery لجدة؟", "neu"),
        ],
    },
}

# Product indexes by category, so templates stay realistic (no "broken jacket")
ELECTRONICS = [0, 1, 2, 5, 6, 8]
SCREENS = [0, 1, 6]
CLOTHING = [3, 4, 7]

PRODUCTS = {
    "en": ["phone", "laptop", "headphones", "jacket", "shoes", "blender", "watch", "t-shirt", "charger", "backpack"],
    "ar": ["موبايل", "لابتوب", "سماعات", "جاكيت", "جزمة", "خلاط", "ساعة", "تيشيرت", "شاحن", "شنطة"],
    "franco": ["mobile", "laptop", "sama3at", "jacket", "shoes", "blender", "sa3a", "t-shirt", "charger", "shanta"],
}

# Tone phrases per dialect: (phrase, position)
TONES = {
    "en": {
        "polite": ["Hello,", "Hi there,", "Good morning,", "Hi team,"],
        "thankful": ["Thanks a lot, you guys are great!", "Love your store, thank you!", "Really appreciate your help 🙏"],
        "annoyed": ["This is really frustrating.", "Not happy about this.", "Honestly disappointed."],
        "angry": ["This is unacceptable!!", "Worst service ever.", "I'm extremely angry right now."],
        "urgent": ["Please reply ASAP.", "I need this sorted today.", "URGENT!", "Need an answer right now please."],
    },
    "eg": {
        "polite": ["السلام عليكم،", "مساء الخير،", "صباح الخير،"],
        "thankful": ["شكراً جداً ليكم، خدمتكم ممتازة", "متشكر جداً ليكم ❤️", "بجد انتو أحسن متجر"],
        "annoyed": ["بصراحة ده مضايقني", "مش مبسوط خالص", "حاجة تزهق"],
        "angry": ["ده كلام مش مقبول!!", "أسوأ خدمة شفتها في حياتي", "أنا متضايق جداً ومش هتعامل معاكم تاني"],
        "urgent": ["محتاج رد النهارده ضروري", "بسرعة لو سمحت", "ضروري جداً"],
    },
    "gulf": {
        "polite": ["السلام عليكم", "هلا والله،", "مساكم الله بالخير،"],
        "thankful": ["مشكورين ما قصرتوا", "يعطيكم العافية على الخدمة الحلوة"],
        "annoyed": ["والله شي يضايق", "مو راضي أبد"],
        "angry": ["هذا شي ما ينقبل!!", "خدمة سيئة جداً ولا عاد أطلب منكم"],
        "urgent": ["أبي رد اليوم ضروري", "تكفون بسرعة", "مستعجل مرة"],
    },
    "msa": {
        "polite": ["مرحباً،", "تحية طيبة،", "السلام عليكم ورحمة الله،"],
        "thankful": ["شكراً جزيلاً لكم على خدماتكم الرائعة", "أشكركم على حسن التعامل"],
        "annoyed": ["هذا الأمر مزعج للغاية.", "لست راضياً عن ذلك."],
        "angry": ["هذا غير مقبول إطلاقاً!", "خدمة سيئة للغاية!"],
        "urgent": ["أرجو الرد بشكل عاجل.", "الأمر عاجل جداً."],
    },
    "mixed": {
        "polite": ["salam,", "hi ya gama3a,", "ezayoko,", "hello يا جماعة،"],
        "thankful": ["merci awy, ento gamdeen", "shokran gedan 🙏", "thanks بجد انتو جامدين"],
        "annoyed": ["bgd da mdaye2ny", "msh mabsoot khales", "honestly ده مضايقني"],
        "angry": ["da msh ma2bool!!", "worst service بجد!!", "ana met3asab gedan"],
        "urgent": ["3ayez rad enharda darory", "please bsor3a", "urgent لو سمحت"],
    },
}

TONE_WEIGHTS = {"none": 35, "polite": 20, "thankful": 15, "annoyed": 15, "angry": 15}
DIALECT_LANGUAGE = {"en": "en", "eg": "ar", "gulf": "ar", "msa": "ar", "mixed": "mixed"}
# Target mix: ~50% English, ~35% Arabic, ~15% mixed/Franco
DIALECT_WEIGHTS = {"en": 50, "eg": 17, "gulf": 9, "msa": 9, "mixed": 15}


def make_order(rng: random.Random, dialect: str) -> tuple[str, str]:
    """Return (text as written in the message, normalized label)."""
    if rng.random() < 0.25:
        label = f"ORD-{rng.randint(10000, 99999)}"
    else:
        label = str(rng.randint(1000, 999999))
    shown = label
    if dialect in ("eg", "gulf", "msa") and rng.random() < 0.5:
        shown = label.translate(ARABIC_DIGITS)
    if dialect == "en":
        text = rng.choice(["#{}", "order {}", "order #{}", "Order number: {}", "order no. {}"]).format(shown)
    elif dialect == "mixed":
        text = rng.choice(["order {}", "el order raqam {}", "#{}", "رقم الأوردر {}"]).format(shown)
    else:
        text = rng.choice(["رقم الطلب {}", "طلب رقم {}", "الأوردر رقم {}", "#{}"]).format(shown)
    return text, label


def make_example(rng: random.Random) -> dict:
    intent = rng.choice(list(TEMPLATES))
    by_dialect = TEMPLATES[intent]
    dialect = rng.choices(list(DIALECT_WEIGHTS), weights=list(DIALECT_WEIGHTS.values()))[0]
    template, base = rng.choice(by_dialect[dialect])

    order_text, order_id = make_order(rng, dialect)
    if "{order}" not in template:
        order_id = None
    product_pool = "en" if dialect == "en" else "franco" if template.isascii() else "ar"
    if "screen" in template:
        allowed = SCREENS
    elif intent == "damaged_product" or "warranty" in template or "ضمان" in template:
        allowed = ELECTRONICS
    elif intent == "return_exchange" or "size" in template or "مقاس" in template:
        allowed = CLOTHING
    else:
        allowed = range(len(PRODUCTS["en"]))
    product = PRODUCTS[product_pool][rng.choice(list(allowed))]
    text = template.format(order=order_text, product=product, days=rng.randint(2, 20))

    tones = dict(TONE_WEIGHTS)
    if base == "neg":
        tones.pop("thankful")  # "thanks!" + "it's broken" gives a muddled label
    tone = rng.choices(list(tones), weights=list(tones.values()))[0]
    phrases = TONES[dialect]
    if tone == "polite":
        text = f"{rng.choice(phrases['polite'])} {text}"
    elif tone != "none":
        text = f"{text} {rng.choice(phrases[tone])}"
    urgent = rng.random() < 0.25
    if urgent:
        text = f"{text} {rng.choice(phrases['urgent'])}"

    # Light noise so the model doesn't rely on perfect formatting
    if dialect in ("en", "mixed") and rng.random() < 0.2:
        text = text.lower().replace(order_text.lower(), order_text)  # keep the order ID as written
    if rng.random() < 0.15:
        text = text.rstrip(".!?؟")

    if base == "neg" or tone in ("annoyed", "angry"):
        sentiment = "negative"
    elif tone == "thankful":
        sentiment = "positive"
    else:
        sentiment = "neutral"
    if urgent or tone == "angry":
        urgency = "high"
    elif sentiment == "negative":
        urgency = "medium"
    else:
        urgency = "low"

    return {
        "text": text.strip(),
        "label": {
            "intent": intent,
            "order_id": order_id,
            "sentiment": sentiment,
            "urgency": urgency,
            "language": DIALECT_LANGUAGE[dialect],
        },
    }


def main():
    rng = random.Random(SEED)
    test_texts = {
        json.loads(line)["text"] for line in (OUT / "test.jsonl").read_text(encoding="utf-8").splitlines() if line
    } if (OUT / "test.jsonl").exists() else set()

    seen, examples = set(test_texts), []
    while len(examples) < N_TRAIN + N_VAL:
        ex = make_example(rng)
        if ex["text"] not in seen:
            seen.add(ex["text"])
            examples.append(ex)

    for name, rows in (("train", examples[:N_TRAIN]), ("val", examples[N_TRAIN:])):
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} examples to data/{name}.jsonl")


if __name__ == "__main__":
    main()
