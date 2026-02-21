import os
import telebot
from telebot.types import ReplyKeyboardMarkup
import sqlite3
import threading
import time
from datetime import datetime
import json
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import random
import gc
import hashlib
from detoxify import Detoxify
from flask import Flask
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== ТВОИ ДАННЫЕ ИЗ ПЕРЕМЕННЫХ ОКРУЖЕНИЯ =====
BOT_TOKEN = os.environ.get('BOT_TOKEN')
LESSONS_CHANNEL_ID = int(os.environ.get('LESSONS_CHANNEL_ID', '-1003849222505'))
KNOWLEDGE_CHANNEL_ID = int(os.environ.get('KNOWLEDGE_CHANNEL_ID', '-1003790164516'))
ADMIN_ID = int(os.environ.get('ADMIN_ID', '1393455996'))
MODEL_NAME = os.environ.get('MODEL_NAME', 'sberbank-ai/rugpt3large_based_on_gpt2')  # можно переопределить

bot = telebot.TeleBot(BOT_TOKEN)
START_TIME = time.time()

# ===== SQLite (локальный файл) =====
DB_PATH = 'yuki_bot.db'
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS lessons
             (lesson_num INTEGER PRIMARY KEY, title TEXT, content TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS channel_messages
             (message_id INTEGER PRIMARY KEY, chat_id INTEGER, text TEXT, date TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS users
             (user_id TEXT PRIMARY KEY, name TEXT, current_lesson INTEGER DEFAULT 1,
              messages_count INTEGER DEFAULT 0, xp INTEGER DEFAULT 0,
              badges TEXT DEFAULT '["🎌"]', lessons_completed TEXT DEFAULT '[]')''')
c.execute('''CREATE TABLE IF NOT EXISTS logs
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              user_id TEXT,
              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
              action TEXT,
              details TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS settings
             (key TEXT PRIMARY KEY, value TEXT)''')
conn.commit()

# ===== ЛОГГЕР =====
def log(msg):
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[{now}] {msg}")

def log_user_action(user_id, action, details=""):
    try:
        c.execute("INSERT INTO logs (user_id, action, details) VALUES (?,?,?)",
                  (user_id, action, details))
        conn.commit()
    except Exception as e:
        print(f"Logging error: {e}")

def get_setting(key, default="0"):
    c.execute("SELECT value FROM settings WHERE key=?", (key,))
    row = c.fetchone()
    if row:
        return row[0]
    else:
        c.execute("INSERT INTO settings (key, value) VALUES (?,?)", (key, default))
        conn.commit()
        return default

def set_setting(key, value):
    c.execute("REPLACE INTO settings (key, value) VALUES (?,?)", (key, value))
    conn.commit()

# ===== УРОКИ =====
def load_lessons():
    c.execute("SELECT lesson_num, title, content FROM lessons")
    rows = c.fetchall()
    lessons = {}
    for num, title, content in rows:
        lessons[num] = {"title": title, "content": content}
    log(f"📚 Уроков загружено: {len(lessons)}")
    return lessons

lessons = load_lessons()

def clean_lesson_content(raw_content):
    if not raw_content:
        return ""
    lines = [line.strip() for line in raw_content.split('\n') if line.strip()]
    cleaned, buffer = [], ""
    for line in lines:
        if '—' in line:
            if buffer:
                cleaned.append(buffer)
                buffer = ""
            cleaned.append(line)
        else:
            if len(line) <= 2 and not re.search(r'[а-яА-Яa-zA-Z0-9]', line):
                buffer += line
            else:
                if buffer:
                    cleaned.append(buffer + line)
                    buffer = ""
                else:
                    cleaned.append(line)
    if buffer:
        cleaned.append(buffer)
    return '\n'.join(cleaned)

# ===== ПОЛЬЗОВАТЕЛИ =====
def get_user(user_id):
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row:
        return {
            "user_id": row[0],
            "name": row[1],
            "current_lesson": row[2],
            "messages_count": row[3],
            "xp": row[4],
            "badges": json.loads(row[5]),
            "lessons_completed": json.loads(row[6])
        }
    else:
        user = {
            "user_id": user_id,
            "name": "",
            "current_lesson": 1,
            "messages_count": 0,
            "xp": 0,
            "badges": ["🎌"],
            "lessons_completed": []
        }
        c.execute("INSERT INTO users VALUES (?,?,?,?,?,?,?)",
                  (user_id, "", 1, 0, 0, json.dumps(["🎌"]), json.dumps([])))
        conn.commit()
        log_user_action(user_id, "new_user")
        return user

def save_user(user):
    c.execute('''UPDATE users SET name=?, current_lesson=?, messages_count=?,
                 xp=?, badges=?, lessons_completed=? WHERE user_id=?''',
              (user["name"], user["current_lesson"], user["messages_count"],
               user["xp"], json.dumps(user["badges"]), json.dumps(user["lessons_completed"]),
               user["user_id"]))
    conn.commit()

# ===== E5-LARGE (ПОИСК) =====
gc.collect()
device = 'cpu'  # принудительно CPU
log(f"🖥️ Используется устройство: {device}")

log("🔄 Загрузка E5-large...")
try:
    embedder = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    log("✅ E5-large загружена")
except Exception as e:
    log(f"❌ Ошибка загрузки E5-large: {e}")
    raise

def encode_queries(queries):
    return embedder.encode(["query: " + q for q in queries], batch_size=8, convert_to_numpy=True)

def encode_corpus(texts):
    return embedder.encode(["passage: " + t for t in texts], batch_size=8, convert_to_numpy=True)

def load_corpus():
    c.execute("SELECT text FROM channel_messages WHERE chat_id=? AND text IS NOT NULL AND text != ''", (KNOWLEDGE_CHANNEL_ID,))
    return [row[0] for row in c.fetchall()]

corpus_texts = load_corpus()
log(f"📄 В корпусе: {len(corpus_texts)}")

def build_index(texts):
    if not texts:
        return None, []
    embeddings = encode_corpus(texts)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, texts

index, corpus = build_index(corpus_texts)
log("✅ Индекс E5 построен")

def search_answer(query, top_k=3, min_sim=0.6):
    if index is None or len(corpus) == 0:
        return []
    q_emb = encode_queries([query])
    faiss.normalize_L2(q_emb)
    dist, idx = index.search(q_emb, top_k)
    results = []
    for i in range(top_k):
        if idx[0][i] != -1 and dist[0][i] >= min_sim:
            results.append(corpus[idx[0][i]])
    return results

def is_answer_relevant(query, answer, threshold=0.6):
    q_emb = encode_queries([query])
    a_emb = encode_corpus([answer])
    faiss.normalize_L2(q_emb)
    faiss.normalize_L2(a_emb)
    sim = np.dot(q_emb, a_emb.T)[0][0]
    log(f"🔍 Сходство: {sim:.3f}")
    return sim >= threshold

e5_check_enabled = get_setting("e5_check_enabled", "1") == "1"

# ===== ruGPT3Large (760M) — НА CPU =====
log(f"🔥 Загрузка {MODEL_NAME} на CPU... (это займёт несколько минут)")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    log(f"✅ Модель загружена на {device}")
except Exception as e:
    log(f"❌ Ошибка загрузки модели: {e}")
    model = None  # если модель не загрузится, будем использовать только E5

# Few-shot примеры для генерации (можно оставить те же)
FEW_SHOT_EXAMPLES = """
Ниже приведены примеры вопросов и ответов. Отвечай кратко, понятно и по делу.

Вопрос: 2+2?
Ответ: 4

Вопрос: Столица России?
Ответ: Москва

Вопрос: Кто написал "Войну и мир"?
Ответ: Лев Толстой

Вопрос: Что такое хирагана?
Ответ: Японская слоговая азбука.

Вопрос: Как тебя зовут?
Ответ: Меня зовут Юки, я помогаю отвечать на вопросы.

Теперь ответь на следующий вопрос, используя информацию, если она есть.
Вопрос: {user_text}
Ответ:
"""

def generate_llm_answer(user_text, context=None, max_new_tokens=80):
    if model is None:
        return None  # если модель не загружена, сразу fallback
    try:
        if context:
            if len(context) > 500:
                context = context[:500] + "…"
            prompt = f"Используя эту информацию: {context}\n\nВопрос: {user_text}\nОтвет:"
        else:
            prompt = FEW_SHOT_EXAMPLES.format(user_text=user_text)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.4,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                num_return_sequences=1,
                early_stopping=True
            )
        answer = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        # простой фильтр мусора
        if sum(c.isdigit() for c in answer) / max(len(answer),1) > 0.3 or 'http' in answer:
            return None
        return answer
    except Exception as e:
        log(f"Ошибка генерации: {e}")
        return None

# ===== УМНЫЙ АНТИМАТ =====
log("🔄 Загрузка detoxify...")
try:
    tox_model = Detoxify('multilingual')
    log("✅ detoxify загружена")
except Exception as e:
    log(f"❌ Ошибка загрузки detoxify: {e}")
    tox_model = None

TOXICITY_THRESHOLD = 0.6

def is_toxic(text):
    if tox_model is None:
        return False
    try:
        results = tox_model.predict(text)
        if results['toxicity'] > TOXICITY_THRESHOLD or results['severe_toxicity'] > TOXICITY_THRESHOLD * 0.7:
            return True
        return False
    except Exception as e:
        log(f"Ошибка проверки токсичности: {e}")
        return False

# ===== ТАЙМАУТ =====
def run_with_timeout(func, timeout, *args, **kwargs):
    result, error = [], []
    def target():
        try:
            result.append(func(*args, **kwargs))
        except Exception as e:
            error.append(e)
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        log(f"⏰ Таймаут {timeout} сек")
        return None
    if error:
        log(f"⚠️ Ошибка: {error[0]}")
        return None
    return result[0] if result else None

# ===== КЛАВИАТУРЫ =====
def main_menu():
    m = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    m.add("📚 Учеба", "🎌 Культура", "👤 Профиль", "🏅 Бейджи", "❓ Помощь", "🔄 Сброс")
    return m

def study_menu():
    m = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    m.add("➡️ Следующий", "⬅️ Предыдущий", "📋 Выбрать урок", "◀️ Назад")
    return m

def culture_menu():
    items = ["🗾 Япония", "🍜 Еда", "🎌 Праздники"]
    m = ReplyKeyboardMarkup(resize_keyboard=True, row_width=2)
    for it in items:
        m.add(it)
    m.add("◀️ Назад")
    return m

# ===== ОБРАБОТКА СООБЩЕНИЙ ИЗ КАНАЛОВ =====
@bot.message_handler(func=lambda m: m.chat.id in [LESSONS_CHANNEL_ID, KNOWLEDGE_CHANNEL_ID])
def handle_channel_message(message):
    global index, corpus, lessons
    text = message.text or ""
    chat_id = message.chat.id
    if text.startswith('/'):
        return
    log(f"📥 В канале {chat_id}: {text[:50]}...")
    c.execute("INSERT OR REPLACE INTO channel_messages VALUES (?,?,?,?)",
              (message.message_id, chat_id, text, str(message.date)))
    conn.commit()

    # Если канал знаний и сообщение от админа, генерируем ответ
    if (chat_id == KNOWLEDGE_CHANNEL_ID and 
        message.from_user.id == ADMIN_ID and 
        message.from_user.id != bot.get_me().id):
        bot.send_chat_action(chat_id, 'typing')
        context_parts = search_answer(text, top_k=3, min_sim=0.6)
        context = "\n".join(context_parts) if context_parts else None
        answer = generate_llm_answer(text, context, max_new_tokens=80)
        if answer:
            msg = bot.send_message(chat_id, answer)
            c.execute("INSERT OR REPLACE INTO channel_messages VALUES (?,?,?,?)",
                      (msg.message_id, chat_id, answer, str(msg.date)))
            conn.commit()
            log(f"✅ Ответ отправлен в канал {chat_id}")
        else:
            log("⚠️ Не удалось сгенерировать ответ")

    # Обработка уроков
    if chat_id == LESSONS_CHANNEL_ID and text.startswith('📗 Урок'):
        lines = text.split('\n')
        title = lines[0].strip()
        raw = '\n'.join(lines[1:]) if len(lines) > 1 else ''
        cleaned = clean_lesson_content(raw)
        match = re.search(r'Урок (\d+)', title)
        num = int(match.group(1)) if match else max(lessons.keys(), default=0)+1
        if not match:
            title = f"📗 Урок {num}: {title.replace('📗 Урок','').strip()}"
        c.execute("INSERT OR REPLACE INTO lessons VALUES (?,?,?)", (num, title, cleaned))
        conn.commit()
        lessons[num] = {"title": title, "content": cleaned}
        log(f"✅ Урок {num} сохранён")

    # Перестроение индекса
    if chat_id == KNOWLEDGE_CHANNEL_ID:
        corpus_texts = load_corpus()
        index, corpus = build_index(corpus_texts)
        log("🔄 Индекс E5 перестроен")

# ===== КОМАНДЫ АДМИНА =====
@bot.message_handler(commands=['stats'])
def stats(message):
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    c.execute("SELECT COUNT(*) FROM users")
    users = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM logs")
    logs = c.fetchone()[0]
    bot.send_message(message.chat.id, f"📊 Пользователей: {users}\n📝 Логов: {logs}")

@bot.message_handler(commands=['clear_cache'])
def clear_cache(message):
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    global search_cache
    search_cache.clear()
    bot.send_message(message.chat.id, "🗑️ Кэш поиска очищен.")

@bot.message_handler(commands=['rebuild_index'])
def rebuild_index(message):
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    global index, corpus
    corpus_texts = load_corpus()
    index, corpus = build_index(corpus_texts)
    bot.send_message(message.chat.id, f"🔄 Индекс перестроен, в корпусе {len(corpus_texts)} сообщений")

@bot.message_handler(commands=['reset_knowledge_db'])
def reset_knowledge_db(message):
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    c.execute("DELETE FROM channel_messages WHERE chat_id=?", (KNOWLEDGE_CHANNEL_ID,))
    conn.commit()
    global index, corpus
    corpus_texts = load_corpus()
    index, corpus = build_index(corpus_texts)
    bot.send_message(message.chat.id, f"🗃️ База знаний очищена, в корпусе {len(corpus_texts)} сообщений")

@bot.message_handler(commands=['toggle_e5_check'])
def toggle_e5_check(message):
    global e5_check_enabled
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    e5_check_enabled = not e5_check_enabled
    set_setting("e5_check_enabled", "1" if e5_check_enabled else "0")
    status = "включена" if e5_check_enabled else "отключена"
    bot.send_message(message.chat.id, f"🔍 Проверка E5 {status}.")

@bot.message_handler(commands=['ping'])
def ping(message):
    bot.send_message(message.chat.id, "pong")

@bot.message_handler(commands=['status'])
def status(message):
    if str(message.from_user.id) != str(ADMIN_ID):
        return
    uptime = time.time() - START_TIME
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    c.execute("SELECT COUNT(*) FROM channel_messages WHERE chat_id=?", (KNOWLEDGE_CHANNEL_ID,))
    corpus_size = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM lessons")
    lessons_count = c.fetchone()[0]
    bot.send_message(message.chat.id,
                     f"⏱ Uptime: {hours}ч {minutes}м\n"
                     f"📚 Уроков: {lessons_count}\n"
                     f"📄 Сообщений в корпусе: {corpus_size}\n"
                     f"👥 Пользователей: {c.execute('SELECT COUNT(*) FROM users').fetchone()[0]}")

# ===== ОСНОВНОЙ ОБРАБОТЧИК =====
user_last_time = {}
search_cache = {}

@bot.message_handler(commands=['start'])
def start(message):
    uid = str(message.from_user.id)
    user = get_user(uid)
    if not user['name']:
        user['name'] = message.from_user.first_name or ''
        save_user(user)
    bot.send_message(message.chat.id,
                     f"🎌 Привет, {user['name']}!\n📚 Уроков: {len(lessons)}\nЯ работаю, но немного медленно (на CPU).",
                     reply_markup=main_menu())

@bot.message_handler(func=lambda m: True)
def handle_user_message(message):
    uid = str(message.from_user.id)
    user = get_user(uid)
    user['messages_count'] += 1
    text = message.text

    # Антифлуд
    ct = time.time()
    if uid in user_last_time and ct - user_last_time[uid] < 3.5:
        bot.send_message(message.chat.id, "⏳ Подожди 3.5 сек", reply_markup=main_menu())
        return
    user_last_time[uid] = ct

    # Умный антимат
    if is_toxic(text):
        bot.send_message(message.chat.id, "🚫 Пожалуйста, общайся культурно!", reply_markup=main_menu())
        log_user_action(uid, "toxicity_blocked", text[:50])
        return

    # МЕНЮ
    if text == "📚 Учеба":
        bot.send_message(message.chat.id, f"📚 Текущий урок: {user['current_lesson']}", reply_markup=study_menu())
    elif text == "➡️ Следующий":
        user['current_lesson'] = min(user['current_lesson'] + 1, 450)
        save_user(user)
        lesson = lessons.get(user['current_lesson'], {"title": f"Урок {user['current_lesson']}", "content": "Нет урока"})
        bot.send_message(message.chat.id, f"{lesson['title']}\n\n{lesson['content']}", reply_markup=study_menu())
    elif text == "⬅️ Предыдущий":
        user['current_lesson'] = max(user['current_lesson'] - 1, 1)
        save_user(user)
        lesson = lessons.get(user['current_lesson'], {"title": f"Урок {user['current_lesson']}", "content": "Нет урока"})
        bot.send_message(message.chat.id, f"{lesson['title']}\n\n{lesson['content']}", reply_markup=study_menu())
    elif text == "📋 Выбрать урок":
        markup = ReplyKeyboardMarkup(resize_keyboard=True, row_width=3)
        for i in range(1,11): markup.add(f"Урок {i}")
        markup.add("◀️ Назад")
        bot.send_message(message.chat.id, "Выбери урок:", reply_markup=markup)
    elif text.startswith("Урок ") and text[5:].isdigit():
        num = int(text[5:])
        if 1 <= num <= 450:
            user['current_lesson'] = num
            save_user(user)
            lesson = lessons.get(num, {"title": f"Урок {num}", "content": "Нет урока"})
            bot.send_message(message.chat.id, f"{lesson['title']}\n\n{lesson['content']}", reply_markup=study_menu())
    elif text == "🎌 Культура":
        bot.send_message(message.chat.id, "Выбери тему:", reply_markup=culture_menu())
    elif text in ["🗾 Япония","🍜 Еда","🎌 Праздники"]:
        facts = {"🗾 Япония":["🏯 Токио","🗻 Фудзи"], "🍜 Еда":["🍣 Суши","🍜 Рамен"], "🎌 Праздники":["🎉 Ханами","🎊 Сёгацу"]}
        bot.send_message(message.chat.id, random.choice(facts[text]), reply_markup=main_menu())
    elif text == "👤 Профиль":
        bot.send_message(message.chat.id, f"👤 {user['name']}\n📚 Урок: {user['current_lesson']}\n🏅 {' '.join(user['badges'])}")
    elif text == "🏅 Бейджи":
        bot.send_message(message.chat.id, f"🏅 {' '.join(user['badges'])}")
    elif text == "❓ Помощь":
        bot.send_message(message.chat.id, "👑 Используй меню.")
    elif text == "🔄 Сброс":
        c.execute("DELETE FROM users WHERE user_id=?", (uid,))
        conn.commit()
        bot.send_message(message.chat.id, "🔄 Данные сброшены.", reply_markup=main_menu())
    elif text == "◀️ Назад":
        bot.send_message(message.chat.id, "Главное меню:", reply_markup=main_menu())
    else:
        # ОСНОВНОЙ ЗАПРОС
        log_user_action(uid, "query", text[:200])
        bot.send_chat_action(message.chat.id, 'typing')

        context_parts = search_answer(text, top_k=3, min_sim=0.6)
        context = "\n".join(context_parts) if context_parts else None
        if context:
            log(f"📚 Контекст найден ({len(context_parts)} фрагментов)")

        try:
            # Увеличим таймаут до 60 секунд для CPU
            llm_ans = run_with_timeout(generate_llm_answer, 60, text, context)
            if llm_ans:
                if e5_check_enabled and context_parts:
                    if is_answer_relevant(text, llm_ans):
                        answer = llm_ans
                    else:
                        log("❌ Ответ нерелевантен, ищу в канале")
                        e5_ans = search_answer(text, top_k=1)
                        answer = f"🔍 Нашёл в канале:\n\n{e5_ans[0]}" if e5_ans else "🤖 Не уверен. Попробуй перефразировать."
                else:
                    answer = llm_ans
            else:
                e5_ans = search_answer(text, top_k=1)
                answer = f"🔍 Нашёл в канале:\n\n{e5_ans[0]}" if e5_ans else "🤖 Не уверен. Попробуй перефразировать."
        except Exception as e:
            log(f"Ошибка: {e}")
            e5_ans = search_answer(text, top_k=1)
            answer = f"🔍 Нашёл в канале:\n\n{e5_ans[0]}" if e5_ans else "🤖 Ошибка, попробуй позже."

        bot.send_message(message.chat.id, answer, reply_markup=main_menu())

    save_user(user)

# ===== УВЕДОМЛЕНИЕ О ЗАПУСКЕ =====
def notify_admin_start():
    try:
        bot.send_message(ADMIN_ID, "🚀 Бот запущен на Render (CPU)")
    except Exception as e:
        log(f"Не удалось отправить уведомление: {e}")

# ===== HEALTH CHECK (FLASK) =====
app = Flask(__name__)

@app.route('/')
def home():
    return "Yuki bot is running", 200

def run_flask():
    app.run(host='0.0.0.0', port=10000, debug=False, use_reloader=False)

threading.Thread(target=run_flask, daemon=True).start()
log("🌐 Flask сервер запущен на порту 10000")

# ===== KEEPALIVE (не нужен, но можно оставить для логов) =====
def keep_alive():
    cnt = 0
    while True:
        time.sleep(60)
        cnt += 1
        log(f"💓 KeepAlive #{cnt}")

threading.Thread(target=keep_alive, daemon=True).start()
log("💪 KeepAlive запущен")

# ===== ЗАПУСК =====
if __name__ == "__main__":
    log("="*50)
    log("🚀 ЮКИ ШИРАКАВА — CPU ВЕРСИЯ ДЛЯ RENDER")
    log(f"📚 Уроков: {len(lessons)}")
    log(f"📄 В корпусе: {len(corpus_texts)}")
    log(f"🔍 Проверка E5: {'вкл' if e5_check_enabled else 'выкл'}")
    log("="*50)

    notify_admin_start()

    while True:
        try:
            bot.polling(none_stop=True, interval=1, timeout=30)
        except Exception as e:
            error_msg = f"⚠️ Бот упал с ошибкой: {e}"
            log(error_msg)
            try:
                bot.send_message(ADMIN_ID, error_msg)
            except:
                pass
            log("🔄 Перезапуск через 10 секунд...")
            time.sleep(10)
