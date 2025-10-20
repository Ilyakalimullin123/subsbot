"""
Telegram бот для напоминаний о подписках (ежемесячные списания)

Функции:
- /start, /help — краткая инструкция
- /add — мастером собирает 4 поля: дата начала, день списания, сумма, карта (+ необязательное имя)
- /list — показывает ваши подписки
- /delete — удалить подписку по списку
- Авто-напоминание: за 1 день до списания (с учётом часового пояса)

Технологии:
- Python 3.12+
- aiogram 3.x
- APScheduler (фоновые джобы)
- SQLite локально (sqlite+aiosqlite:///subscriptions.sqlite3) или Postgres на проде (postgresql+asyncpg://...)

Развёртывание (локально):
1) Создайте бота у @BotFather, получите BOT_TOKEN
2) Создайте .env с переменными ниже
3) Установите зависимости (см. requirements.txt)
4) Запустите: python bot123.py

Переменные окружения (.env):
BOT_TOKEN=123456:ABC...
TZ_DEFAULT=Asia/Almaty
# Локально:
DB_URL=sqlite+aiosqlite:///subscriptions.sqlite3
# На Railway/Render/Neon (пример):
# DB_URL=postgresql+asyncpg://USER:PASSWORD@HOST/DBNAME

Хостинг (кратко):
- Railway/Render/Fly.io/Heroku — запустите один worker-процесс (python bot123.py)
- В переменных окружения задайте BOT_TOKEN, TZ_DEFAULT, DB_URL
- Для продакшена используйте Postgres

Метрики:
- Логи уже есть (logging). Для продакшена можно подключить Amplitude/PostHog через HTTP API (TODO)
"""
from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton
from aiogram.utils.formatting import as_marked_section, Bold
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

# SQLAlchemy (универсальная работа с SQLite и Postgres)
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from urllib.parse import urlsplit, urlunsplit, quote

# -------------------- Конфиг --------------------
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
TZ_DEFAULT = os.getenv("TZ_DEFAULT", "Asia/Almaty")
DB_URL_ENV = os.getenv("DB_URL", "sqlite+aiosqlite:///subscriptions.sqlite3")

if not BOT_TOKEN:
    raise RuntimeError("Не задан BOT_TOKEN в окружении")

# -------------------- Логирование --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("subsbot")

# -------------------- Утилиты дат --------------------
def last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        nxt = date(year + 1, 1, 1)
    else:
        nxt = date(year, month + 1, 1)
    return (nxt - timedelta(days=1)).day


def normalize_day(year: int, month: int, day: int) -> int:
    return min(day, last_day_of_month(year, month))


def next_charge_date(ref: date, charge_day: int) -> date:
    """Ближайшая дата списания (>= ref) по правилу "каждый месяц в charge_day"."""
    d = normalize_day(ref.year, ref.month, charge_day)
    candidate = date(ref.year, ref.month, d)
    if candidate >= ref:
        return candidate
    # следующий месяц
    year = ref.year + (1 if ref.month == 12 else 0)
    month = 1 if ref.month == 12 else ref.month + 1
    d = normalize_day(year, month, charge_day)
    return date(year, month, d)


# -------------------- Модель --------------------
@dataclass
class Subscription:
    id: int | None
    user_id: int
    name: str | None
    start_date: date
    charge_day: int  # 1..31
    amount: float
    card: str
    tz: str  # IANA timezone
    created_at: datetime | None


# -------------------- Хранилище --------------------
# Универсальный слой БД: локально SQLite-файл, в облаке — Postgres.
# Примеры URL:
#  - sqlite+aiosqlite:///subscriptions.sqlite3 (локально)
#  - postgresql+asyncpg://user:pass@host:port/dbname (облако)

CREATE_SQL_PG = """
CREATE TABLE IF NOT EXISTS subscriptions (
    id SERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    name TEXT,
    start_date TEXT NOT NULL,
    charge_day INTEGER NOT NULL,
    amount DOUBLE PRECISION NOT NULL,
    card TEXT NOT NULL,
    tz TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_subs_user ON subscriptions(user_id);
"""

engine: AsyncEngine | None = None


def _normalize_db_url() -> str:
    # Поддержим старое значение: DB_URL=sqlite+file и подчистим asyncpg-url от параметров
    url = DB_URL_ENV.strip() if DB_URL_ENV else ""
    if url in ("", "sqlite+file"):
        return "sqlite+aiosqlite:///subscriptions.sqlite3"
    if url.startswith("postgresql+asyncpg://"):
        # 1) убираем query (?sslmode=..., channel_binding=...) — asyncpg их не понимает
        scheme, netloc, path, query, frag = urlsplit(url)
        if query:
            url = urlunsplit((scheme, netloc, path, "", ""))
        # 2) если в пароле есть спецсимволы — percent-encode
        scheme, netloc, path, query, frag = urlsplit(url)
        if "@" in netloc and ":" in netloc.split("@", 1)[0]:
            creds, host = netloc.split("@", 1)
            user, pwd = creds.split(":", 1)
            enc_pwd = quote(pwd, safe="")
            netloc = f"{user}:{enc_pwd}@{host}"
            url = urlunsplit((scheme, netloc, path, "", ""))
    return url


async def db_init():
    """Создаёт таблицы при необходимости и инициализирует движок."""
    global engine
    url = _normalize_db_url()
    engine = create_async_engine(url, echo=False, pool_pre_ping=True)
    async with engine.begin() as conn:
        if url.startswith("sqlite"):
            # Совместимый DDL для SQLite
            await conn.exec_driver_sql(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    name TEXT,
                    start_date TEXT NOT NULL,
                    charge_day INTEGER NOT NULL,
                    amount REAL NOT NULL,
                    card TEXT NOT NULL,
                    tz TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            await conn.exec_driver_sql(
                "CREATE INDEX IF NOT EXISTS idx_subs_user ON subscriptions(user_id);"
            )
        else:
            # asyncpg не принимает несколько команд одним prepared statement — выполним по отдельности
            for _stmt in CREATE_SQL_PG.strip().split(';'):
                _s = _stmt.strip()
                if _s:
                    await conn.exec_driver_sql(_s + ';')


async def db_add(sub: Subscription) -> int:
    assert engine is not None
    url = _normalize_db_url()
    async with engine.begin() as conn:
        if url.startswith("sqlite"):
            await conn.exec_driver_sql(
                """
                INSERT INTO subscriptions (user_id, name, start_date, charge_day, amount, card, tz, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sub.user_id,
                    sub.name,
                    sub.start_date.isoformat(),
                    sub.charge_day,
                    sub.amount,
                    sub.card,
                    sub.tz,
                    datetime.now(ZoneInfo("UTC")).isoformat(),
                ),
            )
            row = await conn.exec_driver_sql("SELECT last_insert_rowid()")
            last_id = (await row.fetchone())[0]
            return int(last_id)
        else:
            res = await conn.execute(
                text(
                    """
                    INSERT INTO subscriptions (user_id, name, start_date, charge_day, amount, card, tz, created_at)
                    VALUES (:user_id, :name, :start_date, :charge_day, :amount, :card, :tz, :created_at)
                    RETURNING id
                    """
                ),
                {
                    "user_id": sub.user_id,
                    "name": sub.name,
                    "start_date": sub.start_date.isoformat(),
                    "charge_day": sub.charge_day,
                    "amount": sub.amount,
                    "card": sub.card,
                    "tz": sub.tz,
                    "created_at": datetime.now(ZoneInfo("UTC")).isoformat(),
                },
            )
            return int(res.scalar_one())


async def db_list(user_id: int) -> list[Subscription]:
    assert engine is not None
    url = _normalize_db_url()
    async with engine.begin() as conn:
        if url.startswith("sqlite"):
            res = await conn.exec_driver_sql(
                "SELECT * FROM subscriptions WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            )
            rows = await res.fetchall()
            cols = [c[0] for c in res.cursor.description]
        else:
            res = await conn.execute(
                text("SELECT * FROM subscriptions WHERE user_id = :u ORDER BY created_at DESC"),
                {"u": user_id},
            )
            rows = res.fetchall()
            cols = res.keys()

    def row_to_sub(r):
        d = dict(zip(cols, r))
        return Subscription(
            id=d["id"],
            user_id=d["user_id"],
            name=d["name"],
            start_date=date.fromisoformat(d["start_date"]),
            charge_day=d["charge_day"],
            amount=d["amount"],
            card=d["card"],
            tz=d["tz"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    return [row_to_sub(r) for r in rows]


async def db_delete(user_id: int, sub_id: int) -> bool:
    assert engine is not None
    url = _normalize_db_url()
    async with engine.begin() as conn:
        if url.startswith("sqlite"):
            res = await conn.exec_driver_sql(
                "DELETE FROM subscriptions WHERE user_id = ? AND id = ?",
                (user_id, sub_id),
            )
            return res.rowcount > 0
        else:
            res = await conn.execute(
                text("DELETE FROM subscriptions WHERE user_id = :u AND id = :i"),
                {"u": user_id, "i": sub_id},
            )
            return res.rowcount > 0


async def db_for_notification(target_date: date) -> list[Subscription]:
    assert engine is not None
    url = _normalize_db_url()
    async with engine.begin() as conn:
        if url.startswith("sqlite"):
            res = await conn.exec_driver_sql("SELECT * FROM subscriptions")
            rows = await res.fetchall()
            cols = [c[0] for c in res.cursor.description]
        else:
            res = await conn.execute(text("SELECT * FROM subscriptions"))
            rows = res.fetchall()
            cols = res.keys()

    def row_to_sub(r):
        d = dict(zip(cols, r))
        return Subscription(
            id=d["id"],
            user_id=d["user_id"],
            name=d["name"],
            start_date=date.fromisoformat(d["start_date"]),
            charge_day=d["charge_day"],
            amount=d["amount"],
            card=d["card"],
            tz=d["tz"],
            created_at=datetime.fromisoformat(d["created_at"]),
        )

    subs = [row_to_sub(r) for r in rows]
    out: list[Subscription] = []
    for s in subs:
        if s.start_date > target_date:
            continue
        due = next_charge_date(target_date, s.charge_day)
        if due == target_date:
            out.append(s)
    return out


# -------------------- FSM формы --------------------
class AddSub(StatesGroup):
    waiting_for_name = State()  # опционально
    waiting_for_start = State()
    waiting_for_charge_day = State()
    waiting_for_amount = State()
    waiting_for_card = State()


# -------------------- Бот --------------------
bot = Bot(BOT_TOKEN)
dp = Dispatcher()
scheduler = AsyncIOScheduler()

# Отображение "красивых" номеров в /list: user_id -> [real_db_id]
LAST_LIST_MAP: dict[int, list[int]] = {}

main_kb = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="➕ Добавить подписку"), KeyboardButton(text="📜 Мои подписки")],
        [KeyboardButton(text="🗑 Удалить подписку"), KeyboardButton(text="ℹ️ Помощь")],
    ],
    resize_keyboard=True,
)


@dp.message(CommandStart())
async def cmd_start(message: Message, state: FSMContext):
    tz = TZ_DEFAULT
    txt = (
        "Привет! Я напомню за 1 день до списания по твоим подпискам.\n\n"
        "Команды:\n"
        "/add — добавить подписку (мастер)\n"
        "/list — список подписок\n"
        "/delete — удалить подписку\n"
        "/help — справка\n\n"
        f"Твой часовой пояс по умолчанию: {tz}. Напоминания приходят в 10:00."
    )
    await message.answer(txt, reply_markup=main_kb)


@dp.message(Command("help"))
@dp.message(F.text.in_({"ℹ️ Помощь"}))
async def cmd_help(message: Message):
    await message.answer(
        as_marked_section(
            Bold("Как пользоваться"),
            "Нажми ➕ Добавить подписку и заполни поля.",
            "Мы пришлём напоминание за 1 день до списания.",
            "Для теста: /notify_now — отправить напоминания прямо сейчас (для подписок на завтра).",
            marker="•",
        ).as_html()
    )


@dp.message(Command("add"))
@dp.message(F.text.in_({"➕ Добавить подписку"}))
async def add_start(message: Message, state: FSMContext):
    await state.clear()
    await state.set_state(AddSub.waiting_for_name)
    await state.update_data(tz=TZ_DEFAULT)
    await message.answer(
        "Как назовём подписку? (например, Netflix). Можно пропустить — напиши '-'"
    )


@dp.message(AddSub.waiting_for_name)
async def add_name(message: Message, state: FSMContext):
    name = None if message.text.strip() == "-" else message.text.strip()
    await state.update_data(name=name)
    await state.set_state(AddSub.waiting_for_start)
    await message.answer("Укажи дату начала подписки в формате ГГГГ-ММ-ДД (например, 2025-10-01)")


@dp.message(AddSub.waiting_for_start)
async def add_start_date(message: Message, state: FSMContext):
    try:
        dt = date.fromisoformat(message.text.strip())
    except Exception:
        await message.answer("Не понял дату. Пример: 2025-10-01. Попробуй ещё раз.")
        return
    await state.update_data(start_date=dt)
    await state.set_state(AddSub.waiting_for_charge_day)
    await message.answer(
        "Укажи ДЕНЬ МЕСЯЦА списания (1..31). Если месяц короче — возьмём последний день."
    )


@dp.message(AddSub.waiting_for_charge_day)
async def add_charge_day(message: Message, state: FSMContext):
    try:
        day = int(message.text.strip())
        if not (1 <= day <= 31):
            raise ValueError
    except Exception:
        await message.answer("Нужно целое число от 1 до 31. Попробуй ещё раз.")
        return
    await state.update_data(charge_day=day)
    await state.set_state(AddSub.waiting_for_amount)
    await message.answer("Сумма списания (например, 2990.00)")


@dp.message(AddSub.waiting_for_amount)
async def add_amount(message: Message, state: FSMContext):
    try:
        amount = float(message.text.replace(",", ".").strip())
        if amount <= 0:
            raise ValueError
    except Exception:
        await message.answer("Нужно положительное число. Пример: 2990.00")
        return
    await state.update_data(amount=amount)
    await state.set_state(AddSub.waiting_for_card)
    await message.answer("С какой карты списывается? (например, Kaspi **** 1234)")


@dp.message(AddSub.waiting_for_card)
async def add_card(message: Message, state: FSMContext):
    data = await state.get_data()
    sub = Subscription(
        id=None,
        user_id=message.from_user.id,
        name=data.get("name"),
        start_date=data["start_date"],
        charge_day=data["charge_day"],
        amount=data["amount"],
        card=message.text.strip(),
        tz=data.get("tz", TZ_DEFAULT),
        created_at=None,
    )
    sub_id = await db_add(sub)
    await state.clear()
    nxt = next_charge_date(date.today(), sub.charge_day)
    await message.answer(
        (
            "Готово!\n"
            f"ID: {sub_id}\n"
            f"Название: {sub.name or '—'}\n"
            f"Старт: {sub.start_date.isoformat()}\n"
            f"День списания: {sub.charge_day} числа каждого месяца\n"
            f"Сумма: {sub.amount}\n"
            f"Карта: {sub.card}\n"
            f"Ближайшее списание: {nxt.isoformat()}\n"
        ),
        reply_markup=main_kb,
    )
    log.info("subscription_added user=%s id=%s", message.from_user.id, sub_id)
    # TODO: метрики Amplitude/PostHog


@dp.message(Command("list"))
@dp.message(F.text.in_({"📜 Мои подписки"}))
async def list_subs(message: Message):
    subs = await db_list(message.from_user.id)
    if not subs:
        await message.answer("У тебя пока нет подписок. Нажми ➕ Добавить подписку.")
        return
    # сохраняем отображение "красивого" номера -> реальный ID в БД
    LAST_LIST_MAP[message.from_user.id] = [s.id for s in subs]
    lines = []
    for i, s in enumerate(subs, start=1):
        lines.append(
            (
                f"{i}) {s.name or 'Без названия'}\n"
                f"• Старт: {s.start_date.isoformat()}\n"
                f"• День списания: {s.charge_day} числа\n"
                f"• Сумма: {s.amount}\n"
                f"• Карта: {s.card}\n"
                f"• (ID в БД: {s.id})\n"
            )
        )
    await message.answer("\n\n".join(lines))


@dp.message(Command("delete"))
@dp.message(F.text.in_({"🗑 Удалить подписку"}))
async def delete_prompt(message: Message):
    subs = await db_list(message.from_user.id)
    if not subs:
        await message.answer("Список пуст. Добавь подписку сначала.")
        return
    # Можно удалять по красивому номеру из последнего /list
    LAST_LIST_MAP[message.from_user.id] = [s.id for s in subs]
    await message.answer(
        "Напиши номер подписки из последнего списка (1, 2, 3, ...) или реальный ID из БД.\nНапример: 2"
    )


@dp.message(F.text.regexp(r"^\d+$"))
async def delete_by_id(message: Message):
    entered = int(message.text)
    user_id = message.from_user.id
    # Если есть отображение последнего /list — трактуем введённое число как порядковый номер
    mapped_id = None
    mapping = LAST_LIST_MAP.get(user_id)
    if mapping and 1 <= entered <= len(mapping):
        mapped_id = mapping[entered - 1]
    # Иначе — считаем, что это реальный ID в БД
    target_id = mapped_id or entered

    ok = await db_delete(user_id, target_id)
    if ok:
        await message.answer("Удалено.")
        log.info("subscription_deleted user=%s id=%s (entered=%s)", user_id, target_id, entered)
        # Подчищаем отображение, если удаляли по красивому номеру
        if mapping and mapped_id:
            try:
                mapping.remove(target_id)
                LAST_LIST_MAP[user_id] = mapping
            except ValueError:
                pass
    else:
        await message.answer("Не нашёл такую подписку у тебя. Отправь /list и попробуй снова.")


# -------------------- Планировщик уведомлений --------------------
@dp.message(Command("notify_now"))
async def notify_now(message: Message):
    """Ручной тест: отправляем напоминания за завтрашние списания прямо сейчас."""
    await send_tomorrow_notifications()
    await message.answer("Ок! Я отправил тест-уведомления за завтрашние списания (если такие есть).")


async def send_tomorrow_notifications():
    """Отправляем напоминания за 1 день до списания.
    Вызов планировщика делается ежедневно в 10:00 TZ_DEFAULT.
    """
    tz = ZoneInfo(TZ_DEFAULT)
    now_local = datetime.now(tz)
    target = (now_local + timedelta(days=1)).date()  # завтра
    subs = await db_for_notification(target)
    if not subs:
        log.info("no_due_tomorrow target=%s", target)
        return
    for s in subs:
        try:
            text_msg = (
                "Напоминание: завтра списание по подписке!\n\n"
                f"Название: {s.name or '—'}\n"
                f"Сумма: {s.amount}\n"
                f"Карта: {s.card}\n"
                f"Дата списания: {target.isoformat()} (день {s.charge_day})\n"
            )
            await bot.send_message(chat_id=s.user_id, text=text_msg)
            log.info("notified user=%s sub=%s date=%s", s.user_id, s.id, target)
        except Exception as e:
            log.exception("failed_to_notify user=%s sub=%s: %s", s.user_id, s.id, e)


# -------------------- Точка входа --------------------
async def on_startup():
    await db_init()
    # Планируем ежедневную задачу в 10:00 локального TZ_DEFAULT
    scheduler.add_job(
        send_tomorrow_notifications,
        CronTrigger(hour=10, minute=0, timezone=TZ_DEFAULT),
        id="notify_daily",
        replace_existing=True,
    )
    scheduler.start()
    log.info("scheduler_started TZ=%s", TZ_DEFAULT)


async def main():
    await on_startup()
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("Bot stopped")
