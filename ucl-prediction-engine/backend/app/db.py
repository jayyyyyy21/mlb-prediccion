"""Esquema de base de datos (SQLAlchemy 2). Funciona con PostgreSQL o SQLite."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import DATA_DIR, settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)  # id ESPN
    name: Mapped[str] = mapped_column(String(120))
    short_name: Mapped[str | None] = mapped_column(String(60))
    abbreviation: Mapped[str | None] = mapped_column(String(10))
    logo: Mapped[str | None] = mapped_column(String(300))
    color: Mapped[str | None] = mapped_column(String(10))
    domestic_league: Mapped[str | None] = mapped_column(String(40))
    in_ucl: Mapped[bool] = mapped_column(Boolean, default=False)


class Match(Base):
    __tablename__ = "matches"
    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # id evento ESPN
    league: Mapped[str] = mapped_column(String(40), index=True)
    season: Mapped[str | None] = mapped_column(String(60))
    phase: Mapped[str | None] = mapped_column(String(60))
    kickoff: Mapped[datetime] = mapped_column(DateTime, index=True)
    status: Mapped[str] = mapped_column(String(40))
    completed: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    home_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    away_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer)
    away_goals: Mapped[int | None] = mapped_column(Integer)
    neutral: Mapped[bool] = mapped_column(Boolean, default=False)
    venue: Mapped[str | None] = mapped_column(String(160))
    city: Mapped[str | None] = mapped_column(String(80))
    country: Mapped[str | None] = mapped_column(String(80))
    # Cuotas verificadas (proveedor que publica ESPN). JSON con decimales.
    odds: Mapped[dict | None] = mapped_column(JSON)
    detail_fetched: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class TeamMatchStats(Base):
    """Estadísticas observadas de un equipo en un partido (una fila por equipo y partido)."""

    __tablename__ = "team_match_stats"
    __table_args__ = (UniqueConstraint("match_id", "team_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    is_home: Mapped[bool] = mapped_column(Boolean)
    goals: Mapped[int | None] = mapped_column(Integer)
    shots: Mapped[float | None] = mapped_column(Float)
    shots_on_target: Mapped[float | None] = mapped_column(Float)
    corners: Mapped[float | None] = mapped_column(Float)
    possession: Mapped[float | None] = mapped_column(Float)
    fouls: Mapped[float | None] = mapped_column(Float)
    # Del boxscore detallado (puede faltar)
    passes: Mapped[float | None] = mapped_column(Float)
    accurate_passes: Mapped[float | None] = mapped_column(Float)
    crosses: Mapped[float | None] = mapped_column(Float)
    long_balls: Mapped[float | None] = mapped_column(Float)
    tackles: Mapped[float | None] = mapped_column(Float)
    interceptions: Mapped[float | None] = mapped_column(Float)
    clearances: Mapped[float | None] = mapped_column(Float)
    blocked_shots: Mapped[float | None] = mapped_column(Float)
    offsides: Mapped[float | None] = mapped_column(Float)
    saves: Mapped[float | None] = mapped_column(Float)
    penalty_goals: Mapped[float | None] = mapped_column(Float)
    red_cards: Mapped[float | None] = mapped_column(Float)
    formation: Mapped[str | None] = mapped_column(String(20))


class Appearance(Base):
    """Alineación por partido (titular / suplente) con goles y asistencias."""

    __tablename__ = "appearances"
    __table_args__ = (UniqueConstraint("match_id", "player_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    player_id: Mapped[str] = mapped_column(String(20), index=True)
    player_name: Mapped[str] = mapped_column(String(120))
    position: Mapped[str | None] = mapped_column(String(10))
    starter: Mapped[bool] = mapped_column(Boolean)
    subbed_in: Mapped[bool] = mapped_column(Boolean, default=False)
    goals: Mapped[float] = mapped_column(Float, default=0)
    assists: Mapped[float] = mapped_column(Float, default=0)
    shots: Mapped[float] = mapped_column(Float, default=0)
    shots_on_target: Mapped[float] = mapped_column(Float, default=0)


class Availability(Base):
    """Bajas (lesiones/suspensiones). status: confirmed | probable | rumor."""

    __tablename__ = "availability"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    team_id: Mapped[str] = mapped_column(ForeignKey("teams.id"), index=True)
    match_id: Mapped[str | None] = mapped_column(String(20), index=True)
    player_name: Mapped[str] = mapped_column(String(120))
    player_id: Mapped[str | None] = mapped_column(String(20))
    position: Mapped[str | None] = mapped_column(String(40))
    reason: Mapped[str | None] = mapped_column(String(300))
    status: Mapped[str] = mapped_column(String(20))
    source: Mapped[str] = mapped_column(String(80))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Weather(Base):
    __tablename__ = "weather"
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), primary_key=True)
    temperature_c: Mapped[float | None] = mapped_column(Float)
    precipitation_mm: Mapped[float | None] = mapped_column(Float)
    wind_kmh: Mapped[float | None] = mapped_column(Float)
    elevation_m: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(80))
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Prediction(Base):
    """Historial de pronósticos: cada recálculo se guarda con su fecha y motivo."""

    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    match_id: Mapped[str] = mapped_column(ForeignKey("matches.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
    model_version: Mapped[str] = mapped_column(String(40))
    trigger: Mapped[str] = mapped_column(String(120))
    inputs_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    model_version: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict] = mapped_column(JSON)


class ModelState(Base):
    """Parámetros ajustados (hiperparámetros elegidos por validación, etc.)."""

    __tablename__ = "model_state"
    key: Mapped[str] = mapped_column(String(60), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class FetchLog(Base):
    """Registro de cada consulta a fuentes externas (transparencia)."""

    __tablename__ = "fetch_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(60))
    url: Mapped[str] = mapped_column(Text)
    status: Mapped[int | None] = mapped_column(Integer)
    ok: Mapped[bool] = mapped_column(Boolean)
    error: Mapped[str | None] = mapped_column(Text)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


def make_engine(url: str | None = None):
    url = url or settings.database_url
    if url.startswith("sqlite"):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return create_engine(url, connect_args={"check_same_thread": False})
    return create_engine(url, pool_pre_ping=True)


engine = make_engine()
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db(eng=None) -> None:
    eng = eng or engine
    Base.metadata.create_all(eng)
    # Migración ligera: añade columnas nuevas (nullable) a tablas ya existentes.
    from sqlalchemy import inspect, text
    insp = inspect(eng)
    for table in Base.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have and col.nullable:
                with eng.begin() as conn:
                    conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN {col.name} '
                                      f'{col.type.compile(eng.dialect)}'))
