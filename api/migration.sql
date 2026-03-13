-- ============================================================
-- PlusOne Backend SQL Migrations
-- Run once in the Supabase SQL Editor
-- ============================================================

-- ─── 1. ml_models table (DC Model Supabase Persistence) ───────────────────────
-- Stores the serialised DCPredictor (joblib bytes) so it survives Railway redeploys.

CREATE TABLE IF NOT EXISTS ml_models (
    id             SERIAL PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE DEFAULT 'dc_model',
    model_bytes    BYTEA NOT NULL,
    n_samples      INT,
    train_accuracy FLOAT,
    cv_accuracy    FLOAT,
    n_features     INT,
    created_at     TIMESTAMPTZ DEFAULT NOW()
);


-- ─── 2. prediction_log UNIQUE constraint on match_id ──────────────────────────
-- Required for ON CONFLICT (match_id) DO NOTHING deduplication in /upcoming
-- Only run this if the constraint doesn't already exist:

ALTER TABLE prediction_log
  ADD CONSTRAINT prediction_log_match_id_unique UNIQUE (match_id);


-- ─── 3. prediction_log table (if not created yet) ─────────────────────────────
-- Run this block only if prediction_log doesn't exist in your database.

CREATE TABLE IF NOT EXISTS prediction_log (
    id               SERIAL PRIMARY KEY,
    match_id         INT UNIQUE,           -- references matches(id)
    home_team        TEXT NOT NULL,
    away_team        TEXT NOT NULL,
    league           TEXT,
    match_date       DATE,
    predicted        TEXT NOT NULL,        -- 'Home Win' / 'Draw' / 'Away Win'
    confidence       TEXT,                 -- 'High' / 'Medium' / 'Low'
    confidence_score FLOAT,
    home_win_prob    FLOAT,
    draw_prob        FLOAT,
    away_win_prob    FLOAT,
    actual           TEXT,                 -- filled in by evaluate after match
    correct          BOOLEAN,              -- filled in by evaluate
    evaluated_at     TIMESTAMPTZ,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS prediction_log_match_id_idx ON prediction_log(match_id);
CREATE INDEX IF NOT EXISTS prediction_log_correct_idx  ON prediction_log(correct);
CREATE INDEX IF NOT EXISTS prediction_log_date_idx     ON prediction_log(match_date);
 
 
 -- ─── 4. teams table logo_url column ─────────────────────────────────────────
 -- Add column for storing team logos scraped from FBref
 
 ALTER TABLE teams
   ADD COLUMN IF NOT EXISTS logo_url TEXT;
