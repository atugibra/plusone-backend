-- ============================================================
-- PlusOne Database Migration
-- Run this ONCE in the Supabase SQL Editor
-- ============================================================

-- 1. Add updated_at to matches table (for incremental sync tracking)
ALTER TABLE matches
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Backfill existing rows with match_date as a reasonable updated_at value
UPDATE matches
  SET updated_at = COALESCE(match_date::timestamptz, NOW())
  WHERE updated_at IS NULL;

-- Index for fast incremental sync queries
CREATE INDEX IF NOT EXISTS matches_updated_at_idx ON matches(updated_at DESC);

-- 2. Verify prediction_log has all required columns
-- (Already confirmed — this is just a safety check)
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'home_win_prob'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN home_win_prob FLOAT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'draw_prob'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN draw_prob FLOAT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'away_win_prob'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN away_win_prob FLOAT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'home_xg'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN home_xg FLOAT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'away_xg'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN away_xg FLOAT;
  END IF;
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_name = 'prediction_log' AND column_name = 'predicted_score'
  ) THEN
    ALTER TABLE prediction_log ADD COLUMN predicted_score TEXT;
  END IF;
END $$;

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS prediction_log_match_id_idx  ON prediction_log(match_id);
CREATE INDEX IF NOT EXISTS prediction_log_correct_idx   ON prediction_log(correct);
CREATE INDEX IF NOT EXISTS prediction_log_date_idx      ON prediction_log(match_date);
CREATE INDEX IF NOT EXISTS prediction_log_created_idx   ON prediction_log(created_at DESC);

-- 3. Verify the matches table now has updated_at
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'matches' AND column_name = 'updated_at';
-- Should return 1 row: updated_at | timestamp with time zone
