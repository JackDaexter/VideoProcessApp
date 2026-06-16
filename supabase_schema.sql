-- ────────────────────────────────────────────────────────────────────────────
-- Supabase Schema — VideoProcessApp Jobs Table
-- Run this in the Supabase SQL Editor to set up your database.
-- ────────────────────────────────────────────────────────────────────────────

-- Enable UUID extension (enabled by default in Supabase)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ── Jobs Table ────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.jobs (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type        TEXT NOT NULL CHECK (type IN ('clip_generator', 'ai_shorts', 'youtube_studio')),
    status      TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    input_url   TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    options     JSONB NOT NULL DEFAULT '{}',
    result      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ── Indexes ───────────────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_jobs_status   ON public.jobs (status);
CREATE INDEX IF NOT EXISTS idx_jobs_type     ON public.jobs (type);
CREATE INDEX IF NOT EXISTS idx_jobs_created  ON public.jobs (created_at DESC);

-- ── Auto-update updated_at ────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS jobs_updated_at ON public.jobs;
CREATE TRIGGER jobs_updated_at
    BEFORE UPDATE ON public.jobs
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ── Row Level Security ────────────────────────────────────────────────────────
-- The backend uses the service role key which bypasses RLS.
-- Enable RLS to prevent direct client access.

ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- No public policies — only service role can access.
-- Add policies here if you add user-scoped access later.

-- ── Example Queries ───────────────────────────────────────────────────────────

-- Check all jobs:
-- SELECT id, type, status, created_at FROM public.jobs ORDER BY created_at DESC;

-- Check pending jobs:
-- SELECT * FROM public.jobs WHERE status = 'pending';
