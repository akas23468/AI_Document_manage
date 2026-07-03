-- Create document_approvals table
CREATE TABLE IF NOT EXISTS document_approvals (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID,
  doc_name TEXT NOT NULL,
  old_version TEXT DEFAULT 'v0.0',
  proposed_version TEXT DEFAULT 'v1.0',
  old_content TEXT,
  new_content TEXT,
  file_path TEXT,
  file_name TEXT,
  file_type TEXT,
  file_size BIGINT,
  submitted_by TEXT,
  status TEXT DEFAULT 'Pending',
  ai_score INTEGER,
  ai_decision TEXT,
  ai_reasoning TEXT,
  decided_by TEXT,
  decided_at TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE SET NULL
);

-- Create documents table if it doesn't exist
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  icon TEXT,
  version TEXT DEFAULT 'v1.0',
  status TEXT DEFAULT 'Active',
  summary TEXT,
  content TEXT,
  file_path TEXT,
  file_name TEXT,
  file_type TEXT,
  file_size BIGINT,
  uploaded_by TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create document_history table if it doesn't exist
CREATE TABLE IF NOT EXISTS document_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  document_id UUID NOT NULL,
  version TEXT,
  updated_by TEXT,
  changes TEXT,
  content_snapshot TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create activity_log table if it doesn't exist
CREATE TABLE IF NOT EXISTS activity_log (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  actor TEXT,
  action TEXT,
  target TEXT,
  details TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create maintenance_tickets table if it doesn't exist
CREATE TABLE IF NOT EXISTS maintenance_tickets (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  ticket_id TEXT UNIQUE,
  system_name TEXT,
  issue_type TEXT,
  severity TEXT,
  description TEXT,
  reported_by TEXT,
  status TEXT DEFAULT 'Open',
  resolution_notes TEXT,
  resolved_by TEXT,
  file_path TEXT,
  file_name TEXT,
  file_type TEXT,
  file_size BIGINT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  resolved_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create users table if it doesn't exist
CREATE TABLE IF NOT EXISTS users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  employee_id TEXT UNIQUE NOT NULL,
  name TEXT,
  email TEXT UNIQUE,
  department TEXT,
  designation TEXT,
  role TEXT,
  password_hash TEXT,
  twofa_enabled BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create ai_query_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS ai_query_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  question TEXT,
  answer TEXT,
  sources TEXT,
  ai_model TEXT,
  queried_by TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_document_approvals_status ON document_approvals(status);
CREATE INDEX IF NOT EXISTS idx_document_approvals_submitted_by ON document_approvals(submitted_by);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_activity_log_actor ON activity_log(actor);
CREATE INDEX IF NOT EXISTS idx_maintenance_tickets_status ON maintenance_tickets(status);

2
alter table documents
  add column if not exists icon      text,
  add column if not exists file_path text,
  add column if not exists file_name text,
  add column if not exists file_type text,
  add column if not exists file_size bigint;

-- Also needed by document_history / approvals endpoints — safe to run even if already present
alter table document_history
  add column if not exists content_snapshot text;

create table if not exists document_approvals (
  id                uuid primary key default gen_random_uuid(),
  document_id       uuid references documents(id) on delete set null,
  doc_name          text not null,
  old_version       text,
  proposed_version  text,
  old_content       text,
  new_content       text,
  file_path         text,
  file_name         text,
  file_type         text,
  file_size         bigint,
  submitted_by      text,
  status            text default 'Pending',
  created_at        timestamptz default now(),
  decided_at        timestamptz,
  decided_by        text
);

create table if not exists activity_log (
  id          uuid primary key default gen_random_uuid(),
  actor       text,
  action      text,
  target      text,
  details     text,
  created_at  timestamptz default now()
);

insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

notify pgrst, 'reload schema';
3
alter table maintenance_tickets
  add column if not exists file_path    text,
  add column if not exists file_name    text,
  add column if not exists file_type    text,
  add column if not exists file_size    bigint,
  add column if not exists resolved_by  text;

  4
  -- ============================================================
-- OnlyTech Portal — Supabase Schema
-- Run this in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- 1) USERS (auth: employees + admins)
create table if not exists users (
  employee_id     text primary key,
  name            text not null,
  email           text not null unique,
  department      text,
  designation     text,
  role            text not null default 'employee', -- 'employee' or 'admin'
  password_hash   text not null,
  twofa_enabled   boolean default false,
  created_at      timestamptz default now()
);

-- 2) DOCUMENTS (knowledge base files)
create table if not exists documents (
  id           uuid primary key default gen_random_uuid(),
  name         text not null,
  icon         text,
  version      text default 'v1.0',
  status       text default 'Active',   -- Active / Draft / Updated
  summary      text,
  content      text,
  file_path    text,
  file_name    text,
  file_type    text,
  file_size    bigint,
  uploaded_by  text,
  created_at   timestamptz default now(),
  updated_at   timestamptz default now()
);

-- 3) DOCUMENT HISTORY (version timeline per document)
create table if not exists document_history (
  id                uuid primary key default gen_random_uuid(),
  document_id       uuid references documents(id) on delete cascade,
  version           text,
  updated_by        text,
  changes           text,
  content_snapshot  text,
  created_at        timestamptz default now()
);

-- 4) DOCUMENT APPROVALS (employee-submitted edits awaiting admin review)
create table if not exists document_approvals (
  id                uuid primary key default gen_random_uuid(),
  document_id       uuid references documents(id) on delete set null,
  doc_name          text not null,
  old_version       text,
  proposed_version  text,
  old_content       text,
  new_content       text,
  file_path         text,
  file_name         text,
  file_type         text,
  file_size         bigint,
  submitted_by      text,
  status            text default 'Pending', -- Pending / Approved / Rejected
  created_at        timestamptz default now(),
  decided_at        timestamptz,
  decided_by        text
);

-- 5) MAINTENANCE TICKETS (IT incidents)
create table if not exists maintenance_tickets (
  id                uuid primary key default gen_random_uuid(),
  ticket_id         text unique,
  system_name       text not null,
  issue_type        text,
  severity          text,        -- e.g. P1, P2, P3
  description       text,
  reported_by       text,
  status            text default 'Open', -- Open / Resolved
  resolution_notes  text,
  resolved_by       text,
  created_at        timestamptz default now(),
  resolved_at       timestamptz
);

-- 6) ACTIVITY LOG (drives the Recent Activity feed)
create table if not exists activity_log (
  id          uuid primary key default gen_random_uuid(),
  actor       text,
  action      text,   -- uploaded_document / proposed_edit / approved_edit / etc.
  target      text,
  details     text,
  created_at  timestamptz default now()
);

-- 7) AI QUERY LOGS (PlantMind AI usage, drives Top Queries chart)
create table if not exists ai_query_logs (
  id          uuid primary key default gen_random_uuid(),
  question    text,
  mode        text,
  answer      text,
  sources     text,
  asked_by    text,
  created_at  timestamptz default now()
);

-- ============================================================
-- Helpful indexes
-- ============================================================
create index if not exists idx_doc_history_docid on document_history(document_id);
create index if not exists idx_approvals_status on document_approvals(status);
create index if not exists idx_tickets_status on maintenance_tickets(status);
create index if not exists idx_activity_created on activity_log(created_at desc);

-- ============================================================
-- Storage bucket for uploaded files (documents + approval attachments)
-- ============================================================
insert into storage.buckets (id, name, public)
values ('documents', 'documents', false)
on conflict (id) do nothing;

5
-- Run this once in your Supabase project's SQL Editor
-- (Project -> SQL Editor -> New Query -> paste -> Run)

create extension if not exists "uuid-ossp";

-- 1. Users (login / signup)
create table if not exists users (
    employee_id   text primary key,
    name          text not null,
    email         text not null,
    department    text not null,
    designation   text,
    role          text not null,
    password_hash text not null,
    twofa_enabled boolean default false,
    created_at    timestamptz default now()
);

-- 2. Documents (knowledge base)
create table if not exists documents (
    id           uuid primary key default uuid_generate_v4(),
    name         text not null,
    version      text default 'v1.0',
    status       text default 'Active',
    summary      text,
    content      text,
    uploaded_by  text,
    created_at   timestamptz default now(),
    updated_at   timestamptz default now()
);

-- 3. Document version history (audit trail)
create table if not exists document_history (
    id          uuid primary key default uuid_generate_v4(),
    document_id uuid references documents(id) on delete cascade,
    version     text,
    updated_by  text,
    changes     text,
    created_at  timestamptz default now()
);

-- 4. Maintenance / IT incident tickets
create table if not exists maintenance_tickets (
    id                uuid primary key default uuid_generate_v4(),
    ticket_id         text unique,
    system_name       text,
    issue_type        text,
    severity          text,
    description       text,
    reported_by       text,
    status            text default 'Open',
    resolution_notes  text,
    created_at        timestamptz default now(),
    resolved_at       timestamptz
);

-- 5. AI query logs (optional analytics for PlantMind AI)
create table if not exists ai_query_logs (
    id         uuid primary key default uuid_generate_v4(),
    question   text,
    mode       text,
    answer     text,
    sources    text,
    created_at timestamptz default now()
);